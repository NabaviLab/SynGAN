import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    """
    A single fusion block for the Transformer-based decoder:

        Z1     = LN(Z_latent)
        Z_self = MSA(Z1, Z1, Z1) + Z_latent
        Z2     = LN(Z_self)
        K,V    = Proj(F_fused)
        Z_ctx  = CrossAttn(Q=Z2, K, V) + Z_self
        Z3     = LN(Z_ctx)
        Z_out  = PFFN(Z3) + Z_ctx
    """
    def __init__(self, D: int, n_heads: int, d_ff: int, twoD: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=D, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(D)
        self.k_proj = nn.Linear(twoD, D)
        self.v_proj = nn.Linear(twoD, D)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=D, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ln3 = nn.LayerNorm(D)
        self.ffn = nn.Sequential(
            nn.Linear(D, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, D),
        )
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.k_proj.weight); nn.init.zeros_(self.k_proj.bias)
        nn.init.xavier_uniform_(self.v_proj.weight); nn.init.zeros_(self.v_proj.bias)

    def forward(self, Z_latent: torch.Tensor, F_fused: torch.Tensor) -> torch.Tensor:
        # Self-attention over latent tokens
        Z1 = self.ln1(Z_latent)
        A_self, _ = self.self_attn(Z1, Z1, Z1)                  # (B, N, D)
        Z_self = Z_latent + self.drop(A_self)

        # Cross-attention with encoder tokens (fused prior+current)
        Z2 = self.ln2(Z_self)
        K = self.k_proj(F_fused)                                # (B, N, D)
        V = self.v_proj(F_fused)                                # (B, N, D)
        A_cross, _ = self.cross_attn(Z2, K, V)
        Z_ctx = Z_self + self.drop(A_cross)

        # Position-wise FFN
        Z3 = self.ln3(Z_ctx)
        F = self.ffn(Z3)
        Z_out = Z_ctx + self.drop(F)
        return Z_out


class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder that integrates latent z and fused encoder features:

    Steps:
      1) Project z ∈ R^{d} -> tokens Z_latent ∈ R^{N×D}
      2) DecoderBlock (MSA + MCA + PFFN)
      3) Heads map tokens → patches (P^2), then Fold to images:
           - T_gen   : tumor region (no activation)
           - T_hat   : tumor probability map (sigmoid)
           - M_blend : soft blending mask in [0,1] (sigmoid)
    """
    def __init__(self, N: int, P: int, D: int, twoD: int, d_latent: int,
                 H: int, W: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.N, self.P, self.D, self.H, self.W = N, P, D, H, W

        # Latent z → tokens
        self.z_to_tokens = nn.Linear(d_latent, N * D)
        nn.init.xavier_uniform_(self.z_to_tokens.weight); nn.init.zeros_(self.z_to_tokens.bias)

        # One fusion block (can be stacked if desired; paper describes the sequence once)
        self.block = DecoderBlock(D=D, n_heads=n_heads, d_ff=d_ff, twoD=twoD, dropout=dropout)

        # Token → patch pixels (1 channel)
        self.token_to_pixels_T = nn.Linear(D, P * P)  # tumor image head
        self.token_to_pixels_S = nn.Linear(D, P * P)  # tumor prob head
        self.token_to_pixels_M = nn.Linear(D, P * P)  # blend mask head
        for m in [self.token_to_pixels_T, self.token_to_pixels_S, self.token_to_pixels_M]:
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

        # Fold patches back to H×W
        self.fold = nn.Fold(output_size=(H, W), kernel_size=P, stride=P)

    def _tokens_to_image(self, tokens: torch.Tensor, head: nn.Linear, activation: str = "none") -> torch.Tensor:
        """
        tokens: (B, N, D) → (B, N, P^2) → (B, P^2, N) → Fold → (B, 1, H, W)
        """
        patches = head(tokens)                # (B, N, P^2)
        patches = patches.transpose(1, 2)     # (B, P^2, N)
        img = self.fold(patches)              # (B, 1, H, W)
        if activation == "sigmoid":
            img = torch.sigmoid(img)
        return img

    def forward(self, z: torch.Tensor, F_fused: torch.Tensor):
        """
        Inputs:
          z        : (B, d_latent)
          F_fused  : (B, N, 2D)

        Returns:
          Z_out      : (B, N, D) decoder tokens
          T_gen      : (B, 1, H, W)
          T_hat      : (B, 1, H, W) ∈ [0,1]
          M_blend    : (B, 1, H, W) ∈ [0,1]
        """
        B = z.size(0)
        Z_latent = self.z_to_tokens(z).view(B, self.N, self.D)  # (B, N, D)

        Z_out = self.block(Z_latent, F_fused)                   # (B, N, D)

        T_gen   = self._tokens_to_image(Z_out, self.token_to_pixels_T, activation="none")
        T_hat   = self._tokens_to_image(Z_out, self.token_to_pixels_S, activation="sigmoid")
        M_blend = self._tokens_to_image(Z_out, self.token_to_pixels_M, activation="sigmoid")

        return Z_out, T_gen, T_hat, M_blend