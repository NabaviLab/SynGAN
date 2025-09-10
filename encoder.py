import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    Implements: Z0 = X_patches W_E + E_pos
    - Non-overlapping P×P patches via Unfold
    - Linear projection from P^2 * in_ch -> D
    - Learnable positional encoding E_pos ∈ R^{N×D}
    """
    def __init__(self, in_ch: int, P: int, D: int, H: int, W: int):
        super().__init__()
        assert H % P == 0 and W % P == 0, "H and W must be divisible by P"
        self.P, self.D, self.H, self.W = P, D, H, W
        self.N = (H // P) * (W // P)

        self.unfold = nn.Unfold(kernel_size=P, stride=P)     # (B, P^2*in_ch, N)
        self.W_E    = nn.Linear(P * P * in_ch, D)            # learnable patch embedding
        self.pos    = nn.Parameter(torch.zeros(1, self.N, D))# E_pos

        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.xavier_uniform_(self.W_E.weight); nn.init.zeros_(self.W_E.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, H, W)
        patches = self.unfold(x).transpose(1, 2)  # (B, N, P^2*in_ch)
        Z0 = self.W_E(patches) + self.pos         # (B, N, D)
        return Z0


class TransformerEncoderLayer(nn.Module):
    """
    One encoder layer (pre-LN):
        Z'  = MSA(LN(Z)) + Z
        Z'' = PFFN(LN(Z')) + Z'
    """
    def __init__(self, D: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.msa = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(D)
        self.ffn = nn.Sequential(
            nn.Linear(D, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, D),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        Z1 = self.ln1(Z)
        A, _ = self.msa(Z1, Z1, Z1)  # self-attention
        Z = Z + self.drop(A)

        Z2 = self.ln2(Z)
        F = self.ffn(Z2)
        Z = Z + self.drop(F)
        return Z


class EncoderE(nn.Module):
    """
    Complete Transformer encoder E(·) with L_enc layers.
    Shared instance is used for both X_prior and X_current.
    """
    def __init__(self, in_ch: int, P: int, D: int, H: int, W: int,
                 L_enc: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.embed = PatchEmbed(in_ch=in_ch, P=P, D=D, H=H, W=W)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(D=D, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(L_enc)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.embed(x)  # (B, N, D)
        for layer in self.layers:
            Z = layer(Z)
        return Z           # (B, N, D)