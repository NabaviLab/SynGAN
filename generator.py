import torch
import torch.nn as nn
from .encoder import EncoderE
from .decoder import TransformerDecoder

class VariationalLatent(nn.Module):
    """
    Variational Latent Space for sampling z:
      μ = W_μ * F_fused_flat
      logσ² = W_σ * F_fused_flat
      z = μ + σ * ε, where ε ~ N(0, I)
    """
    def __init__(self, N: int, D: int, d_latent: int):
        super().__init__()
        in_dim = N * (2 * D)
        self.W_mu = nn.Linear(in_dim, d_latent)
        self.W_logvar = nn.Linear(in_dim, d_latent)

        nn.init.xavier_uniform_(self.W_mu.weight); nn.init.zeros_(self.W_mu.bias)
        nn.init.xavier_uniform_(self.W_logvar.weight); nn.init.zeros_(self.W_logvar.bias)

    def forward(self, F_fused: torch.Tensor):
        B, N, twoD = F_fused.shape
        flat = F_fused.reshape(B, N * twoD)   # (B, N*2D)
        mu = self.W_mu(flat)                 # (B, d_latent)
        logvar = self.W_logvar(flat)         # (B, d_latent)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps                  # reparameterization trick
        return z, mu, logvar


class Generator(nn.Module):
    """
    Generator Architecture:
      - Shared dual Transformer encoders for X_prior & X_current
      - Variational latent space for sampling z
      - Transformer-based decoder for tumor generation
      - Differentiable blending module
    """
    def __init__(self, cfg, in_ch: int = 1):
        super().__init__()
        self.cfg = cfg
        self.N = (cfg.H // cfg.P) * (cfg.W // cfg.P)

        # Shared dual encoders
        self.encoder = EncoderE(
            in_ch=in_ch, P=cfg.P, D=cfg.D, H=cfg.H, W=cfg.W,
            L_enc=cfg.L_enc, n_heads=cfg.n_heads,
            d_ff=cfg.d_ff, dropout=cfg.dropout
        )

        # Variational latent space
        self.latent = VariationalLatent(N=self.N, D=cfg.D, d_latent=cfg.d_latent)

        # Transformer-based decoder
        self.decoder = TransformerDecoder(
            N=self.N, P=cfg.P, D=cfg.D, twoD=2 * cfg.D, d_latent=cfg.d_latent,
            H=cfg.H, W=cfg.W, n_heads=cfg.n_heads,
            d_ff=cfg.d_ff, dropout=cfg.dropout
        )

    def forward(self, X_prior: torch.Tensor, X_current: torch.Tensor):
        """
        Inputs:
          X_prior   : Prior mammogram   (B, 1, H, W)
          X_current : Current mammogram (B, 1, H, W)

        Returns:
          {
            F_prior, F_current, F_fused,
            z, mu, logvar,
            Z_out, T_gen, T_hat, M_blend,
            X_synthetic
          }
        """
        # Dual encoding
        F_prior = self.encoder(X_prior)         # (B, N, D)
        F_current = self.encoder(X_current)     # (B, N, D)
        F_fused = torch.cat([F_prior, F_current], dim=-1)  # (B, N, 2D)

        # Latent sampling
        z, mu, logvar = self.latent(F_fused)

        # Decode to tumor + mask + blend map
        Z_out, T_gen, T_hat, M_blend = self.decoder(z, F_fused)

        # Differentiable blending: insert tumor into current image
        X_synthetic = (1.0 - M_blend) * X_current + M_blend * T_gen

        return {
            "F_prior": F_prior,
            "F_current": F_current,
            "F_fused": F_fused,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "Z_out": Z_out,
            "T_gen": T_gen,
            "T_hat": T_hat,
            "M_blend": M_blend,
            "X_synthetic": X_synthetic
        }