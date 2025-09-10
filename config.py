from dataclasses import dataclass

@dataclass
class ModelConfig:
    H: int                 # image height (e.g., 1024)
    W: int                 # image width  (e.g., 1024)
    P: int                 # patch size
    D: int                 # token dim
    L_enc: int             # encoder layers (4)
    n_heads: int           # attention heads
    d_ff: int              # FFN hidden dim
    dropout: float         # (0.0 per paper unless needed)
    d_latent: int          # latent dim d
    swin_name: str         # e.g., "swin_tiny_patch4_window7_224"
    epsilon: float         # small number for GAN log stability

@dataclass
class TrainConfig:
    epochs: int            # 200
    batch_size: int        # 4
    optimizer: str         # "adam"
    lr_max: float          # 1e-2
    lr_min: float          # 1e-5
    betas: tuple           # (0.9, 0.999)
    weight_decay: float    # 0.0
    num_workers: int       # dataloader workers
    save_every: int        # checkpoint interval
    val_every: int         # eval interval
    device: str            # "cuda" or "cpu"

@dataclass
class LossConfig:
    lambda_kl_base: float  # 1.0
    lambda_tumor: float    # 1.0
    beta_mode: str         # "kl" or "off" (β applied to KL term if "kl")