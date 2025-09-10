"""
Single-split training script (clear & patient-folder friendly).

DATA LAYOUT (per split):
data/
  train/
    PATIENT001/
      prior/   LCC.png LMLO.png RCC.png RMLO.png
      current/ LCC.png LMLO.png RCC.png RMLO.png
      masks/   LCC.png (others optional)
    PATIENT002/
      ...
  val/
    PATIENTxxx/ ...
  test/
    PATIENTyyy/ ...

This script:
  • loads data from data/<split>/PATIENT... folders
  • builds Generator + Discriminator exactly per your methodology
  • uses per-sample tumor loss ONLY where a mask exists
  • applies KL loss ONLY for masked samples (as your text specifies)
  • schedules LR: 1e-2 → 1e-5 over 200 epochs
  • schedules β_KL: 0.3 → 2.0 over 200 epochs
  • saves checkpoints in ./checkpoints and CSV logs in ./logs
"""

import os
import csv
import math
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---- repo imports ----
from src.utils.seed import set_seed
from src.utils.losses import loss_recon, gan_terms
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.data.dataset import PairedMammoDataset


# ---------------------------
# Schedules (simple & explicit)
# ---------------------------
def lr_from_epoch(epoch: int, total_epochs: int, lr_max: float, lr_min: float) -> float:
    """Smooth exponential decay lr_max → lr_min across epochs."""
    if total_epochs <= 1:
        return lr_min
    ratio = lr_min / lr_max
    t = epoch / (total_epochs - 1)
    return lr_max * (ratio ** t)

def beta_kl_from_epoch(epoch: int, total_epochs: int, start: float = 0.3, end: float = 2.0) -> float:
    """Linear anneal β_KL from 0.3 → 2.0."""
    if total_epochs <= 1:
        return end
    t = epoch / (total_epochs - 1)
    return start + (end - start) * t


# ---------------------------
# Per-sample KL and Tumor BCE (mask-aware)
# ---------------------------
def kl_per_sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL_i = -1/2 * sum(1 + logσ²_i − μ_i² − σ_i²) for each sample i
    returns (B,)
    """
    kl_vec = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl_vec  # (B,)

def tumor_bce_per_sample(T_hat: torch.Tensor, M_gt: torch.Tensor) -> torch.Tensor:
    """
    BCE per sample with reduction='none' then mean over (C,H,W).
    returns (B,)
    """
    bce = F.binary_cross_entropy(T_hat, M_gt, reduction='none')  # (B,1,H,W)
    return bce.mean(dim=(1, 2, 3))  # (B,)


# ---------------------------
# CSV logger (tiny & explicit)
# ---------------------------
class CSVLogger:
    def __init__(self, path: str, fieldnames):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.fieldnames = fieldnames
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()

    def log(self, row: dict):
        row = {k: (v if not isinstance(v, float) else float(v)) for k, v in row.items()}
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writerow(row)


# ---------------------------
# Training
# ---------------------------
def train(args):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    os.makedirs(args.checkpoints, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)

    # Data
    train_ds = PairedMammoDataset(root_dir=args.data_root, split="train", img_size=(args.H, args.W))
    if len(train_ds) == 0:
        raise RuntimeError(f"No training samples found under {args.data_root}/train")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    # Models
    class Cfg:  # minimal struct so models take what they need
        H=args.H; W=args.W; P=args.P; D=args.D; L_enc=4; n_heads=args.n_heads
        d_ff=args.d_ff; dropout=0.0; d_latent=args.d_latent
        swin_name=args.swin_name; epsilon=1e-7

    G = Generator(Cfg, in_ch=1).to(device)
    D = Discriminator(swin_name=args.swin_name).to(device)

    # Opt
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_max, betas=(0.9, 0.999), weight_decay=0.0)
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_max, betas=(0.9, 0.999), weight_decay=0.0)

    # Logger
    log_fields = ["epoch","time","lr","beta_kl","L_G","L_D","L_recon","L_kl","L_tumor","y_real","y_fake"]
    logger = CSVLogger(os.path.join(args.logs, "train_single_split.csv"), log_fields)

    for epoch in range(args.epochs):
        G.train(); D.train()

        # schedules
        lr_now = lr_from_epoch(epoch, args.epochs, args.lr_max, args.lr_min)
        for pg in opt_G.param_groups: pg["lr"] = lr_now
        for pg in opt_D.param_groups: pg["lr"] = lr_now
        beta_kl = beta_kl_from_epoch(epoch, args.epochs, 0.3, 2.0)

        # running metrics
        m = dict(L_G=0., L_D=0., L_recon=0., L_kl=0., L_tumor=0., y_real=0., y_fake=0.)
        steps = 0

        for batch in train_loader:
            X_prior   = batch["X_prior"].to(device)   # (B,1,H,W)
            X_current = batch["X_current"].to(device) # (B,1,H,W)
            M_gt      = batch["M_gt"].to(device)      # (B,1,H,W)
            has_mask  = batch["has_mask"].to(device).view(-1)  # (B,)

            # Forward G
            out = G(X_prior, X_current)
            X_synth = out["X_synthetic"]
            T_hat   = out["T_hat"]
            mu      = out["mu"]
            logvar  = out["logvar"]

            # Recon (always)
            L_recon = loss_recon(X_synth, X_current)

            # Per-sample KL (only where masks exist)
            kl_vec = kl_per_sample(mu, logvar)                   # (B,)
            mask_sum = has_mask.sum()
            if mask_sum > 0:
                L_kl = (kl_vec * has_mask).sum() / (mask_sum + 1e-8)
            else:
                L_kl = torch.zeros((), device=device)

            # Per-sample tumor BCE (only where masks exist)
            bce_vec = tumor_bce_per_sample(T_hat, M_gt)          # (B,)
            if mask_sum > 0:
                L_tumor = (bce_vec * has_mask).sum() / (mask_sum + 1e-8)
            else:
                L_tumor = torch.zeros((), device=device)

            # GAN terms
            y_real, y_fake, L_GAN, L_D = gan_terms(D, X_prior, X_current, X_synth, Cfg.epsilon)

            # Total G loss:
            #   with mask:   L_recon + beta_kl*L_kl + 1.0*L_tumor + L_GAN
            #   without:     L_recon + L_GAN  (since mask_sum=0 → L_kl=L_tumor=0)
            L_G = L_recon + beta_kl * L_kl + 1.0 * L_tumor + L_GAN

            # Update D
            opt_D.zero_grad(set_to_none=True)
            L_D.backward(retain_graph=True)
            opt_D.step()

            # Update G
            opt_G.zero_grad(set_to_none=True)
            L_G.backward()
            opt_G.step()

            # accumulate
            m["L_G"]     += float(L_G.detach().cpu())
            m["L_D"]     += float(L_D.detach().cpu())
            m["L_recon"] += float(L_recon.detach().cpu())
            m["L_kl"]    += float(L_kl.detach().cpu())
            m["L_tumor"] += float(L_tumor.detach().cpu())
            m["y_real"]  += float(y_real.detach().cpu().mean())
            m["y_fake"]  += float(y_fake.detach().cpu().mean())
            steps += 1

        # epoch means
        for k in m: m[k] = m[k] / max(steps, 1)

        # log
        logger.log({
            "epoch": epoch+1,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lr": lr_now,
            "beta_kl": beta_kl,
            **m
        })

        # console
        print(f"[train] epoch {epoch+1}/{args.epochs} "
              f"lr={lr_now:.6f} beta_kl={beta_kl:.3f} "
              f"L_G={m['L_G']:.4f} L_D={m['L_D']:.4f} "
              f"Recon={m['L_recon']:.4f} KL={m['L_kl']:.4f} Tumor={m['L_tumor']:.4f} "
              f"y_real={m['y_real']:.3f} y_fake={m['y_fake']:.3f}"
        )

        # save ckpt
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.checkpoints, f"dual_timepoint_train_e{epoch+1}.pt")
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "epoch": epoch+1}, ckpt_path)
            print(f"[ckpt] saved → {ckpt_path}")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Dual-Timepoint Tumor Synthesis (single split)")
    # data
    p.add_argument("--data_root", type=str, default="data", help="root that contains train/val/test folders")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    # image/model sizes
    p.add_argument("--H", type=int, default=1024)
    p.add_argument("--W", type=int, default=1024)
    p.add_argument("--P", type=int, default=16)
    p.add_argument("--D", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--d_latent", type=int, default=512)
    p.add_argument("--swin_name", type=str, default="swin_tiny_patch4_window7_224")
    # train
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr_max", type=float, default=1e-2)
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--save_every", type=int, default=10)
    # io
    p.add_argument("--checkpoints", type=str, default="checkpoints")
    p.add_argument("--logs", type=str, default="logs")

    args = p.parse_args()
    train(args)
