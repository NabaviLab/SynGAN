"""
Simple, clear inference/evaluation for a single split.

Runs on data/test patient folders and:
  • loads latest checkpoint from ./checkpoints (or a provided path)
  • generates X_synthetic, T_gen, T_hat, M_blend
  • saves stitched visuals per item
  • reports average reconstruction MSE

Output → ./eval_out/
"""

import os
import glob
import json
import argparse

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.utils.losses import loss_recon
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.data.dataset import PairedMammoDataset


def latest_ckpt(ckpt_dir: str):
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "dual_timepoint_train_e*.pt")))
    return cands[-1] if cands else None


@torch.no_grad()
def evaluate(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # Data
    ds = PairedMammoDataset(root_dir=args.data_root, split=args.split, img_size=(args.H, args.W))
    if len(ds) == 0:
        raise RuntimeError(f"No samples found under {args.data_root}/{args.split}")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # Minimal cfg for models
    class Cfg:
        H=args.H; W=args.W; P=args.P; D=args.D; L_enc=4; n_heads=args.n_heads
        d_ff=args.d_ff; dropout=0.0; d_latent=args.d_latent
        swin_name=args.swin_name; epsilon=1e-7

    # Models
    G = Generator(Cfg, in_ch=1).to(device)
    D = Discriminator(swin_name=args.swin_name).to(device)

    # Load checkpoint
    ckpt_path = args.ckpt if args.ckpt else latest_ckpt(args.checkpoints)
    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise FileNotFoundError("No checkpoint found. Train first or pass --ckpt path.")
    state = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(state["G"]); D.load_state_dict(state["D"])
    G.eval(); D.eval()
    print(f"[ckpt] loaded → {ckpt_path}")

    # Output dir
    os.makedirs(args.out_dir, exist_ok=True)

    # Loop
    total_mse = 0.0
    n = 0
    for i, batch in enumerate(loader):
        X_prior   = batch["X_prior"].to(device)
        X_current = batch["X_current"].to(device)

        out = G(X_prior, X_current)
        X_synth = out["X_synthetic"]
        T_gen   = out["T_gen"]
        T_hat   = out["T_hat"]
        M_blend = out["M_blend"]

        # recon metric
        mse = loss_recon(X_synth, X_current).item()
        total_mse += mse
        n += 1

        # save a stitched panel: [prior | current | synthetic | T_gen(norm) | T_hat | M_blend]
        T_gen_norm = (T_gen - T_gen.min()) / (T_gen.max() - T_gen.min() + 1e-8)
        panel = torch.cat([X_prior, X_current, X_synth, T_gen_norm, T_hat, M_blend], dim=3)
        vutils.save_image(panel, os.path.join(args.out_dir, f"sample_{i:05d}.png"))

    # summary
    avg_mse = total_mse / max(n, 1)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"avg_mse_recon": avg_mse, "count": n}, f, indent=2)
    print(f"[eval] {args.split}: n={n} | avg MSE(recon)={avg_mse:.6f}")
    print(f"[eval] outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate Dual-Timepoint model (single split)")
    # data
    p.add_argument("--data_root", type=str, default="data", help="root that contains train/val/test folders")
    p.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    p.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    # image/model cfg
    p.add_argument("--H", type=int, default=1024)
    p.add_argument("--W", type=int, default=1024)
    p.add_argument("--P", type=int, default=16)
    p.add_argument("--D", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--d_latent", type=int, default=512)
    p.add_argument("--swin_name", type=str, default="swin_tiny_patch4_window7_224")
    # io
    p.add_argument("--checkpoints", type=str, default="checkpoints")
    p.add_argument("--ckpt", type=str, default="", help="optional explicit checkpoint path")
    p.add_argument("--out_dir", type=str, default="eval_out")

    args = p.parse_args()
    evaluate(args)
