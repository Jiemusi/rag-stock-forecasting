import argparse
import math
import os
import pickle
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model_baseline import BaselineTSFM


FEATURE_COLS = [
    "log_ret",
    "volatility_20d",
    "volume_change",
    "ps_ratio",
    "pe_ratio",
    "rev_growth_qoq",
    "real_rate",
    "yield_curve",
    "unemployment_change",
    "close",
    "is_trading_day",
]


class BaselineDataset(Dataset):
    """
    Loads baseline training samples from a pickle file produced by
    construct_dataset_baseline.py.

    Each record is expected to have:
      - "x_struct": (T, F) numpy array of features
      - "y":       (H,)   numpy array of future log returns
    """

    def __init__(self, path: str):
        super().__init__()
        with open(path, "rb") as f:
            self.records = pickle.load(f)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        x_struct = torch.tensor(rec["x_struct"], dtype=torch.float32)  # (T, F)
        y = torch.tensor(rec["y"], dtype=torch.float32)                # (H,)
        return x_struct, y


def shape_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Simple shape loss: compare normalized trajectory shapes (up to scale/shift).
    pred, target: (B, H)
    """
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    tgt_c = target - target.mean(dim=1, keepdim=True)

    pred_n = pred_c / (pred_c.norm(dim=1, keepdim=True) + eps)
    tgt_n = tgt_c / (tgt_c.norm(dim=1, keepdim=True) + eps)

    return nn.functional.mse_loss(pred_n, tgt_n)


def warmup_cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        scale = float(step + 1) / max(1, warmup_steps)
    else:
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * scale


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    base_lr: float,
    total_steps: int,
    global_step: int,
    warmup_steps: int,
    shape_loss_weight: float,
) -> int:
    model.train()
    mse_loss_fn = nn.MSELoss()

    for x_struct, y in loader:
        x_struct = x_struct.to(device)  # (B, T, F)
        y = y.to(device)                # (B, H)

        # LR schedule
        lr = warmup_cosine_lr(global_step, total_steps, base_lr, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        pred = model(x_struct)  # (B, H)

        mse = mse_loss_fn(pred, y)
        if shape_loss_weight > 0.0:
            sl = shape_loss(pred, y)
        else:
            sl = torch.tensor(0.0, device=device)

        loss = mse + shape_loss_weight * sl
        loss.backward()
        optimizer.step()

        global_step += 1

    return global_step


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    mse_loss_fn = nn.MSELoss()
    losses = []

    for x_struct, y in loader:
        x_struct = x_struct.to(device)
        y = y.to(device)

        pred = model(x_struct)
        loss = mse_loss_fn(pred, y)
        losses.append(loss.item())

    return float(np.mean(losses)) if losses else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Train baseline TSFM (no RAG).")
    parser.add_argument("--train_path", type=str, default="../data/tsfm_dataset_baseline/train.pkl")
    parser.add_argument("--val_path", type=str, default="../data/tsfm_dataset_baseline/val.pkl")
    parser.add_argument("--input_dim", type=int, default=len(FEATURE_COLS))
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--shape_loss_weight", type=float, default=0.2)
    parser.add_argument("--out_path", type=str, default="tsfm_baseline_best_val.pt")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = BaselineDataset(args.train_path)
    val_ds = BaselineDataset(args.val_path)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = BaselineTSFM(
        input_dim=args.input_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        horizon=args.horizon,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    global_step = 0
    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.lr,
            total_steps,
            global_step,
            args.warmup_steps,
            args.shape_loss_weight,
        )
        val_mse = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch:03d} | val MSE = {val_mse:.6f}")

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        torch.save(best_state, args.out_path)
        print(f"Saved best baseline model (val MSE={best_val:.6f}) to {args.out_path}")
    else:
        print("No best_state recorded; training dataset might be empty.")


if __name__ == "__main__":
    main()
