import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import TSFM

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.swa_utils import AveragedModel, SWALR

import wandb
wandb.init(project="tsfm_rag", name="tsfm_experiment", config={"epochs": 15, "batch_size": 64})
config = wandb.config


# ============================
# Dataset Class
# ============================
class TSFMDataset(Dataset):
    def __init__(self, path):
        self.data = pickle.load(open(path, "rb"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]

        x_struct = torch.tensor(s["x_struct"], dtype=torch.float32)
        x_query  = torch.tensor(s["x_query"], dtype=torch.float32)
        x_keys   = torch.tensor(s["x_keys"], dtype=torch.float32)
        x_values = torch.tensor(s["x_values"], dtype=torch.float32)
        y        = torch.tensor(s["y"], dtype=torch.float32)

        return x_struct, x_query, x_keys, x_values, y


# ============================
# Loss Functions
# ============================
mse_loss = nn.MSELoss()


def shape_loss(pred, target):
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    tgt_norm  = target / (target.norm(dim=1, keepdim=True) + 1e-8)
    cos = (pred_norm * tgt_norm).sum(dim=1)
    return 1 - cos.mean()


# ============================
# Training Loop
# ============================
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for x_struct, x_query, x_keys, x_values, y in loader:
        x_struct = x_struct.to(device)
        x_query  = x_query.to(device)
        x_keys   = x_keys.to(device)
        x_values = x_values.to(device)
        y        = y.to(device)

        optimizer.zero_grad()
        with autocast():
            y_pred, attn = model(x_struct, x_query, x_keys, x_values)
            attn = attn * neighbor_scale
            loss1 = mse_loss(y_pred, y)
            loss2 = shape_loss(y_pred, y)
            loss  = loss1 + shape_loss_weight * loss2

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.update()

        for name, p in model.named_parameters():
            ema_shadow[name] = ema_decay * ema_shadow[name] + (1 - ema_decay) * p.data

        total_loss += loss.item()

    if hasattr(attn, "shape"):
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(attn.detach().cpu().numpy(), cmap="viridis")
        ax.set_title("Attention Weights")
        fig.tight_layout()
        writer.add_figure(f"train/attn_epoch_{epoch}", fig)
        fig.savefig(f"runs/tsfm/attn_epoch_{epoch}.png")
        plt.close(fig)
        top_neighbor = int(torch.argmax(attn))
        writer.add_scalar(f"train/top_neighbor_epoch_{epoch}", top_neighbor, epoch)

    if attn is not None:
        contrib = attn.detach().cpu().numpy()
        wandb.log({"neighbor_contribution": wandb.Histogram(contrib), "top_neighbor": int(contrib.argmax())})

        # Generate neighbor explanation HTML
        html_path = f"runs/tsfm/neighbor_report_epoch_{epoch}.html"
        with open(html_path, "w") as f:
            f.write("<html><body><h2>Neighbor Contribution Report</h2>")
            f.write("<table border='1'><tr><th>Neighbor</th><th>Weight</th></tr>")
            contrib = attn.detach().cpu().numpy().flatten()
            for i, w in enumerate(contrib):
                f.write(f"<tr><td>{i}</td><td>{w:.4f}</td></tr>")
            f.write("</table></body></html>")

        table = wandb.Table(columns=["neighbor_id", "weight"])
        for i, w in enumerate(contrib):
            table.add_data(i, float(w))
        wandb.log({"neighbor_table": table})

    return total_loss / len(loader)


# ============================
# Validation Loop
# ============================
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x_struct, x_query, x_keys, x_values, y in loader:
            x_struct = x_struct.to(device)
            x_query  = x_query.to(device)
            x_keys   = x_keys.to(device)
            x_values = x_values.to(device)
            y        = y.to(device)

            backup = {}
            for name, p in model.named_parameters():
                backup[name] = p.data.clone()
                p.data = ema_shadow[name].clone()

            y_pred, _ = model(x_struct, x_query, x_keys, x_values)

            for name, p in model.named_parameters():
                p.data = backup[name]

            loss1 = mse_loss(y_pred, y)
            loss2 = shape_loss(y_pred, y)
            total_loss += (loss1 + 0.2 * loss2).item()

    return total_loss / len(loader)


# ============================
# Main Training Script
# ============================
def main():

    TRAIN_PATH = "data/tsfm_dataset/train.pkl"
    VAL_PATH   = "data/tsfm_dataset/val.pkl"

    LR = config.lr
    BATCH_SIZE = config.batch_size
    EPOCHS = config.epochs
    # warmup_steps = 500
    warmup_steps = config.warmup_steps
    shape_loss_weight = config.shape_loss_weight
    neighbor_scale = config.neighbor_scale

    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    train_ds = TSFMDataset(TRAIN_PATH)
    val_ds   = TSFMDataset(VAL_PATH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("Initializing model...")
    model = TSFM().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    swa_model = AveragedModel(model)
    swa_start = int(EPOCHS * 0.6)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5)
    ema_decay = 0.999
    ema_shadow = {name: p.data.clone() for name, p in model.named_parameters()}

    def warmup_cosine(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + np.cos((step - warmup_steps) / (EPOCHS * len(train_loader) - warmup_steps) * np.pi))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    writer = SummaryWriter(log_dir="runs/tsfm")
    wandb.watch(model, log="all", log_freq=100)
    scaler = GradScaler()
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_loss   = eval_epoch(model, val_loader, DEVICE)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"], "epoch": epoch})

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"tsfm_best_val_{epoch}.pt")
            print("✓ Saved new best model.")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    torch.optim.swa_utils.update_bn(train_loader, swa_model)
    torch.save(swa_model.state_dict(), "tsfm_swa_final.pt")

    print("\nTraining Complete!")
    writer.close()
    wandb.finish()


if __name__ == "__main__":
    main()
