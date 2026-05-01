#!/usr/bin/env python
"""
Randomized hyperparameter tuning for the ConvLSTM drought forecast model.

Search strategy
---------------
- Randomly samples configurations from a predefined search space.
- Trains each trial with early stopping on validation loss.
- Logs all trial outcomes and saves the best config/model.

Outputs
-------
  outputs/convlstm_tuning_results.csv
  outputs/convlstm_tuning_best_config.json
  outputs/convlstm_tuned_model.pt
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


BASE_DIR = Path(__file__).resolve().parents[1]
PROC = BASE_DIR / "data" / "processed"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

IGNORE_IDX = -99
NUM_CLASSES = 3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(
            in_channels + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            bias=True,
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch: int, lat: int, lon: int):
        return (
            torch.zeros(batch, self.hidden_dim, lat, lon),
            torch.zeros(batch, self.hidden_dim, lat, lon),
        )


class ConvLSTMForecast(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        kernel_size: int,
        num_layers: int,
        num_classes: int,
        spatial_dropout: float,
    ):
        super().__init__()
        self.num_layers = num_layers

        cells = []
        for layer in range(num_layers):
            inp = in_channels if layer == 0 else hidden_dim
            cells.append(ConvLSTMCell(inp, hidden_dim, kernel_size))
        self.cells = nn.ModuleList(cells)

        self.bn = nn.BatchNorm2d(hidden_dim)
        self.drop = nn.Dropout2d(p=spatial_dropout)
        self.head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _, lat, lon = x.shape
        h = [None] * self.num_layers
        c = [None] * self.num_layers
        for li, cell in enumerate(self.cells):
            h[li], c[li] = cell.init_hidden(bsz, lat, lon)
            h[li], c[li] = h[li].to(x.device), c[li].to(x.device)

        out = None
        for t in range(seq_len):
            inp = x[:, t]
            for li, cell in enumerate(self.cells):
                h[li], c[li] = cell(inp, h[li], c[li])
                inp = h[li]
            out = h[-1]

        return self.head(self.drop(self.bn(out)))


@dataclass
class TrialConfig:
    hidden_dim: int
    num_layers: int
    spatial_dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    kernel_size: int


def compute_class_weights(y_train: torch.Tensor, device: torch.device) -> torch.Tensor:
    valid = y_train[y_train != IGNORE_IDX]
    counts = torch.bincount(valid.view(-1), minlength=NUM_CLASSES).float()
    total = counts.sum().item()
    weights = []
    for i in range(NUM_CLASSES):
        w = total / (NUM_CLASSES * counts[i].item()) if counts[i] > 0 else 0.0
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def sample_config(rng: random.Random, focused: bool = False) -> TrialConfig:
    if focused:
        return TrialConfig(
            hidden_dim=32,
            num_layers=2,
            spatial_dropout=rng.choice([0.3, 0.4, 0.5]),
            lr=rng.choice([5e-4, 1e-3, 1.5e-3]),
            weight_decay=rng.choice([1e-6, 1e-5, 3e-5]),
            batch_size=4,
            kernel_size=3,
        )
    return TrialConfig(
        hidden_dim=rng.choice([16, 24, 32]),
        num_layers=rng.choice([1, 2]),
        spatial_dropout=rng.choice([0.1, 0.2, 0.3, 0.4]),
        lr=rng.choice([3e-4, 1e-3, 2e-3]),
        weight_decay=rng.choice([1e-5, 1e-4, 5e-4]),
        batch_size=rng.choice([4, 8]),
        kernel_size=3,
    )


def train_one_trial(
    cfg: TrialConfig,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    in_channels: int,
    class_weights: torch.Tensor,
    device: torch.device,
    max_epochs: int,
    patience: int,
):
    model = ConvLSTMForecast(
        in_channels=in_channels,
        hidden_dim=cfg.hidden_dim,
        kernel_size=cfg.kernel_size,
        num_layers=cfg.num_layers,
        num_classes=NUM_CLASSES,
        spatial_dropout=cfg.spatial_dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_IDX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-5
    )

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val), batch_size=cfg.batch_size, shuffle=False
    )

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item()
        val_loss /= max(1, len(val_loader))
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    params_count = sum(p.numel() for p in model.parameters())
    return best_val, best_epoch, best_state, params_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomized tuning for ConvLSTM forecast model")
    parser.add_argument("--trials", type=int, default=24, help="Number of random trials")
    parser.add_argument("--max-epochs", type=int, default=80, help="Max epochs per trial")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--focused",
        action="store_true",
        help="Use focused search space around current best 2-layer ConvLSTM region",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Search mode: {'focused' if args.focused else 'broad'}")
    print("Loading arrays ...")
    x_train = torch.from_numpy(np.load(PROC / "convlstm_X_train.npy"))
    y_train = torch.from_numpy(np.load(PROC / "convlstm_y_train.npy"))
    x_val = torch.from_numpy(np.load(PROC / "convlstm_X_val.npy"))
    y_val = torch.from_numpy(np.load(PROC / "convlstm_y_val.npy"))
    print(f"  X_train {tuple(x_train.shape)}  y_train {tuple(y_train.shape)}")
    print(f"  X_val   {tuple(x_val.shape)}    y_val   {tuple(y_val.shape)}")

    in_channels = int(x_train.shape[2])
    class_weights = compute_class_weights(y_train, device)
    print("Class weights:", [round(float(w), 4) for w in class_weights.tolist()])

    results = []
    global_best = float("inf")
    global_best_payload = None

    for trial in range(1, args.trials + 1):
        cfg = sample_config(rng, focused=args.focused)
        print(f"Trial {trial}/{args.trials}: {cfg}")
        best_val, best_epoch, best_state, params_count = train_one_trial(
            cfg=cfg,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            in_channels=in_channels,
            class_weights=class_weights,
            device=device,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        row = {
            "trial": trial,
            **asdict(cfg),
            "best_val_loss": float(best_val),
            "best_epoch": int(best_epoch),
            "params_count": int(params_count),
        }
        results.append(row)
        print(
            f"  -> best_val_loss={best_val:.5f}, best_epoch={best_epoch}, params={params_count:,}"
        )

        if best_val < global_best:
            global_best = best_val
            global_best_payload = {
                "config": asdict(cfg),
                "best_val_loss": float(best_val),
                "best_epoch": int(best_epoch),
                "params_count": int(params_count),
                "model_state": best_state,
                "meta": {
                    "in_channels": in_channels,
                    "num_classes": NUM_CLASSES,
                    "seed": args.seed,
                },
            }

    results_df = pd.DataFrame(results).sort_values("best_val_loss", ascending=True)
    results_path = OUT_DIR / "convlstm_tuning_results.csv"
    results_df.to_csv(results_path, index=False)
    print("Wrote:", results_path)

    assert global_best_payload is not None
    best_cfg_path = OUT_DIR / "convlstm_tuning_best_config.json"
    best_cfg_serializable = {
        **global_best_payload["config"],
        "best_val_loss": global_best_payload["best_val_loss"],
        "best_epoch": global_best_payload["best_epoch"],
        "params_count": global_best_payload["params_count"],
        **global_best_payload["meta"],
    }
    best_cfg_path.write_text(json.dumps(best_cfg_serializable, indent=2))
    print("Wrote:", best_cfg_path)

    model_path = OUT_DIR / "convlstm_tuned_model.pt"
    torch.save(
        {
            "model_state": global_best_payload["model_state"],
            "config": global_best_payload["config"],
            "meta": global_best_payload["meta"],
        },
        model_path,
    )
    print("Wrote:", model_path)

    print("Best trial:")
    print(json.dumps(best_cfg_serializable, indent=2))


if __name__ == "__main__":
    main()
