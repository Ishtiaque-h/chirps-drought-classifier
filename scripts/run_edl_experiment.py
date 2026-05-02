#!/usr/bin/env python
"""
Evidential Deep Learning (EDL) experiment on the canonical forecast dataset.

This is non-destructive:
  - reads data/processed/dataset_forecast.parquet
  - writes outputs/edl_* artifacts

Outputs:
  outputs/edl_model.pt
  outputs/edl_feature_scaler.npz
  outputs/edl_test_probs.npz
  outputs/edl_uncertainty_monthly.csv
  outputs/edl_metrics.txt
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from feature_config import get_feature_columns


DATA = Path("data/processed/dataset_forecast.parquet")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

TARGET = "target_label"
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
N_CLASSES = 3


class EDLNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, N_CLASSES),
        )
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        evidence = self.softplus(self.net(x))
        return evidence


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--anneal-epochs", type=int, default=10)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap on training rows for faster iteration.",
    )
    return parser.parse_args()


def kl_divergence(alpha: torch.Tensor) -> torch.Tensor:
    k = alpha.shape[1]
    alpha0 = torch.ones((1, k), device=alpha.device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    sum_alpha0 = torch.sum(alpha0, dim=1, keepdim=True)

    ln_b = torch.lgamma(sum_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    ln_b0 = torch.lgamma(sum_alpha0) - torch.sum(torch.lgamma(alpha0), dim=1, keepdim=True)

    digamma_sum = torch.digamma(sum_alpha)
    digamma_alpha = torch.digamma(alpha)

    kl = torch.sum((alpha - alpha0) * (digamma_alpha - digamma_sum), dim=1, keepdim=True)
    kl = kl + ln_b - ln_b0
    return kl.squeeze(1)


def edl_mse_loss(
    y_onehot: torch.Tensor,
    alpha: torch.Tensor,
    epoch: int,
    anneal_epochs: int,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    s = torch.sum(alpha, dim=1, keepdim=True)
    p = alpha / s
    err = torch.sum((y_onehot - p) ** 2, dim=1)
    var = torch.sum(alpha * (s - alpha) / (s * s * (s + 1.0)), dim=1)
    loss = err + var

    kl = kl_divergence(alpha)
    anneal = min(1.0, float(epoch) / float(max(1, anneal_epochs)))
    loss = loss + anneal * kl

    if sample_weight is not None:
        loss = loss * sample_weight

    return loss.mean()


def to_onehot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def predict_dirichlet(alpha: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / s

    # Predictive entropy (total uncertainty)
    p = probs.clamp(min=1e-9)
    total_uncert = -torch.sum(p * torch.log(p), dim=1)

    # Expected entropy (aleatoric)
    digamma_sum = torch.digamma(s)
    digamma_alpha = torch.digamma(alpha + 1.0)
    expected_entropy = -torch.sum((alpha / s) * (digamma_alpha - torch.digamma(s + 1.0)), dim=1)

    epistemic = total_uncert - expected_entropy

    return (
        probs.detach().cpu().numpy(),
        total_uncert.detach().cpu().numpy(),
        expected_entropy.detach().cpu().numpy(),
        epistemic.detach().cpu().numpy(),
    )


def main() -> None:
    args = parse_args()

    print(f"Loading dataset: {DATA}")
    df = pd.read_parquet(DATA)
    df["year"] = df["year"].astype(int)
    features = get_feature_columns(df.columns)

    train = df[df["year"] <= 2016].copy()
    val = df[(df["year"] >= 2017) & (df["year"] <= 2020)].copy()
    test = df[df["year"] >= 2021].copy()

    if args.max_train_rows is not None and len(train) > args.max_train_rows:
        train = train.sample(args.max_train_rows, random_state=42)

    X_train = train[features].to_numpy(dtype=np.float32)
    X_val = val[features].to_numpy(dtype=np.float32)
    X_test = test[features].to_numpy(dtype=np.float32)

    y_train = train[TARGET].map(LABEL_MAP).to_numpy()
    y_val = val[TARGET].map(LABEL_MAP).to_numpy()
    y_test = test[TARGET].map(LABEL_MAP).to_numpy()

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    scaler_path = OUT_DIR / "edl_feature_scaler.npz"
    np.savez(scaler_path, mean=mean, std=std, features=np.array(features))

    class_counts = np.bincount(y_train, minlength=N_CLASSES)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    sample_weights = class_weights[y_train]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EDLNet(n_features=X_train.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_tensor = torch.tensor(X_train, device=device)
    val_tensor = torch.tensor(X_val, device=device)
    test_tensor = torch.tensor(X_test, device=device)
    y_train_oh = torch.tensor(to_onehot(y_train, N_CLASSES), device=device)
    y_val_oh = torch.tensor(to_onehot(y_val, N_CLASSES), device=device)
    y_test_oh = torch.tensor(to_onehot(y_test, N_CLASSES), device=device)
    weight_tensor = torch.tensor(sample_weights.astype(np.float32), device=device)

    batch_size = args.batch_size
    n_train = train_tensor.shape[0]

    print(f"Training EDL model on {n_train:,} samples ({len(features)} features)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            x_batch = train_tensor[idx]
            y_batch = y_train_oh[idx]
            w_batch = weight_tensor[idx]

            optimizer.zero_grad()
            evidence = model(x_batch)
            alpha = evidence + 1.0
            loss = edl_mse_loss(y_batch, alpha, epoch, args.anneal_epochs, w_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(idx)

        avg_loss = total_loss / n_train
        model.eval()
        with torch.no_grad():
            val_evidence = model(val_tensor)
            val_alpha = val_evidence + 1.0
            val_loss = edl_mse_loss(y_val_oh, val_alpha, epoch, args.anneal_epochs).item()
        print(f"Epoch {epoch:02d} | train loss {avg_loss:.4f} | val loss {val_loss:.4f}")

    model_path = OUT_DIR / "edl_model.pt"
    torch.save(model.state_dict(), model_path)

    model.eval()
    with torch.no_grad():
        test_evidence = model(test_tensor)
        test_alpha = test_evidence + 1.0
        test_probs, total_u, aleatoric_u, epistemic_u = predict_dirichlet(test_alpha)

    y_pred = np.array([INV_LABEL_MAP[i] for i in test_probs.argmax(axis=1)])
    report = classification_report(
        test[TARGET].to_numpy(),
        y_pred,
        target_names=["dry(-1)", "normal(0)", "wet(+1)"],
        digits=3,
    )

    test_df = test.copy()
    test_df["is_dry"] = (test_df[TARGET] == -1).astype(float)
    val_df = val.copy()
    val_df["is_dry"] = (val_df[TARGET] == -1).astype(float)

    train_monthly_dry = train.groupby("month")["target_label"].apply(lambda s: (s == -1).mean())
    global_dry = float((train[TARGET] == -1).mean())
    val_df["clim_prob_dry"] = val_df["month"].map(train_monthly_dry).fillna(global_dry)
    test_df["clim_prob_dry"] = test_df["month"].map(train_monthly_dry).fillna(global_dry)

    val_df["target_time"] = (pd.to_datetime(val_df["time"]) + pd.DateOffset(months=1)).dt.to_period("M").dt.to_timestamp()
    test_df["target_time"] = (pd.to_datetime(test_df["time"]) + pd.DateOffset(months=1)).dt.to_period("M").dt.to_timestamp()

    # Recompute validation probs from the model for clean calibration.
    with torch.no_grad():
        val_evidence = model(val_tensor)
        val_alpha = val_evidence + 1.0
        val_probs, _, _, _ = predict_dirichlet(val_alpha)
        val_raw = val_probs[:, LABEL_MAP[-1]]

    test_raw = test_probs[:, LABEL_MAP[-1]]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_raw, val_df["is_dry"].to_numpy())
    val_iso = iso.predict(val_raw)
    test_iso = iso.predict(test_raw)

    platt = LogisticRegression(solver="lbfgs")
    platt.fit(val_raw.reshape(-1, 1), val_df["is_dry"].to_numpy())
    val_platt = platt.predict_proba(val_raw.reshape(-1, 1))[:, 1]
    test_platt = platt.predict_proba(test_raw.reshape(-1, 1))[:, 1]

    val_df["edl_raw_prob_dry"] = val_raw
    val_df["edl_isotonic_prob_dry"] = val_iso
    val_df["edl_platt_prob_dry"] = val_platt
    test_df["edl_raw_prob_dry"] = test_raw
    test_df["edl_isotonic_prob_dry"] = test_iso
    test_df["edl_platt_prob_dry"] = test_platt

    def brier(y: np.ndarray, p: np.ndarray) -> float:
        return float(np.mean((p - y) ** 2))

    def bss(y: np.ndarray, p: np.ndarray, ref: np.ndarray) -> float:
        ref_bs = brier(y, ref)
        return float(1.0 - brier(y, p) / ref_bs) if ref_bs > 0 else float("nan")

    def monthly_bs(frame: pd.DataFrame, prob_col: str) -> float:
        monthly = (
            frame.groupby("target_time")
            .agg(
                y_true_dry_frac=("is_dry", "mean"),
                pred_prob_dry=(prob_col, "mean"),
            )
            .reset_index()
        )
        return brier(monthly["y_true_dry_frac"].to_numpy(), monthly["pred_prob_dry"].to_numpy())

    calibration_cols = {
        "none": "edl_raw_prob_dry",
        "isotonic": "edl_isotonic_prob_dry",
        "platt": "edl_platt_prob_dry",
    }
    val_bs_by_method = {
        method: monthly_bs(val_df, col)
        for method, col in calibration_cols.items()
    }
    best_method = min(val_bs_by_method, key=val_bs_by_method.get)
    best_col = calibration_cols[best_method]
    test_df["edl_selected_prob_dry"] = test_df[best_col]

    test_df["total_u"] = total_u
    test_df["aleatoric_u"] = aleatoric_u
    test_df["epistemic_u"] = epistemic_u

    monthly = (
        test_df.groupby("target_time")
        .agg(
            y_true_dry_frac=("is_dry", "mean"),
            edl_raw_prob_dry=("edl_raw_prob_dry", "mean"),
            edl_isotonic_prob_dry=("edl_isotonic_prob_dry", "mean"),
            edl_platt_prob_dry=("edl_platt_prob_dry", "mean"),
            edl_selected_prob_dry=("edl_selected_prob_dry", "mean"),
            clim_prob_dry=("clim_prob_dry", "mean"),
            total_u=("total_u", "mean"),
            aleatoric_u=("aleatoric_u", "mean"),
            epistemic_u=("epistemic_u", "mean"),
        )
        .reset_index()
    )

    y = monthly["y_true_dry_frac"].to_numpy()
    clim = monthly["clim_prob_dry"].to_numpy()
    raw = monthly["edl_raw_prob_dry"].to_numpy()
    iso_pred = monthly["edl_isotonic_prob_dry"].to_numpy()
    platt_pred = monthly["edl_platt_prob_dry"].to_numpy()
    selected = monthly["edl_selected_prob_dry"].to_numpy()

    bs_clim = brier(y, clim)
    bs_raw = brier(y, raw)
    bs_iso = brier(y, iso_pred)
    bs_platt = brier(y, platt_pred)
    bs_selected = brier(y, selected)
    bss_raw = bss(y, raw, clim)
    bss_iso = bss(y, iso_pred, clim)
    bss_platt = bss(y, platt_pred, clim)
    bss_selected = bss(y, selected, clim)

    metrics = (
        "EDL Experiment (MLP baseline)\n"
        f"{'=' * 60}\n"
        "Design: canonical SPI-1[t+1] target with EDL multi-class MLP.\n"
        f"Features: {features}\n"
        f"Train rows: {len(train):,}  Val rows: {len(val):,}  Test rows: {len(test):,}\n"
        f"Best calibration (val monthly BS): {best_method}\n\n"
        "Monthly dry-fraction Brier Scores\n"
        f"  Climatology         : {bs_clim:.5f}\n"
        f"  EDL raw             : {bs_raw:.5f}\n"
        f"  EDL isotonic        : {bs_iso:.5f}\n"
        f"  EDL Platt           : {bs_platt:.5f}\n"
        f"  EDL selected        : {bs_selected:.5f}\n\n"
        "Brier Skill Score vs monthly climatology\n"
        f"  EDL raw             : {bss_raw:.5f}\n"
        f"  EDL isotonic        : {bss_iso:.5f}\n"
        f"  EDL Platt           : {bss_platt:.5f}\n"
        f"  EDL selected        : {bss_selected:.5f}\n\n"
        "Pixel-level classification report (secondary diagnostic)\n"
        f"{report}\n"
    )

    metrics_path = OUT_DIR / "edl_metrics.txt"
    metrics_path.write_text(metrics)
    print(metrics)

    # Attach uncertainty arrays to test_df for monthly aggregation
    test_df["total_u"] = total_u
    test_df["aleatoric_u"] = aleatoric_u
    test_df["epistemic_u"] = epistemic_u

    monthly_path = OUT_DIR / "edl_uncertainty_monthly.csv"
    monthly.to_csv(monthly_path, index=False)

    probs_path = OUT_DIR / "edl_test_probs.npz"
    np.savez_compressed(
        probs_path,
        probs=test_probs.astype("float32"),
        dry_probs_raw=test_raw.astype("float32"),
        dry_probs_isotonic=test_iso.astype("float32"),
        dry_probs_platt=test_platt.astype("float32"),
        dry_probs_selected=test_df[best_col].to_numpy(dtype="float32"),
        total_u=total_u.astype("float32"),
        aleatoric_u=aleatoric_u.astype("float32"),
        epistemic_u=epistemic_u.astype("float32"),
        y_true=test[TARGET].to_numpy(),
        times=test["time"].to_numpy(),
        target_times=test_df["target_time"].to_numpy(),
        latitude=test["latitude"].to_numpy(),
        longitude=test["longitude"].to_numpy(),
        features=np.array(features),
        best_calibration=best_method,
    )

    print("Wrote:")
    print(f"  {model_path}")
    print(f"  {scaler_path}")
    print(f"  {metrics_path}")
    print(f"  {monthly_path}")
    print(f"  {probs_path}")


if __name__ == "__main__":
    main()
