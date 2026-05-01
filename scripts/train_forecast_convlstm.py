#!/usr/bin/env python
"""
ConvLSTM spatiotemporal drought forecast model.

Architecture
------------
  Input:  (batch, seq_len=3, C=4, lat, lon)
          channels: [spi1, spi3, spi6, pr_norm]

    ConvLSTM stack (2 layers, hidden_dim=32, kernel 3×3)
        → takes the last hidden state  (batch, 32, lat, lon)
    Spatial dropout (p=0.4)
    Conv2d head (hidden_dim → 3 classes, kernel 1×1)
    → output logits  (batch, 3, lat, lon)

Loss: cross-entropy with ignore_index=-99
  (masked / out-of-domain pixels carry label -99 and are excluded)

Training details
  Optimiser : AdamW  lr=1e-3, weight_decay=1e-4
  Scheduler : ReduceLROnPlateau(patience=5, factor=0.5)
  Max epochs: 80  (early stopping after 10 val-loss non-improvement epochs)
  Batch size: 8   (one mini-batch = 8 consecutive time steps)

Outputs
  outputs/convlstm_metrics.txt
  outputs/convlstm_cm.png
  outputs/convlstm_model.pt
  outputs/convlstm_test_preds.npz   — per-pixel predicted class + true label
  outputs/convlstm_test_probs.npz   — per-pixel softmax probabilities (N_test, 3, lat, lon)

Inputs
  data/processed/convlstm_X_train.npy   (build_dataset_convlstm.py must run first)
  data/processed/convlstm_y_train.npy
  data/processed/convlstm_X_val.npy
  data/processed/convlstm_y_val.npy
  data/processed/convlstm_X_test.npy
  data/processed/convlstm_y_test.npy
  data/processed/convlstm_meta.npz
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

BASE_DIR  = Path(__file__).resolve().parents[1]
PROC      = BASE_DIR / "data" / "processed"
OUT_DIR   = BASE_DIR / "outputs"; OUT_DIR.mkdir(exist_ok=True)

HIDDEN_DIM  = 32
KERNEL_SIZE = 3
NUM_LAYERS  = 2
SPATIAL_DROPOUT = 0.4
NUM_CLASSES = 3
BATCH_SIZE  = 4
MAX_EPOCHS  = 80
PATIENCE    = 10          # early-stopping patience (val-loss epochs)
LR          = 1e-3
WEIGHT_DECAY= 1e-5
IGNORE_IDX  = -99         # label for masked / out-of-domain pixels

# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# --------------------------------------------------------------------------
# 1. Load arrays
# --------------------------------------------------------------------------
print("Loading dataset arrays ...")
X_train = torch.from_numpy(np.load(PROC / "convlstm_X_train.npy"))   # float32
y_train = torch.from_numpy(np.load(PROC / "convlstm_y_train.npy"))   # int64
X_val   = torch.from_numpy(np.load(PROC / "convlstm_X_val.npy"))
y_val   = torch.from_numpy(np.load(PROC / "convlstm_y_val.npy"))
X_test  = torch.from_numpy(np.load(PROC / "convlstm_X_test.npy"))
y_test  = torch.from_numpy(np.load(PROC / "convlstm_y_test.npy"))
meta    = np.load(PROC / "convlstm_meta.npz", allow_pickle=True)

print(f"  X_train {tuple(X_train.shape)}  y_train {tuple(y_train.shape)}")
print(f"  X_val   {tuple(X_val.shape)}    y_val   {tuple(y_val.shape)}")
print(f"  X_test  {tuple(X_test.shape)}   y_test  {tuple(y_test.shape)}")

# Compute class weights from training labels (excluding ignore_index)
train_valid = y_train[y_train != IGNORE_IDX]
class_counts = torch.bincount(train_valid.view(-1), minlength=NUM_CLASSES).float()
total_count = class_counts.sum().item()
w_dry = total_count / (NUM_CLASSES * class_counts[0].item()) if class_counts[0] > 0 else 0.0
w_normal = total_count / (NUM_CLASSES * class_counts[1].item()) if class_counts[1] > 0 else 0.0
w_wet = total_count / (NUM_CLASSES * class_counts[2].item()) if class_counts[2] > 0 else 0.0
class_weights = torch.tensor([w_dry, w_normal, w_wet], dtype=torch.float32).to(device)
print("Class counts (encoded [0=dry,1=normal,2=wet]):", class_counts.int().tolist())
print("Class weights:", [round(float(w), 4) for w in class_weights.tolist()])

train_loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False,
)

# --------------------------------------------------------------------------
# 2. ConvLSTM cell
# --------------------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell.

    References
    ----------
    Shi et al. (2015) "Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting", NeurIPS.
    """

    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_dim = hidden_dim
        # Combined gate convolution (input + hidden → 4 * hidden_dim)
        self.gates = nn.Conv2d(
            in_channels + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=1)       # (B, C+H, lat, lon)
        gates    = self.gates(combined)            # (B, 4H, lat, lon)
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(
        self, batch: int, lat: int, lon: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch, self.hidden_dim, lat, lon),
            torch.zeros(batch, self.hidden_dim, lat, lon),
        )


# --------------------------------------------------------------------------
# 3. Full ConvLSTM model
# --------------------------------------------------------------------------
class ConvLSTMForecast(nn.Module):
    """
    Multi-layer ConvLSTM encoder followed by a 1×1 Conv classifier.

    Forward input : (B, seq_len, C, lat, lon)
    Forward output: (B, num_classes, lat, lon)  — raw logits
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        kernel_size: int,
        num_layers: int,
        num_classes: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        cells = []
        for layer in range(num_layers):
            inp = in_channels if layer == 0 else hidden_dim
            cells.append(ConvLSTMCell(inp, hidden_dim, kernel_size))
        self.cells = nn.ModuleList(cells)

        # BatchNorm + spatial dropout help regularize small-data training.
        self.bn   = nn.BatchNorm2d(hidden_dim)
        self.drop = nn.Dropout2d(p=SPATIAL_DROPOUT)
        self.head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, seq_len, C, lat, lon = x.shape

        # Initialise hidden / cell states on correct device
        h = [None] * self.num_layers
        c = [None] * self.num_layers
        for l, cell in enumerate(self.cells):
            h[l], c[l] = cell.init_hidden(B, lat, lon)
            h[l], c[l] = h[l].to(x.device), c[l].to(x.device)

        # Unroll across sequence length
        out = None
        for t in range(seq_len):
            inp = x[:, t]          # (B, C, lat, lon)
            for l, cell in enumerate(self.cells):
                h[l], c[l] = cell(inp, h[l], c[l])
                inp = h[l]         # next layer receives current hidden state
            out = h[-1]            # last layer's hidden state at final step

        logits = self.head(self.drop(self.bn(out)))    # (B, num_classes, lat, lon)
        return logits


# --------------------------------------------------------------------------
# 4. Instantiate model and optimiser
# --------------------------------------------------------------------------
in_channels = X_train.shape[2]    # C dimension
model = ConvLSTMForecast(
    in_channels=in_channels,
    hidden_dim=HIDDEN_DIM,
    kernel_size=KERNEL_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_IDX)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-5,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# --------------------------------------------------------------------------
# 5. Training loop
# --------------------------------------------------------------------------
print("Training ...")
best_val_loss  = float("inf")
best_state     = None
no_improve     = 0
train_losses   = []
val_losses     = []

for epoch in range(1, MAX_EPOCHS + 1):
    # --- train ---
    model.train()
    epoch_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(X_b)                    # (B, 3, lat, lon)
        loss   = criterion(logits, y_b)        # CrossEntropyLoss flattens internally
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    train_loss = epoch_loss / len(train_loader)
    train_losses.append(train_loss)

    # --- validate ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits   = model(X_b)
            val_loss += criterion(logits, y_b).item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    scheduler.step(val_loss)

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve    = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}  (best val={best_val_loss:.4f})")
            break

# --------------------------------------------------------------------------
# 6. Evaluate on test set with best weights
# --------------------------------------------------------------------------
model.load_state_dict(best_state)
model.eval()

all_probs = []
all_preds = []
all_true  = []

with torch.no_grad():
    for X_b, y_b in DataLoader(
        TensorDataset(X_test, y_test), batch_size=BATCH_SIZE
    ):
        logits    = model(X_b.to(device))                    # (B, 3, lat, lon)
        probs     = torch.softmax(logits, dim=1).cpu()       # (B, 3, lat, lon)
        preds     = logits.argmax(dim=1).cpu()               # (B, lat, lon)  — encoded labels
        all_probs.append(probs.numpy())
        all_preds.append(preds.numpy())
        all_true.append(y_b.numpy())

prob_arr = np.concatenate(all_probs).astype("float32")  # (N_test, 3, lat, lon)
pred_enc = np.concatenate(all_preds)   # (N_test, lat, lon)
true_enc = np.concatenate(all_true)    # (N_test, lat, lon)

# Mask out invalid pixels (-99) for reporting
LABEL_DEC = {0: -1, 1: 0, 2: 1}
valid_mask = true_enc != IGNORE_IDX

y_pred_flat = np.array([LABEL_DEC.get(p, 0) for p in pred_enc[valid_mask].ravel()])
y_true_flat = np.array([LABEL_DEC.get(t, 0) for t in true_enc[valid_mask].ravel()])

report = classification_report(
    y_true_flat, y_pred_flat,
    labels=[-1, 0, 1],
    target_names=["Dry (−1)", "Normal (0)", "Wet (+1)"],
)
acc = (y_pred_flat == y_true_flat).mean()
print(report)

metrics_text = (
    f"ConvLSTM Forecast — Test Metrics (valid pixels only)\n"
    f"{'='*52}\n"
    f"Hidden dim   : {HIDDEN_DIM}    Layers: {NUM_LAYERS}    "
    f"Seq len: {X_train.shape[1]}    Channels: {in_channels}\n"
    f"Kernel size  : {KERNEL_SIZE}×{KERNEL_SIZE}\n"
    f"Spatial dropout p: {SPATIAL_DROPOUT:.2f}\n"
    f"Total params : {total_params:,}\n"
    f"Best val loss: {best_val_loss:.4f}\n\n"
    f"Overall Accuracy (valid pixels): {acc:.4f}\n\n"
    f"{report}\n"
)
print(metrics_text)
(OUT_DIR / "convlstm_metrics.txt").write_text(metrics_text)

# --------------------------------------------------------------------------
# 7. Confusion matrix
# --------------------------------------------------------------------------
cm  = confusion_matrix(y_true_flat, y_pred_flat, labels=[-1, 0, 1])
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Dry (−1)", "Normal (0)", "Wet (+1)"],
).plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("ConvLSTM — Test Confusion Matrix")
plt.tight_layout()
fig.savefig(OUT_DIR / "convlstm_cm.png", dpi=150)
plt.close(fig)

# --------------------------------------------------------------------------
# 8. Training-curve plot
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(train_losses, label="Train loss")
ax.plot(val_losses,   label="Val loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-entropy loss")
ax.set_title("ConvLSTM — Training Curve")
ax.legend()
plt.tight_layout()
fig.savefig(OUT_DIR / "convlstm_training_curve.png", dpi=150)
plt.close(fig)

# --------------------------------------------------------------------------
# 9. Save model and predictions
# --------------------------------------------------------------------------
torch.save(
    {"model_state": best_state, "config": {
        "in_channels": in_channels, "hidden_dim": HIDDEN_DIM,
        "kernel_size": KERNEL_SIZE, "num_layers": NUM_LAYERS,
        "num_classes": NUM_CLASSES, "seq_len": int(X_train.shape[1]),
    }},
    OUT_DIR / "convlstm_model.pt",
)

np.savez(
    OUT_DIR / "convlstm_test_probs.npz",
    proba=prob_arr,    # (N_test, 3, lat, lon); class axis: [dry(-1), normal(0), wet(+1)]
    test_feature_times=meta["test_feature_times"],
    test_target_times=meta["test_target_times"],
    target_alignment_version=meta["target_alignment_version"],
)

np.savez(
    OUT_DIR / "convlstm_test_preds.npz",
    pred_enc=pred_enc.astype("int8"),
    true_enc=true_enc.astype("int8"),
    test_feature_times=meta["test_feature_times"],
    test_target_times=meta["test_target_times"],
    target_alignment_version=meta["target_alignment_version"],
)

print("Saved outputs to", OUT_DIR)
