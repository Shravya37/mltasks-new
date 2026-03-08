import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_squared_error, r2_score


def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name": "linreg_lvl1_synth_mse",
        "task_type": "regression",
        "input_type": "tabular",
        "output_type": "continuous",
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get computation device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(n_samples=4000, test_size=0.2, random_state=42, batch_size=128):
    """Create synthetic dataset and dataloaders."""
    rng = np.random.RandomState(random_state)

    d = 10
    X = rng.randn(n_samples, d).astype(np.float32)
    w = rng.randn(d, 1).astype(np.float32)
    y = X @ w + 0.5 * rng.randn(n_samples, 1).astype(np.float32)

    split = int((1.0 - test_size) * n_samples)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def build_model(device, input_dim=10):
    """Build PyTorch linear regression model."""
    model = nn.Linear(input_dim, 1).to(device)
    return model


def train(model, train_loader, val_loader=None, epochs=60, lr=0.05, verbose=True):
    """Train the linear regression model."""
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * xb.size(0)

        if verbose and (ep == 0 or (ep + 1) % 10 == 0):
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"Epoch {ep+1}/{epochs} - train_loss: {avg_loss:.4f}")

    return model


def evaluate(model, loader, device):
    """Evaluate model and return metrics."""
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds = model(xb).cpu().numpy()
            y_pred.append(preds)
            y_true.append(yb.numpy())

    y_true = np.vstack(y_true).ravel()
    y_pred = np.vstack(y_pred).ravel()

    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {"loss": mse, "mse": mse, "r2": r2}


def predict(model, loader, device):
    """Make predictions on a loader (kept for protocol completeness)."""
    model.eval()
    preds_all = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            preds = model(xb).cpu()
            preds_all.append(preds)
    return torch.cat(preds_all, dim=0)


def save_artifacts(model, metrics, save_dir="output"):
    """Save model artifacts."""
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"Artifacts saved to {save_dir}")


def main():
    print("Starting PyTorch Linear Regression (Synthetic, MSE) Task...")

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    print("Creating dataloaders...")
    train_loader, val_loader = make_dataloaders(
        n_samples=4000, test_size=0.2, random_state=42, batch_size=128
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    print("Building model...")
    model = build_model(device, input_dim=10)

    print("Training model...")
    model = train(model, train_loader, epochs=60, lr=0.05, verbose=True)

    print("Evaluating on training set...")
    train_metrics = evaluate(model, train_loader, device)
    print("Train Metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("Evaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)
    print("Validation Metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("Saving artifacts...")
    save_artifacts(model, val_metrics, save_dir="output")

    print("\n" + "=" * 60)
    print("Final Results:")
    print("=" * 60)
    print(f"MSE (Train): {train_metrics['mse']:.4f}")
    print(f"MSE (Val):   {val_metrics['mse']:.4f}")
    print(f"R²  (Train): {train_metrics['r2']:.4f}")
    print(f"R²  (Val):   {val_metrics['r2']:.4f}")
    print("=" * 60)

    print("\nQuality Check:")
    passed = val_metrics["mse"] < 1.0
    if passed:
        print(f"PASS: Val MSE < 1.0 ({val_metrics['mse']:.4f})")
    else:
        print(f"FAIL: Val MSE >= 1.0 ({val_metrics['mse']:.4f})")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())