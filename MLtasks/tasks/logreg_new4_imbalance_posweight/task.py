"""
Logistic Regression using Raw PyTorch Tensors (Imbalanced Binary Classification + Pos Weight)

- Data: synthetic imbalanced labels (e.g., ~20% positives)
- Model: p(y=1|x) = sigmoid(X @ w + b)
- Loss: Weighted BCE (pos_weight)
- Uses ONLY PyTorch tensors (no torch.nn, no torch.optim, no autograd)
"""

import os
import sys
import json
import numpy as np
import torch

OUTPUT_DIR = "/Developer/AIserver/output/tasks/logreg_new4_imbalance_posweight"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    return {
        "task_name": "logreg_new4_imbalance_posweight",
        "description": "Imbalanced logistic regression with weighted BCE (pos_weight) using raw PyTorch tensors",
        "input_dim": 20,
        "output_dim": 1,
        "model_type": "logistic_regression_raw_tensors",
        "loss_type": "weighted_binary_cross_entropy",
        "optimization": "manual_gradient_descent_with_pos_weight",
        "metrics": ["accuracy", "bce_weighted"],
    }


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(
    n_samples=6000,
    train_ratio=0.8,
    batch_size=128,
    n_features=20,
    pos_ratio=0.2,
    seed=42,
):
    rng = np.random.RandomState(seed)

    X = rng.randn(n_samples, n_features).astype(np.float32)
    true_w = rng.randn(n_features, 1).astype(np.float32)
    true_b = -0.5  # shift to reduce positives a bit

    logits = X @ true_w + true_b + 0.5 * rng.randn(n_samples, 1).astype(np.float32)

    # Choose threshold so that about pos_ratio are positive
    thresh = np.quantile(logits, 1.0 - pos_ratio)
    y = (logits >= thresh).astype(np.float32)

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    n_train = int(n_samples * train_ratio)
    X_train, X_val = X_tensor[:n_train], X_tensor[n_train:]
    y_train, y_val = y_tensor[:n_train], y_tensor[n_train:]

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_ds = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Compute pos_weight = (#neg / #pos) on train set
    pos = float(y_train.sum().item())
    neg = float(y_train.numel() - pos)
    pos_weight = (neg / max(pos, 1.0))

    return train_loader, val_loader, pos_weight


def sigmoid(z):
    return 1.0 / (1.0 + torch.exp(-z))


class LogisticRegressionRaw:
    def __init__(self, input_dim, device=None):
        self.device = device if device is not None else get_device()
        self.w = torch.zeros((input_dim, 1), device=self.device)
        self.b = torch.zeros((), device=self.device)

    def forward_logits(self, X):
        return X @ self.w + self.b

    def forward_proba(self, X):
        return sigmoid(self.forward_logits(X))

    def weighted_bce_loss(self, p, y, pos_weight):
        """
        Weighted BCE:
          L = - [ pos_weight*y*log(p) + (1-y)*log(1-p) ]
        """
        eps = 1e-7
        p = torch.clamp(p, eps, 1 - eps)
        return torch.mean(-(pos_weight * y * torch.log(p) + (1 - y) * torch.log(1 - p)))

    def compute_gradients(self, X, p, y, pos_weight):
        """
        Gradient for weighted BCE:
          e = (p - y) * w_y
        where w_y = pos_weight if y=1 else 1
        """
        N = X.shape[0]
        w_y = torch.where(y > 0.5, torch.tensor(pos_weight, device=X.device), torch.tensor(1.0, device=X.device))
        e = (p - y) * w_y
        grad_w = (1.0 / N) * (X.T @ e)
        grad_b = torch.mean(e)
        return grad_w, grad_b

    def update(self, grad_w, grad_b, lr):
        self.w = self.w - lr * grad_w
        self.b = self.b - lr * grad_b

    def fit(self, train_loader, epochs=250, lr=0.1, pos_weight=1.0, verbose=True):
        for ep in range(epochs):
            total = 0.0
            n_batches = 0
            for Xb, yb in train_loader:
                Xb = Xb.to(self.device)
                yb = yb.to(self.device)

                p = self.forward_proba(Xb)
                loss = self.weighted_bce_loss(p, yb, pos_weight=pos_weight)

                grad_w, grad_b = self.compute_gradients(Xb, p, yb, pos_weight=pos_weight)
                self.update(grad_w, grad_b, lr)

                total += float(loss.item())
                n_batches += 1

            if verbose and (ep == 0 or (ep + 1) % 50 == 0):
                print(f"Epoch {ep+1}/{epochs} - weighted_bce: {total / max(n_batches,1):.4f}")

    def evaluate(self, loader, pos_weight=1.0):
        probs = []
        targets = []
        for Xb, yb in loader:
            Xb = Xb.to(self.device)
            yb = yb.to(self.device)
            with torch.no_grad():
                probs.append(self.forward_proba(Xb))
                targets.append(yb)
        p = torch.cat(probs, dim=0)
        y = torch.cat(targets, dim=0)

        wbce = self.weighted_bce_loss(p, y, pos_weight=pos_weight).item()
        y_hat = (p >= 0.5).float()
        acc = torch.mean((y_hat == y).float()).item()
        return {"bce_weighted": float(wbce), "accuracy": float(acc)}

    def state_dict(self):
        return {"w": self.w.detach().cpu(), "b": float(self.b.detach().cpu().item())}


def build_model(device=None):
    device = device if device is not None else get_device()
    return LogisticRegressionRaw(input_dim=20, device=device)


def train(model, train_loader, epochs=250, lr=0.1, pos_weight=1.0):
    model.fit(train_loader, epochs=epochs, lr=lr, pos_weight=pos_weight, verbose=True)
    return model


def evaluate(model, loader, device=None, pos_weight=1.0):
    return model.evaluate(loader, pos_weight=pos_weight)


def predict(model, X, device=None):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    X = X.to(model.device)
    with torch.no_grad():
        return model.forward_proba(X).cpu()


def save_artifacts(model, metrics, save_dir=OUTPUT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, pos_weight = make_dataloaders(
        n_samples=6000,
        train_ratio=0.8,
        batch_size=128,
        n_features=20,
        pos_ratio=0.2,
        seed=42,
    )
    print(f"Computed pos_weight (neg/pos): {pos_weight:.3f}")

    model = build_model(device=device)
    train(model, train_loader, epochs=250, lr=0.1, pos_weight=pos_weight)

    train_metrics = evaluate(model, train_loader, pos_weight=pos_weight)
    val_metrics = evaluate(model, val_loader, pos_weight=pos_weight)

    print("TRAIN:", train_metrics)
    print("VAL:", val_metrics)

    save_artifacts(model, {"train": train_metrics, "val": val_metrics, "pos_weight": pos_weight}, save_dir=OUTPUT_DIR)

    # With imbalance, accuracy can still be high, but we require decent > 0.80
    passed = val_metrics["accuracy"] >= 0.80
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()