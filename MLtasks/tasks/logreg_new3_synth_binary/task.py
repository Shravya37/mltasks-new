"""
Linear Regression using Raw PyTorch Tensors (Multivariate + Weight Decay)

- Data: y = X @ w + b + noise
- Uses ONLY PyTorch tensors (no torch.nn, no torch.optim, no autograd)
- Trains with manual gradient descent + L2 regularization (weight decay)
"""

import os
import sys
import json
import numpy as np
import torch

OUTPUT_DIR = "/Developer/AIserver/output/tasks/linreg_new2_weight_decay"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    return {
        "task_name": "linreg_new2_weight_decay",
        "description": "Multivariate linear regression (20 features) with L2 weight decay using raw PyTorch tensors",
        "input_dim": 20,
        "output_dim": 1,
        "model_type": "linear_regression_raw_tensors",
        "loss_type": "mse",
        "optimization": "manual_gradient_descent_with_weight_decay",
    }


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(
    n_samples=2000,
    train_ratio=0.8,
    noise_std=0.5,
    batch_size=64,
    n_features=20,
    seed=42,
):
    rng = np.random.RandomState(seed)

    X = rng.uniform(-5, 5, size=(n_samples, n_features)).astype(np.float32)
    true_w = rng.uniform(-2, 2, size=(n_features, 1)).astype(np.float32)
    true_b = 3.0

    y = X @ true_w + true_b + rng.normal(0, noise_std, size=(n_samples, 1)).astype(np.float32)

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    n_train = int(n_samples * train_ratio)
    X_train, X_val = X_tensor[:n_train], X_tensor[n_train:]
    y_train, y_val = y_tensor[:n_train], y_tensor[n_train:]

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_ds = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class LinearRegressionRaw:
    """
    y_hat = X @ w + b
    w: (D, 1)
    b: scalar
    """

    def __init__(self, input_dim, device=None):
        self.device = device if device is not None else get_device()
        self.w = torch.zeros((input_dim, 1), device=self.device)
        self.b = torch.zeros((), device=self.device)

    def forward(self, X):
        return X @ self.w + self.b

    def compute_loss(self, y_pred, y_true, weight_decay=0.01):
        """
        Loss = MSE + weight_decay * ||w||^2
        """
        mse = torch.mean((y_pred - y_true) ** 2)
        l2 = torch.sum(self.w ** 2)
        return mse + weight_decay * l2

    def compute_gradients(self, X, y_pred, y_true, weight_decay=0.01):
        """
        For MSE part:
          grad_w = (2/N) * X^T @ (y_hat - y)
          grad_b = (2/N) * sum(y_hat - y)

        For L2 part (weight_decay * ||w||^2):
          grad_w += 2 * weight_decay * w
        """
        N = X.shape[0]
        e = (y_pred - y_true)
        grad_w = (2.0 / N) * (X.T @ e)
        grad_b = (2.0 / N) * torch.sum(e)

        # L2 weight decay gradient
        grad_w = grad_w + 2.0 * weight_decay * self.w
        return grad_w, grad_b

    def update(self, grad_w, grad_b, lr):
        self.w = self.w - lr * grad_w
        self.b = self.b - lr * grad_b

    def fit(self, train_loader, epochs=200, lr=0.01, weight_decay=0.01, verbose=True):
        for ep in range(epochs):
            total = 0.0
            n_batches = 0

            for Xb, yb in train_loader:
                Xb = Xb.to(self.device)
                yb = yb.to(self.device)

                y_pred = self.forward(Xb)
                loss = self.compute_loss(y_pred, yb, weight_decay=weight_decay)

                grad_w, grad_b = self.compute_gradients(Xb, y_pred, yb, weight_decay=weight_decay)
                self.update(grad_w, grad_b, lr)

                total += float(loss.item())
                n_batches += 1

            if verbose and (ep == 0 or (ep + 1) % 50 == 0):
                print(f"Epoch {ep+1}/{epochs} - loss: {total / max(n_batches,1):.4f}")

    def evaluate(self, loader):
        preds, targets = [], []
        for Xb, yb in loader:
            Xb = Xb.to(self.device)
            yb = yb.to(self.device)
            with torch.no_grad():
                preds.append(self.forward(Xb))
                targets.append(yb)

        y_pred = torch.cat(preds, dim=0)
        y_true = torch.cat(targets, dim=0)

        mse = torch.mean((y_pred - y_true) ** 2).item()

        ss_res = torch.sum((y_true - y_pred) ** 2).item()
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {"mse": float(mse), "r2": float(r2)}

    def state_dict(self):
        return {"w": self.w.detach().cpu(), "b": float(self.b.detach().cpu().item())}


def build_model(device=None):
    device = device if device is not None else get_device()
    return LinearRegressionRaw(input_dim=20, device=device)


def train(model, train_loader, epochs=200, lr=0.01, weight_decay=0.01):
    model.fit(train_loader, epochs=epochs, lr=lr, weight_decay=weight_decay, verbose=True)
    return model


def evaluate(model, loader, device=None):
    return model.evaluate(loader)


def predict(model, X, device=None):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    X = X.to(model.device)
    with torch.no_grad():
        return (X @ model.w + model.b).cpu()


def save_artifacts(model, metrics, save_dir=OUTPUT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader = make_dataloaders(
        n_samples=2000,
        train_ratio=0.8,
        noise_std=0.5,
        batch_size=64,
        n_features=20,
        seed=42,
    )

    model = build_model(device=device)
    train(model, train_loader, epochs=200, lr=0.01, weight_decay=0.01)

    train_metrics = evaluate(model, train_loader)
    val_metrics = evaluate(model, val_loader)

    print("TRAIN:", train_metrics)
    print("VAL:", val_metrics)

    save_artifacts(model, {"train": train_metrics, "val": val_metrics}, save_dir=OUTPUT_DIR)

    passed = val_metrics["mse"] < 2.0
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()