import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


def get_task_metadata():
    return {
        "task_id": "mlp_new1_xor_classifier",
        "series": "Neural Networks",
        "task_type": "binary_classification",
        "description": "MLP for XOR classification using PyTorch"
    }


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_dataloaders(batch_size=16):
    X = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ], dtype=torch.float32)

    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

    X = X.repeat(100, 1)
    y = y.repeat(100, 1)

    dataset = TensorDataset(X, y)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def build_model():
    return nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )


def train(model, train_loader, device, epochs=200, lr=0.05):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        history.append(total_loss / len(train_loader))

    return history


def evaluate(model, data_loader, device):
    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return {
        "loss": total_loss / len(data_loader),
        "accuracy": correct / total
    }


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        outputs = model(X)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
    return preds


def save_artifacts():
    torch.save({"status": "done"}, "xor_model_artifact.pt")


if __name__ == "__main__":
    try:
        set_seed(42)
        device = get_device()

        train_loader, val_loader = make_dataloaders()
        model = build_model().to(device)

        history = train(model, train_loader, device)
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        print("Task:", get_task_metadata()["task_id"])
        print("Train metrics:", train_metrics)
        print("Validation metrics:", val_metrics)

        assert val_metrics["accuracy"] > 0.95, "Validation accuracy is too low"

        save_artifacts()
        sys.exit(0)

    except Exception as e:
        print("Error:", e)
        sys.exit(1)
