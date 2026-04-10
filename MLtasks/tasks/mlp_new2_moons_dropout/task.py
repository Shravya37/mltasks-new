import sys
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader, random_split


def get_task_metadata():
    return {
        "task_id": "mlp_new2_moons_dropout",
        "series": "Neural Networks",
        "task_type": "binary_classification",
        "description": "MLP with dropout on make_moons dataset"
    }


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_dataloaders(batch_size=32):
    X, y = make_moons(n_samples=1000, noise=0.20, random_state=42)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X, y)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    metadata = {
        "input_dim": 2,
        "output_dim": 1,
        "train_size": train_size,
        "val_size": val_size
    }

    return train_loader, val_loader, metadata


def build_model(input_dim, output_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Dropout(0.20),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Dropout(0.20),
        nn.Linear(16, output_dim)
    )
    return model


def train(model, train_loader, val_loader, device, epochs=200, lr=0.01):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)

    return history


def evaluate(model, data_loader, device):
    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_targets.extend(y_batch.cpu().numpy().flatten().tolist())

    acc = accuracy_score(all_targets, all_preds)

    return {
        "loss": total_loss / len(data_loader),
        "accuracy": acc
    }


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        outputs = model(X)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
    return preds


def save_artifacts(history, train_metrics, val_metrics):
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("mlp_new2_moons_dropout Loss Curve")
    plt.legend()
    plt.savefig("mlp_new2_moons_dropout_loss.png")
    plt.close()

    torch.save(
        {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        },
        "mlp_new2_moons_dropout_artifacts.pt"
    )


if __name__ == "__main__":
    try:
        set_seed(42)
        device = get_device()

        train_loader, val_loader, metadata = make_dataloaders()
        model = build_model(metadata["input_dim"], metadata["output_dim"]).to(device)

        history = train(model, train_loader, val_loader, device, epochs=200, lr=0.01)

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        print("Task:", get_task_metadata()["task_id"])
        print("Train metrics:", train_metrics)
        print("Validation metrics:", val_metrics)

        assert val_metrics["accuracy"] > 0.88, "Validation accuracy is too low"

        save_artifacts(history, train_metrics, val_metrics)
        sys.exit(0)

    except Exception as e:
        print("Error:", e)
        sys.exit(1)
