import sys
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split


def get_task_metadata():
    return {
        "task_id": "cnn_new3_synthetic_shapes",
        "series": "Convolutional Neural Networks",
        "task_type": "binary_image_classification",
        "description": "CNN on synthetic 16x16 images with vertical vs horizontal bars"
    }


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_image(label, size=16):
    image = torch.zeros((1, size, size), dtype=torch.float32)

    if label == 0:
        col = random.randint(5, 10)
        image[0, :, col - 1:col + 1] = 1.0
    else:
        row = random.randint(5, 10)
        image[0, row - 1:row + 1, :] = 1.0

    noise = torch.randn((1, size, size)) * 0.15
    image = image + noise
    image = torch.clamp(image, 0.0, 1.0)

    return image


def make_dataloaders(batch_size=32):
    images = []
    labels = []

    for _ in range(500):
        images.append(generate_image(0))
        labels.append(0)

        images.append(generate_image(1))
        labels.append(1)

    X = torch.stack(images)
    y = torch.tensor(labels, dtype=torch.long)

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
        "input_channels": 1,
        "num_classes": 2,
        "image_size": 16,
        "train_size": train_size,
        "val_size": val_size
    }

    return train_loader, val_loader, metadata


def build_model(input_channels, num_classes):
    model = nn.Sequential(
        nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes)
    )
    return model


def train(model, train_loader, val_loader, device, epochs=15, lr=0.001):
    criterion = nn.CrossEntropyLoss()
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
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    confusion = torch.zeros((2, 2), dtype=torch.int32)

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            for true_label, pred_label in zip(y_batch.cpu(), preds.cpu()):
                confusion[int(true_label), int(pred_label)] += 1

    return {
        "loss": total_loss / len(data_loader),
        "accuracy": correct / total,
        "confusion_matrix": confusion.tolist()
    }


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        outputs = model(X)
        preds = torch.argmax(outputs, dim=1)
    return preds


def save_artifacts(history, metrics, sample_batch):
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("cnn_new3_synthetic_shapes Loss Curve")
    plt.legend()
    plt.savefig("cnn_new3_synthetic_shapes_loss.png")
    plt.close()

    images, labels = sample_batch

    plt.figure(figsize=(8, 4))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i][0].cpu(), cmap="gray")
        plt.title("Label: " + str(int(labels[i].item())))
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("cnn_new3_synthetic_shapes_samples.png")
    plt.close()

    confusion = metrics["confusion_matrix"]
    plt.figure(figsize=(4, 4))
    plt.imshow(confusion, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(confusion[i][j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig("cnn_new3_synthetic_shapes_confusion.png")
    plt.close()

    torch.save(
        {
            "history": history,
            "metrics": metrics
        },
        "cnn_new3_synthetic_shapes_artifacts.pt"
    )


if __name__ == "__main__":
    try:
        set_seed(42)
        device = get_device()

        train_loader, val_loader, metadata = make_dataloaders()
        model = build_model(metadata["input_channels"], metadata["num_classes"]).to(device)

        history = train(model, train_loader, val_loader, device, epochs=15, lr=0.001)

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        print("Task:", get_task_metadata()["task_id"])
        print("Train metrics:", train_metrics)
        print("Validation metrics:", val_metrics)

        sample_batch = next(iter(val_loader))
        save_artifacts(history, val_metrics, sample_batch)

        assert val_metrics["accuracy"] > 0.95, "Validation accuracy is too low"

        sys.exit(0)

    except Exception as e:
        print("Error:", e)
        sys.exit(1)
