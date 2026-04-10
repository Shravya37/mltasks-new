import sys
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split


def get_task_metadata():
    return {
        "task_id": "ae_new4_denoising_synthetic",
        "series": "Autoencoders",
        "task_type": "denoising_autoencoder",
        "description": "Denoising autoencoder on synthetic vectors"
    }


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_dataloaders(batch_size=32):
    clean_data = torch.randn(1000, 20)
    noisy_data = clean_data + 0.30 * torch.randn(1000, 20)

    dataset = TensorDataset(noisy_data, clean_data)

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
        "input_dim": 20,
        "latent_dim": 8,
        "train_size": train_size,
        "val_size": val_size
    }

    return train_loader, val_loader, metadata


def build_model(input_dim, latent_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, 16),
        nn.ReLU(),
        nn.Linear(16, latent_dim),
        nn.ReLU(),
        nn.Linear(latent_dim, 16),
        nn.ReLU(),
        nn.Linear(16, input_dim)
    )
    return model


def train(model, train_loader, val_loader, device, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for noisy_batch, clean_batch in train_loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)

            optimizer.zero_grad()
            outputs = model(noisy_batch)
            loss = criterion(outputs, clean_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for noisy_batch, clean_batch in val_loader:
                noisy_batch = noisy_batch.to(device)
                clean_batch = clean_batch.to(device)

                outputs = model(noisy_batch)
                loss = criterion(outputs, clean_batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)

    return history


def evaluate(model, data_loader, device):
    criterion = nn.MSELoss()
    model.eval()

    total_loss = 0.0
    noisy_baseline_loss = 0.0

    with torch.no_grad():
        for noisy_batch, clean_batch in data_loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)

            outputs = model(noisy_batch)

            loss = criterion(outputs, clean_batch)
            baseline = criterion(noisy_batch, clean_batch)

            total_loss += loss.item()
            noisy_baseline_loss += baseline.item()

    mse = total_loss / len(data_loader)
    baseline_mse = noisy_baseline_loss / len(data_loader)

    return {
        "mse": mse,
        "baseline_mse": baseline_mse,
        "improvement": baseline_mse - mse
    }


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        outputs = model(X)
    return outputs


def save_artifacts(history, sample_noisy, sample_clean, sample_recon):
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("ae_new4_denoising_synthetic Loss Curve")
    plt.legend()
    plt.savefig("ae_new4_denoising_synthetic_loss.png")
    plt.close()

    plt.figure(figsize=(10, 6))

    for i in range(3):
        plt.subplot(3, 3, i + 1)
        plt.plot(sample_noisy[i].cpu().numpy())
        plt.title("Noisy " + str(i + 1))

        plt.subplot(3, 3, i + 4)
        plt.plot(sample_recon[i].cpu().numpy())
        plt.title("Reconstructed " + str(i + 1))

        plt.subplot(3, 3, i + 7)
        plt.plot(sample_clean[i].cpu().numpy())
        plt.title("Clean " + str(i + 1))

    plt.tight_layout()
    plt.savefig("ae_new4_denoising_synthetic_recon.png")
    plt.close()

    torch.save(
        {
            "history": history
        },
        "ae_new4_denoising_synthetic_artifacts.pt"
    )


if __name__ == "__main__":
    try:
        set_seed(42)
        device = get_device()

        train_loader, val_loader, metadata = make_dataloaders()
        model = build_model(metadata["input_dim"], metadata["latent_dim"]).to(device)

        history = train(model, train_loader, val_loader, device, epochs=100, lr=0.001)

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        print("Task:", get_task_metadata()["task_id"])
        print("Train metrics:", train_metrics)
        print("Validation metrics:", val_metrics)

        sample_noisy, sample_clean = next(iter(val_loader))
        sample_noisy = sample_noisy[:3].to(device)
        sample_clean = sample_clean[:3].to(device)
        sample_recon = predict(model, sample_noisy, device)

        save_artifacts(history, sample_noisy, sample_clean, sample_recon)

        assert val_metrics["mse"] < val_metrics["baseline_mse"], "Model did not improve over noisy baseline"
        assert val_metrics["improvement"] > 0.02, "Denoising improvement is too small"

        sys.exit(0)

    except Exception as e:
        print("Error:", e)
        sys.exit(1)
