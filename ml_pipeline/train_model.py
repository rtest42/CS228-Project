import numpy as np
import random
import torch

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Final

from device import get_device
from fingerprint_dataset import (
    FingerprintPairDataset,
    generate_pairs,
    load_manifest,
    split_by_impression,
    ImageRecord,
    FingerprintPair
)
from siamese_model import SiameseNetwork, ContrastiveLoss


SEED: Final[int] = 42
BATCH_SIZE: Final[int] = 32
EPOCHS: Final[int] = 99
LEARNING_RATE: Final[float] = 1e-3
PATIENCE: Final[int] = 5

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
CHECKPOINT_DIR: Final[Path] = PROJECT_ROOT / "checkpoints"
BEST_MODEL_PATH: Final[Path] = CHECKPOINT_DIR / "siamese_best_enhanced.pt"

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
MANIFEST_PATH: Final[Path] = PROJECT_ROOT / "data" / "manifest_enhanced.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def run_epoch(
        model: SiameseNetwork,
        loader: DataLoader,
        criterion: ContrastiveLoss,
        device: torch.device,
        optimizer: torch.optim.Optimizer | None = None,
        description: str = "Training"
) -> float:
    training: bool = optimizer is not None
    model.train(training)

    total_loss: float = 0.0
    total_samples: int = 0

    with tqdm(loader, desc=description, leave=False, dynamic_ncols=True, position=1) as progress:
        with torch.set_grad_enabled(training):
            for image1, image2, labels in progress:
                image1 = image1.to(device)
                image2 = image2.to(device)
                labels = labels.to(device)

                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)

                embedding1, embedding2 = model(image1, image2)
                loss = criterion(embedding1, embedding2, labels)

                if optimizer is not None:
                    loss.backward()
                    optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                progress.set_postfix(
                    loss=f"{total_loss / total_samples:.4f}"
                )

    if total_samples == 0:
        raise ValueError("Cannot run an epoch with an empty DataLoader.")

    return total_loss / total_samples


def main() -> None:
    set_seed(SEED)

    device: torch.device = get_device()
    print(f"Using device: {device}", flush=True)

    records: list[ImageRecord] = load_manifest(MANIFEST_PATH)
    splits: dict[str, list[ImageRecord]] = split_by_impression(records, seed=SEED)

    train_pairs: list[FingerprintPair] = generate_pairs(splits["train"], seed=SEED)
    val_pairs: list[FingerprintPair] = generate_pairs(splits["validation"], seed=SEED)

    train_dataset: FingerprintPairDataset = FingerprintPairDataset(train_pairs)
    val_dataset: FingerprintPairDataset = FingerprintPairDataset(val_pairs)

    # Start with num_workers=0 for compatibility with Windows.
    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda"
    )

    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda"
    )

    model: SiameseNetwork = SiameseNetwork(embedding_dim=128).to(device)
    criterion: ContrastiveLoss = ContrastiveLoss(margin=1.0)
    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss: float = float("inf")
    epochs_without_improvement: int = 0

    with tqdm(
        range(1, EPOCHS + 1),
        desc="Epochs",
        position=0,
        dynamic_ncols=True
    ) as epoch_bar:
        for epoch in epoch_bar:
            train_loss = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                description="Training"
            )

            val_loss = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                description="Validation"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "embedding_dim": 128,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "seed": SEED,
                    },
                    BEST_MODEL_PATH,
                )

                status = "Best checkpoint saved"
                # print(f"Saved best checkpoint: {BEST_MODEL_PATH}", flush=True)
            else:
                epochs_without_improvement += 1
                status = f"No improvement {epochs_without_improvement}/{PATIENCE}"
                # print(
                #    "No meaningful validation improvement: "
                #    f"{epochs_without_improvement}/{PATIENCE}",
                #    flush=True
                # )

            epoch_bar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                best=f"{best_val_loss:.4f}",
                status=status
            )

            if epochs_without_improvement >= PATIENCE:
                epoch_bar.set_description("Early stopping")
                # print(f"Early stopping at epoch {epoch}.", flush=True)
                break

    print(f"Best validation loss: {best_val_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
