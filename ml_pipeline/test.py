
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from device import get_device
from fingerprint_dataset import (
    FingerprintPairDataset,
    generate_pairs,
    load_manifest,
    split_by_impression,
)
from siamese_model import SiameseNetwork


SEED = 42
BATCH_SIZE = 32

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "siamese_best.pt"


def collect_distances(
        model: SiameseNetwork,
        loader: DataLoader,
        device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pairwise embedding distances and binary labels."""
    model.eval()

    distances: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    with torch.inference_mode():
        for image1, image2, labels in loader:
            image1 = image1.to(device)
            image2 = image2.to(device)

            embedding1, embedding2 = model(image1, image2)

            batch_distances = torch.linalg.vector_norm(
                embedding1 - embedding2,
                ord=2,
                dim=1,
                )

            distances.append(batch_distances.cpu().numpy())
            labels_list.append(labels.numpy())

    if not distances:
        raise ValueError("No pairs were evaluated.")

    return np.concatenate(distances), np.concatenate(labels_list)


def choose_eer_threshold(
        distances: np.ndarray,
        labels: np.ndarray,
) -> tuple[float, float]:
    """
    Select the validation threshold closest to equal error rate.

    Lower distance means a more likely genuine match.
    Label 1 = genuine; label 0 = impostor.
    """
    if len(np.unique(labels)) != 2:
        raise ValueError("Threshold calibration requires both pair classes.")

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels,
        -distances,
        pos_label=1,
    )

    false_negative_rate = 1.0 - true_positive_rate
    index = int(np.argmin(np.abs(false_positive_rate - false_negative_rate)))

    # roc_curve used -distance, so convert the selected score threshold
    # back to a distance threshold.
    distance_threshold = -float(thresholds[index])

    eer = float(
        (false_positive_rate[index] + false_negative_rate[index]) / 2
    )

    return distance_threshold, eer


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()
    print(f"Using device: {device}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. Run train.py first."
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    embedding_dim = int(checkpoint["embedding_dim"])
    model = SiameseNetwork(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    records = load_manifest()
    splits = split_by_impression(records, seed=SEED)

    # Generate pairs independently within each split.
    val_pairs = generate_pairs(splits["validation"], seed=SEED)
    test_pairs = generate_pairs(splits["test"], seed=SEED)

    val_loader = DataLoader(
        FingerprintPairDataset(val_pairs),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    test_loader = DataLoader(
        FingerprintPairDataset(test_pairs),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # Calibrate on validation data only.
    val_distances, val_labels = collect_distances(
        model, val_loader, device
    )
    threshold, val_eer = choose_eer_threshold(
        val_distances, val_labels
    )

    # Evaluate the frozen threshold on test data.
    test_distances, test_labels = collect_distances(
        model, test_loader, device
    )

    predictions = (test_distances <= threshold).astype(np.int64)

    test_accuracy = accuracy_score(test_labels, predictions)
    test_auc = roc_auc_score(test_labels, -test_distances)

    genuine = test_labels == 1
    impostor = test_labels == 0

    false_accept_rate = float(
        np.mean(test_distances[impostor] <= threshold)
    )
    false_reject_rate = float(
        np.mean(test_distances[genuine] > threshold)
    )

    print("\nVerification results")
    print(f"Checkpoint epoch:       {checkpoint['epoch']}")
    print(f"Validation EER:         {val_eer:.4f}")
    print(f"Distance threshold:     {threshold:.6f}")
    print(f"Test accuracy:          {test_accuracy:.4f}")
    print(f"Test ROC-AUC:           {test_auc:.4f}")
    print(f"Test false accept rate: {false_accept_rate:.4f}")
    print(f"Test false reject rate: {false_reject_rate:.4f}")


if __name__ == "__main__":
    main()
