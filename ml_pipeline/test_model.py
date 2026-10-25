import random
from pathlib import Path

import numpy as np
import torch
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
            labels_list.append(labels.cpu().numpy())

    if not distances:
        raise ValueError("No pairs were evaluated.")

    return np.concatenate(distances), np.concatenate(labels_list).astype(
        np.int64
    )


def choose_eer_threshold(
        distances: np.ndarray,
        labels: np.ndarray,
) -> tuple[float, float]:
    """
    Find the distance threshold where FAR and FRR are closest.

    Label 1 = genuine; label 0 = impostor.
    Lower distances indicate more likely genuine matches.
    """
    genuine = labels == 1
    impostor = labels == 0

    if not genuine.any() or not impostor.any():
        raise ValueError(
            "Threshold calibration requires genuine and impostor pairs."
        )

    # Test each distinct validation distance as a threshold.
    thresholds = np.unique(distances)

    genuine_distances = np.sort(distances[genuine])
    impostor_distances = np.sort(distances[impostor])

    genuine_accepted = np.searchsorted(
        genuine_distances, thresholds, side="right"
    )
    impostor_accepted = np.searchsorted(
        impostor_distances, thresholds, side="right"
    )

    far = impostor_accepted / impostor.sum()
    frr = (genuine.sum() - genuine_accepted) / genuine.sum()

    index = int(np.argmin(np.abs(far - frr)))

    threshold = float(thresholds[index])
    eer = float((far[index] + frr[index]) / 2.0)

    return threshold, eer


def precision_recall_metrics(
        distances: np.ndarray,
        labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Calculate a precision-recall curve, Average Precision, and
    trapezoidal PR-AUC without scikit-learn.

    Smaller distances indicate genuine matches.
    """
    positives = int(np.sum(labels == 1))

    if positives == 0 or positives == len(labels):
        raise ValueError(
            "PR metrics require both genuine and impostor pairs."
        )

    # Higher scores should indicate genuine pairs.
    scores = -distances

    # Sort scores from highest to lowest.
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    cumulative_tp = np.cumsum(sorted_labels == 1)
    cumulative_fp = np.cumsum(sorted_labels == 0)

    # Evaluate only at the end of each group of tied scores.
    group_ends = np.r_[
        np.flatnonzero(np.diff(sorted_scores) != 0),
        len(sorted_scores) - 1,
    ]

    true_positives = cumulative_tp[group_ends]
    false_positives = cumulative_fp[group_ends]

    precision = true_positives / (
            true_positives + false_positives
    )
    recall = true_positives / positives

    # Average Precision: sum of recall increments times precision.
    previous_recall = np.r_[0.0, recall[:-1]]
    average_precision = float(
        np.sum((recall - previous_recall) * precision)
    )

    # Add the conventional starting point for trapezoidal integration.
    curve_precision = np.r_[1.0, precision]
    curve_recall = np.r_[0.0, recall]

    # Manual trapezoidal integration; no sklearn or SciPy required.
    pr_auc = float(
        np.sum(
            np.diff(curve_recall)
            * (curve_precision[:-1] + curve_precision[1:])
            / 2.0
        )
    )

    return curve_precision, curve_recall, average_precision, pr_auc


def calculate_metrics(
        distances: np.ndarray,
        labels: np.ndarray,
        threshold: float,
) -> dict[str, float | int | np.ndarray]:
    """Calculate threshold-based verification metrics."""
    genuine = labels == 1
    impostor = labels == 0

    if not genuine.any() or not impostor.any():
        raise ValueError(
            "Evaluation requires both genuine and impostor pairs."
        )

    predictions = (distances <= threshold).astype(np.int64)

    # Confusion matrix: rows = actual, columns = predicted.
    #                 Predicted impostor   Predicted genuine
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))

    confusion_matrix = np.array([
        [tn, fp],
        [fn, tp],
    ], dtype=np.int64)

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    far = fp / (fp + tn)
    frr = fn / (fn + tp)

    _, _, average_precision, pr_auc = precision_recall_metrics(
        distances, labels
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "far": float(far),
        "frr": float(frr),
        "average_precision": average_precision,
        "pr_auc": pr_auc,
        "confusion_matrix": confusion_matrix,
    }


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

    # Calibrate the threshold using validation data only.
    val_distances, val_labels = collect_distances(
        model, val_loader, device
    )
    threshold, val_eer = choose_eer_threshold(
        val_distances, val_labels
    )

    # Evaluate on the test set with the frozen threshold.
    test_distances, test_labels = collect_distances(
        model, test_loader, device
    )

    metrics = calculate_metrics(
        test_distances,
        test_labels,
        threshold,
    )

    print("\nVerification results")
    print("-" * 40)
    print(f"Checkpoint epoch:       {checkpoint['epoch']}")
    print(f"Validation EER:         {val_eer:.4f}")
    print(f"Distance threshold:     {threshold:.6f}")
    print(f"Test accuracy:          {metrics['accuracy']:.4f}")
    print(f"Test precision:         {metrics['precision']:.4f}")
    print(f"Test recall:            {metrics['recall']:.4f}")
    print(f"Test F1 score:          {metrics['f1']:.4f}")
    print(f"Test Average Precision: {metrics['average_precision']:.4f}")
    print(f"Test PR-AUC:            {metrics['pr_auc']:.4f}")
    print(f"Test FAR:               {metrics['far']:.4f}")
    print(f"Test FRR:               {metrics['frr']:.4f}")

    print("\nConfusion matrix")
    print("Rows = actual; columns = predicted")
    print("                 Predicted")
    print("                 Impostor  Genuine")
    cm = metrics["confusion_matrix"]
    print(f"Actual impostor  {cm[0, 0]:8d}  {cm[0, 1]:7d}")
    print(f"Actual genuine   {cm[1, 0]:8d}  {cm[1, 1]:7d}")


if __name__ == "__main__":
    main()
