import csv
import random

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
MANIFEST_PATH: Final[Path] = PROJECT_ROOT / "data" / "manifest.csv"

IMAGE_SIZE: Final[tuple[int, int]] = (256, 256)


class ImageRecord(TypedDict):
    image_path: Path
    class_id: str
    impression_id: int


class FingerprintPair(TypedDict):
    image1: ImageRecord
    image2: ImageRecord
    label: int


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> list[ImageRecord]:
    """Load the CSV manifest and resolve image paths."""
    records_: list[ImageRecord] = []

    with open(manifest_path, newline="", encoding="utf-8") as file:
        reader: csv.DictReader[str] = csv.DictReader(file)

        for row in reader:
            relative_path = Path(row["image_path"])

            records_.append({
                "image_path": (
                    relative_path
                    if relative_path.is_absolute()
                    else PROJECT_ROOT / relative_path
                ),
                "class_id": row["class_id"],
                "impression_id": int(row["impression_id"]),
            })

    missing = [
        str(record["image_path"])
        for record in records_
        if not record["image_path"].is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest image(s) are missing. "
            f"First missing path: {missing[0]}"
        )

    if not records_:
        raise ValueError(f"No image records found in {manifest_path}")

    return records_


def split_by_impression(records_: list[ImageRecord], seed: int = 42) -> dict[str, list[ImageRecord]]:
    """
    Split impressions within each finger identity.

    Default split: 4 train, 2 validation, 2 test impressions.
    All images of an individual impression stay in one split.
    """
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)

    for record in records_:
        grouped[record["class_id"]].append(record)

    rng = random.Random(seed)

    splits_: dict[str, list[ImageRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for class_id, images in sorted(grouped.items()):
        images = sorted(
            images,
            key=lambda record__: record__["impression_id"],
        )

        impression_ids = [image["impression_id"] for image in images]
        if len(impression_ids) != len(set(impression_ids)):
            raise ValueError(
                f"Duplicate impression IDs for {class_id}. "
                "Check the manifest before splitting."
            )

        if len(images) < 8:
            raise ValueError(
                f"{class_id} has only {len(images)} images; "
                "this split requires at least 8."
            )

        rng.shuffle(images)

        splits_["train"].extend(images[:4])
        splits_["validation"].extend(images[4:6])
        splits_["test"].extend(images[6:8])

    return splits_


def generate_pairs(records_: list[ImageRecord], seed: int = 42) -> list[FingerprintPair]:
    """
    Generate balanced positive and negative pairs.

    Label 1 = same finger.
    Label 0 = different fingers.
    """
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)

    for record in records_:
        grouped[record["class_id"]].append(record)

    identities = sorted(grouped)

    if len(identities) < 2:
        raise ValueError("At least two finger identities are required.")

    positive_pairs: list[FingerprintPair] = []
    negative_candidates: list[FingerprintPair] = []

    for class_id in identities:
        images = grouped[class_id]

        for image1_, image2_ in combinations(images, 2):
            positive_pairs.append({
                "image1": image1_,
                "image2": image2_,
                "label": 1,
            })

    for index, finger1 in enumerate(identities):
        for finger2 in identities[index + 1:]:
            for image1_ in grouped[finger1]:
                for image2_ in grouped[finger2]:
                    negative_candidates.append({
                        "image1": image1_,
                        "image2": image2_,
                        "label": 0,
                    })

    if not positive_pairs:
        raise ValueError(
            "No positive pairs available. Each identity needs "
            "at least two images in this split."
        )

    rng = random.Random(seed)
    negative_count = min(
        len(negative_candidates),
        len(positive_pairs),
    )
    negative_pairs = rng.sample(
        negative_candidates,
        k=negative_count,
    )

    pairs_ = positive_pairs + negative_pairs
    rng.shuffle(pairs_)

    return pairs_


def load_image(path: Path) -> torch.Tensor:
    """Load a TIFF as a normalized grayscale tensor [1, H, W]."""
    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR,
        )
        array = np.array(image, dtype=np.float32, copy=True) / 255.0

    return torch.from_numpy(array).unsqueeze(0)


class FingerprintPairDataset(Dataset):
    """PyTorch Dataset returning image pairs and binary labels."""
    def __init__(self, pairs_: list[FingerprintPair]):
        self.pairs = pairs_

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair = self.pairs[index]

        image1_ = load_image(pair["image1"]["image_path"])
        image2_ = load_image(pair["image2"]["image_path"])
        label_ = torch.tensor(pair["label"], dtype=torch.float32)

        return image1_, image2_, label_


if __name__ == "__main__":
    records = load_manifest()
    splits = split_by_impression(records)

    for split_name, split_records in splits.items():
        pairs = generate_pairs(
            split_records,
            seed=42,
        )
        positives = sum(pair["label"] == 1 for pair in pairs)
        negatives = sum(pair["label"] == 0 for pair in pairs)

        print(
            f"{split_name}: {len(split_records)} images, "
            f"{len(pairs)} pairs "
            f"({positives} genuine, {negatives} impostor)"
        )

    train_pairs = generate_pairs(splits["train"])
    dataset = FingerprintPairDataset(train_pairs)

    image1, image2, label = dataset[0]
    print("Tensor shape:", image1.shape)
    print("Pair label:", label.item())
