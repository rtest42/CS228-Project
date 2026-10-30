"""
Exploratory data analysis for the FVC 2000/2002/2004 "B" fingerprint sets.

Reads data/manifest.csv (built by download_datasets.py), computes simple
image statistics, runs sanity checks, and writes figures and tables to
results/eda/. Nothing here is random, so the output is deterministic.

Usage:
    python eda.py                    # raw images
    python eda.py --variant enhanced # FpEnhancer output
"""

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Final, TypedDict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from fingerprint_dataset import (
    DEFAULT_VARIANT,
    PROJECT_ROOT,
    VARIANTS,
    ImageRecord,
    generate_pairs,
    load_manifest,
    manifest_for,
    split_by_impression,
)

RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results" / "eda"
SEED: Final[int] = 42

# The TIFFs carry no usable DPI metadata, so these are the nominal sensor
# resolutions published on the FVC competition pages (bias.csr.unibo.it).
# DB4 in every year is synthetic (SFinGe), generated at about 500 dpi.
NOMINAL_DPI: Final[dict[str, int]] = {
    "fvc2002/DB2_B": 569,
    "fvc2004/DB3_B": 512,
}
DEFAULT_DPI: Final[int] = 500
SYNTHETIC_DATABASE: Final[str] = "DB4_B"

# Pixel blocks whose standard deviation exceeds this are treated as
# fingerprint (ridges present) rather than flat background.
BLOCK_SIZE: Final[int] = 16
FOREGROUND_STD: Final[float] = 20.0


class ImageStats(TypedDict):
    image_path: str
    group: str
    dataset: str
    database: str
    finger_id: str
    impression_id: int
    width: int
    height: int
    nominal_dpi: int
    synthetic: bool
    mean: float
    contrast: float
    sharpness: float
    foreground: float
    md5: str


def group_label(dataset: str, database: str) -> str:
    """Short label such as 'FVC2000 DB1'."""
    return f"{dataset.upper()} {database.split('_')[0]}"


def laplacian_variance(pixels: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian; higher means sharper ridges."""
    laplacian = (
        -4.0 * pixels[1:-1, 1:-1]
        + pixels[:-2, 1:-1] + pixels[2:, 1:-1]
        + pixels[1:-1, :-2] + pixels[1:-1, 2:]
    )
    return float(laplacian.var())


def foreground_fraction(pixels: np.ndarray) -> float:
    """Fraction of 16x16 blocks that contain ridge texture."""
    height = pixels.shape[0] - pixels.shape[0] % BLOCK_SIZE
    width = pixels.shape[1] - pixels.shape[1] % BLOCK_SIZE
    blocks = pixels[:height, :width].reshape(
        height // BLOCK_SIZE, BLOCK_SIZE, width // BLOCK_SIZE, BLOCK_SIZE
    )
    return float((blocks.std(axis=(1, 3)) > FOREGROUND_STD).mean())


def compute_stats(record: ImageRecord) -> ImageStats:
    """Load one image at native resolution and summarize it."""
    path = record["image_path"]
    data = path.read_bytes()

    with Image.open(path) as image:
        width, height = image.size
        pixels = np.asarray(image.convert("L"), dtype=np.float64)

    key = f"{record['dataset']}/{record['database']}"

    return {
        "image_path": path.relative_to(PROJECT_ROOT).as_posix(),
        "group": group_label(record["dataset"], record["database"]),
        "dataset": record["dataset"],
        "database": record["database"],
        "finger_id": record["finger_id"],
        "impression_id": record["impression_id"],
        "width": width,
        "height": height,
        "nominal_dpi": NOMINAL_DPI.get(key, DEFAULT_DPI),
        "synthetic": record["database"] == SYNTHETIC_DATABASE,
        "mean": float(pixels.mean()),
        "contrast": float(pixels.std()),
        "sharpness": laplacian_variance(pixels),
        "foreground": foreground_fraction(pixels),
        "md5": hashlib.md5(data).hexdigest(),
    }


def sanity_checks(stats: list[ImageStats]) -> list[str]:
    """Return human-readable findings about the loaded data."""
    findings: list[str] = []

    per_finger = Counter(s["finger_id"] for s in stats)
    counts = Counter(per_finger.values())
    findings.append(
        f"{len(stats)} images, {len(per_finger)} finger identities; "
        f"impressions per finger: {dict(counts)}"
    )

    by_hash: dict[str, list[str]] = defaultdict(list)
    for s in stats:
        by_hash[s["md5"]].append(s["image_path"])
    duplicates = [paths for paths in by_hash.values() if len(paths) > 1]
    findings.append(f"Byte-identical duplicate images: {len(duplicates)}")
    for paths in duplicates[:5]:
        findings.append(f"  duplicate: {', '.join(paths)}")

    sizes_per_group: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for s in stats:
        sizes_per_group[s["group"]].add((s["width"], s["height"]))
    mixed = [g for g, sizes in sizes_per_group.items() if len(sizes) > 1]
    findings.append(
        f"Distinct image sizes: {len({(s['width'], s['height']) for s in stats})} "
        f"(databases with mixed sizes: {mixed or 'none'})"
    )

    off_dpi = sorted({(s["group"], s["nominal_dpi"]) for s in stats
                      if s["nominal_dpi"] != DEFAULT_DPI})
    findings.append(f"Databases not at {DEFAULT_DPI} dpi: {off_dpi}")
    findings.append(
        "Synthetic (SFinGe) images: "
        f"{sum(s['synthetic'] for s in stats)} "
        f"({sorted({s['group'] for s in stats if s['synthetic']})})"
    )

    # Flag images whose foreground is unusually small for their database.
    for group in sorted(sizes_per_group):
        values = np.array([s["foreground"] for s in stats if s["group"] == group])
        q1, q3 = np.percentile(values, [25, 75])
        low = q1 - 1.5 * (q3 - q1)
        outliers = [s for s in stats
                    if s["group"] == group and s["foreground"] < low]
        if outliers:
            names = ", ".join(Path(s["image_path"]).name for s in outliers[:6])
            findings.append(
                f"{group}: {len(outliers)} low-foreground outlier(s) "
                f"(< {low:.2f}): {names}"
            )

    return findings


def write_table(stats: list[ImageStats], output_path: Path) -> list[dict]:
    """Write per-database summary statistics and return the rows."""
    rows: list[dict] = []

    for group in sorted({s["group"] for s in stats}):
        subset = [s for s in stats if s["group"] == group]
        rows.append({
            "database": group,
            "fingers": len({s["finger_id"] for s in subset}),
            "images": len(subset),
            "size": f"{subset[0]['width']}x{subset[0]['height']}",
            "nominal_dpi": subset[0]["nominal_dpi"],
            "synthetic": subset[0]["synthetic"],
            "mean_intensity": round(float(np.mean([s["mean"] for s in subset])), 1),
            "contrast": round(float(np.mean([s["contrast"] for s in subset])), 1),
            "sharpness": round(float(np.median([s["sharpness"] for s in subset])), 1),
            "foreground": round(float(np.mean([s["foreground"] for s in subset])), 3),
        })

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def plot_samples(stats: list[ImageStats], output_path: Path) -> None:
    groups = sorted({s["group"] for s in stats})
    fig, axes = plt.subplots(3, 4, figsize=(12, 10))

    for axis, group in zip(axes.flat, groups):
        sample = next(s for s in stats
                      if s["group"] == group and s["impression_id"] == 1)
        with Image.open(PROJECT_ROOT / sample["image_path"]) as image:
            axis.imshow(np.asarray(image.convert("L")), cmap="gray")
        tag = " (synthetic)" if sample["synthetic"] else ""
        axis.set_title(f"{group}{tag}\n{sample['width']}x{sample['height']}, "
                       f"{sample['nominal_dpi']} dpi", fontsize=9)
        axis.set_xlabel("x (pixels)", fontsize=8)
        axis.set_ylabel("y (pixels)", fontsize=8)
        axis.tick_params(labelsize=7)

    fig.suptitle("Sample impression (finger 101, impression 1) from each database")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_boxplots(stats: list[ImageStats], output_path: Path) -> None:
    groups = sorted({s["group"] for s in stats})
    metrics = [
        ("mean", "Mean intensity (0-255)"),
        ("contrast", "Contrast: pixel std (0-255)"),
        ("sharpness", "Sharpness: Laplacian variance (log)"),
        ("foreground", "Foreground fraction (ridge blocks)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for axis, (key, label) in zip(axes.flat, metrics):
        axis.boxplot([[s[key] for s in stats if s["group"] == g] for g in groups])
        axis.set_xticks(range(1, len(groups) + 1), groups, rotation=45,
                        ha="right", fontsize=8)
        axis.set_xlabel("Database")
        axis.set_ylabel(label)
        axis.set_title(f"{label.split(':')[0].split(' (')[0]} by database")
        if key == "sharpness":
            axis.set_yscale("log")

    fig.suptitle("Image quality statistics per database (80 images each)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_histograms(stats: list[ImageStats], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for dataset in sorted({s["dataset"] for s in stats}):
        subset = [s for s in stats if s["dataset"] == dataset]
        axes[0].hist([s["foreground"] for s in subset], bins=20, range=(0, 1),
                     alpha=0.5, label=dataset.upper())
        axes[1].hist([s["contrast"] for s in subset], bins=20, alpha=0.5,
                     label=dataset.upper())

    axes[0].set_title("Distribution of foreground fraction")
    axes[0].set_xlabel("Fraction of 16x16 blocks containing ridges")
    axes[0].set_ylabel("Number of images")
    axes[1].set_title("Distribution of image contrast")
    axes[1].set_xlabel("Pixel standard deviation (0-255)")
    axes[1].set_ylabel("Number of images")

    for axis in axes:
        axis.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_sizes(stats: list[ImageStats], output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))

    for group in sorted({s["group"] for s in stats}):
        sample = next(s for s in stats if s["group"] == group)
        axis.scatter(sample["width"], sample["height"], s=60)
        axis.annotate(f"{group}\n{sample['nominal_dpi']} dpi",
                      (sample["width"], sample["height"]),
                      textcoords="offset points", xytext=(6, 4), fontsize=7)

    axis.scatter([256], [256], marker="x", color="black", s=80)
    axis.annotate("CNN input\n256x256", (256, 256), textcoords="offset points",
                  xytext=(6, -22), fontsize=7)
    axis.set_title("Native image size per database vs. CNN input size")
    axis.set_xlabel("Width (pixels)")
    axis.set_ylabel("Height (pixels)")
    axis.set_xlim(200, 700)
    axis.set_ylim(200, 620)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_correlation(stats: list[ImageStats], output_path: Path) -> None:
    keys = ["width", "height", "nominal_dpi", "mean", "contrast",
            "sharpness", "foreground"]
    matrix = np.array([[float(s[k]) for k in keys] for s in stats])
    corr = np.corrcoef(matrix, rowvar=False)

    fig, axis = plt.subplots(figsize=(7.5, 6.5))
    image = axis.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(keys)), keys, rotation=45, ha="right")
    axis.set_yticks(range(len(keys)), keys)

    for i in range(len(keys)):
        for j in range(len(keys)):
            axis.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                      fontsize=8)

    axis.set_title("Pearson correlation between image statistics (960 images)")
    axis.set_xlabel("Image statistic")
    axis.set_ylabel("Image statistic")
    fig.colorbar(image, ax=axis, label="Correlation coefficient")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def class_balance(records: list[ImageRecord]) -> dict[str, dict[str, int]]:
    """Genuine and impostor pair counts for each evaluation setting."""
    per_finger = Counter(r["finger_id"] for r in records)
    genuine = sum(comb(n, 2) for n in per_finger.values())
    total = comb(len(records), 2)
    counts = {"All pairs (SourceAFIS)": {"genuine": genuine,
                                         "impostor": total - genuine}}

    splits = split_by_impression(records, seed=SEED)
    for name, split in splits.items():
        pairs = generate_pairs(split, seed=SEED)
        positives = sum(pair["label"] for pair in pairs)
        counts[f"CNN {name} (balanced)"] = {
            "genuine": positives, "impostor": len(pairs) - positives
        }

    return counts


def plot_class_balance(counts: dict[str, dict[str, int]],
                       records: list[ImageRecord], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8),
                             gridspec_kw={"width_ratios": [1, 1.4]})

    per_group = Counter(group_label(r["dataset"], r["database"]) for r in records)
    groups = sorted(per_group)
    axes[0].bar(groups, [per_group[g] for g in groups])
    axes[0].set_title("Images per database (10 fingers x 8 impressions)")
    axes[0].set_xlabel("Database")
    axes[0].set_ylabel("Number of images")
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)

    names = list(counts)
    x = np.arange(len(names))
    width = 0.38
    genuine = [counts[n]["genuine"] for n in names]
    impostor = [counts[n]["impostor"] for n in names]
    axes[1].bar(x - width / 2, genuine, width, label="Genuine (same finger)")
    axes[1].bar(x + width / 2, impostor, width, label="Impostor (different finger)")

    for i in range(len(names)):
        axes[1].text(x[i] - width / 2, genuine[i], f"{genuine[i]:,}",
                     ha="center", va="bottom", fontsize=7)
        axes[1].text(x[i] + width / 2, impostor[i], f"{impostor[i]:,}",
                     ha="center", va="bottom", fontsize=7)

    axes[1].set_yscale("log")
    axes[1].set_xticks(x, names, fontsize=8)
    axes[1].set_title("Genuine vs. impostor pairs (log scale)")
    axes[1].set_xlabel("Evaluation setting")
    axes[1].set_ylabel("Number of pairs")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fingerprint dataset EDA.")
    parser.add_argument("--variant", choices=VARIANTS, default="raw",
                        help="raw images or FpEnhancer output (default: %(default)s)")
    args = parser.parse_args()

    output_dir = RESULTS_DIR if args.variant == "raw" else RESULTS_DIR.with_name(
        f"eda_{args.variant}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_manifest(manifest_for(args.variant))
    stats = [compute_stats(record) for record in records]

    with open(output_dir / "image_stats.csv", "w", newline="",
              encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)

    print("Sanity checks")
    print("-" * 40)
    findings = sanity_checks(stats)
    for line in findings:
        print(line)
    (output_dir / "sanity_checks.txt").write_text("\n".join(findings) + "\n")

    print("\nPer-database summary")
    print("-" * 40)
    rows = write_table(stats, output_dir / "database_summary.csv")
    header = list(rows[0])
    print(" | ".join(header))
    for row in rows:
        print(" | ".join(str(row[h]) for h in header))

    counts = class_balance(records)
    print("\nClass balance (pairs)")
    print("-" * 40)
    for name, values in counts.items():
        ratio = values["impostor"] / values["genuine"]
        print(f"{name}: {values['genuine']:,} genuine, "
              f"{values['impostor']:,} impostor (1:{ratio:.0f})")

    plot_samples(stats, output_dir / "fig_samples.png")
    plot_boxplots(stats, output_dir / "fig_quality_boxplots.png")
    plot_histograms(stats, output_dir / "fig_histograms.png")
    plot_sizes(stats, output_dir / "fig_image_sizes.png")
    plot_correlation(stats, output_dir / "fig_correlation.png")
    plot_class_balance(counts, records, output_dir / "fig_class_balance.png")

    print(f"\nFigures and tables written to {output_dir}")


if __name__ == "__main__":
    main()
