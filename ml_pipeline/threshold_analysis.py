"""
Threshold and error analysis for pairwise fingerprint comparison scores.

Accepts the CSVs written by BaselineFingerprintPairwiseTest.java (column
"score", higher = more likely genuine) and by test_model.py (column
"distance", lower = more likely genuine). For each run it reports:

* confusion-matrix counts and rates at the operating threshold,
* the EER and the FAR/FRR trade-off across all thresholds,
* thresholds that meet target false-accept rates,
* per-database error rates, and the worst false rejects/accepts.

FVC2000 DB1_B and DB2_B share physical fingers (see fingerprint_dataset.py).
Their same-finger pairs are labelled impostor in the score files but are
really cross-sensor genuine pairs, so they are excluded from the metrics by
default and reported separately; --keep-shared-fingers keeps them.

Usage:
    python threshold_analysis.py \
        --scores data/pairwise_scores_raw.csv --name sourceafis_raw

    # SourceAFIS and the CNN on the same test-split pairs:
    python threshold_analysis.py \
        --scores data/pairwise_scores_raw.csv data/cnn_test_distances.csv \
        --name sourceafis_test cnn_test --split test --compare raw_test
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from fingerprint_dataset import (
    MANIFEST_PATH,
    PROJECT_ROOT,
    load_manifest,
    physical_finger,
    split_by_impression,
)

RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results" / "thresholds"
SEED: Final[int] = 42

# SourceAFIS documents 40 as its default threshold (about 0.01% FMR).
SOURCEAFIS_THRESHOLD: Final[float] = 40.0
TARGET_FARS: Final[tuple[float, ...]] = (0.01, 0.001, 0.0001)
WORST_COUNT: Final[int] = 10


@dataclass
class Pairs:
    """Comparison results, with similarity oriented so higher = genuine."""
    path1: np.ndarray
    path2: np.ndarray
    label: np.ndarray
    similarity: np.ndarray
    kind: str  # "score" or "distance"

    def native(self, similarity: float | np.ndarray) -> float | np.ndarray:
        """Convert an oriented similarity back to the file's own units."""
        return similarity if self.kind == "score" else -similarity


def normalize_path(raw: str) -> str:
    """Make a path relative to ml_pipeline/, e.g. data/processed/..."""
    parts = Path(raw).parts
    return Path(*parts[parts.index("data"):]).as_posix()


def image_key(path: str) -> tuple[str, int]:
    """(finger_id, impression_id) for data/<variant>/<year>/<DB>/<f>_<i>.tif"""
    parts = Path(path).parts
    finger, impression = Path(parts[-1]).stem.split("_")
    return f"{parts[-3]}/{parts[-2]}/finger_{finger}", int(impression)


def database_of(path: str) -> str:
    parts = Path(path).parts
    return f"{parts[-3].upper()} {parts[-2].split('_')[0]}"


def load_pairs(csv_path: Path, split: str,
               keep_shared: bool) -> tuple[Pairs, np.ndarray]:
    """
    Read a score/distance CSV, dropping self-pairs and filtering by split.

    Returns the pairs to evaluate and the oriented similarities of the
    excluded cross-sensor same-finger pairs.
    """
    keep: set[tuple[str, int]] | None = None
    if split != "all":
        splits = split_by_impression(load_manifest(MANIFEST_PATH), seed=SEED)
        keep = {(r["finger_id"], r["impression_id"]) for r in splits[split]}

    path1: list[str] = []
    path2: list[str] = []
    labels: list[int] = []
    values: list[float] = []
    shared: list[float] = []
    self_pairs = 0

    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        kind = "score" if "score" in fields else "distance"

        for row in reader:
            a = normalize_path(row["image_path_1"])
            b = normalize_path(row["image_path_2"])

            # Older baseline runs compared every image with itself.
            if a == b:
                self_pairs += 1
                continue

            if keep is not None and (image_key(a) not in keep
                                     or image_key(b) not in keep):
                continue

            label = int(row["label"])
            value = float(row[kind])

            if (not keep_shared and label == 0
                    and physical_finger(image_key(a)[0])
                    == physical_finger(image_key(b)[0])):
                shared.append(value)
                continue

            path1.append(a)
            path2.append(b)
            labels.append(label)
            values.append(value)

    if self_pairs:
        print(f"  dropped {self_pairs:,} self-pairs from {csv_path.name}")

    similarity = np.array(values)
    shared_similarity = np.array(shared)
    if kind == "distance":
        similarity = -similarity
        shared_similarity = -shared_similarity

    return Pairs(np.array(path1), np.array(path2), np.array(labels),
                 similarity, kind), shared_similarity


def error_curves(pairs: Pairs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    FAR and FRR for every candidate threshold.

    A pair is accepted when similarity >= threshold.
    FAR = accepted impostors / impostors; FRR = rejected genuines / genuines.
    """
    genuine = np.sort(pairs.similarity[pairs.label == 1])
    impostor = np.sort(pairs.similarity[pairs.label == 0])

    thresholds = np.unique(pairs.similarity)
    thresholds = np.r_[thresholds, np.inf]

    far = (len(impostor) - np.searchsorted(impostor, thresholds, side="left")) \
        / len(impostor)
    frr = np.searchsorted(genuine, thresholds, side="left") / len(genuine)

    return thresholds, far, frr


def confusion(pairs: Pairs, threshold: float) -> dict[str, float]:
    accepted = pairs.similarity >= threshold
    genuine = pairs.label == 1

    tp = int(np.sum(genuine & accepted))
    fn = int(np.sum(genuine & ~accepted))
    fp = int(np.sum(~genuine & accepted))
    tn = int(np.sum(~genuine & ~accepted))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    far = fp / (fp + tn)
    frr = fn / (fn + tp)

    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "far": far, "frr": frr,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall else 0.0,
        "accuracy": (tp + tn) / (tp + fn + fp + tn),
        "balanced_accuracy": 1.0 - (far + frr) / 2.0,
    }


def per_database(pairs: Pairs, threshold: float) -> list[dict]:
    """FRR over genuine pairs and FAR over same-database impostor pairs."""
    db1 = np.array([database_of(p) for p in pairs.path1])
    db2 = np.array([database_of(p) for p in pairs.path2])
    accepted = pairs.similarity >= threshold
    rows: list[dict] = []

    for db in sorted(set(db1)):
        genuine = (db1 == db) & (pairs.label == 1)
        impostor = (db1 == db) & (db2 == db) & (pairs.label == 0)
        false_rejects = int(np.sum(genuine & ~accepted))
        false_accepts = int(np.sum(impostor & accepted))

        rows.append({
            "database": db,
            "genuine_pairs": int(genuine.sum()),
            "false_rejects": false_rejects,
            "frr": false_rejects / max(1, int(genuine.sum())),
            "impostor_pairs": int(impostor.sum()),
            "false_accepts": false_accepts,
            "far": false_accepts / max(1, int(impostor.sum())),
        })

    cross = (db1 != db2) & (pairs.label == 0)
    cross_accepts = int(np.sum(cross & accepted))
    rows.append({
        "database": "Cross-database impostors",
        "genuine_pairs": 0, "false_rejects": 0, "frr": 0.0,
        "impostor_pairs": int(cross.sum()),
        "false_accepts": cross_accepts,
        "far": cross_accepts / max(1, int(cross.sum())),
    })
    return rows


def worst_errors(pairs: Pairs) -> tuple[list[int], list[int]]:
    """Indices of the lowest-similarity genuines and highest impostors."""
    genuine = np.flatnonzero(pairs.label == 1)
    impostor = np.flatnonzero(pairs.label == 0)
    lowest = genuine[np.argsort(pairs.similarity[genuine], kind="stable")]
    highest = impostor[np.argsort(-pairs.similarity[impostor], kind="stable")]
    return lowest[:WORST_COUNT].tolist(), highest[:WORST_COUNT].tolist()


def plot_distributions(pairs: Pairs, threshold: float, eer_threshold: float,
                       name: str, output_path: Path) -> None:
    values = pairs.native(pairs.similarity)
    unit = "SourceAFIS similarity score" if pairs.kind == "score" \
        else "Embedding distance"
    bins = np.linspace(values.min(), np.percentile(values, 99.9), 80)

    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.hist(values[pairs.label == 0], bins=bins, alpha=0.6, density=True,
              label=f"Impostor ({np.sum(pairs.label == 0):,} pairs)")
    axis.hist(values[pairs.label == 1], bins=bins, alpha=0.6, density=True,
              label=f"Genuine ({np.sum(pairs.label == 1):,} pairs)")
    axis.axvline(pairs.native(threshold), color="black", linestyle="--",
                 label=f"Operating threshold = {pairs.native(threshold):.3g}")
    axis.axvline(pairs.native(eer_threshold), color="gray", linestyle=":",
                 label=f"EER threshold = {pairs.native(eer_threshold):.3g}")
    axis.set_yscale("log")
    axis.set_title(f"Genuine vs. impostor {unit.lower()} distributions ({name})")
    axis.set_xlabel(unit)
    axis.set_ylabel("Density (log scale)")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_far_frr(pairs: Pairs, thresholds: np.ndarray, far: np.ndarray,
                 frr: np.ndarray, eer_index: int, threshold: float, name: str,
                 output_path: Path) -> None:
    finite = np.isfinite(thresholds)
    x = pairs.native(thresholds[finite])
    unit = "SourceAFIS score threshold" if pairs.kind == "score" \
        else "Distance threshold"

    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(x, far[finite] * 100, label="FAR (false accept rate)")
    axis.plot(x, frr[finite] * 100, label="FRR (false reject rate)")
    axis.scatter([pairs.native(thresholds[eer_index])],
                 [(far[eer_index] + frr[eer_index]) * 50], color="black",
                 zorder=3, label=f"EER = {(far[eer_index] + frr[eer_index]) * 50:.2f}%")
    axis.axvline(pairs.native(threshold), color="black", linestyle="--",
                 linewidth=1, label="Operating threshold")
    if pairs.kind == "score":
        axis.set_xlim(0, np.percentile(x, 99.5))
    axis.set_title(f"FAR and FRR vs. threshold ({name})")
    axis.set_xlabel(unit)
    axis.set_ylabel("Error rate (%)")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_det(curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
             title: str, output_path: Path) -> None:
    """DET-style plot: FRR against FAR, both on log axes."""
    fig, axis = plt.subplots(figsize=(6.5, 5.5))

    for name, (far, frr, eer) in curves.items():
        floor = 1e-6
        axis.plot(np.maximum(far, floor) * 100, np.maximum(frr, floor) * 100,
                  label=f"{name} (EER {eer * 100:.2f}%)")

    axis.plot([1e-3, 100], [1e-3, 100], color="gray", linestyle=":",
              linewidth=1, label="FAR = FRR")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1e-3, 100)
    axis.set_ylim(1e-1, 100)
    axis.set_title(title)
    axis.set_xlabel("False accept rate, FAR (%)")
    axis.set_ylabel("False reject rate, FRR (%)")
    axis.legend(fontsize=8)
    axis.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_per_database(rows: list[dict], name: str, output_path: Path) -> None:
    rows = [r for r in rows if r["genuine_pairs"]]
    names = [r["database"] for r in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(x, [r["frr"] * 100 for r in rows])
    axes[0].set_title(f"False reject rate by database ({name})")
    axes[0].set_ylabel("FRR (%)")
    axes[1].bar(x, [r["far"] * 100 for r in rows], color="tab:orange")
    axes[1].set_title(f"False accept rate within each database ({name})")
    axes[1].set_ylabel("FAR (%)")

    for axis, key, count in ((axes[0], "frr", "false_rejects"),
                             (axes[1], "far", "false_accepts")):
        axis.set_xticks(x, names, rotation=45, ha="right", fontsize=8)
        axis.set_xlabel("Database")
        for i, r in enumerate(rows):
            axis.text(i, r[key] * 100, str(r[count]), ha="center",
                      va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_examples(pairs: Pairs, rejects: list[int], accepts: list[int],
                  name: str, output_path: Path) -> None:
    examples = [("False reject", i) for i in rejects[:3]] + \
               [("False accept", i) for i in accepts[:3]]
    fig, axes = plt.subplots(len(examples), 2, figsize=(6, 3 * len(examples)))

    for row, (kind, index) in enumerate(examples):
        value = pairs.native(pairs.similarity[index])
        for col, path in enumerate((pairs.path1[index], pairs.path2[index])):
            with Image.open(PROJECT_ROOT / path) as image:
                axes[row, col].imshow(np.asarray(image.convert("L")), cmap="gray")
            parts = Path(path).parts
            axes[row, col].set_title(
                f"{database_of(path)} {parts[-1]}", fontsize=8)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(f"{kind}\n{pairs.kind} = {value:.3g}",
                                fontsize=9)

    fig.suptitle(f"Hardest errors ({name}): lowest-scoring genuine and "
                 f"highest-scoring impostor pairs", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def analyze(csv_path: Path, name: str, split: str, threshold_arg: float | None,
            keep_shared: bool) -> tuple[np.ndarray, np.ndarray, float]:
    print(f"\n=== {name} ({csv_path.name}, split = {split}) ===")
    pairs, shared = load_pairs(csv_path, split, keep_shared)
    output_dir = RESULTS_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds, far, frr = error_curves(pairs)
    eer_index = int(np.argmin(np.abs(far - frr)))
    eer = float((far[eer_index] + frr[eer_index]) / 2)
    eer_threshold = float(thresholds[eer_index])

    # The operating threshold: SourceAFIS's documented default for scores,
    # otherwise an explicit value, otherwise the EER point.
    if threshold_arg is not None:
        threshold = threshold_arg if pairs.kind == "score" else -threshold_arg
    elif pairs.kind == "score":
        threshold = SOURCEAFIS_THRESHOLD
    else:
        threshold = eer_threshold

    at_threshold = confusion(pairs, threshold)

    targets = []
    for target in TARGET_FARS:
        index = int(np.argmax(far <= target))
        targets.append({
            "target_far": target,
            "threshold": float(pairs.native(thresholds[index])),
            "far": float(far[index]),
            "frr": float(frr[index]),
        })

    rows = per_database(pairs, threshold)
    rejects, accepts = worst_errors(pairs)

    genuine = int(np.sum(pairs.label == 1))
    impostor = int(np.sum(pairs.label == 0))
    shared_accepted = int(np.sum(shared >= threshold))
    print(f"Pairs: {genuine:,} genuine, {impostor:,} impostor")
    print(f"Operating threshold: {pairs.native(threshold):.6g}")
    if len(shared):
        print(f"  excluded {len(shared):,} FVC2000 DB1/DB2 cross-sensor "
              f"same-finger pairs; {shared_accepted:,} "
              f"({shared_accepted / len(shared) * 100:.1f}%) would be accepted")
    c = at_threshold
    print(f"  TP {c['tp']:,} | FN {c['fn']:,} | FP {c['fp']:,} | TN {c['tn']:,}")
    print(f"  FAR {c['far'] * 100:.4f}% ({c['fp']:,}/{c['fp'] + c['tn']:,})  "
          f"FRR {c['frr'] * 100:.2f}% ({c['fn']:,}/{c['fn'] + c['tp']:,})")
    print(f"  precision {c['precision']:.4f}  recall {c['recall']:.4f}  "
          f"F1 {c['f1']:.4f}  accuracy {c['accuracy']:.4f}  "
          f"balanced accuracy {c['balanced_accuracy']:.4f}")
    print(f"EER: {eer * 100:.2f}% at threshold {pairs.native(eer_threshold):.6g}")
    for t in targets:
        print(f"  FAR <= {t['target_far'] * 100:g}%: threshold {t['threshold']:.6g}, "
              f"FAR {t['far'] * 100:.4f}%, FRR {t['frr'] * 100:.2f}%")
    print("Per database (operating threshold):")
    for r in rows:
        print(f"  {r['database']:<26} FRR {r['frr'] * 100:6.2f}% "
              f"({r['false_rejects']}/{r['genuine_pairs']})  "
              f"FAR {r['far'] * 100:7.4f}% "
              f"({r['false_accepts']}/{r['impostor_pairs']})")

    with open(output_dir / "per_database.csv", "w", newline="",
              encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with open(output_dir / "worst_errors.csv", "w", newline="",
              encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["error", "image_path_1", "image_path_2", pairs.kind])
        for kind, indices in (("false_reject", rejects),
                              ("false_accept", accepts)):
            for i in indices:
                writer.writerow([kind, pairs.path1[i], pairs.path2[i],
                                 pairs.native(pairs.similarity[i])])

    summary = {
        "scores": str(csv_path.relative_to(PROJECT_ROOT)),
        "split": split,
        "genuine_pairs": genuine,
        "impostor_pairs": impostor,
        "excluded_cross_sensor_pairs": int(len(shared)),
        "excluded_cross_sensor_accepted": shared_accepted,
        "operating_threshold": float(pairs.native(threshold)),
        "at_operating_threshold": at_threshold,
        "eer": eer,
        "eer_threshold": float(pairs.native(eer_threshold)),
        "target_far": targets,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    plot_distributions(pairs, threshold, eer_threshold, name,
                       output_dir / "fig_score_distributions.png")
    plot_far_frr(pairs, thresholds, far, frr, eer_index, threshold, name,
                 output_dir / "fig_far_frr.png")
    plot_det({name: (far, frr, eer)}, f"DET curve ({name})",
             output_dir / "fig_det.png")
    plot_per_database(rows, name, output_dir / "fig_per_database.png")
    plot_examples(pairs, rejects, accepts, name,
                  output_dir / "fig_error_examples.png")

    print(f"Results: {output_dir}")
    return far, frr, eer


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold and error analysis.")
    parser.add_argument("--scores", nargs="+", required=True, type=Path,
                        help="score/distance CSVs, relative to ml_pipeline/")
    parser.add_argument("--name", nargs="+", required=True,
                        help="one output name per CSV")
    parser.add_argument("--split", choices=("all", "train", "validation", "test"),
                        default="all",
                        help="keep only pairs whose images are both in this split")
    parser.add_argument("--threshold", type=float, default=None,
                        help="operating threshold in the CSV's own units "
                             "(default: 40 for scores, EER for distances)")
    parser.add_argument("--keep-shared-fingers", action="store_true",
                        help="count FVC2000 DB1/DB2 same-finger pairs as "
                             "impostors, as the raw score files label them")
    parser.add_argument("--compare", default=None,
                        help="also write a combined DET plot under this name")
    args = parser.parse_args()

    if len(args.scores) != len(args.name):
        parser.error("--scores and --name need the same number of values")

    curves = {}
    for path, name in zip(args.scores, args.name):
        csv_path = path if path.is_absolute() else PROJECT_ROOT / path
        curves[name] = analyze(csv_path, name, args.split, args.threshold,
                               args.keep_shared_fingers)

    if args.compare:
        output_dir = RESULTS_DIR / args.compare
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_det(curves, f"DET comparison ({args.split} pairs)",
                 output_dir / "fig_det_comparison.png")
        print(f"\nComparison: {output_dir / 'fig_det_comparison.png'}")


if __name__ == "__main__":
    main()
