"""
Run the original FpEnhancer Enhancer model on every image listed in
data/manifest.csv and save results to data/enhanced/, preserving paths.

Place this file in the root of the cloned XiongjunGuan/FpEnhancer repository,
where the `models/` directory and `best.pth` are available.

This is inference only; it does not train or modify the model.
"""

import csv
from pathlib import Path
from tqdm import tqdm
from typing import Final

import numpy as np
import torch
from PIL import Image
from FpEnhancer_main.models.Enhancer import Enhancer


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
MANIFEST_PATH: Final[Path] = PROJECT_ROOT / "data" / "manifest.csv"
INPUT_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "data" / "enhanced"
MODEL_PATH: Final[Path] = PROJECT_ROOT / "checkpoints" / "best.pth"

SUPPORTED_EXTENSIONS: Final[set[str]] = {
    ".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"
}

# These values match configs/config_enhancer.yaml in the original repository
# and the tensor shapes in the supplied best.pth checkpoint.
MODEL_CONFIG = {
    "img_channel": 1,
    "width": 32,
    "mid_blk_num": 2,
    "enc_blk_nums": [2, 2, 2],
    "dec_blk_nums": [2, 2, 2],
    "dw_expand": 1,
    "ffn_expand": 2,
}


def load_model(device: torch.device) -> torch.nn.Module:
    """Construct the repository's original model and load its trained weights."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}\n"
            "Put best.pth in the project root or update MODEL_PATH."
        )

    model = Enhancer(**MODEL_CONFIG)
    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_manifest_paths() -> list[Path]:
    """Read image paths from the existing manifest and validate them."""
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    image_paths: list[Path] = []
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "image_path" not in reader.fieldnames:
            raise ValueError("manifest.csv must contain an image_path column")

        for row in reader:
            raw_path = Path(row["image_path"])
            image_path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path

            try:
                image_path.resolve().relative_to(INPUT_DIR.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"Manifest path is outside data/processed: {image_path}"
                ) from exc

            if image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_paths.append(image_path)

    if not image_paths:
        raise ValueError(f"No supported image paths found in {MANIFEST_PATH}")

    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest image(s) are missing. "
            f"First missing file: {missing[0]}"
        )

    return image_paths


def enhance_image(
        model: torch.nn.Module,
        input_path: Path,
        device: torch.device,
) -> Path:
    """Enhance one fingerprint using the repository's input/output convention."""
    relative_path = input_path.relative_to(INPUT_DIR)
    output_path = OUTPUT_DIR / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as image:
        # The original FpEnhancer test loader reads grayscale and inverts it:
        # model input = (255 - grayscale_pixel) / 255.
        grayscale = image.convert("L")
        original_size = grayscale.size
        pixels = np.asarray(grayscale, dtype=np.float32).copy()
        normalized = (255.0 - pixels) / 255.0

    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(device)

    with torch.inference_mode():
        prediction = model(tensor)

    # The original inference script converts predictions back using
    # 255 * (1 - prediction), yielding conventional grayscale output.
    prediction = prediction.squeeze(0).squeeze(0).float().cpu().numpy()
    output_pixels = np.clip(255.0 * (1.0 - prediction), 0, 255)
    output_pixels = np.rint(output_pixels).astype(np.uint8)

    # Preserve original width/height; the model itself pads to a multiple of 8
    # and crops back to the original dimensions.
    if (output_pixels.shape[1], output_pixels.shape[0]) != original_size:
        raise RuntimeError(
            f"Unexpected output size for {input_path}: "
            f"{output_pixels.shape[::-1]} vs {original_size}"
        )

    Image.fromarray(output_pixels, mode="L").save(output_path)
    return output_path


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    model = load_model(device)
    image_paths = load_manifest_paths()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed = 0

    with tqdm(image_paths, desc="Enhancing fingerprints", unit="image", dynamic_ncols=True) as pbar:
        for image_path in pbar:
            try:
                output_path = enhance_image(model, image_path, device)
                succeeded += 1
            except Exception as e:
                failed += 1
                tqdm.write(f"FAILED {image_path}: {e}")

            pbar.set_postfix(
                success=succeeded,
                failed=failed
            )

    print(f"\nComplete. Success: {succeeded}; failed: {failed}")
    print(f"Outputs saved under: {OUTPUT_DIR}")

    create_enhanced_manifest()


def create_enhanced_manifest():
    manifest_path = PROJECT_ROOT / "data" / "manifest.csv"
    output_path = PROJECT_ROOT / "data" / "manifest_enhanced.csv"

    with manifest_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        if not fieldnames or "image_path" not in fieldnames:
            raise ValueError("manifest.csv must contain an image_path column")

        rows = list(reader)

    for row in rows:
        path = Path(row["image_path"])

        # Replace only the processed directory component.
        parts = list(path.parts)
        try:
            index = parts.index("processed")
            parts[index] = "enhanced"
        except ValueError:
            raise ValueError(
                f"Could not find 'processed' in image path: {path}"
            )

        row["image_path"] = str(Path(*parts))

    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created enhanced manifest: {output_path}")
    print(f"Updated {len(rows)} image paths.")


if __name__ == "__main__":
    main()
