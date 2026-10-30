import argparse
import torch

from pathlib import Path
from typing import Final

from siamese_model import FingerprintEncoder


ROOT: Final[Path] = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Train Siamese Network on Fingerprints.")
parser.add_argument("--enhanced", action="store_true", help="Use enhanced datasets and checkpoints.")
args, _ = parser.parse_known_args()
suffix = "_enhanced" if args.enhanced else ""

CHECKPOINT: Final[Path] = ROOT / "checkpoints" / f"siamese_best{suffix}.pt"
OUTPUT: Final[Path] = ROOT / "checkpoints" / f"fingerprint_encoder{suffix}.pt"


def main():
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    embedding_dim = int(checkpoint["embedding_dim"])

    encoder = FingerprintEncoder(embedding_dim=embedding_dim)
    state_dict = checkpoint["model_state_dict"]

    # Extract the encoder weights from SiameseNetwork.
    encoder_state_dict = {
        key.removeprefix("encoder."): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }

    if not encoder_state_dict:
        raise ValueError("No encoder weights found in checkpoint.")

    encoder.load_state_dict(encoder_state_dict)
    encoder.eval()

    example = torch.zeros(1, 1, 256, 256)

    with torch.inference_mode():
        scripted_encoder = torch.jit.trace(encoder, example)
        scripted_encoder.save(str(OUTPUT))

    print(f"Encoder exported to: {OUTPUT}")
    print(f"Embedding dimension: {embedding_dim}")


if __name__ == "__main__":
    main()
