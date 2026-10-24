import torch
from torch import nn


class FingerprintEncoder(nn.Module):
    """CNN that maps a grayscale fingerprint to an embedding."""
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.projection = nn.Linear(128, embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.projection(self.features(images))
        return nn.functional.normalize(embeddings, p=2, dim=1)


class SiameseNetwork(nn.Module):
    """Two inputs processed by the same encoder."""
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.encoder = FingerprintEncoder(embedding_dim)

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(image1), self.encoder(image2)


class ContrastiveLoss(nn.Module):
    """
    Label 1: same finger, minimize distance.
    Label 0: different fingers, separate by at least the margin.
    """
    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        distances = nn.functional.pairwise_distance(
            embedding1,
            embedding2,
        )

        positive_loss = labels * distances.square()
        negative_loss = (1.0 - labels) * torch.clamp(
            self.margin - distances,
            min=0.0,
        ).square()

        return (positive_loss + negative_loss).mean()
