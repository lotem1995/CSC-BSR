# simple_models_attempt.py
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResNet18SquareClassifier(nn.Module):
    """
    ResNet18-based classifier for chessboard *square* images.

    - Pretrained on ImageNet (optional)
    - Replaces final FC layer with (512 -> num_classes)
    - Exposes a `features()` method for OOD methods (Mahalanobis etc.)
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Save everything except the FC head as a feature extractor
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # -> (B, 512, 1, 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return penultimate features (B, 512).
        """
        feats = self.feature_extractor(x)          # (B, 512, 1, 1)
        feats = torch.flatten(feats, 1)            # (B, 512)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return logits (B, num_classes).
        """
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits


@torch.no_grad()
def msp_score(logits: torch.Tensor) -> torch.Tensor:
    """
    Maximum Softmax Probability (MSP) confidence score.
    Higher = more in-distribution (usually).
    """
    probs = torch.softmax(logits, dim=1)
    return probs.max(dim=1).values


def energy_score(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """
    Energy-based OOD score.
    Lower energy => more confident/in-distribution (common convention).
    Energy(x) = -T * logsumexp(logits / T)

    NOTE: Some papers flip the sign. Pick one convention and keep it consistent
    with your thresholding.
    """
    return -T * torch.logsumexp(logits / T, dim=1)


if __name__ == "__main__":
    # quick sanity check
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNet18SquareClassifier(num_classes=13, pretrained=True).to(device)
    x = torch.randn(4, 3, 96, 96).to(device)

    logits = model(x)
    print("logits shape:", logits.shape)  # (4, 13)

    print("MSP:", msp_score(logits).cpu())
    print("Energy:", energy_score(logits, T=1.0).detach().cpu())
