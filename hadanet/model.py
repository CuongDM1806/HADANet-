"""HADANet architecture reconstructed from the published specification.

The public research code contains partial variants of the architecture.  This
module keeps the paper's complete data flow in one testable implementation:

five-band DE -> hierarchical CNN -> channel/spatial attention -> residual
feature alignment -> adversarial and MK-MMD alignment -> MLP classifier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        return -ctx.alpha * gradient, None


class GradientReversal(nn.Module):
    def forward(self, features: Tensor, alpha: float = 1.0) -> Tensor:
        return _GradientReversalFunction.apply(features, alpha)


class HierarchicalCNN(nn.Module):
    """Paper equations (2)-(4): decoupled spectral and temporal branches."""

    def __init__(self, channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.horizontal = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(5, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.vertical = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 4), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: Tensor) -> Tensor:
        horizontal = self.horizontal(features)  # [B, C, 1, 4]
        vertical = self.vertical(features)  # [B, C, 5, 1]
        return self.fusion(horizontal * vertical)  # broadcast to [B, C, 5, 4]


class HybridAttention(nn.Module):
    """Paper equations (5)-(7): channel gate followed by spatial attention."""

    def __init__(self, channels: int = 64, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        self.spatial_depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.spatial_pointwise = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, features: Tensor) -> Tensor:
        channel_weights = self.channel_gate(features).unsqueeze(-1).unsqueeze(-1)
        channel_features = features * channel_weights
        spatial_weights = torch.sigmoid(
            self.spatial_pointwise(self.spatial_depthwise(channel_features))
        )
        return channel_features * spatial_weights


class ResidualFeatureAligner(nn.Module):
    """Paper equations (8)-(9): Z = F_A + Delta F."""

    def __init__(self, feature_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.correction = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return features + self.correction(features)


class DomainDiscriminator(nn.Module):
    """Three-layer domain classifier used with gradient reversal."""

    def __init__(self, feature_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class MultiKernelMMD(nn.Module):
    """Unbiased five-Gaussian-kernel MMD estimator from paper equation (12)."""

    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        if source.size(0) < 2 or target.size(0) < 2:
            return source.new_zeros(())

        source = F.normalize(source, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)
        total = torch.cat((source, target), dim=0)
        distances = torch.cdist(total, total, p=2).square()
        count = total.size(0)
        bandwidth = distances.detach().sum() / max(count * (count - 1), 1)
        bandwidth = bandwidth.clamp_min(1e-6)
        bandwidth = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))
        kernels = sum(
            torch.exp(-distances / (bandwidth * self.kernel_mul**index))
            for index in range(self.kernel_num)
        )

        source_count = source.size(0)
        target_count = target.size(0)
        source_kernel = kernels[:source_count, :source_count]
        target_kernel = kernels[source_count:, source_count:]
        cross_kernel = kernels[:source_count, source_count:]
        source_term = (source_kernel.sum() - source_kernel.diagonal().sum()) / (
            source_count * (source_count - 1)
        )
        target_term = (target_kernel.sum() - target_kernel.diagonal().sum()) / (
            target_count * (target_count - 1)
        )
        return source_term + target_term - 2.0 * cross_kernel.mean()


class MotionClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 640,
        classes: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, classes)

    def forward(self, features: Tensor) -> Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(features))))


class HADANet(nn.Module):
    """Complete four-class HADANet for DE tensors shaped [B, 64, 5, 4]."""

    def __init__(
        self,
        channels: int = 64,
        bands: int = 5,
        temporal_segments: int = 4,
        classes: int = 4,
        aligner_hidden_dim: int = 512,
        domain_hidden_dim: int = 512,
    ):
        super().__init__()
        if bands != 5 or temporal_segments != 4:
            raise ValueError("The published HADANet H-CNN requires 5 bands and 4 segments.")
        self.feature_dim = channels * bands * temporal_segments
        self.hierarchical_cnn = HierarchicalCNN(channels)
        self.attention = HybridAttention(channels)
        self.aligner = ResidualFeatureAligner(self.feature_dim, aligner_hidden_dim)
        self.grl = GradientReversal()
        self.domain_discriminator = DomainDiscriminator(
            self.feature_dim, domain_hidden_dim
        )
        self.mmd = MultiKernelMMD()
        self.classifier = MotionClassifier(self.feature_dim, classes=classes)

    def extract_features(self, inputs: Tensor) -> Tensor:
        features = self.hierarchical_cnn(inputs)
        features = self.attention(features)
        features = features.flatten(start_dim=1)
        return self.aligner(features)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.extract_features(inputs))

    def forward_domains(
        self, source: Tensor, target: Tensor, alpha: float
    ) -> dict[str, Tensor]:
        source_features = self.extract_features(source)
        target_features = self.extract_features(target)
        source_logits = self.classifier(source_features)
        domain_features = torch.cat((source_features, target_features), dim=0)
        domain_logits = self.domain_discriminator(self.grl(domain_features, alpha))
        return {
            "source_features": source_features,
            "target_features": target_features,
            "source_logits": source_logits,
            "domain_logits": domain_logits,
        }

    def orthogonal_loss(self) -> Tensor:
        # PyTorch stores the paper's W as W.T.  W W.T therefore implements
        # the published W.T W constraint between class weight vectors.
        weight = self.classifier.fc2.weight
        gram = weight @ weight.transpose(0, 1)
        identity = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        return (gram - identity).square().sum() / float(self.feature_dim**2)

    @staticmethod
    def grl_alpha(epoch: int, epochs: int) -> float:
        progress = epoch / max(epochs - 1, 1)
        return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


@dataclass(frozen=True)
class HADANetLoss:
    adversarial_weight: float = 1.0
    mmd_weight: float = 0.5
    orthogonal_weight: float = 0.1

    def __call__(
        self,
        model: HADANet,
        outputs: dict[str, Tensor],
        source_labels: Tensor,
    ) -> dict[str, Tensor]:
        source_count = outputs["source_features"].size(0)
        target_count = outputs["target_features"].size(0)
        domain_targets = torch.cat(
            (
                outputs["domain_logits"].new_zeros((source_count, 1)),
                outputs["domain_logits"].new_ones((target_count, 1)),
            ),
            dim=0,
        )
        classification = F.cross_entropy(outputs["source_logits"], source_labels)
        adversarial = F.binary_cross_entropy_with_logits(
            outputs["domain_logits"], domain_targets
        )
        mmd = model.mmd(outputs["source_features"], outputs["target_features"])
        orthogonal = model.orthogonal_loss()
        total = (
            classification
            + self.adversarial_weight * adversarial
            + self.mmd_weight * mmd
            + self.orthogonal_weight * orthogonal
        )
        return {
            "total": total,
            "classification": classification,
            "adversarial": adversarial,
            "mmd": mmd,
            "orthogonal": orthogonal,
        }
