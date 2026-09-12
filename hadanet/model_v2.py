"""Deeper HCNN + Hybrid Attention variant for HADANet.

The published HADANet specification (equations 2-7) uses one convolution per
branch. This variant adds a same-size residual stage before the full-span
collapsing convolutions and widens the attention bottleneck. All downstream
domain-adaptation and classifier components are reused from ``hadanet.model``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hadanet.model import (
    DomainDiscriminator,
    GradientReversal,
    MotionClassifier,
    MultiKernelMMD,
    ResidualFeatureAligner,
)


class HierarchicalCNNv2(nn.Module):
    """Deeper HCNN that preserves the published dual collapsing branches."""

    def __init__(self, channels: int = 64, widen: int = 128, dropout: float = 0.2):
        super().__init__()
        # Stage 0 preserves [B, C, 5, 4] while widening the feature maps.
        self.stage0_h = nn.Sequential(
            nn.Conv2d(
                channels,
                widen,
                kernel_size=(3, 1),
                padding=(1, 0),
                bias=False,
            ),
            nn.BatchNorm2d(widen),
            nn.GELU(),
        )
        self.stage0_v = nn.Sequential(
            nn.Conv2d(
                channels,
                widen,
                kernel_size=(1, 3),
                padding=(0, 1),
                bias=False,
            ),
            nn.BatchNorm2d(widen),
            nn.GELU(),
        )
        self.stage0_fuse = nn.Sequential(
            nn.Conv2d(widen, widen, kernel_size=1, bias=False),
            nn.BatchNorm2d(widen),
        )
        self.stage0_residual = nn.Conv2d(
            channels, widen, kernel_size=1, bias=False
        )
        self.stage0_dropout = nn.Dropout(dropout)

        # Stage 1 retains the original full-span frequency/time convolutions.
        self.horizontal = nn.Sequential(
            nn.Conv2d(widen, channels, kernel_size=(5, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.vertical = nn.Sequential(
            nn.Conv2d(widen, channels, kernel_size=(1, 4), bias=False),
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
        h0 = self.stage0_h(features)
        v0 = self.stage0_v(features)
        fused0 = self.stage0_fuse(h0 * v0) + self.stage0_residual(features)
        fused0 = self.stage0_dropout(F.gelu(fused0))

        horizontal = self.horizontal(fused0)  # [B, C, 1, 4]
        vertical = self.vertical(fused0)  # [B, C, 5, 1]
        return self.fusion(horizontal * vertical)  # [B, C, 5, 4]


class HybridAttentionV2(nn.Module):
    """Wider channel gate followed by local and dilated spatial filtering."""

    def __init__(self, channels: int = 64, reduction: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        self.spatial_depthwise_1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.spatial_depthwise_2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=channels,
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm2d(channels)
        self.spatial_pointwise = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, features: Tensor) -> Tensor:
        channel_weights = self.channel_gate(features).unsqueeze(-1).unsqueeze(-1)
        channel_features = features * channel_weights
        spatial = self.spatial_depthwise_1(channel_features)
        spatial = spatial + self.spatial_depthwise_2(channel_features)
        spatial = F.gelu(self.spatial_bn(spatial))
        spatial_weights = torch.sigmoid(self.spatial_pointwise(spatial))
        return channel_features * spatial_weights


class HADANetV2(nn.Module):
    """HADANet with the deeper HCNN and attention front-end."""

    def __init__(
        self,
        channels: int = 64,
        bands: int = 5,
        temporal_segments: int = 4,
        classes: int = 4,
        widen: int = 128,
        aligner_hidden_dim: int = 512,
        domain_hidden_dim: int = 512,
        hcnn_dropout: float = 0.2,
        attn_dropout: float = 0.1,
        aligner_dropout: float = 0.3,
        domain_dropout: float = 0.3,
        classifier_dropout: float = 0.5,
    ):
        super().__init__()
        if bands != 5 or temporal_segments != 4:
            raise ValueError(
                "The published HADANet H-CNN requires 5 bands and 4 segments."
            )
        self.feature_dim = channels * bands * temporal_segments
        self.hierarchical_cnn = HierarchicalCNNv2(
            channels, widen=widen, dropout=hcnn_dropout
        )
        self.attention = HybridAttentionV2(channels, dropout=attn_dropout)
        self.aligner = ResidualFeatureAligner(
            self.feature_dim, aligner_hidden_dim, dropout=aligner_dropout
        )
        self.grl = GradientReversal()
        self.domain_discriminator = DomainDiscriminator(
            self.feature_dim, domain_hidden_dim, dropout=domain_dropout
        )
        self.mmd = MultiKernelMMD()
        self.classifier = MotionClassifier(
            self.feature_dim, classes=classes, dropout=classifier_dropout
        )

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
        source_count = source.size(0)
        all_features = self.extract_features(torch.cat((source, target), dim=0))
        source_features = all_features[:source_count]
        target_features = all_features[source_count:]
        source_logits = self.classifier(source_features)
        domain_logits = self.domain_discriminator(self.grl(all_features, alpha))
        return {
            "source_features": source_features,
            "target_features": target_features,
            "source_logits": source_logits,
            "domain_logits": domain_logits,
        }

    def orthogonal_loss(self) -> Tensor:
        weight = self.classifier.fc2.weight
        gram = weight @ weight.transpose(0, 1)
        identity = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        classifier_feature_dim = weight.size(1)
        return (gram - identity).square().sum() / float(
            classifier_feature_dim**2
        )

    @staticmethod
    def grl_alpha(epoch: int, epochs: int) -> float:
        from hadanet.model import HADANet as _Base

        return _Base.grl_alpha(epoch, epochs)
