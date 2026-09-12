"""Raw-EEG HADANet variant for 4.1-second PhysioNet trials.

This front-end accepts tensors shaped ``[batch, 64, 656]``.  It deliberately
does not require differential entropy, a filter bank, or hand-crafted temporal
segments.  The downstream attention, residual feature correction, domain
alignment, and classifier follow the existing HADANet implementation.
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


class RawHierarchicalCNN(nn.Module):
    """Learn short- and long-range temporal features directly from raw EEG."""

    def __init__(self, channels: int = 64, widen: int = 128, dropout: float = 0.2):
        super().__init__()
        # A pointwise projection learns interactions between all electrodes.
        self.stem = nn.Sequential(
            nn.Conv1d(channels, widen, kernel_size=1, bias=False),
            nn.BatchNorm1d(widen),
            nn.GELU(),
        )
        self.short_branch = nn.Sequential(
            nn.Conv1d(
                widen,
                widen,
                kernel_size=15,
                padding=7,
                groups=widen,
                bias=False,
            ),
            nn.BatchNorm1d(widen),
            nn.GELU(),
            nn.Conv1d(widen, channels, kernel_size=1, bias=False),
        )
        self.long_branch = nn.Sequential(
            nn.Conv1d(
                widen,
                widen,
                kernel_size=63,
                padding=31,
                groups=widen,
                bias=False,
            ),
            nn.BatchNorm1d(widen),
            nn.GELU(),
            nn.Conv1d(widen, channels, kernel_size=1, bias=False),
        )
        self.residual = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.fusion = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.stem(inputs)
        short = self.short_branch(features)
        long = self.long_branch(features)
        return self.fusion(short * long + self.residual(inputs))


class RawHybridAttention(nn.Module):
    """Channel attention followed by local and dilated temporal attention."""

    def __init__(self, channels: int = 64, reduction: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        self.temporal_local = nn.Conv1d(
            channels,
            channels,
            kernel_size=15,
            padding=7,
            groups=channels,
            bias=False,
        )
        self.temporal_dilated = nn.Conv1d(
            channels,
            channels,
            kernel_size=15,
            padding=28,
            dilation=4,
            groups=channels,
            bias=False,
        )
        self.temporal_bn = nn.BatchNorm1d(channels)
        self.temporal_pointwise = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, features: Tensor) -> Tensor:
        channel_weights = self.channel_gate(features).unsqueeze(-1)
        channel_features = features * channel_weights
        temporal = self.temporal_local(channel_features)
        temporal = temporal + self.temporal_dilated(channel_features)
        temporal = F.gelu(self.temporal_bn(temporal))
        temporal_weights = torch.sigmoid(self.temporal_pointwise(temporal))
        return channel_features * temporal_weights


class HADANetRaw(nn.Module):
    """HADANet domain adaptation operating on raw 64-channel EEG."""

    def __init__(
        self,
        channels: int = 64,
        samples: int = 656,
        classes: int = 4,
        widen: int = 128,
        pooled_samples: int = 20,
        aligner_hidden_dim: int = 512,
        domain_hidden_dim: int = 512,
        hcnn_dropout: float = 0.2,
        attention_dropout: float = 0.1,
        aligner_dropout: float = 0.3,
        domain_dropout: float = 0.3,
        classifier_dropout: float = 0.5,
    ):
        super().__init__()
        self.channels = channels
        self.samples = samples
        self.feature_dim = channels * pooled_samples
        self.hierarchical_cnn = RawHierarchicalCNN(
            channels, widen=widen, dropout=hcnn_dropout
        )
        self.attention = RawHybridAttention(
            channels, dropout=attention_dropout
        )
        self.temporal_pool = nn.AdaptiveAvgPool1d(pooled_samples)
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
        if inputs.ndim != 3 or inputs.shape[1:] != (
            self.channels,
            self.samples,
        ):
            raise ValueError(
                f"Expected [batch, {self.channels}, {self.samples}], "
                f"got {tuple(inputs.shape)}."
            )
        features = self.hierarchical_cnn(inputs)
        features = self.attention(features)
        features = self.temporal_pool(features).flatten(start_dim=1)
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
        return (gram - identity).square().sum() / float(weight.size(1) ** 2)

    @staticmethod
    def grl_alpha(epoch: int, epochs: int) -> float:
        from hadanet.model import HADANet as _Base

        return _Base.grl_alpha(epoch, epochs)
