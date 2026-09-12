"""Reproducible HADANet components and PhysioNet LOSO utilities."""

from .model import HADANet as HADANetV1, HADANetLoss
from .model_v2 import HADANetV2

# V2 is the active architecture; V1 remains importable for controlled A/B tests.
HADANet = HADANetV2

__all__ = ["HADANet", "HADANetV1", "HADANetV2", "HADANetLoss"]
