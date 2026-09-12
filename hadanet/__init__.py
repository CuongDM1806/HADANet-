"""Reproducible HADANet components and PhysioNet LOSO utilities."""

from .model import HADANet as HADANetV1, HADANetLoss
from .model_raw import HADANetRaw
from .model_v2 import HADANetV2

# The raw-EEG model is active on this branch. DE-based V1/V2 remain importable
# for controlled comparisons.
HADANet = HADANetRaw

__all__ = ["HADANet", "HADANetRaw", "HADANetV1", "HADANetV2", "HADANetLoss"]
