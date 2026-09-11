"""PhysioNet EEGMMIDB preprocessing and leakage-free LOSO assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


IMAGERY_RUNS = (4, 6, 8, 10, 12, 14)
BANDS = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 50.0),
)
CLASS_NAMES = ("left_hand", "right_hand", "both_hands", "both_feet")
CACHE_VERSION = 1


def differential_entropy_features(
    trials: np.ndarray,
    sfreq: float = 160.0,
    segment_seconds: float = 1.0,
) -> np.ndarray:
    """Convert four-second trials to [trial, channel, 5 bands, 4 segments]."""
    from scipy.signal import butter, sosfiltfilt

    if trials.ndim != 3:
        raise ValueError(f"Expected [trials, channels, samples], got {trials.shape}.")
    segment_samples = int(round(segment_seconds * sfreq))
    temporal_segments = 4
    required_samples = segment_samples * temporal_segments
    if trials.shape[-1] < required_samples:
        raise ValueError(
            f"Each trial needs {required_samples} samples, got {trials.shape[-1]}."
        )
    trials = trials[..., :required_samples].astype(np.float64, copy=False)
    features = np.empty(
        (trials.shape[0], trials.shape[1], len(BANDS), temporal_segments),
        dtype=np.float32,
    )
    nyquist = sfreq / 2.0
    for band_index, (_, low, high) in enumerate(BANDS):
        high = min(high, nyquist - 1e-3)
        sos = butter(4, (low, high), btype="bandpass", fs=sfreq, output="sos")
        filtered = sosfiltfilt(sos, trials, axis=-1)
        segments = filtered.reshape(
            trials.shape[0], trials.shape[1], temporal_segments, segment_samples
        )
        variance = segments.var(axis=-1, ddof=1)
        features[:, :, band_index, :] = (
            0.5 * np.log(2.0 * np.pi * np.e * np.maximum(variance, 1e-12))
        ).astype(np.float32)
    return features


def _load_imagery_trials(
    subject_id: int,
    data_path: Path,
    verbose: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Download/read all six imagery runs while preserving run-specific labels."""
    import mne
    from mne.datasets import eegbci

    data_path.mkdir(parents=True, exist_ok=True)
    file_paths = eegbci.load_data(
        subject_id,
        runs=list(IMAGERY_RUNS),
        path=str(data_path),
        update_path=False,
        verbose=verbose,
    )
    all_trials = []
    all_labels = []
    for run, file_path in zip(IMAGERY_RUNS, file_paths):
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=verbose)
        eegbci.standardize(raw)
        raw.pick_types(eeg=True, meg=False, stim=False)
        if len(raw.ch_names) != 64:
            raise RuntimeError(
                f"PhysioNet S{subject_id:03d} run {run} has "
                f"{len(raw.ch_names)} EEG channels; expected 64."
            )
        raw.notch_filter(50.0, verbose=verbose)
        raw.filter(0.5, 50.0, verbose=verbose)
        events, _ = mne.events_from_annotations(
            raw, event_id={"T1": 1, "T2": 2}, verbose=verbose
        )
        epochs = mne.Epochs(
            raw,
            events,
            event_id={"T1": 1, "T2": 2},
            tmin=0.0,
            tmax=4.0 - 1.0 / raw.info["sfreq"],
            baseline=None,
            preload=True,
            reject_by_annotation=True,
            verbose=verbose,
        )
        trials = epochs.get_data(copy=True)
        event_codes = epochs.events[:, 2]
        if run in (4, 8, 12):
            labels = np.where(event_codes == 1, 0, 1)
        else:
            labels = np.where(event_codes == 1, 2, 3)
        all_trials.append(trials)
        all_labels.append(labels.astype(np.int64))

    trials = np.concatenate(all_trials, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    observed = np.unique(labels)
    if not np.array_equal(observed, np.arange(4)):
        raise RuntimeError(
            f"PhysioNet S{subject_id:03d} classes are {observed.tolist()}, expected 0..3."
        )
    return trials, labels


def load_subject_de(
    subject_id: int,
    data_path: str | Path,
    cache_path: str | Path,
    verbose: str = "WARNING",
) -> tuple[np.ndarray, np.ndarray]:
    """Load a subject's DE tensor, using a versioned on-disk cache."""
    data_path = Path(data_path)
    cache_path = Path(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"physionet_s{subject_id:03d}_de_v{CACHE_VERSION}.npz"
    if cache_file.is_file():
        cached = np.load(cache_file)
        features = cached["features"].astype(np.float32, copy=False)
        labels = cached["labels"].astype(np.int64, copy=False)
    else:
        trials, labels = _load_imagery_trials(subject_id, data_path, verbose)
        features = differential_entropy_features(trials, sfreq=160.0)
        np.savez_compressed(cache_file, features=features, labels=labels)

    if features.ndim != 4 or features.shape[1:] != (64, 5, 4):
        raise RuntimeError(
            f"Invalid cached shape for S{subject_id:03d}: {features.shape}."
        )
    if len(features) != len(labels):
        raise RuntimeError(f"Feature/label count mismatch for S{subject_id:03d}.")
    return features, labels


def load_subject_pool(
    subject_ids: Iterable[int],
    data_path: str | Path,
    cache_path: str | Path,
    verbose: str = "WARNING",
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    pool = {}
    for subject_id in subject_ids:
        features, labels = load_subject_de(
            subject_id, data_path, cache_path, verbose=verbose
        )
        counts = np.bincount(labels, minlength=4).tolist()
        print(
            f"[data] S{subject_id:03d} | trials={len(labels)} | "
            f"classes={counts} | shape={features.shape}",
            flush=True,
        )
        pool[subject_id] = (features, labels)
    return pool


@dataclass(frozen=True)
class LOSOFold:
    target_subject: int
    source_subjects: tuple[int, ...]
    source_train_x: np.ndarray
    source_train_y: np.ndarray
    source_val_x: np.ndarray
    source_val_y: np.ndarray
    target_train_x: np.ndarray
    target_test_x: np.ndarray
    target_test_y: np.ndarray


def build_loso_fold(
    subject_pool: dict[int, tuple[np.ndarray, np.ndarray]],
    target_subject: int,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> LOSOFold:
    """Create one fold without exposing target labels to the training datasets."""
    from sklearn.model_selection import train_test_split

    if target_subject not in subject_pool:
        raise KeyError(f"Target S{target_subject:03d} is absent from the subject pool.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")
    source_subjects = tuple(
        subject for subject in sorted(subject_pool) if subject != target_subject
    )
    if not source_subjects:
        raise ValueError("LOSO requires at least two subjects.")

    train_x_parts, train_y_parts = [], []
    val_x_parts, val_y_parts = [], []
    for subject in source_subjects:
        features, labels = subject_pool[subject]
        indices = np.arange(len(labels))
        train_indices, val_indices = train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=seed + subject,
            stratify=labels,
        )
        train_x_parts.append(features[train_indices])
        train_y_parts.append(labels[train_indices])
        val_x_parts.append(features[val_indices])
        val_y_parts.append(labels[val_indices])

    source_train_x = np.concatenate(train_x_parts).astype(np.float32, copy=False)
    source_train_y = np.concatenate(train_y_parts).astype(np.int64, copy=False)
    source_val_x = np.concatenate(val_x_parts).astype(np.float32, copy=False)
    source_val_y = np.concatenate(val_y_parts).astype(np.int64, copy=False)
    target_x, target_y = subject_pool[target_subject]

    # Fit normalization only on labeled source training trials.  Target EEG is
    # transformed but never used to estimate supervised preprocessing state.
    mean = source_train_x.mean(axis=0, keepdims=True)
    std = source_train_x.std(axis=0, keepdims=True).clip(min=1e-6)

    def standardize(array: np.ndarray) -> np.ndarray:
        return ((array - mean) / std).astype(np.float32, copy=False)

    target_x = standardize(target_x)
    return LOSOFold(
        target_subject=target_subject,
        source_subjects=source_subjects,
        source_train_x=standardize(source_train_x),
        source_train_y=source_train_y,
        source_val_x=standardize(source_val_x),
        source_val_y=source_val_y,
        target_train_x=target_x,
        target_test_x=target_x.copy(),
        target_test_y=target_y.astype(np.int64, copy=True),
    )
