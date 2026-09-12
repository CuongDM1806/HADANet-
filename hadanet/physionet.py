"""Raw PhysioNet EEGMMIDB loading and leakage-free LOSO assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


IMAGERY_RUNS = (4, 6, 8, 10, 12, 14)
CLASS_NAMES = ("left_hand", "right_hand", "both_hands", "both_feet")
TRIAL_SECONDS = 4.1
SFREQ = 160.0
TRIAL_SAMPLES = int(round(TRIAL_SECONDS * SFREQ))
CACHE_VERSION = 4


def _load_imagery_trials(
    subject_id: int,
    data_path: Path,
    verbose: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Download and crop six imagery runs to raw 4.1-second EEG trials."""
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
        events, _ = mne.events_from_annotations(
            raw, event_id={"T1": 1, "T2": 2}, verbose=verbose
        )
        epochs = mne.Epochs(
            raw,
            events,
            event_id={"T1": 1, "T2": 2},
            tmin=0.0,
            tmax=TRIAL_SECONDS - 1.0 / raw.info["sfreq"],
            baseline=None,
            preload=True,
            reject_by_annotation=True,
            verbose=verbose,
        )
        trials = epochs.get_data(copy=True).astype(np.float32, copy=False)
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


def load_subject_trials(
    subject_id: int,
    data_path: str | Path,
    cache_path: str | Path,
    verbose: str = "WARNING",
) -> tuple[np.ndarray, np.ndarray]:
    """Load raw [trial, 64, 656] EEG using a versioned on-disk cache."""
    data_path = Path(data_path)
    cache_path = Path(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = (
        cache_path / f"physionet_s{subject_id:03d}_raw4p1s_v{CACHE_VERSION}.npz"
    )
    if cache_file.is_file():
        cached = np.load(cache_file)
        trials = cached["trials"].astype(np.float32, copy=False)
        labels = cached["labels"].astype(np.int64, copy=False)
    else:
        trials, labels = _load_imagery_trials(subject_id, data_path, verbose)
        np.savez_compressed(cache_file, trials=trials, labels=labels)

    if trials.ndim != 3 or trials.shape[1:] != (64, TRIAL_SAMPLES):
        raise RuntimeError(
            f"Invalid cached shape for S{subject_id:03d}: {trials.shape}."
        )
    if len(trials) != len(labels):
        raise RuntimeError(f"Trial/label count mismatch for S{subject_id:03d}.")
    return trials, labels


def load_subject_pool(
    subject_ids: Iterable[int],
    data_path: str | Path,
    cache_path: str | Path,
    verbose: str = "WARNING",
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    pool = {}
    for subject_id in subject_ids:
        trials, labels = load_subject_trials(
            subject_id, data_path, cache_path, verbose=verbose
        )
        counts = np.bincount(labels, minlength=4).tolist()
        print(
            f"[data] S{subject_id:03d} | trials={len(labels)} | "
            f"classes={counts} | raw_shape={trials.shape}",
            flush=True,
        )
        pool[subject_id] = (trials, labels)
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
    validation_fraction: float = 0.05,
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
        class_count = np.unique(labels).size
        validation_size = max(
            int(np.ceil(len(labels) * validation_fraction)), class_count
        )
        if len(labels) - validation_size < class_count:
            raise ValueError(
                f"S{subject:03d} has too few trials for a stratified "
                f"{validation_fraction:.1%} validation split."
            )
        train_indices, val_indices = train_test_split(
            indices,
            test_size=validation_size,
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

    return LOSOFold(
        target_subject=target_subject,
        source_subjects=source_subjects,
        source_train_x=source_train_x,
        source_train_y=source_train_y,
        source_val_x=source_val_x,
        source_val_y=source_val_y,
        target_train_x=target_x.astype(np.float32, copy=False),
        target_test_x=target_x.copy(),
        target_test_y=target_y.astype(np.int64, copy=True),
    )
