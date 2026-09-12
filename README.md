# HADANet-Raw: PhysioNet LOSO runner

This branch evaluates a raw-EEG variant of HADANet. It keeps the strict
unlabeled-target LOSO protocol and HADANet domain-adaptation objectives, but
replaces the paper's hand-crafted DE input with three-second EEG trials.

## What is implemented

The active model in `hadanet/model_raw.py` uses:

1. raw three-second EEG, shaped `[B, 64, 480]`;
2. short- and long-range learned temporal convolutions;
3. element-wise temporal fusion;
4. channel and local/dilated temporal attention;
5. residual feature alignment `Z = F_A + Delta F`;
6. a gradient-reversal domain discriminator;
7. five-kernel MK-MMD alignment;
8. a two-layer motion classifier with orthogonal regularization.

The optimized objective is:

```text
L = L_cls + 1.0 L_adv + 0.5 L_mmd + 0.1 L_ortho
```

`L_cls` receives source labels only. Target labels never enter preprocessing,
training, validation, early stopping, or checkpoint selection.

## Strict LOSO protocol

The default benchmark uses PhysioNet subjects S001-S020. For target S001,
S002-S020 are labeled sources; the complete EEG set of S001 is the unlabeled
target domain. This is repeated for all 20 target subjects.

Each source subject contributes a stratified 95/5 train/validation split. The
source validation accuracy selects the checkpoint. The held-out target labels
are read once, after training, for the final fold score. Target EEG is used
without labels during training, so this is transductive UDA rather than pure
domain generalization.

The PhysioNet loader uses imagery runs 4, 6, 8, 10, 12, and 14. It excludes
rest and executed-movement runs and crops the canonical `[0, 3)` interval from
cue onset. No notch filter, frequency-band filter bank, temporal segmentation,
or differential entropy is applied. The cached input is raw EEG with shape
`[trial, 64, 480]`.

## Installation and training

```bash
python -m pip install -r requirements_loso.txt
python train_physionet_loso.py \
  --subjects 1-20 \
  --targets 1-20 \
  --epochs 150 \
  --batch-size 40 \
  --validation-fraction 0.05 \
  --device cuda
```

For a quick end-to-end check, run one target fold:

```bash
python train_physionet_loso.py --subjects 1-20 --targets 1 --epochs 2
```

Raw EDF files and versioned three-second trials are cached under `data/`. Results are
written to `results/physionet_loso/seed_42/`, including:

- one source-validation-selected checkpoint per target;
- epoch histories;
- per-fold accuracy, kappa, recall, and macro F1;
- the mean and standard deviation across completed LOSO folds.

## Verification

```bash
python -m unittest discover -s tests -v
```

The tests cover architecture forward/backward, DE tensor construction, LOSO
subject exclusion, and the absence of labels from the target training loader.

One-cell cloud runners are provided in `colab_train_physionet_loso.ipynb` and
`kaggle_train_physionet_loso.ipynb`.
