"""Train the published HADANet design with strict PhysioNet LOSO UDA.

Target EEG is available without labels during optimization.  Target labels are
held out until the single final evaluation of each fold.  Model selection uses
only labeled source-validation trials.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from hadanet.model import HADANet, HADANetLoss
from hadanet.physionet import build_loso_fold, load_subject_pool


def parse_subjects(value: str) -> list[int]:
    subjects = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = (int(item) for item in part.split("-", maxsplit=1))
            subjects.extend(range(start, stop + 1))
        else:
            subjects.append(int(part))
    subjects = sorted(set(subjects))
    if not subjects or any(subject < 1 or subject > 109 for subject in subjects):
        raise argparse.ArgumentTypeError("Subjects must be in the PhysioNet range 1..109.")
    return subjects


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HADANet PhysioNet strict LOSO (unlabeled-target UDA)."
    )
    parser.add_argument("--subjects", type=parse_subjects, default=parse_subjects("1-20"))
    parser.add_argument(
        "--targets",
        type=parse_subjects,
        default=None,
        help="Target folds to run; default is every subject in --subjects.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/physionet_raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/physionet_de"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/physionet_loso"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.05,
        help="Per-source validation fraction (default: 0.05, i.e. 95/5).",
    )
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--verbose", default="WARNING")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU.")
    return device


def make_loaders(fold, batch_size: int, num_workers: int):
    source_train = TensorDataset(
        torch.from_numpy(fold.source_train_x), torch.from_numpy(fold.source_train_y)
    )
    source_val = TensorDataset(
        torch.from_numpy(fold.source_val_x), torch.from_numpy(fold.source_val_y)
    )
    # Deliberately contains no label tensor.
    target_unlabeled = TensorDataset(torch.from_numpy(fold.target_train_x))
    target_test = TensorDataset(
        torch.from_numpy(fold.target_test_x), torch.from_numpy(fold.target_test_y)
    )
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    return (
        DataLoader(source_train, shuffle=True, drop_last=True, **common),
        DataLoader(source_val, shuffle=False, **common),
        DataLoader(target_unlabeled, shuffle=True, drop_last=True, **common),
        DataLoader(target_test, shuffle=False, **common),
    )


@torch.no_grad()
def evaluate(model: HADANet, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    labels, predictions = [], []
    loss_sum = 0.0
    sample_count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss_sum += torch.nn.functional.cross_entropy(
            logits, targets, reduction="sum"
        ).item()
        labels.extend(targets.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        sample_count += len(targets)
    return {
        "loss": loss_sum / sample_count,
        "accuracy": accuracy_score(labels, predictions),
        "kappa": cohen_kappa_score(labels, predictions),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
        "f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "samples": sample_count,
    }


def train_fold(args, fold, device: torch.device, fold_dir: Path) -> dict:
    source_loader, val_loader, target_loader, test_loader = make_loaders(
        fold, args.batch_size, args.num_workers
    )
    if len(source_loader) == 0 or len(target_loader) == 0:
        raise RuntimeError("Batch size is larger than a training split.")

    model = HADANet().to(device)
    objective = HADANetLoss(
        adversarial_weight=1.0, mmd_weight=0.5, orthogonal_weight=0.1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_state = None
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    started_at = time.perf_counter()

    print(
        f"[fold S{fold.target_subject:03d}] source_subjects={len(fold.source_subjects)} "
        f"| source_train={len(fold.source_train_y)} | source_val={len(fold.source_val_y)} "
        f"| target_unlabeled={len(fold.target_train_x)} | batch={args.batch_size}",
        flush=True,
    )
    for epoch in range(args.epochs):
        model.train()
        alpha = model.grl_alpha(epoch, args.epochs)
        target_iterator = iter(target_loader)
        totals = {name: 0.0 for name in ("total", "classification", "adversarial", "mmd", "orthogonal")}
        correct = 0
        seen = 0
        for source_x, source_y in source_loader:
            try:
                (target_x,) = next(target_iterator)
            except StopIteration:
                target_iterator = iter(target_loader)
                (target_x,) = next(target_iterator)
            source_x = source_x.to(device, non_blocking=True)
            source_y = source_y.to(device, non_blocking=True)
            target_x = target_x.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model.forward_domains(source_x, target_x, alpha)
            losses = objective(model, outputs, source_y)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_count = len(source_y)
            for name in totals:
                totals[name] += float(losses[name].detach()) * batch_count
            correct += int((outputs["source_logits"].argmax(dim=1) == source_y).sum())
            seen += batch_count

        validation = evaluate(model, val_loader, device)
        epoch_stats = {
            "epoch": epoch + 1,
            "alpha": alpha,
            "train_accuracy": correct / seen,
            "val_accuracy": validation["accuracy"],
            "val_loss": validation["loss"],
            **{f"train_{name}": value / seen for name, value in totals.items()},
        }
        history.append(epoch_stats)
        elapsed = (time.perf_counter() - started_at) / 60.0
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"loss={epoch_stats['train_total']:.4f} "
            f"cls={epoch_stats['train_classification']:.4f} "
            f"adv={epoch_stats['train_adversarial']:.4f} "
            f"mmd={epoch_stats['train_mmd']:.4f} "
            f"ortho={epoch_stats['train_orthogonal']:.6f} | "
            f"train_acc={epoch_stats['train_accuracy'] * 100:.2f}% "
            f"val_acc={validation['accuracy'] * 100:.2f}% "
            f"val_loss={validation['loss']:.4f} | alpha={alpha:.3f} "
            f"elapsed={elapsed:.1f}m",
            flush=True,
        )

        improved = validation["accuracy"] > best_val_accuracy or (
            validation["accuracy"] == best_val_accuracy
            and validation["loss"] < best_val_loss
        )
        if improved:
            best_val_accuracy = validation["accuracy"]
            best_val_loss = validation["loss"]
            best_epoch = epoch + 1
            stale_epochs = 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if args.patience > 0 and stale_epochs >= args.patience:
                print(
                    f"[fold S{fold.target_subject:03d}] early stop at epoch "
                    f"{epoch + 1}; best source-val epoch={best_epoch}",
                    flush=True,
                )
                break

    if best_state is None:
        raise RuntimeError("Training produced no source-validation checkpoint.")
    model.load_state_dict(best_state)
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "target_subject": fold.target_subject,
            "source_subjects": fold.source_subjects,
            "best_epoch": best_epoch,
            "best_source_val_accuracy": best_val_accuracy,
        },
        fold_dir / "best_source_val.pt",
    )
    with (fold_dir / "history.json").open("w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2)

    # This is the first and only point at which target labels are consumed.
    test_started = time.perf_counter()
    target_result = evaluate(model, test_loader, device)
    target_result["test_seconds"] = time.perf_counter() - test_started
    target_result["target_subject"] = fold.target_subject
    target_result["best_epoch"] = best_epoch
    target_result["source_val_accuracy"] = best_val_accuracy
    print(
        f"TARGET S{fold.target_subject:03d} FINAL | samples={target_result['samples']} "
        f"| acc={target_result['accuracy'] * 100:.2f}% "
        f"| kappa={target_result['kappa']:.4f} | f1={target_result['f1']:.4f} "
        f"| selected_epoch={best_epoch}",
        flush=True,
    )
    return target_result


def main() -> None:
    args = arguments()
    targets = args.subjects if args.targets is None else args.targets
    missing_targets = sorted(set(targets) - set(args.subjects))
    if missing_targets:
        raise ValueError(f"Targets are absent from --subjects: {missing_targets}.")
    if len(args.subjects) < 2:
        raise ValueError("LOSO requires at least two subjects.")
    set_seed(args.seed)
    device = choose_device(args.device)
    print(
        f"HADANet strict LOSO | subjects={args.subjects} | targets={targets} "
        f"| device={device} | seed={args.seed}",
        flush=True,
    )
    subject_pool = load_subject_pool(
        args.subjects, args.data_dir, args.cache_dir, verbose=args.verbose
    )
    if args.download_only:
        print("Download and DE cache creation completed.", flush=True)
        return

    run_dir = args.results_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for target_subject in targets:
        set_seed(args.seed + target_subject)
        fold = build_loso_fold(
            subject_pool,
            target_subject,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        result = train_fold(
            args, fold, device, run_dir / f"subject_{target_subject:03d}"
        )
        results.append(result)
        mean_accuracy = np.mean([item["accuracy"] for item in results])
        print(
            f"PARTIAL LOSO | folds={len(results)}/{len(targets)} "
            f"| mean_acc={mean_accuracy * 100:.2f}%",
            flush=True,
        )

    fieldnames = list(results[0])
    with (run_dir / "fold_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "subjects": args.subjects,
        "targets": targets,
        "seed": args.seed,
        "mean_accuracy": float(np.mean([item["accuracy"] for item in results])),
        "std_accuracy": float(np.std([item["accuracy"] for item in results])),
        "mean_kappa": float(np.mean([item["kappa"] for item in results])),
        "mean_f1": float(np.mean([item["f1"] for item in results])),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(
        f"LOSO COMPLETE | mean_acc={summary['mean_accuracy'] * 100:.2f}% "
        f"+- {summary['std_accuracy'] * 100:.2f}% "
        f"| mean_kappa={summary['mean_kappa']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
