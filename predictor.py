from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils import set_seed,freeze_module, clear_memory, atomic_torch_save, atomic_json_dump, load_torch_file, capture_torch_rng_state, restore_torch_rng_state
from fm_models import  sample_intermediate_states
from build_dataset_predictor_from_zero import make_paired_stratified_splits, make_paired_loader


EXPECTED_FM_POSITIVE_SAMPLING_STRATEGY = "single_seeded_permutation_v1"


# ============================================================
# Validity Predictor
# ============================================================


class ValidityPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout

        layers: list[nn.Module] = []
        input_dim = latent_dim + 1

        for layer_index in range(num_hidden_layers):
            layers.extend(
                [
                    nn.Linear(input_dim if layer_index == 0 else hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat((x_t, t), dim=-1))


def make_random_splits(
    positive: torch.Tensor,
    negative: torch.Tensor,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, TensorDataset]:
    if validation_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("Split fractions must be non-negative.")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError(
            "validation_fraction + test_fraction must be smaller than 1."
        )

    # Match the original training script: build the balanced dataset first,
    # shuffle it once, then slice the shuffled dataset into splits.
    endpoints = torch.cat((positive, negative), dim=0).float()
    labels = torch.cat(
        (
            torch.ones(len(positive), 1),
            torch.zeros(len(negative), 1),
        ),
        dim=0,
    )
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(endpoints), generator=generator)
    endpoints = endpoints[permutation]
    labels = labels[permutation]

    n_validation = int(round(len(endpoints) * validation_fraction))
    n_test = int(round(len(endpoints) * test_fraction))
    n_train = len(endpoints) - n_validation - n_test

    splits = {
        "train": TensorDataset(endpoints[:n_train], labels[:n_train]),
        "validation": TensorDataset(
            endpoints[n_train:n_train + n_validation],
            labels[n_train:n_train + n_validation],
        ),
    }
    if n_test > 0:
        splits["test"] = TensorDataset(
            endpoints[n_train + n_validation:],
            labels[n_train + n_validation:],
        )
    return splits


# Backward-compatible name for callers outside this pipeline.
make_stratified_splits = make_random_splits


def make_loader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "persistent_workers": num_workers > 0,
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)


def binary_metrics_from_arrays(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    predictions = probabilities >= 0.5
    labels_bool = labels >= 0.5

    true_positive = int(np.logical_and(predictions, labels_bool).sum())
    true_negative = int(np.logical_and(~predictions, ~labels_bool).sum())
    false_positive = int(np.logical_and(predictions, ~labels_bool).sum())
    false_negative = int(np.logical_and(~predictions, labels_bool).sum())
    total = max(len(labels), 1)

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    result = {
        "accuracy": (true_positive + true_negative) / total,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }

    return result


@torch.no_grad()
def evaluate_predictor_random_t(
    model: ValidityPredictor,
    loader: DataLoader,
    device: torch.device,
    fixed_seed: int,
) -> dict[str, float]:
    rng_state = capture_torch_rng_state()

    try:
        set_seed(fixed_seed)
        model.eval()
        loss_sum = 0.0
        sample_count = 0
        all_probabilities: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for endpoints, labels in loader:
            endpoints = endpoints.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            x_t, t = sample_intermediate_states(endpoints)
            logits = model(x_t, t)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="sum",
            )
            probabilities = torch.sigmoid(logits)

            loss_sum += float(loss.item())
            sample_count += len(endpoints)
            all_probabilities.append(probabilities.cpu().numpy().reshape(-1))
            all_labels.append(labels.cpu().numpy().reshape(-1))

        probabilities_np = np.concatenate(all_probabilities)
        labels_np = np.concatenate(all_labels)
        metrics = binary_metrics_from_arrays(probabilities_np, labels_np)
        metrics["bce"] = loss_sum / max(sample_count, 1)
        return metrics
    finally:
        restore_torch_rng_state(rng_state)


@torch.no_grad()
def evaluate_predictor_at_time(
    model: ValidityPredictor,
    loader: DataLoader,
    device: torch.device,
    time_value: float,
    fixed_seed: int,
) -> dict[str, float]:
    rng_state = capture_torch_rng_state()

    try:
        set_seed(fixed_seed)
        model.eval()
        all_probabilities: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        loss_sum = 0.0
        sample_count = 0

        for endpoints, labels in loader:
            endpoints = endpoints.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            source = torch.randn_like(endpoints)
            t = torch.full(
                (len(endpoints), 1),
                time_value,
                device=device,
                dtype=endpoints.dtype,
            )
            x_t = (1.0 - t) * source + t * endpoints
            logits = model(x_t, t)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="sum",
            )
            probabilities = torch.sigmoid(logits)

            loss_sum += float(loss.item())
            sample_count += len(endpoints)
            all_probabilities.append(probabilities.cpu().numpy().reshape(-1))
            all_labels.append(labels.cpu().numpy().reshape(-1))

        probabilities_np = np.concatenate(all_probabilities)
        labels_np = np.concatenate(all_labels)
        metrics = binary_metrics_from_arrays(probabilities_np, labels_np)
        metrics["bce"] = loss_sum / max(sample_count, 1)
        return metrics
    finally:
        restore_torch_rng_state(rng_state)


def save_training_plot(history: list[dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"Plot skipped because matplotlib is unavailable: {error}")
        return

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_bce"] for item in history]
    validation_loss = [item["validation_bce"] for item in history]
    validation_accuracy = [item["validation_accuracy"] for item in history]

    figure, axis_loss = plt.subplots(figsize=(9, 5.5))
    line_train = axis_loss.plot(epochs, train_loss, label="Train BCE")[0]
    line_validation = axis_loss.plot(epochs, validation_loss, label="Validation BCE")[0]
    axis_loss.set_xlabel("Epoch")
    axis_loss.set_ylabel("BCE")
    axis_loss.grid(True, alpha=0.3)

    axis_accuracy = axis_loss.twinx()
    line_accuracy = axis_accuracy.plot(
        epochs,
        validation_accuracy,
        label="Validation accuracy",
    )[0]
    axis_accuracy.set_ylabel("Accuracy")
    axis_accuracy.set_ylim(0.0, 1.0)

    figure.legend(
        handles=[line_train, line_validation, line_accuracy],
        loc="upper center",
        ncol=3,
    )
    plt.title("Validity predictor training")
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)



def train_validity_predictor(
    args: argparse.Namespace,
    dataset_path: Path,
    predictor_path: Path,
    output_dir: Path,
    device: torch.device,
) -> Path:
    if predictor_path.is_file() and not args.overwrite_predictor:
        print(f"Predictor already exists; reusing it: {predictor_path}")
        return predictor_path

    data = load_torch_file(dataset_path)
    positive = data["positive_endpoints_normalized"].float()
    negative = data["hard_negative_endpoints_normalized"].float()
    latent_dim = int(positive.shape[1])
    transform_payload = data.get("latent_transform")

    if args.predictor_dataset_source == "fm":
        sampling_strategy = data.get("metadata", {}).get(
            "positive_sampling_strategy"
        )
        if sampling_strategy != EXPECTED_FM_POSITIVE_SAMPLING_STRATEGY:
            raise RuntimeError(
                "The FM predictor dataset was created with the old positive "
                "sampling strategy and may leak duplicate endpoints across "
                "splits. Rebuild the dataset before training the predictor."
            )

        splits = make_random_splits(
            positive=positive,
            negative=negative,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        loaders = {
        name: make_loader(
            dataset=split,
            batch_size=args.predictor_batch_size,
            shuffle=name == "train",
            num_workers=args.num_workers,
        )
        for name, split in splits.items()
    }
    elif args.predictor_dataset_source == "gaussian":
        splits = make_paired_stratified_splits(
            positive=positive,
            negative=negative,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        loaders = {
        name: make_paired_loader(
            dataset=split,
            batch_size=args.predictor_batch_size,
            shuffle=name == "train",
            num_workers=args.num_workers,
            seed=args.seed,
        )
        for name, split in splits.items()
    }

    print("\nPredictor split sizes:")
    for name, split in splits.items():
        print(f"  {name:<10}: {len(split):,}")

    model = ValidityPredictor(
        latent_dim=latent_dim,
        hidden_dim=args.predictor_hidden_dim,
        num_hidden_layers=args.predictor_num_hidden_layers,
        dropout=args.predictor_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.predictor_learning_rate,
        weight_decay=args.predictor_weight_decay,
    )

    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print("\nStarting validity predictor training.")

    for epoch in range(1, args.predictor_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        train_correct = 0

        for endpoints, labels in loaders["train"]:
            endpoints = endpoints.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            x_t, t = sample_intermediate_states(endpoints)
            logits = model(x_t, t)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.predictor_grad_clip)
            optimizer.step()

            with torch.no_grad():
                predictions = (torch.sigmoid(logits) >= 0.5).float()
                train_correct += int((predictions == labels).sum().item())

            train_loss_sum += float(loss.item()) * len(endpoints)
            train_count += len(endpoints)

        train_bce = train_loss_sum / max(train_count, 1)
        train_accuracy = train_correct / max(train_count, 1)
        validation_metrics = evaluate_predictor_random_t(
            model=model,
            loader=loaders["validation"],
            device=device,
            fixed_seed=args.seed + 20_000,
        )
        is_best = validation_metrics["bce"] < best_validation_loss

        if is_best:
            best_validation_loss = validation_metrics["bce"]
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_result = {
            "epoch": epoch,
            "train_bce": train_bce,
            "train_accuracy": train_accuracy,
            "validation_bce": validation_metrics["bce"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_f1": validation_metrics["f1"],
            # "validation_roc_auc": validation_metrics["roc_auc"],
            "is_best": is_best,
        }
        history.append(epoch_result)
        atomic_json_dump(history, output_dir / "predictor_training_history.json")

        if is_best:
            atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "predictor_config": {
                        "latent_dim": latent_dim,
                        "hidden_dim": args.predictor_hidden_dim,
                        "num_hidden_layers": args.predictor_num_hidden_layers,
                        "dropout": args.predictor_dropout,
                    },
                    "latent_transform": transform_payload,
                    "best_validation_loss": best_validation_loss,
                    "best_epoch": best_epoch,
                    "dataset_path": str(dataset_path),
                    "dataset_metadata": data.get("metadata", {}),
                    "fm_checkpoint": (str(Path(args.fm_checkpoint)) if args.fm_checkpoint else None),                    
                    "training_history": history,
                },
                predictor_path,
            )

        print(
            f"Epoch {epoch:03d}/{args.predictor_epochs:03d} | "
            f"train_bce={train_bce:.5f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_bce={validation_metrics['bce']:.5f} | "
            f"val_acc={validation_metrics['accuracy']:.4f}",
            # f"val_auc={validation_metrics['roc_auc']:.4f}",
            flush=True,
        )

        if epochs_without_improvement >= args.predictor_patience:
            print(
                "Early stopping: no improvement for "
                f"{args.predictor_patience} epochs."
            )
            break

    best_checkpoint = load_torch_file(predictor_path)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_random = None
    test_by_time = None
    if "test" in loaders and len(splits["test"]) > 0:
        test_random = evaluate_predictor_random_t(
            model=model,
            loader=loaders["test"],
            device=device,
            fixed_seed=args.seed + 30_000,
        )
        test_by_time = {
            f"t={time_value:g}": evaluate_predictor_at_time(
                model=model,
                loader=loaders["test"],
                device=device,
                time_value=time_value,
                fixed_seed=args.seed + 40_000 + int(1000 * time_value),
            )
            for time_value in args.predictor_test_times
        }

    report = {
        "best_epoch": int(best_checkpoint["best_epoch"]),
        "best_validation_loss": float(best_checkpoint["best_validation_loss"]),
        "test_random_t": test_random,
        "test_by_time": test_by_time,
        "split_sizes": {name: len(split) for name, split in splits.items()},
        "predictor_config": best_checkpoint["predictor_config"],
    }
    atomic_json_dump(report, output_dir / "predictor_test_metrics.json")
    save_training_plot(history, output_dir / "predictor_training_curves.png")

    print("\nPredictor trained.")
    print(f"Checkpoint : {predictor_path}")
    print(f"Best epoch : {report['best_epoch']}")
    if test_random is not None:
        print(
            "Test random t: "
            f"accuracy={test_random['accuracy']:.4f}, "
            f"F1={test_random['f1']:.4f}"
            # f"AUC={test_random['roc_auc']:.4f}"
        )

    del model, optimizer, best_checkpoint, data, positive, negative, splits, loaders
    clear_memory()
    return predictor_path


def load_validity_predictor(
    path: Path,
    device: torch.device,
) -> tuple[ValidityPredictor, dict[str, Any]]:
    checkpoint = load_torch_file(path)

    state_dict = checkpoint["model_state_dict"]
    config = dict(checkpoint["predictor_config"])

    model = ValidityPredictor(
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_hidden_layers=int(
            config["num_hidden_layers"]
        ),
        dropout=float(config["dropout"]),
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(device)
    freeze_module(model)

    transform_payload = checkpoint.get(
        "latent_transform"
    )
    metadata = {
        "predictor_config": config,
        "best_epoch": int(
            checkpoint["best_epoch"]
        ),
        "best_validation_loss": float(
            checkpoint["best_validation_loss"]
        ),
        "fm_checkpoint": checkpoint.get(
            "fm_checkpoint"
        ),
        "latent_transform": dict(
            transform_payload["config"]
        ),
    }

    return model, metadata
