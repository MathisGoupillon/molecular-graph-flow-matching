from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from fm_models import strip_module_prefix, sample_ode
from utils import set_seed, LatentSpaceTransform,freeze_module, clear_memory, atomic_torch_save, atomic_json_dump,load_vae, load_torch_file, load_any_latent_tensor, capture_torch_rng_state,restore_torch_rng_state, load_latent_transform_from_checkpoint, float_tag
from evaluations import (
    compute_generation_metrics_from_latents,
    load_training_smiles,
    summarize_trials,
)
from predictor import load_validity_predictor, ValidityPredictor
from fcd_evaluation import (
    FCDReference,
    compute_fcd_result,
    empty_fcd_result,
    prepare_fcd_reference,
)



def build_training_latent_transform(
    latents: torch.Tensor,
    predictor_path: Path,
    device: torch.device,
) -> LatentSpaceTransform:
    transform = load_latent_transform_from_checkpoint(
        predictor_path,
        device,
    )
    print(
        "Using the exact latent transform stored in the predictor "
        f"checkpoint: {predictor_path}"
    )
    return transform

# ============================================================
# Variational Flow Matching endpoint-based
# ============================================================

class VFMEndpointDistributionMLP(nn.Module):
    """
    q_theta(x_1 | x_t, t) = N(mu_theta, diag(sigma_theta^2)).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        min_logvar: float,
        max_logvar: float,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be greater than or equal to 1.")
        if min_logvar >= max_logvar:
            raise ValueError("min_logvar must be strictly less than max_logvar.")

        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        layers: list[nn.Module] = []

        for layer_index in range(num_layers):
            input_dim = dim + 1 if layer_index == 0 else hidden_dim
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden_dim, dim)
        self.logvar_head = nn.Linear(hidden_dim, dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if t.ndim == 1:
            t = t[:, None]

        h = self.backbone(torch.cat((x_t, t), dim=-1))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(
            min=self.min_logvar,
            max=self.max_logvar,
        )
        return mu, logvar


def make_vfm_config(args: argparse.Namespace, dim: int) -> dict[str, Any]:
    return {
        "dim": int(dim),
        "hidden_dim": int(args.vfm_hidden_dim),
        "num_layers": int(args.vfm_num_layers),
        "dropout": float(args.vfm_dropout),
        "min_logvar": float(args.vfm_min_logvar),
        "max_logvar": float(args.vfm_max_logvar),
    }


def build_vfm_model(config: dict[str, Any]) -> VFMEndpointDistributionMLP:
    return VFMEndpointDistributionMLP(
        dim=int(config["dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config.get("dropout", 0.0)),
        min_logvar=float(config.get("min_logvar", -6.0)),
        max_logvar=float(config.get("max_logvar", 2.0)),
    )


def vfm_loss_terms(
    model: VFMEndpointDistributionMLP,
    endpoints: torch.Tensor,
    predictor: Optional[ValidityPredictor],
    validity_weight: float,
    validity_samples: int,
    eps_time: float,
) -> dict[str, torch.Tensor]:
    source = torch.randn_like(endpoints)
    t = torch.empty(
        len(endpoints),
        1,
        device=endpoints.device,
        dtype=endpoints.dtype,
    ).uniform_(0, 1.0 - eps_time)
    x_t = (1.0 - t) * source + t * endpoints
    mu, logvar = model(x_t, t)

    inverse_variance = torch.exp(-logvar)
    gaussian_nll = 0.5 * (
        math.log(2.0 * math.pi)
        + logvar
        + (endpoints - mu).square() * inverse_variance
    ).mean()

    endpoint_mse = F.mse_loss(mu, endpoints)
    predicted_std = torch.exp(0.5 * logvar).mean()

    validity_loss = gaussian_nll.new_zeros(())
    mean_validity_probability = gaussian_nll.new_full((), float("nan"))

    if validity_weight > 0.0:
        standard_deviation = torch.exp(0.5 * logvar)
        noise = torch.randn(
            validity_samples,
            *mu.shape,
            device=mu.device,
            dtype=mu.dtype,
        )
        proposed_endpoints = (
            mu.unsqueeze(0)
            + standard_deviation.unsqueeze(0) * noise
        )
        proposed_flat = proposed_endpoints.reshape(-1, model.dim)
        endpoint_time = torch.full(
            (len(proposed_flat), 1),
            1.0 - eps_time,
            device=proposed_flat.device,
            dtype=proposed_flat.dtype,
        )
        logits = predictor(proposed_flat, endpoint_time)
        validity_loss = F.softplus(-logits).mean()
        mean_validity_probability = torch.sigmoid(logits).mean()

    total_loss = gaussian_nll + validity_weight * validity_loss

    return {
        "total": total_loss,
        "gaussian_nll": gaussian_nll,
        "validity_loss": validity_loss,
        "endpoint_mse": endpoint_mse,
        "predicted_std": predicted_std,
        "mean_validity_probability": mean_validity_probability,
    }


def make_vfm_loader(
    latents: torch.Tensor,
    indices: range | torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
) -> DataLoader:
    dataset = Subset(TensorDataset(latents), indices)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": drop_last,
        "persistent_workers": num_workers > 0,
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)


def aggregate_vfm_metrics(
    sums: dict[str, float],
    sample_count: int,
) -> dict[str, float]:
    return {
        key: value / max(sample_count, 1)
        for key, value in sums.items()
    }

@torch.no_grad()
def evaluate_vfm_training_loss(
    model: VFMEndpointDistributionMLP,
    loader: DataLoader,
    transform: LatentSpaceTransform,
    predictor: Optional[ValidityPredictor],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    rng_state = capture_torch_rng_state()

    try:
        set_seed(seed)
        model.eval()
        sums = {
            "total": 0.0,
            "gaussian_nll": 0.0,
            "validity_loss": 0.0,
            "endpoint_mse": 0.0,
            "predicted_std": 0.0,
            "mean_validity_probability": 0.0,
        }
        valid_probability_count = 0
        sample_count = 0

        for (raw_endpoints,) in loader:
            raw_endpoints = raw_endpoints.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )
            endpoints = transform(raw_endpoints)
            terms = vfm_loss_terms(
                model=model,
                endpoints=endpoints,
                predictor=predictor,
                validity_weight=args.vfm_validity_weight,
                validity_samples=args.vfm_validity_samples,
                eps_time=args.vfm_eps_time,
            )
            batch_count = len(endpoints)
            sample_count += batch_count

            for key in (
                "total",
                "gaussian_nll",
                "validity_loss",
                "endpoint_mse",
                "predicted_std",
            ):
                sums[key] += float(terms[key].item()) * batch_count

            probability = terms["mean_validity_probability"]

            if torch.isfinite(probability):
                sums["mean_validity_probability"] += (
                    float(probability.item()) * batch_count
                )
                valid_probability_count += batch_count

        metrics = aggregate_vfm_metrics(sums, sample_count)

        if valid_probability_count == 0:
            metrics["mean_validity_probability"] = float("nan")
        else:
            metrics["mean_validity_probability"] = (
                sums["mean_validity_probability"] / valid_probability_count
            )

        return metrics
    finally:
        restore_torch_rng_state(rng_state)


def save_vfm_training_plot(
    history: list[dict[str, Any]],
    path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        return

    epochs = [entry["epoch"] for entry in history]
    train_total = [entry["train_total"] for entry in history]
    validation_total = [entry["validation_total"] for entry in history]
    train_nll = [entry["train_gaussian_nll"] for entry in history]
    validation_nll = [entry["validation_gaussian_nll"] for entry in history]
    validation_validity = [
        entry["validation_validity_loss"]
        for entry in history
    ]

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(epochs, train_total, label="Train total")
    axis.plot(epochs, validation_total, label="Validation total")
    axis.plot(epochs, train_nll, label="Train Gaussian NLL")
    axis.plot(epochs, validation_nll, label="Validation Gaussian NLL")

    if any(value != 0.0 for value in validation_validity):
        axis.plot(
            epochs,
            validation_validity,
            label="Validation validity loss",
        )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)

def save_vfm_checkpoint(
    path: Path,
    model: VFMEndpointDistributionMLP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_validation_loss: float,
    epochs_without_improvement: int,
    history: list[dict[str, Any]],
    transform: LatentSpaceTransform,
    args: argparse.Namespace,
    predictor_path: Path,
    train_count: int,
    validation_count: int,
) -> None:
    model_config = make_vfm_config(
        args=args,
        dim=model.dim,
    )

    state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    checkpoint = {
        "format_version": 2,
        "model_type": "vfm_endpoint_distribution",

        # Model
        "model_state_dict": state_dict,
        "vfm_config": model_config,

        # Training state
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": int(epoch),
        "best_validation_loss": float(best_validation_loss),
        "epochs_without_improvement": int(
            epochs_without_improvement
        ),
        "training_history": history,

        # Latent-space information
        "latent_transform": transform.checkpoint_payload(),
        "predictor_path": str(predictor_path),

        # Dataset information
        "dataset": args.dataset,
        "latents_path": str(Path(args.latents_path)),
        "train_count": int(train_count),
        "validation_count": int(validation_count),

        # Relevant training arguments
        "training_config": {
            key: value
            for key, value in vars(args).items()
            if key.startswith("vfm_")
            or key in {
                "dataset",
                "seed",
            }
        },
    }
    atomic_torch_save(
        checkpoint,
        path,
    )

def train_vfm(
    args: argparse.Namespace,
    predictor_path: Path,
    vfm_checkpoint_path: Path,
    vfm_last_checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
) -> Path:
    if vfm_checkpoint_path.is_file() and not args.overwrite_vfm:
        print(f"VFM already exists; reusing it: {vfm_checkpoint_path}")
        return vfm_checkpoint_path

    latents = load_any_latent_tensor(Path(args.latents_path), mmap=True)
    available_count = len(latents)
    used_count = (
        available_count
        if args.vfm_max_samples <= 0
        else min(available_count, args.vfm_max_samples)
    )

    validation_count = max(
        1,
        int(round(used_count * args.vfm_validation_fraction)),
    )
    train_count = used_count - validation_count

    if args.vfm_split_mode == "random":
        generator = torch.Generator().manual_seed(args.seed)
        permutation = torch.randperm(used_count, generator=generator)
        train_indices: range | torch.Tensor = permutation[:train_count]
        validation_indices: range | torch.Tensor = permutation[train_count:]
    else:
        train_indices = range(0, train_count)
        validation_indices = range(train_count, used_count)

    transform = build_training_latent_transform(
        latents=latents,
        predictor_path=predictor_path,
        device=device,
    )
    transformed_dim = transform.output_dim

    predictor: Optional[ValidityPredictor] = None
    predictor_metadata: Optional[dict[str, Any]] = None

    if args.vfm_validity_weight > 0.0:
        predictor, predictor_metadata = load_validity_predictor(
            predictor_path,
            device,
        )

    train_loader = make_vfm_loader(
        latents=latents,
        indices=train_indices,
        batch_size=args.vfm_batch_size,
        shuffle=args.vfm_shuffle_train,
        num_workers=args.num_workers,
        drop_last=args.vfm_drop_last,
    )
    validation_loader = make_vfm_loader(
        latents=latents,
        indices=validation_indices,
        batch_size=args.vfm_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    config = make_vfm_config(args, transformed_dim)
    model = build_vfm_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.vfm_learning_rate, weight_decay=args.vfm_weight_decay,)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.vfm_epochs,eta_min=args.vfm_min_learning_rate,)
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    if args.vfm_resume_checkpoint:
        resume_path = Path(args.vfm_resume_checkpoint)
        checkpoint = load_torch_file(resume_path)
        resume_config = checkpoint.get("vfm_config", {})

        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("model_state"),
        )
        model.load_state_dict(strip_module_prefix(state_dict), strict=True)

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            print(
                "Warning: the resume checkpoint has no scheduler "
                "state; the scheduler restarts from its initial state.")

        history = list(checkpoint.get("training_history", []))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_validation_loss = float(
            checkpoint.get("best_validation_loss", float("inf"))
        )
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )

        print(f"Resuming VFM training from: {resume_path}")
        print(f"Next epoch: {start_epoch}")

    print("\nVFM configuration:")
    print(f"  dataset              : {args.dataset}")
    print(f"  latent vectors used  : {used_count:,}")
    print(f"  train / validation   : {train_count:,} / {validation_count:,}")
    print(f"  transformation       : {transform.mode}")
    print(f"  VFM dimension        : {transformed_dim}")
    print(f"  hidden / layers      : {args.vfm_hidden_dim} / {args.vfm_num_layers}")
    print(f"  validity weight      : {args.vfm_validity_weight}")
    print(f"  predictor            : {predictor_path}")

    for epoch in range(start_epoch, args.vfm_epochs + 1):
        epoch_start = time.time()
        model.train()

        sums = {
            "total": 0.0,
            "gaussian_nll": 0.0,
            "validity_loss": 0.0,
            "endpoint_mse": 0.0,
            "predicted_std": 0.0,
            "mean_validity_probability": 0.0,
        }
        probability_count = 0
        sample_count = 0
        learning_rate = float(optimizer.param_groups[0]["lr"])

        for (raw_endpoints,) in train_loader:
            raw_endpoints = raw_endpoints.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )
            endpoints = transform(raw_endpoints)

            terms = vfm_loss_terms(
                model=model,
                endpoints=endpoints,
                predictor=predictor,
                validity_weight=args.vfm_validity_weight,
                validity_samples=args.vfm_validity_samples,
                eps_time=args.vfm_eps_time,
            )

            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                args.vfm_grad_clip,
            )
            optimizer.step()

            batch_count = len(endpoints)
            sample_count += batch_count

            for key in (
                "total",
                "gaussian_nll",
                "validity_loss",
                "endpoint_mse",
                "predicted_std",
            ):
                sums[key] += float(terms[key].detach().item()) * batch_count

            probability = terms["mean_validity_probability"].detach()

            if torch.isfinite(probability):
                sums["mean_validity_probability"] += (
                    float(probability.item()) * batch_count
                )
                probability_count += batch_count

        train_metrics = aggregate_vfm_metrics(sums, sample_count)
        train_metrics["mean_validity_probability"] = (
            float("nan")
            if probability_count == 0
            else sums["mean_validity_probability"] / probability_count
        )

        validation_metrics = evaluate_vfm_training_loss(
            model=model,
            loader=validation_loader,
            transform=transform,
            predictor=predictor,
            args=args,
            device=device,
            seed=args.seed + 50_000,
        )
        scheduler.step()

        elapsed_seconds = time.time() - epoch_start
        is_best = validation_metrics["total"] < best_validation_loss

        if is_best:
            best_validation_loss = validation_metrics["total"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_result = {
            "epoch": epoch,
            **{
                f"train_{key}": value
                for key, value in train_metrics.items()
            },
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            },
            "learning_rate": learning_rate,
            "next_learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "elapsed_seconds": elapsed_seconds,
            "is_best": is_best,
        }
        history.append(epoch_result)

        atomic_json_dump(
            history,
            output_dir / "vfm_training_history.json",
        )

        save_vfm_checkpoint(
            path=vfm_last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_validation_loss=best_validation_loss,
            epochs_without_improvement=epochs_without_improvement,
            history=history,
            transform=transform,
            args=args,
            predictor_path=predictor_path,
            train_count=train_count,
            validation_count=validation_count,
        )

        if is_best:
            save_vfm_checkpoint(
                path=vfm_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_validation_loss=best_validation_loss,
                epochs_without_improvement=epochs_without_improvement,
                history=history,
                transform=transform,
                args=args,
                predictor_path=predictor_path,
                train_count=train_count,
                validation_count=validation_count,
            )

        validity_text = (
            "N/A"
            if math.isnan(train_metrics["mean_validity_probability"])
            else f"{train_metrics['mean_validity_probability']:.4f}"
        )

        print(
            f"Epoch {epoch:03d}/{args.vfm_epochs:03d} | "
            f"train={train_metrics['total']:.5f} | "
            f"val={validation_metrics['total']:.5f} | "
            f"train_nll={train_metrics['gaussian_nll']:.5f} | "
            f"val_nll={validation_metrics['gaussian_nll']:.5f} | "
            f"valid_prob={validity_text} | "
            f"{elapsed_seconds:.1f}s",
            flush=True,
        )

        if epochs_without_improvement >= args.vfm_patience:
            print(
                "VFM early stopping: no improvement for "
                f"{args.vfm_patience} epochs."
            )
            break

    save_vfm_training_plot(
        history,
        output_dir / "vfm_training_curves.png",
    )

    metadata_path = output_dir / "vfm_training_summary.json"

    atomic_json_dump(
        {
            "best_checkpoint": str(vfm_checkpoint_path),
            "last_checkpoint": str(vfm_last_checkpoint_path),
            "best_validation_loss": best_validation_loss,
            "epochs_completed": history[-1]["epoch"] if history else 0,
            "predictor_metadata": predictor_metadata,
            "latent_transform": transform.checkpoint_payload()["config"],
        },
        metadata_path,
    )

    print("\nVFM training completed.")
    print(f"Best checkpoint: {vfm_checkpoint_path}")
    print(f"Last checkpoint: {vfm_last_checkpoint_path}")

    del model, optimizer, scheduler, train_loader, validation_loader, transform, latents

    if predictor is not None:
        del predictor

    clear_memory()
    return vfm_checkpoint_path


class VFMEndpointFlow(nn.Module):
    def __init__(
        self,
        endpoint_model: VFMEndpointDistributionMLP,
        eps_time: float,
    ) -> None:
        super().__init__()
        self.endpoint_model = endpoint_model
        self.dim = endpoint_model.dim
        self.eps_time = float(eps_time)

    @torch.no_grad()
    def endpoint_mean_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        mu, _ = self.endpoint_model(x, t)
        denominator = (1.0 - t).clamp_min(self.eps_time)
        return (mu - x) / denominator

    @torch.no_grad()
    def endpoint_reweighted_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        predictor: ValidityPredictor,
        num_endpoints: int,
        beta: float,
    ) -> torch.Tensor:
        mu, logvar = self.endpoint_model(x, t)
        standard_deviation = torch.exp(0.5 * logvar)

        if num_endpoints <= 1 or beta <= 0.0:
            selected_endpoint = mu
        else:
            noise = torch.randn(
                num_endpoints,
                *mu.shape,
                device=mu.device,
                dtype=mu.dtype,
            )
            endpoints = (
                mu.unsqueeze(0)
                + standard_deviation.unsqueeze(0) * noise
            )
            endpoints_flat = endpoints.reshape(-1, self.dim)
            endpoint_time = torch.full(
                (len(endpoints_flat), 1),
                1.0 - self.eps_time,
                device=x.device,
                dtype=x.dtype,
            )
            logits = predictor(endpoints_flat, endpoint_time).reshape(
                num_endpoints,
                len(x),
                1,
            )
            weights = torch.softmax(beta * logits, dim=0)
            selected_endpoint = (weights * endpoints).sum(dim=0)

        denominator = (1.0 - t).clamp_min(self.eps_time)
        return (selected_endpoint - x) / denominator

    @torch.no_grad()
    def velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        predictor: Optional[ValidityPredictor],
        num_endpoints: int,
        beta: float,
    ) -> torch.Tensor:
        if predictor is None or beta <= 0.0 or num_endpoints <= 1:
            return self.endpoint_mean_velocity(x, t)

        return self.endpoint_reweighted_velocity(
            x=x,
            t=t,
            predictor=predictor,
            num_endpoints=num_endpoints,
            beta=beta,
        )

   
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        num_steps: int,
        device: torch.device,
        temperature: float,
        predictor: Optional[ValidityPredictor],
        num_endpoints: int,
        beta: float,
        solver: str,
    ) -> torch.Tensor:
        if solver == "euler":
            # Keep this branch identical to the original VFM evaluation code.
            # In particular, dt is exactly 1 / num_steps and the velocity is
            # never evaluated at the final time 1 - eps_time.
            x = (
                torch.randn(
                    num_samples,
                    self.dim,
                    device=device,
                )
                * temperature
            )
            dt = 1.0 / num_steps

            for step in range(num_steps):
                time_value = (
                    self.eps_time
                    + (1.0 - 2.0 * self.eps_time) * step / num_steps
                )
                t = torch.full(
                    (num_samples, 1),
                    time_value,
                    device=device,
                    dtype=x.dtype,
                )
                x = x + dt * self.velocity(
                    x=x,
                    t=t,
                    predictor=predictor,
                    num_endpoints=num_endpoints,
                    beta=beta,
                )

            return x

        return sample_ode(
            vector_field=lambda x, t: self.velocity(
                x=x,
                t=t,
                predictor=predictor,
                num_endpoints=num_endpoints,
                beta=beta,
            ),
            num_samples=num_samples,
            latent_dim=self.dim,
            num_steps=num_steps,
            device=device,
            temperature=temperature,
            time_start=self.eps_time,
            time_end=1.0 - self.eps_time,
            solver=solver,
        )

def load_vfm_endpoint_flow(
    path: Path,
    eps_time: float,
    predictor_path: Path,
    device: torch.device,
) -> tuple[
    VFMEndpointFlow,
    LatentSpaceTransform,
    dict[str, Any],
]:
    checkpoint = load_torch_file(path)

    state_dict = strip_module_prefix(
        checkpoint["model_state_dict"]
    )
    config = dict(checkpoint["vfm_config"])
    transform = load_latent_transform_from_checkpoint(
        predictor_path,
        device,
    )

    endpoint_model = build_vfm_model(config)
    endpoint_model.load_state_dict(
        state_dict,
        strict=True,
    )

    flow = VFMEndpointFlow(
        endpoint_model=endpoint_model,
        eps_time=eps_time,
    ).to(device)

    freeze_module(flow)

    metadata = {
        "vfm_config": config,
        "epoch": int(checkpoint.get("epoch", -1)),
        "best_validation_loss": float(
            checkpoint.get(
                "best_validation_loss",
                float("nan"),
            )
        ),
        "dataset": checkpoint.get("dataset"),
        "predictor_path_at_training": checkpoint.get(
            "predictor_path"
        ),
        "latent_transform": (
            transform.checkpoint_payload()["config"]
        ),
    }

    del checkpoint, state_dict
    clear_memory()

    print(f"VFM loaded: {path}")
    print(
        "Latent transform loaded from predictor: "
        f"{predictor_path}"
    )
    print(
        f"  dim={config['dim']}, "
        f"hidden={config['hidden_dim']}, "
        f"layers={config['num_layers']}, "
        f"transform={transform.mode}"
    )

    return flow, transform, metadata


def vfm_evaluation_config_key(
    use_predictor: bool,
    num_endpoints: int,
    beta: float,
    solver: str,
) -> str:
    solver_prefix = "" if solver == "heun" else f"{solver}_"

    if not use_predictor:
        return f"{solver_prefix}integrated_mean_no_sampling_predictor"

    return (
        f"{solver_prefix}endpoint_reweight_k{num_endpoints}_"
        f"beta{float_tag(beta)}"
    )



def evaluate_one_vfm_config(
    args: argparse.Namespace,
    flow: VFMEndpointFlow,
    transform: LatentSpaceTransform,
    vae: nn.Module,
    predictor: Optional[ValidityPredictor],
    training_smiles: Optional[set[str]],
    device: torch.device,
    use_predictor: bool,
    num_endpoints: int,
    beta: float,
    fcd_reference: Optional[FCDReference],
) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    all_generated_valid_smiles: list[str] = []

    for trial in range(args.vfm_num_trials):
        set_seed(args.seed + trial)

        raw_latent_chunks: list[torch.Tensor] = []
        remaining = args.vfm_num_molecules

        while remaining > 0:
            batch_count = min(
                args.vfm_eval_batch_size,
                remaining,
            )

            model_space_latents = flow.sample(
                num_samples=batch_count,
                num_steps=args.vfm_eval_steps,
                device=device,
                temperature=args.vfm_temperature,
                predictor=(
                    predictor
                    if use_predictor
                    else None
                ),
                num_endpoints=num_endpoints,
                beta=beta,
                solver=args.vfm_eval_solver,
            )

            raw_latents = transform.inverse(
                model_space_latents
            )

            raw_latent_chunks.append(
                raw_latents
                .detach()
                .float()
                .cpu()
            )

            remaining -= batch_count

            del model_space_latents
            del raw_latents
            clear_memory()

        trial_raw_latents = torch.cat(
            raw_latent_chunks,
            dim=0,
        )

        metrics = compute_generation_metrics_from_latents(
            vae=vae,
            latents=trial_raw_latents,
            dataset=args.dataset,
            batch_size=args.decode_batch_size,
            device=device,
            training_smiles=training_smiles,
            return_valid_smiles=fcd_reference is not None,
        )
        generated_valid_smiles = metrics.pop("valid_smiles",[],)
        if fcd_reference is not None:
            all_generated_valid_smiles.extend(generated_valid_smiles)

        trial_result = {
            "trial": trial + 1,
            "seed": args.seed + trial,
            **metrics,
        }
        trials.append(trial_result)

        novelty_text = (
            "N/A"
            if metrics["novelty"] is None
            else f"{metrics['novelty']:.4f}"
        )

        print(
            f"  Trial {trial + 1:02d}/"
            f"{args.vfm_num_trials:02d} | "
            f"validity={metrics['validity']:.4f} | "
            f"uniqueness={metrics['uniqueness']:.4f} | "
            f"novelty={novelty_text} | "
            f"valid={metrics['valid_count']:,}/"
            f"{metrics['total_count']:,}",
            flush=True,
        )

        del raw_latent_chunks
        del trial_raw_latents
        clear_memory()

    fcd_result = (
        compute_fcd_result(
            args=args,
            device=device,
            reference=fcd_reference,
            generated_valid_smiles=all_generated_valid_smiles,
        )
        if fcd_reference is not None
        else empty_fcd_result()
    )

    return {
        "use_predictor_at_sampling": use_predictor,
        "num_endpoints": num_endpoints,
        "beta": beta,
        "num_trials": args.vfm_num_trials,
        "num_molecules": args.vfm_num_molecules,
        "eval_steps": args.vfm_eval_steps,
        "temperature": args.vfm_temperature,
        "solver": args.vfm_eval_solver,
        **fcd_result,
        "latent_transform": (
            transform
            .checkpoint_payload()["config"]
        ),
        "trials": trials,
        "summary": summarize_trials(trials),
    }

def write_vfm_report(
    results: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "Endpoint-based VFM evaluation with integrated validity",
        "=" * 78,
        "",
    ]

    for key, result in results.items():
        lines.extend(
            [
                key,
                "-" * 50,
                f"predictor at sampling  : "
                f"{result['use_predictor_at_sampling']}",
                f"solver                 : {result['solver']}",
                f"K endpoints            : {result['num_endpoints']}",
                f"beta                   : {result['beta']}",
            ]
        )

        for metric_name in (
            "validity",
            "uniqueness",
            "novelty",
        ):
            values = result["summary"].get(metric_name)

            if values is None:
                lines.append(f"{metric_name:<11}: N/A")
            else:
                lines.append(
                    f"{metric_name:<11}: "
                    f"{values['mean']:.4f} ± {values['std']:.4f} "
                    f"(min={values['min']:.4f}, "
                    f"max={values['max']:.4f})"
                )

        fcd_value = result.get("fcd")

        if fcd_value is None:
            lines.append(f"{'fcd':<11}: N/A")
        else:
            lines.append(
                f"{'fcd':<11}: {fcd_value:.6f} "
                f"(generated="
                f"{result['fcd_generated_count']:,}, "
                f"reference="
                f"{result['fcd_reference_count']:,})"
            )

        lines.append("")

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def evaluate_vfm(
    args: argparse.Namespace,
    predictor_path: Path,
    vfm_checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
) -> None:
    evaluation_dir = output_dir / args.vfm_evaluation_subdir
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    flow, transform, vfm_metadata = load_vfm_endpoint_flow(
    path=vfm_checkpoint_path,
    eps_time=args.vfm_eps_time,
    predictor_path=predictor_path,
    device=device,
)
    vae = load_vae(Path(args.vae_path), device)
    fcd_reference = (
        None
        if args.skip_fcd
        else prepare_fcd_reference(args, device)
    )

    training_smiles = (
        None
        if args.skip_novelty
        else load_training_smiles(Path(args.training_smiles_path))
    )

    predictor: Optional[ValidityPredictor] = None
    predictor_metadata: Optional[dict[str, Any]] = None

    if args.vfm_eval_reweight:
        predictor, predictor_metadata = load_validity_predictor(
            predictor_path,
            device,
        )
    configurations: list[tuple[bool, int, float]] = []

    if args.vfm_eval_include_mean:
        configurations.append((False, 1, 0.0))

    if args.vfm_eval_reweight:
        configurations.extend(
            (True, int(endpoint_count), float(beta))
            for endpoint_count in args.vfm_endpoint_counts
            for beta in args.vfm_betas
            if endpoint_count > 1 and beta > 0.0
        )

    all_results: dict[str, Any] = {}

    for use_predictor, endpoint_count, beta in configurations:
        key = vfm_evaluation_config_key(
            use_predictor=use_predictor,
            num_endpoints=endpoint_count,
            beta=beta,
            solver=args.vfm_eval_solver,
        )
        result_path = evaluation_dir / f"{key}.json"

        if result_path.is_file() and not args.overwrite_vfm_evaluation:
            print(f"VFM result already exists; skipping configuration: {key}")
            all_results[key] = json.loads(
                result_path.read_text(encoding="utf-8")
            )
            continue

        print("\n" + "=" * 80)
        print(f"VFM evaluation : {key}")
        print("=" * 80)
        result = evaluate_one_vfm_config(
            args=args,
            flow=flow,
            transform=transform,
            vae=vae,
            predictor=predictor,
            training_smiles=training_smiles,
            device=device,
            use_predictor=use_predictor,
            num_endpoints=endpoint_count,
            beta=beta,
            fcd_reference=fcd_reference,
        )
        atomic_json_dump(result, result_path)
        all_results[key] = result

        atomic_json_dump(
            {
                "vfm_checkpoint": str(vfm_checkpoint_path),
                "vfm_metadata": vfm_metadata,
                "predictor_path": (
                    None if predictor_path is None else str(predictor_path)
                ),
                "predictor_metadata": predictor_metadata,
                "results": all_results,
            },
            evaluation_dir / "vfm_evaluation_summary.json",
        )
        write_vfm_report(
            all_results,
            evaluation_dir / "vfm_evaluation_summary.txt",
        )

    print("\nVFM evaluation finished.")
    print(f"Résults : {evaluation_dir}")

    del flow, transform, vae
    if fcd_reference is not None:
        del fcd_reference
    if predictor is not None:
        del predictor
    clear_memory()
