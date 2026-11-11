from __future__ import annotations


import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import numpy as np
import os
import warnings
from fm_models import read_fm_checkpoint_components,sample_ode
from utils import freeze_module, clear_memory, atomic_json_dump, set_seed, load_any_latent_tensor, load_vae
from torch.utils.data import DataLoader, Dataset
from evaluations import compute_generation_metrics_from_latents,summarize_trials, load_training_smiles
from fcd_evaluation import (
    FCDReference,
    compute_fcd_result,
    prepare_fcd_reference,
)

import argparse
import contextlib
import json
import time




# ============================================================
# Flow Matching Architecture
# ============================================================


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]

        if self.embedding_dim == 1:
            return t[:, None]

        half_dim = self.embedding_dim // 2

        frequencies = torch.exp(-math.log(10_000.0) * torch.arange(half_dim,device=t.device,dtype=t.dtype,) / max(half_dim - 1, 1))
        angles = (2.0 * math.pi * t[:, None] * frequencies[None, :])
        return torch.cat(
            (
                torch.sin(angles),
                torch.cos(angles),
            ),
            dim=-1,
        )


class PlainResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        residual = h
        h = self.norm(h)
        h = self.fc1(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return residual + h


class FiLMResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        time_embedding_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.time_projection = nn.Linear(time_embedding_size, 2 * hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        residual = h
        scale, shift = self.time_projection(time_embedding).chunk(2, dim=-1)
        h = self.norm(h)
        h = h * (1.0 + scale) + shift
        h = self.fc1(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return residual + h


class ResidualVectorField(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        time_embedding_size: int,
        dropout: float,
        use_film: bool,
    ) -> None:
        super().__init__()
        self.use_film = use_film
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_size, 4 * time_embedding_size),
            nn.SiLU(),
            nn.Linear(4 * time_embedding_size, time_embedding_size),
        )

        if use_film:
            self.input_projection = nn.Linear(latent_dim, hidden_dim)
            self.layers = nn.ModuleList(
                [
                    FiLMResidualBlock(
                        hidden_dim=hidden_dim,
                        time_embedding_size=time_embedding_size,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.input_projection = nn.Linear(
                latent_dim + time_embedding_size,
                hidden_dim,
            )
            self.layers = nn.ModuleList(
                [
                    PlainResidualBlock(
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_mlp(self.time_embedding(t))

        if self.use_film:
            h = self.input_projection(x)

            for layer in self.layers:
                h = layer(h, time_embedding)
        else:
            h = self.input_projection(torch.cat((x, time_embedding), dim=-1))

            for layer in self.layers:
                h = layer(h)

        return self.output_projection(self.output_norm(h))




class FlowMatchingInference(nn.Module):

    def __init__(
        self,
        vector_field: nn.Module,
    ) -> None:
        super().__init__()
        self.vector_field = vector_field


def build_fm_model(
    model_config: dict[str, Any],
) -> FlowMatchingInference:
    latent_dim = int(
        model_config["latent_dim"]
    )

    vector_field = ResidualVectorField(
        latent_dim=latent_dim,
        hidden_dim=int(
            model_config["hidden_dim"]
        ),
        num_layers=int(
            model_config["num_layers"]
        ),
        time_embedding_size=int(
            model_config["time_embedding_size"]
        ),
        dropout=float(
            model_config.get(
                "dropout",
                0.0,
            )
        ),
        use_film=bool(
            model_config.get(
                "use_film",
                False,
            )
        ),
    )

    return FlowMatchingInference(
        vector_field
    )



def load_fm_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    FlowMatchingInference,
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
]:
    print(
        f"FM loading from : {checkpoint_path}"
    )

    (
        model_config,
        state_dict,
        latent_mean_cpu,
        latent_std_cpu,
        metadata,
    ) = read_fm_checkpoint_components(
        checkpoint_path
    )

    model = build_fm_model(
        model_config
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(device)
    freeze_module(model)

    latent_mean = latent_mean_cpu.to(
        device
    )

    latent_std = latent_std_cpu.to(
        device
    )

    metadata = {
        **metadata,
        "model_config": model_config,
    }

    print("Modern FM loaded successfully:")
    print(f"  layers : "
        f"{model_config['num_layers']}")
    print(
        f"  hidden : "
        f"{model_config['hidden_dim']}")
    print(
        f"  FiLM   : "
        f"{model_config.get('use_film', False)}")

    clear_memory()

    return (
        model,
        latent_mean,
        latent_std,
        metadata,
    )

#================================================================
#   Batch OT
#================================================================

class SinkhornBatchOTPlanSampler:
    """
    Échantillonne des couples selon un plan OT régularisé par Sinkhorn.

    L'implémentation reprend la structure de OTPlanSampler de
    torchcfm :

      1. calcul de la matrice de coût euclidienne au carré ;
      2. calcul d'un plan OT diffus avec ot.sinkhorn ;
      3. échantillonnage de couples depuis ce plan.

    Le batch du DataLoader peut être découpé en sous-batches OT afin de
    limiter la taille de la matrice de coût.
    """

    def __init__(
        self,
        ot_batch_size: int,
        reg: float = 10.0,
        max_iterations: int = 1000,
        stop_threshold: float = 1e-6,
        warn: bool = True,
        replace: bool = True,
    ) -> None:

        try:
            import ot as pot
        except ImportError as error:
            raise ImportError(
                "Le Batch OT Sinkhorn nécessite Python Optimal Transport. "
                "Installe-le avec `pip install POT`."
            ) from error

        self.pot = pot
        self.ot_batch_size = ot_batch_size
        self.reg = reg
        self.max_iterations = max_iterations
        # Difficult batches may need more scaling iterations even after
        # switching to log space. Keep the fast first attempt unchanged.
        self.log_max_iterations = 10 * max_iterations
        self.stop_threshold = stop_threshold
        self.warn = warn
        self.replace = replace
        if ot_batch_size < 1 or max_iterations < 1:
            raise ValueError("OT batch size and maximum iterations must be positive.")
        if not np.isfinite(reg) or reg <= 0 or not np.isfinite(stop_threshold) or stop_threshold <= 0:
            raise ValueError("OT regularization and stopping threshold must be finite and positive.")
        self.plan_tolerance = max(float(stop_threshold), 1e-10)
        self.reset_statistics()

    def reset_statistics(self) -> None:
        self._statistics = {
            "requested_plans": 0,
            "accepted_plans": 0,
            "log_retries": 0,
            "failed_plans": 0,
            "max_mass_error": 0.0,
            "max_row_marginal_error": 0.0,
            "max_column_marginal_error": 0.0,
        }

    def statistics(self) -> dict[str, Any]:
        """Errors refer to accepted plans; counters cover all requested plans."""
        return {**self._statistics, "plan_tolerance": self.plan_tolerance,
                "log_max_iterations": self.log_max_iterations}

    def validate_plan(
        self, plan: np.ndarray, a: np.ndarray, b: np.ndarray,
    ) -> dict[str, float]:
        if plan.shape != (len(a), len(b)):
            raise ValueError(f"Unexpected OT plan shape: {plan.shape}")
        if not np.isfinite(plan).all() or (plan < 0).any():
            raise ValueError("OT plan contains nonfinite or negative values.")
        errors = {
            "mass_error": abs(float(plan.sum()) - 1.0),
            "row_marginal_error": float(np.max(np.abs(plan.sum(axis=1) - a))),
            "column_marginal_error": float(np.max(np.abs(plan.sum(axis=0) - b))),
        }
        if max(errors.values()) > self.plan_tolerance:
            raise ValueError(f"OT plan violates marginal constraints: {errors}; tolerance={self.plan_tolerance:g}")
        return errors

    @torch.no_grad()
    def get_map(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> np.ndarray:
        """
        Calcule le plan OT Sinkhorn entre deux minibatches.

        Le plan retourné a la forme ``(len(x0), len(x1))`` et sa somme
        vaut approximativement 1.
        """
        if x0.shape[0] < 1 or x1.shape[0] < 1:
            raise ValueError("Les minibatches OT ne doivent pas être vides.")

        a = self.pot.unif(x0.shape[0])
        b = self.pot.unif(x1.shape[0])

        if x0.ndim > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.ndim > 2:
            x1 = x1.reshape(x1.shape[0], -1)

        self._statistics["requested_plans"] += 1
        # Keep distance computation in float32 on the original device, then
        # promote costs BEFORE exponentiation inside the CPU Sinkhorn solver.
        with torch.autocast(
            device_type=x0.device.type,
            enabled=False,
        ):
            cost = torch.cdist(
                x0.detach().float(),
                x1.detach().float(),
                p=2,
            ).square()

        cost_numpy = cost.cpu().numpy().astype(np.float64)
        if not np.isfinite(cost_numpy).all():
            self._statistics["failed_plans"] += 1
            raise RuntimeError("Nonfinite OT costs; refusing to replace the coupling by a uniform plan.")

        failures = []
        for method in ("sinkhorn", "sinkhorn_log"):
            if method == "sinkhorn_log":
                self._statistics["log_retries"] += 1
                if self.warn:
                    warnings.warn(
                        f"Invalid float64 Sinkhorn plan; retrying in log space with up to "
                        f"{self.log_max_iterations} iterations: {failures[-1]}"
                    )
            try:
                with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
                    plan = np.asarray(self.pot.sinkhorn(
                        a, b, cost_numpy, reg=self.reg, method=method,
                        numItermax=(self.log_max_iterations if method == "sinkhorn_log" else self.max_iterations),
                        stopThr=self.stop_threshold,
                        warn=self.warn,
                    ), dtype=np.float64)
                errors = self.validate_plan(plan, a, b)
            except (ValueError, FloatingPointError, RuntimeError) as error:
                failures.append(f"{method}: {error}")
                continue
            self._statistics["accepted_plans"] += 1
            for name, value in errors.items():
                self._statistics[f"max_{name}"] = max(self._statistics[f"max_{name}"], value)
            return plan

        self._statistics["failed_plans"] += 1
        raise RuntimeError(
            "Both float64 Sinkhorn solvers failed; no uniform fallback is allowed. "
            f"cost range=[{cost_numpy.min():.6g}, {cost_numpy.max():.6g}], reg={self.reg}; "
            f"statistics={self.statistics()}; failures={failures}"
        )

    @staticmethod
    def sample_map(
        plan: np.ndarray,
        batch_size: int,
        replace: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Échantillonne ``batch_size`` couples selon le plan OT."""
        probabilities = plan.reshape(-1).astype(np.float64, copy=False)
        probabilities = probabilities / probabilities.sum()

        choices = np.random.choice(
            plan.shape[0] * plan.shape[1],
            p=probabilities,
            size=batch_size,
            replace=replace,
        )

        return np.divmod(choices, plan.shape[1])

    @torch.no_grad()
    def sample_plan(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calcule le plan Sinkhorn puis échantillonne des couples ``(x0, x1)``.

        Comme dans TorchCFM, l'échantillonnage se fait par défaut avec
        remplacement. Certains éléments peuvent donc apparaître plusieurs
        fois, tandis que d'autres peuvent ne pas apparaître dans une étape.
        """
        if x0.shape != x1.shape:
            raise ValueError(
                "source et target doivent avoir exactement la même forme "
                "pour le couplage Batch OT."
            )

        plan = self.get_map(x0, x1)
        i_np, j_np = self.sample_map(
            plan,
            batch_size=x0.shape[0],
            replace=self.replace,
        )

        i = torch.as_tensor(i_np, device=x0.device, dtype=torch.long)
        j = torch.as_tensor(j_np, device=x1.device, dtype=torch.long)

        return x0.index_select(0, i), x1.index_select(0, j)

    @torch.no_grad()
    def sample_plan_batched(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applique ``sample_plan`` séparément à chaque sous-batch OT."""
        if source.shape != target.shape:
            raise ValueError(
                "source et target doivent avoir exactement la même forme "
                "pour le couplage Batch OT."
            )

        paired_sources = []
        paired_targets = []

        for start in range(0, source.shape[0], self.ot_batch_size):
            end = min(start + self.ot_batch_size, source.shape[0])

            source_chunk, target_chunk = self.sample_plan(
                source[start:end],
                target[start:end],
            )

            paired_sources.append(source_chunk)
            paired_targets.append(target_chunk)

        return (
            torch.cat(paired_sources, dim=0),
            torch.cat(paired_targets, dim=0),
        )


class FlowMatching(nn.Module):
    def __init__(
        self,
        vector_field: ResidualVectorField,
        use_batch_ot: bool,
        batch_ot_size: int,
        batch_ot_reg: float,
        batch_ot_max_iterations: int,
        batch_ot_stop_threshold: float,
    ) -> None:
        super().__init__()
        self.vector_field = vector_field
        self.use_batch_ot = use_batch_ot
        self.batch_ot_sampler = (
            SinkhornBatchOTPlanSampler(
                ot_batch_size=batch_ot_size,
                reg=batch_ot_reg,
                max_iterations=batch_ot_max_iterations,
                stop_threshold=batch_ot_stop_threshold,
                warn=True,
                replace=True,
            )
            if use_batch_ot
            else None
        )

    def compute_loss(
        self,
        target: torch.Tensor,
    ) -> torch.Tensor:
        source = torch.randn_like(target)

        if self.batch_ot_sampler is not None:
            source, target = self.batch_ot_sampler.sample_plan_batched(
                source,
                target,
            )

        t = torch.rand(
            target.shape[0],
            1,
            device=target.device,
            dtype=target.dtype,
        )

        interpolated = (
            (1.0 - t) * source
            + t * target
        )

        target_velocity = target - source

        predicted_velocity = self.vector_field(
            interpolated,
            t,
        )

        return (
            predicted_velocity
            - target_velocity
        ).square().mean()


# ============================================================
# Sampling ODE
# ============================================================

@torch.no_grad()
def sample_fm_ode(
    model: nn.Module,
    num_samples: int,
    latent_dim: int,
    num_steps: int,
    device: torch.device,
    temperature: float,
    solver: str = "heun",
) -> torch.Tensor:
    return sample_ode(
        vector_field=model.vector_field,
        num_samples=num_samples,
        latent_dim=latent_dim,
        num_steps=num_steps,
        device=device,
        temperature=temperature,
        time_start=0.0,
        time_end=1.0,
        solver=solver,
    )


# ============================================================
# Entraînement
# ============================================================

def choose_amp_dtype(
    device: torch.device,
    requested: str,
) -> str:
    if device.type != "cuda":
        return "none"

    if requested != "auto":
        return requested

    return (
        "bf16"
        if torch.cuda.is_bf16_supported()
        else "fp16"
    )


def autocast_context(
    device: torch.device,
    amp_dtype: str,
):
    if device.type != "cuda" or amp_dtype == "none":
        return contextlib.nullcontext()

    dtype = (
        torch.bfloat16
        if amp_dtype == "bf16"
        else torch.float16
    )

    return torch.autocast(
        device_type="cuda",
        dtype=dtype,
    )


def make_grad_scaler(
    device: torch.device,
    amp_dtype: str,
):
    enabled = (
        device.type == "cuda"
        and amp_dtype == "fp16"
    )

    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(
            enabled=enabled
        )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
) -> DataLoader:
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


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
):
    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return (
                step + 1
            ) / max(1, warmup_steps)

        progress = (
            step - warmup_steps
        ) / max(
            1,
            total_steps - warmup_steps,
        )

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        return 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_multiplier,
    )


def create_continuation_scheduler(
    optimizer, learning_rate: float, minimum_rate: float,
    steps_per_epoch: int, restart_epoch: int, end_epoch: int,
    completed_epoch: int,
):
    """Resume a cosine schedule by absolute epoch while retaining Adam moments."""
    if not (0 <= restart_epoch <= completed_epoch < end_epoch):
        raise ValueError("Continuation requires 0 <= restart <= completed < end epoch.")
    if not (math.isfinite(learning_rate) and math.isfinite(minimum_rate)
            and 0 <= minimum_rate <= learning_rate and learning_rate > 0):
        raise ValueError("Continuation LR must satisfy 0 <= min LR <= LR, with finite positive LR.")
    if steps_per_epoch < 1:
        raise ValueError("Continuation requires at least one training step per epoch.")
    offset = (completed_epoch - restart_epoch) * steps_per_epoch
    duration = (end_epoch - restart_epoch) * steps_per_epoch
    floor = minimum_rate / learning_rate
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
        group["initial_lr"] = learning_rate

    def multiplier(step):
        progress = min(max((offset + step) / duration, 0.0), 1.0)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def save_checkpoint_atomic(
    path: Path,
    model: FlowMatching,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_validation_loss: float,
    train_loss: float,
    validation_loss: float,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    parameter_count: int,
) -> None:
    temporary_path = Path(f"{path}.tmp")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "best_validation_loss": best_validation_loss,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "model_config": model_config,
            "training_config": training_config,
            "latent_mean": latent_mean.cpu(),
            "latent_std": latent_std.cpu(),
            "latent_dim": int(latent_mean.numel()),
            "parameter_count": parameter_count,
        },
        temporary_path,
    )

    os.replace(temporary_path, path)


@torch.no_grad()
def evaluate_flow_loss(
    model: FlowMatching,
    loader: DataLoader,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    device: torch.device,
    amp_dtype: str,
) -> float:
    model.eval()

    loss_sum = 0.0
    sample_count = 0

    for raw_batch in loader:
        raw_batch = raw_batch.to(
            device,
            non_blocking=True,
        )

        batch = (
            raw_batch - latent_mean
        ) / latent_std

        with autocast_context(device, amp_dtype):
            loss = model.compute_loss(batch)

        loss_sum += loss.item() * len(batch)
        sample_count += len(batch)

    return loss_sum / sample_count



def prepare_flow_matching_data(
    args: argparse.Namespace,
    output_dir: Path,
    best_path: Path,
    last_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    normalization_path = (
        output_dir / "normalization.pt"
    )

    training_history_path = (
        output_dir / "training_history.json"
    )

    if args.overwrite:
        for path in (
            best_path,
            last_path,
            normalization_path,
            training_history_path,
        ):
            if path.is_file():
                path.unlink()

    latents = load_any_latent_tensor(
        Path(args.latents_path),
        mmap=True,
    )

    available_count = len(latents)
    used_count = (available_count if args.max_samples <= 0 else min(available_count, args.max_samples,))
    latent_dim = int(latents.shape[1])
    validation_count = max(1,min(args.val_size,used_count - 1,),)
    train_count = (used_count - validation_count)

    print("\nFlow Matching dataset:")
    print(f"  available latents : {available_count:,}")
    print(f"  used latents      : {used_count:,}")
    print(f"  train             : {train_count:,}")
    print(f"  validation        : {validation_count:,}")
    print(f"  latent dimension  : {latent_dim}")


    if normalization_path.is_file():
        normalization = torch.load(
            normalization_path,
            map_location="cpu",
            weights_only=False,
        )
        latent_mean_cpu = (normalization["mean"].detach().float().reshape(-1))
        latent_std_cpu = (normalization["std"].detach().float().reshape(-1).clamp_min(1e-8))
        print("Normalisation chargée depuis : "f"{normalization_path}")

    else:
        latent_std_cpu, latent_mean_cpu = torch.std_mean(
        latents[:train_count].float(),
        dim=0,
        correction=0,
    )

        latent_mean_cpu = latent_mean_cpu.cpu()
        latent_std_cpu = latent_std_cpu.clamp_min(1e-8).cpu()

        torch.save(
            {
                "mean": latent_mean_cpu,
                "std": latent_std_cpu,
                "train_count": train_count,
                "validation_count": validation_count,
                "latent_dim": latent_dim,
            },
            normalization_path,
        )


    latent_mean = latent_mean_cpu.to(device)
    latent_std = latent_std_cpu.to(device)

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_dataset = (latents[:train_count])

    validation_dataset = (latents[train_count:used_count])

    train_loader = make_loader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=(
            train_count
            >= args.batch_size
        ),
    )

    validation_loader = make_loader(
        dataset=validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    return {
        "latents": latents,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "latent_mean_cpu": latent_mean_cpu,
        "latent_std_cpu": latent_std_cpu,
        "available_count": available_count,
        "used_count": used_count,
        "train_count": train_count,
        "validation_count": validation_count,
        "latent_dim": latent_dim,
        "normalization_path": normalization_path,
        "training_history_path": training_history_path,
    }

def train_flow_matching(
    args: argparse.Namespace,
    output_dir: Path,
    best_path: Path,
    last_path: Path,
    device: torch.device,
) -> Path:
    restart_epoch = getattr(args, "fm_lr_restart_epoch", None)
    if restart_epoch is not None:
        end_epoch = getattr(args, "fm_lr_end_epoch", None)
        if not args.resume or args.overwrite or end_epoch is None or args.epochs > end_epoch:
            raise ValueError("LR continuation requires --resume, no --overwrite, and --epochs <= --fm_lr_end_epoch.")
    set_seed(args.seed)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_setup = prepare_flow_matching_data(
        args=args,
        output_dir=output_dir,
        best_path=best_path,
        last_path=last_path,
        device=device,
    )
    latents = data_setup["latents"]
    train_loader = data_setup["train_loader"]
    validation_loader = data_setup["validation_loader"]
    latent_mean = data_setup["latent_mean"]
    latent_std = data_setup["latent_std"]
    latent_mean_cpu = data_setup["latent_mean_cpu"]
    latent_std_cpu = data_setup["latent_std_cpu"]
    used_count = data_setup["used_count"]
    train_count = data_setup["train_count"]
    validation_count = data_setup["validation_count"]
    latent_dim = data_setup["latent_dim"]
    training_history_path = data_setup["training_history_path"]

    model_config = {
        "latent_dim": latent_dim,
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "time_embedding_size": (
            args.time_embedding_size
        ),
        "dropout": args.dropout,
        "use_film": args.use_film,
        "use_batch_ot": args.use_batch_ot,
        "batch_ot_size": args.batch_ot_size,
        "batch_ot_reg": args.batch_ot_reg,
        "batch_ot_max_iterations": (
            args.batch_ot_max_iterations
        ),
        "batch_ot_stop_threshold": (
            args.batch_ot_stop_threshold
        ),
    }

    vector_field = ResidualVectorField(
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        time_embedding_size=(
            args.time_embedding_size
        ),
        dropout=args.dropout,
        use_film=args.use_film,
    )

    model = FlowMatching(
        vector_field=vector_field,
        use_batch_ot=args.use_batch_ot,
        batch_ot_size=args.batch_ot_size,
        batch_ot_reg=args.batch_ot_reg,
        batch_ot_max_iterations=(
            args.batch_ot_max_iterations
        ),
        batch_ot_stop_threshold=(
            args.batch_ot_stop_threshold
        ),
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nFlow Matching configuration:")
    print(f"  hidden dimension : {args.hidden_dim}")
    print(f"  layers           : {args.num_layers}")
    print(f"  FiLM             : {args.use_film}")
    print(f"  Batch OT         : {args.use_batch_ot}")
    print(f"  parameters       : {parameter_count:,}")


    serialized_arguments = {
        key: (
            str(value)
            if isinstance(value, Path)
            else value
        )
        for key, value in vars(args).items()
    }

    training_config = {
        **serialized_arguments,
        "train_count": train_count,
        "validation_count": validation_count,
        "used_count": used_count,
        "latent_dim": latent_dim,
        "output_dir": str(output_dir),
    }
    if model.batch_ot_sampler is not None:
        training_config["batch_ot_numerics"] = {
            "cost_dtype": "float32",
            "solver_dtype": "float64",
            "solver": "sinkhorn",
            "retry_solver": "sinkhorn_log",
            "retry_max_iterations": model.batch_ot_sampler.log_max_iterations,
            "uniform_fallback": False,
            "plan_tolerance": model.batch_ot_sampler.plan_tolerance,
        }

    atomic_json_dump(
        {
            "model_config": model_config,
            "training_config": training_config,
        },
        output_dir / "fm_training_config.json",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )

    total_optimization_steps = (
        args.epochs
        * len(train_loader)
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_optimization_steps,
    )

    amp_dtype = choose_amp_dtype(
        device=device,
        requested=args.amp_dtype,
    )

    scaler = make_grad_scaler(
        device=device,
        amp_dtype=amp_dtype,
    )

    best_validation_loss = float(
        "inf"
    )

    start_epoch = 1

    training_history: list[
        dict[str, Any]
    ] = []

    if args.resume:
        checkpoint = torch.load(
            last_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        if restart_epoch is None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            scheduler = create_continuation_scheduler(
                optimizer=optimizer, learning_rate=args.learning_rate,
                minimum_rate=args.fm_min_learning_rate,
                steps_per_epoch=len(train_loader), restart_epoch=restart_epoch,
                end_epoch=args.fm_lr_end_epoch, completed_epoch=int(checkpoint["epoch"]),
            )
            print(f"Continuation LR: {optimizer.param_groups[0]['lr']:.6g} "
                  f"-> {args.fm_min_learning_rate:g} at epoch {args.fm_lr_end_epoch}; "
                  "optimizer state preserved.", flush=True)

        scaler_state = checkpoint.get(
            "scaler_state_dict"
        )

        if scaler_state:
            scaler.load_state_dict(
                scaler_state
            )

        start_epoch = (
            int(checkpoint["epoch"])
            + 1
        )

        best_validation_loss = float(
            checkpoint.get(
                "best_validation_loss",
                float("inf"),
            )
        )

        if training_history_path.is_file():
            with open(
                training_history_path,
                "r",
                encoding="utf-8",
            ) as file:
                training_history = (
                    json.load(file)
                )

        print(
            "\nReprise de l'entraînement depuis : "
            f"{last_path}"
        )

        print(
            f"Prochaine epoch : {start_epoch}"
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nFlow Matching training starts")

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        model.train()
        if model.batch_ot_sampler is not None:
            model.batch_ot_sampler.reset_statistics()

        train_loss_sum = 0.0
        train_sample_count = 0

        epoch_start = time.time()

        for step, raw_batch in enumerate(train_loader,start=1,):
            raw_batch = raw_batch.to(device=device,dtype=torch.float32,non_blocking=True,)

            batch = (raw_batch - latent_mean) / latent_std

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(device=device,amp_dtype=amp_dtype,):
                loss = model.compute_loss(batch)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )

            scaler.step(optimizer)

            scaler.update()

            scheduler.step()

            batch_count = len(batch)

            train_loss_sum += (
                float(loss.detach().item())
                * batch_count
            )

            train_sample_count += (
                batch_count
            )

            if (args.log_every > 0 and step % args.log_every == 0):
                current_loss = (train_loss_sum/ max(train_sample_count,1,))

                print(
                    f"Epoch {epoch:03d} | "
                    f"step {step:,}/"
                    f"{len(train_loader):,} | "
                    f"loss={current_loss:.6f}",
                    flush=True,
                )

        train_loss = (train_loss_sum/ max(train_sample_count,1,))

        train_ot_statistics = None
        if model.batch_ot_sampler is not None:
            train_ot_statistics = model.batch_ot_sampler.statistics()
            model.batch_ot_sampler.reset_statistics()
        validation_loss = evaluate_flow_loss(
            model=model,
            loader=validation_loader,
            latent_mean=latent_mean,
            latent_std=latent_std,
            device=device,
            amp_dtype=amp_dtype,
        )

        is_best = (
            validation_loss
            < best_validation_loss
        )

        if is_best:
            best_validation_loss = (
                validation_loss
            )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": (
                validation_loss
            ),
            "learning_rate": float(
                optimizer
                .param_groups[0]["lr"]
            ),
            "is_best": is_best,
        }
        if model.batch_ot_sampler is not None:
            epoch_result["batch_ot"] = {
                "train": train_ot_statistics,
                "validation": model.batch_ot_sampler.statistics(),
            }
            print(f"Epoch {epoch:03d} | Batch OT: {json.dumps(epoch_result['batch_ot'])}", flush=True)

        training_history.append(
            epoch_result
        )

        atomic_json_dump(
            training_history,
            training_history_path,
        )

        print(
            f"Epoch {epoch:03d}/"
            f"{args.epochs:03d} | "
            f"train={train_loss:.6f} | "
            f"val={validation_loss:.6f} | "
            f"lr="
            f"{optimizer.param_groups[0]['lr']:.3e} | ",
            flush=True,
        )

        save_checkpoint_atomic(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_validation_loss=(
                best_validation_loss
            ),
            train_loss=train_loss,
            validation_loss=(
                validation_loss
            ),
            model_config=model_config,
            training_config=(
                training_config
            ),
            latent_mean=latent_mean_cpu,
            latent_std=latent_std_cpu,
            parameter_count=parameter_count,
        )

        if is_best:
            save_checkpoint_atomic(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_validation_loss=(
                    best_validation_loss
                ),
                train_loss=train_loss,
                validation_loss=(
                    validation_loss
                ),
                model_config=model_config,
                training_config=(
                    training_config
                ),
                latent_mean=latent_mean_cpu,
                latent_std=latent_std_cpu,
                parameter_count=(
                    parameter_count
                ),
            )

            print(
                "Nouveau meilleur checkpoint : "
                f"{best_validation_loss:.6f}",
                flush=True,
            )

    print("\nEntraînement Flow Matching terminé.")
    print(
        "Meilleure loss de validation : "
        f"{best_validation_loss:.6f}"
    )
    print(f"Best checkpoint : {best_path}")
    print(f"Last checkpoint : {last_path}")

    del model
    del optimizer
    del scheduler
    del scaler
    del train_loader
    del validation_loader
    del latent_mean
    del latent_std
    del latents

    clear_memory()
    return best_path


def evaluate_generation_for_steps(
    model: FlowMatching,
    vae,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    training_smiles: set[str] | None,
    args: argparse.Namespace,
    device: torch.device,
    num_steps: int,
    fcd_reference: FCDReference | None = None,
) -> dict[str, Any]:
    trial_metrics: list[dict[str, Any]] = []
    all_generated_valid_smiles: list[str] = []

    latent_dim = int(
        model.vector_field.output_projection.out_features
    )

    for trial in range(args.num_trials):
        trial_seed = (
            args.seed
            + (0 if getattr(args, "fm_paired_seeds", False) else num_steps * 10_000)
            + trial
        )

        set_seed(trial_seed)

        generated_normalized = sample_fm_ode(
            model=model,
            num_samples=args.num_molecules,
            latent_dim=latent_dim,
            num_steps=num_steps,
            device=device,
            temperature=args.temperature,
            solver=args.fm_eval_solver,
        )

        generated_latents = generated_normalized * latent_std + latent_mean
        generated_latents = generated_latents.detach().float().cpu()

        metrics = compute_generation_metrics_from_latents(
            vae=vae,
            latents=generated_latents,
            dataset=args.dataset,
            batch_size=args.decode_batch_size,
            device=device,
            training_smiles=training_smiles,
            return_valid_smiles=fcd_reference is not None,
        )
        generated_valid_smiles = metrics.pop("valid_smiles", [])
        if fcd_reference is not None:
            all_generated_valid_smiles.extend(generated_valid_smiles)

        trial_metrics.append(
            {
                "trial": trial + 1,
                "seed": trial_seed,
                **metrics,
            }
        )

        novelty_text = (
            "N/A"
            if metrics["novelty"] is None
            else f"{metrics['novelty']:.4f}"
        )

        print(
            f"Steps {num_steps:03d} | "
            f"trial {trial + 1:02d}/{args.num_trials:02d} | "
            f"validity={metrics['validity']:.4f} | "
            f"uniqueness={metrics['uniqueness']:.4f} | "
            f"novelty={novelty_text} | "
            f"valid={metrics['valid_count']:,}/"
            f"{metrics['total_count']:,}",
            flush=True,
        )

        del generated_normalized
        del generated_latents
        clear_memory()

    aggregate = summarize_trials(trial_metrics)

    result = {
        "num_steps": num_steps,
        "sample_method": args.fm_eval_solver,
        "paired_seeds": getattr(args, "fm_paired_seeds", False),
        "trials": trial_metrics,
        "aggregate": aggregate,
    }
    if fcd_reference is not None:
        result.update(
            compute_fcd_result(
                args=args,
                device=device,
                reference=fcd_reference,
                generated_valid_smiles=all_generated_valid_smiles,
            )
        )
    return result


def evaluate_flow_matching(
    args: argparse.Namespace,
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:

    evaluation_dir = output_dir / args.fm_evaluation_subdir

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        model,
        latent_mean,
        latent_std,
        fm_metadata,
    ) = load_fm_model(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    vae = load_vae(Path(args.vae_path),device,)
    fcd_reference = (
        prepare_fcd_reference(args, device)
        if not args.skip_fcd
        else None
    )

    training_smiles = (None if args.skip_novelty else load_training_smiles(Path(args.training_smiles_path))
)

    evaluation_results: list[
        dict[str, Any]
    ] = []

    for num_steps in args.num_steps:
        print(
            "\nFlow Matching evaluation"
            f"with {num_steps} steps."
        )

        result = evaluate_generation_for_steps(
            model=model,
            vae=vae,
            latent_mean=latent_mean,
            latent_std=latent_std,
            training_smiles=training_smiles,
            args=args,
            device=device,
            num_steps=num_steps,
            fcd_reference=fcd_reference,
        )

        evaluation_results.append(
            result
        )

        atomic_json_dump(
            result,
            evaluation_dir
            / f"evaluation_steps_{num_steps}.json",
        )

        clear_memory()

    atomic_json_dump(
        {
            "fm_checkpoint": str(
                checkpoint_path
            ),
            "fm_metadata": fm_metadata,
            "evaluation_results": (
                evaluation_results
            ),
        },
        evaluation_dir
        / "evaluation_summary.json",
    )

    print(
        f"Results : {evaluation_dir}"
    )

    del model
    del vae
    del latent_mean
    del latent_std
    del training_smiles
    if fcd_reference is not None:
        del fcd_reference

    clear_memory()

    return evaluation_results
