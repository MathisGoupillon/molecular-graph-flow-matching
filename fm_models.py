from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from collections.abc import Callable


import torch
import torch.nn as nn
from utils import (
    clear_memory,
    load_torch_file
)


def read_fm_checkpoint_components(
    checkpoint_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
]:
    checkpoint = load_torch_file(checkpoint_path)

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The FM checkpoint must be a dictionary."
        )

    required_keys = {
        "model_config",
        "model_state_dict",
        "latent_mean",
        "latent_std",
    }

    missing_keys = required_keys.difference(checkpoint)

    if missing_keys:
        raise KeyError(
            "The FM checkpoint is missing the following keys: "
            f"{sorted(missing_keys)}"
        )

    model_config = dict(checkpoint["model_config"])
    state_dict = dict(checkpoint["model_state_dict"])

    latent_mean = (
        checkpoint["latent_mean"]
        .detach()
        .float()
        .cpu()
        .reshape(-1)
    )

    latent_std = (
        checkpoint["latent_std"]
        .detach()
        .float()
        .cpu()
        .reshape(-1)
        .clamp_min(1e-8)
    )

    if latent_mean.shape != latent_std.shape:
        raise ValueError(
            "latent_mean and latent_std must have the same shape."
        )

    model_config.setdefault(
        "latent_dim",
        latent_mean.numel(),
    )

    metadata = {
        "epoch": int(checkpoint.get("epoch", -1)),
        "best_validation_loss": float(
            checkpoint.get(
                "best_validation_loss",
                float("nan"),
            )
        ),
        "training_config": checkpoint.get(
            "training_config",
            {},
        ),
    }

    return (
        model_config,
        state_dict,
        latent_mean,
        latent_std,
        metadata,
    )

def read_fm_metadata(
    checkpoint_path: Path,
) -> tuple[
    dict[str, Any],
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
]:
    (
        model_config,
        _,
        latent_mean,
        latent_std,
        metadata,
    ) = read_fm_checkpoint_components(checkpoint_path)

    clear_memory()

    return (
        model_config,
        latent_mean,
        latent_std,
        metadata,
    )

@torch.no_grad()
def sample_ode(
    vector_field: Callable[
        [torch.Tensor, torch.Tensor],
        torch.Tensor,
    ],
    num_samples: int,
    latent_dim: int,
    num_steps: int,
    device: torch.device,
    temperature: float,
    time_start: float = 0.0,
    time_end: float = 1.0,
    solver: str = "heun",
) -> torch.Tensor:
    if solver not in {"euler", "heun", "midpoint"}:
        raise ValueError(f"Unsupported ODE solver: {solver}")

    x = (
        torch.randn(
            num_samples,
            latent_dim,
            device=device,
        )
        * temperature
    )

    dt = (time_end - time_start) / num_steps

    for step in range(num_steps):
        t0_value = time_start + step * dt

        t0 = torch.full(
            (num_samples, 1),
            t0_value,
            device=device,
            dtype=x.dtype,
        )

        velocity0 = vector_field(x, t0)

        if solver == "euler":
            x = x + dt * velocity0
            continue

        if solver == "midpoint":
            midpoint = x + 0.5 * dt * velocity0
            t_midpoint = torch.full(
                (num_samples, 1),
                t0_value + 0.5 * dt,
                device=device,
                dtype=x.dtype,
            )
            midpoint_velocity = vector_field(midpoint, t_midpoint)
            x = x + dt * midpoint_velocity
            continue

        predictor = x + dt * velocity0

        t1 = torch.full(
            (num_samples, 1),
            t0_value + dt,
            device=device,
            dtype=x.dtype,
        )

        velocity1 = vector_field(predictor, t1)

        x = x + 0.5 * dt * (velocity0 + velocity1)

    return x

def strip_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if (
        state_dict
        and all(
            key.startswith("module.")
            for key in state_dict
        )
    ):
        return {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }

    return state_dict

def sample_intermediate_states(
    endpoints: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.randn_like(endpoints)
    t = torch.rand(
        endpoints.shape[0],
        1,
        device=endpoints.device,
        dtype=endpoints.dtype,
    ).clamp_(1e-4, 1.0 - 1e-4)
    x_t = (1.0 - t) * source + t * endpoints
    return x_t, t
