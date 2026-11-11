from __future__ import annotations

import gc
import json
import os
import random
import re
from pathlib import Path
from typing import Any,Optional

import numpy as np
import torch
import torch.nn as nn

def clear_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def freeze_module(module: nn.Module) -> None:
    module.eval()

    for parameter in module.parameters():
        parameter.requires_grad_(False)

def load_torch_file(
    path: Path,
    mmap: bool = False,
) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=mmap,
    )

def atomic_torch_save(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(f"{path}.tmp")
    torch.save(data, temporary_path)
    os.replace(temporary_path, path)

def atomic_json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(f"{path}.tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    os.replace(temporary_path, path)

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class LatentSpaceTransform(nn.Module):
    """Serializable map between raw VAE latents and the model space.

    Modes:

    * ``none``: identity;
    * ``standardize``: ``(z - mean) / std``;
    """

    def __init__(
        self,
        mode: str,
        input_dim: int,
        output_dim: int,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        def vector_or_empty(value: Optional[torch.Tensor]) -> torch.Tensor:
            return torch.empty(0) if value is None else value.float().reshape(-1)

        self.register_buffer("mean", vector_or_empty(mean))
        self.register_buffer("std", vector_or_empty(std))

    def forward(self, raw_latents: torch.Tensor) -> torch.Tensor:

        if self.mode == "none":
            return raw_latents

        if self.mode == "standardize":
            return (raw_latents - self.mean) / self.std.clamp_min(1e-8)

    def inverse(self, transformed_latents: torch.Tensor) -> torch.Tensor:
        if transformed_latents.shape[-1] != self.output_dim:
            raise ValueError(
                f"Expected model-space dimension={self.output_dim}, "
                f"obtained={transformed_latents.shape[-1]}."
            )

        if self.mode == "none":
            return transformed_latents

        if self.mode == "standardize":
            return transformed_latents * self.std + self.mean
        
    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "config": {
                "mode": self.mode,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
            },
            "state_dict": {
                key: value.detach().cpu()
                for key, value in self.state_dict().items()
            },
        }

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: dict[str, Any],
        device: torch.device,
    ) -> "LatentSpaceTransform":
        config = dict(payload["config"])
        state = payload["state_dict"]
        transform = cls(
            mode=str(config["mode"]),
            input_dim=int(config["input_dim"]),
            output_dim=int(config["output_dim"]),
            mean=state.get("mean"),
            std=state.get("std"),
        )
        return transform.to(device)


def build_fm_compatible_latent_transform(
    raw_dim: int,
    fm_mean: torch.Tensor,
    fm_std: torch.Tensor,
    device: torch.device,
) -> LatentSpaceTransform:
    """
    Construit la transformation entre les latents bruts du VAE
    et l'espace normalisé attendu par le Flow Matching.

    Aucune PCA n'est utilisée :
        z_normalized = (z_raw - mean) / std
    """
    fm_mean = (
        fm_mean
        .detach()
        .float()
        .cpu()
        .reshape(-1)
    )

    fm_std = (
        fm_std
        .detach()
        .float()
        .cpu()
        .reshape(-1)
        .clamp_min(1e-8)
    )

    if fm_mean.shape != fm_std.shape:
        raise ValueError(
            "The FM mean and standard deviation must have the same shape."
        )

    fm_dim = fm_mean.numel()

    if raw_dim != fm_dim:
        raise ValueError(
            f"The VAE latent dimension is {raw_dim}, "
            f"but the FM expects {fm_dim} dimensions."
        )

    return LatentSpaceTransform(
        mode="standardize",
        input_dim=raw_dim,
        output_dim=raw_dim,
        mean=fm_mean,
        std=fm_std,
    ).to(device)

def load_any_latent_tensor(path: Path, mmap: bool = True) -> torch.Tensor:
    if path.suffix == ".dat":
        return load_memmap_latent_tensor(path)

    latents = torch.load(
        path,
        map_location="cpu",
        mmap=mmap,
    )
    return latents


def load_memmap_latent_tensor(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Latent memmap not found: {path}")

    meta_path = path.with_name(f"{path.stem}_meta.txt")
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Latent memmap metadata not found: {meta_path}"
        )

    parts = meta_path.read_text(encoding="utf-8").split()
    if len(parts) not in {3, 5}:
        raise ValueError(
            f"Expected metadata format '<n_rows> <dim> <dtype>' or "
            f"'<n_rows> <dim> <dtype> <start_idx> <end_idx>' in {meta_path}, "
            f"got: {parts!r}"
        )

    n_rows = int(parts[0])
    latent_dim = int(parts[1])
    dtype = np.dtype(parts[2])
    expected_bytes = n_rows * latent_dim * dtype.itemsize
    actual_bytes = path.stat().st_size

    if actual_bytes < expected_bytes:
        raise ValueError(
            f"Latent memmap is too small: {path} has {actual_bytes} bytes, "
            f"expected at least {expected_bytes} bytes for "
            f"shape=({n_rows}, {latent_dim}) dtype={dtype}."
        )

    mode = "r+" if os.access(path, os.W_OK) else "r"
    array = np.memmap(
        path,
        dtype=dtype,
        mode=mode,
        shape=(n_rows, latent_dim),
    )
    latents = torch.from_numpy(array)
    print(
        f"Loaded latent memmap: {path} "
        f"shape={tuple(latents.shape)} dtype={latents.dtype}"
    )
    return latents


def load_vae(vae_path: Path, device: torch.device):
    if not vae_path.is_file():
        raise FileNotFoundError(f"Checkpoint VAE introuvable : {vae_path}")

    from rdkit import RDLogger
    from src.model.builder import ModelBuilder

    RDLogger.DisableLog("rdApp.*")
    print(f"Chargement du VAE depuis : {vae_path}")
    vae = ModelBuilder.build_from_checkpoint(str(vae_path)).to(device)
    freeze_module(vae)
    print("VAE chargé.")
    return vae


def capture_torch_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_torch_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["cpu"])

    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])

def load_latent_transform_from_checkpoint(
    path: Path,
    device: torch.device,
) -> Optional[LatentSpaceTransform]:
    """Load an exact serialized latent transform when a checkpoint stores one."""
    checkpoint = load_torch_file(path)
    payload = checkpoint.get("latent_transform") if isinstance(checkpoint, dict) else None
    if not isinstance(payload, dict):
        return None
    return LatentSpaceTransform.from_checkpoint_payload(payload, device)


def float_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")
