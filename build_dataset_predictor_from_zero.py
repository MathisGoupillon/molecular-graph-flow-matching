from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable
import torch

from utils import (LatentSpaceTransform,
    atomic_torch_save,
    clear_memory,
    load_any_latent_tensor,
    load_torch_file,
    load_vae
)
from evaluations import decode_latents_to_validity_mask
from torch.utils.data import DataLoader, Sampler, TensorDataset, default_collate


GAUSSIAN_CACHE_FORMAT_VERSION = 1
GaussianProgressCallback = Callable[[dict[str, Any]], None]


def empty_gaussian_component(dim: int) -> dict[str, torch.Tensor]:
    return {
        "positive": torch.empty(0, dim),
        "negative": torch.empty(0, dim),
        "source_indices": torch.empty(0, dtype=torch.long),
        "negative_distances": torch.empty(0),
    }


def gaussian_component_from_chunks(
    positives: list[torch.Tensor],
    negatives: list[torch.Tensor],
    sources: list[torch.Tensor],
    distances: list[torch.Tensor],
    target_count: int,
    dim: int,
) -> dict[str, torch.Tensor]:
    if not positives:
        return empty_gaussian_component(dim)

    return {
        "positive": torch.cat(positives, dim=0)[:target_count].float(),
        "negative": torch.cat(negatives, dim=0)[:target_count].float(),
        "source_indices": torch.cat(sources, dim=0)[:target_count].long(),
        "negative_distances": torch.cat(distances, dim=0)[:target_count].float(),
    }


def validate_gaussian_component_state(
    state: Any,
    component_name: str,
    target_count: int,
    dim: int,
) -> dict[str, Any] | None:
    if state is None:
        return None
    if not isinstance(state, dict):
        raise RuntimeError(
            f"Checkpoint Gaussian invalide : composant {component_name}."
        )

    data = state.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(
            f"Checkpoint Gaussian invalide : données {component_name} absentes."
        )

    expected_dtypes = {
        "positive": torch.float32,
        "negative": torch.float32,
        "source_indices": torch.int64,
        "negative_distances": torch.float32,
    }
    tensors: dict[str, torch.Tensor] = {}
    for key, dtype in expected_dtypes.items():
        value = data.get(key)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(
                "Checkpoint Gaussian invalide : "
                f"{component_name}.{key} est absent."
            )
        tensors[key] = value.detach().cpu().to(dtype=dtype)

    count = len(tensors["positive"])
    if not (
        len(tensors["negative"])
        == len(tensors["source_indices"])
        == len(tensors["negative_distances"])
        == count
    ):
        raise RuntimeError(
            f"Checkpoint Gaussian invalide : tailles incohérentes pour {component_name}."
        )
    if count > target_count:
        raise RuntimeError(
            f"Checkpoint Gaussian invalide : trop de paires {component_name}."
        )
    if (
        tensors["positive"].ndim != 2
        or tensors["negative"].ndim != 2
        or tensors["positive"].shape[1] != dim
        or tensors["negative"].shape[1] != dim
    ):
        raise RuntimeError(
            f"Checkpoint Gaussian invalide : dimension incorrecte pour {component_name}."
        )
    if bool(state.get("complete", False)) and count != target_count:
        raise RuntimeError(
            f"Checkpoint Gaussian invalide : {component_name} marqué complet "
            f"avec {count:,}/{target_count:,} paires."
        )

    validated = dict(state)
    validated["data"] = tensors
    validated["current_count"] = count
    return validated


def gaussian_cache_signature(
    args: argparse.Namespace,
    latents: torch.Tensor,
    local_target: int,
    global_target: int,
) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "dataset_size": int(args.dataset_size),
        "latent_count": int(len(latents)),
        "latent_dim": int(latents.shape[1]),
        "latents_path": str(Path(args.latents_path)),
        "vae_path": str(Path(args.vae_path)),
        "latent_transform": args.latent_transform,
        "negative_dataset_mode": args.negative_dataset_mode,
        "local_target": int(local_target),
        "global_target": int(global_target),
        "local_gaussian_fraction": float(args.local_gaussian_fraction),
        "local_gaussian_sigma": float(args.local_gaussian_sigma),
        "local_gaussian_samples_per_positive": int(
            args.local_gaussian_samples_per_positive
        ),
        "local_gaussian_pair_mode": args.local_gaussian_pair_mode,
        "local_gaussian_positive_batch_size": int(
            args.local_gaussian_positive_batch_size
        ),
        "positive_candidate_batch_size": int(
            args.positive_candidate_batch_size
        ),
        "global_gaussian_batch_size": int(args.global_gaussian_batch_size),
        "global_gaussian_std_scale": float(args.global_gaussian_std_scale),
        "max_noise_negative_candidates": int(
            args.max_noise_negative_candidates
        ),
        "decode_batch_size": int(args.decode_batch_size),
        "seed": int(args.seed),
    }


def load_gaussian_progress_cache(
    cache_path: Path,
    signature: dict[str, Any],
    local_target: int,
    global_target: int,
    dim: int,
) -> dict[str, Any]:
    if not cache_path.is_file():
        return {
            "local": None,
            "global": None,
        }

    cache = load_torch_file(cache_path)
    if (
        not isinstance(cache, dict)
        or cache.get("format_version") != GAUSSIAN_CACHE_FORMAT_VERSION
    ):
        raise RuntimeError(
            f"Checkpoint Gaussian incompatible : {cache_path}. "
            "Utilise --overwrite_dataset pour le supprimer."
        )
    if cache.get("signature") != signature:
        raise RuntimeError(
            f"Le checkpoint Gaussian {cache_path} correspond à une autre "
            "configuration. Utilise --overwrite_dataset pour recommencer."
        )

    local_state = validate_gaussian_component_state(
        cache.get("local"),
        "local",
        local_target,
        dim,
    )
    global_state = validate_gaussian_component_state(
        cache.get("global"),
        "global",
        global_target,
        dim,
    )
    if global_state is not None and (
        local_state is None
        or not bool(local_state.get("complete", False))
    ):
        raise RuntimeError(
            "Checkpoint Gaussian invalide : progression globale présente "
            "avant la fin de la génération locale."
        )

    print(f"Checkpoint Gaussian chargé : {cache_path}")
    if local_state is not None:
        print(
            "  progression locale  : "
            f"{local_state['current_count']:,}/{local_target:,}"
        )
    if global_state is not None:
        print(
            "  progression globale : "
            f"{global_state['current_count']:,}/{global_target:,}"
        )
    return {
        "local": local_state,
        "global": global_state,
    }


def save_gaussian_progress_cache(
    cache_path: Path,
    signature: dict[str, Any],
    local_state: dict[str, Any] | None,
    global_state: dict[str, Any] | None,
) -> None:
    atomic_torch_save(
        {
            "format_version": GAUSSIAN_CACHE_FORMAT_VERSION,
            "signature": signature,
            "local": local_state,
            "global": global_state,
        },
        cache_path,
    )
    print(f"Checkpoint Gaussian sauvegardé : {cache_path}", flush=True)

@torch.no_grad()
def build_latent_transform(
    latents: torch.Tensor,
    mode: str,
) -> LatentSpaceTransform:

    raw_dim = int(latents.shape[1])

    if mode == "none":
        print(
            "Transformation latente : aucune "
            f"({raw_dim}D -> {raw_dim}D)"
        )

        return LatentSpaceTransform(
            mode="none",
            input_dim=raw_dim,
            output_dim=raw_dim,
        )

    if mode == "standardize":
        std, mean = torch.std_mean(
            latents.float(),
            dim=0,
            correction=0,
        )

        std = std.clamp_min(1e-8)

        print(
            "Transformation latente : standardisation "
            f"({raw_dim}D -> {raw_dim}D)"
        )

        return LatentSpaceTransform(
            mode="standardize",
            input_dim=raw_dim,
            output_dim=raw_dim,
            mean=mean.cpu(),
            std=std.cpu(),
        )
    

@torch.no_grad()
def decode_transformed_latents_to_validity_mask(
    vae,
    transformed_latents: torch.Tensor,
    transform: LatentSpaceTransform,
    dataset: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list]:
    raw_latents = transform.inverse(
        transformed_latents
    ).float()

    return decode_latents_to_validity_mask(
        vae=vae,
        latents=raw_latents,
        dataset=dataset,
        batch_size=batch_size,
        device=device,
    )


def build_predictor_dataset_gaussian(
    args: argparse.Namespace,
    dataset_path: Path,
    dataset : str,
    cache_path: Path,
    device: torch.device,
) -> Path:
    target_pairs = args.dataset_size // 2

    (
        local_target,
        global_target,
    ) = compute_gaussian_balanced_counts(
        target_pairs,
        args,
    )

    print("\nConfiguration du dataset prédicteur :")
    print(f"  total positifs              : {target_pairs:,}")
    print(f"  total négatifs              : {target_pairs:,}")
    print(f"  local Gaussian balanced     : {local_target:,}")
    print(f"  global Gaussian balanced    : {global_target:,}")
    print(f"  local sigma                 : {args.local_gaussian_sigma}")

    if (
        dataset_path.is_file()
        and not args.overwrite_dataset
    ):
        data = load_torch_file(dataset_path)

        positive = data.get("positive_endpoints_normalized")
        negative = data.get("hard_negative_endpoints_normalized")
        metadata = data.get("metadata", {})
        existing_mode = metadata.get("negative_dataset_mode",None,)

        if (
            isinstance(positive, torch.Tensor)
            and isinstance(negative, torch.Tensor)
            and len(positive) == target_pairs
            and len(negative) == target_pairs
            and existing_mode == args.negative_dataset_mode
        ):
            print(
                f"Dataset déjà présent : {dataset_path}"
            )
            try:
                cache_path.unlink(missing_ok=True)
            except OSError as error:
                print(
                    "Checkpoint Gaussian obsolète non supprimé : "
                    f"{error}",
                    flush=True,
                )
            return dataset_path

    if args.overwrite_dataset:
        for path in (
            dataset_path,
            cache_path,
        ):
            if path.exists():
                path.unlink()

    latents = load_any_latent_tensor(
        Path(args.latents_path),
        mmap=True,
    )

    transform = build_latent_transform(
        latents=latents,
        mode=args.latent_transform,
    )

    signature = gaussian_cache_signature(
        args=args,
        latents=latents,
        local_target=local_target,
        global_target=global_target,
    )
    progress = load_gaussian_progress_cache(
        cache_path=cache_path,
        signature=signature,
        local_target=local_target,
        global_target=global_target,
        dim=transform.output_dim,
    )

    vae = load_vae(
        Path(args.vae_path),
        device,
    )

    local_state = progress["local"]
    global_state = progress["global"]

    def save_local_progress(state: dict[str, Any]) -> None:
        nonlocal local_state
        local_state = state
        save_gaussian_progress_cache(
            cache_path=cache_path,
            signature=signature,
            local_state=local_state,
            global_state=global_state,
        )

    if local_state is not None and bool(local_state.get("complete", False)):
        local_data = local_state["data"]
        print("Génération locale déjà complète dans le checkpoint.")
    else:
        local_data = mine_local_gaussian_balanced_pairs(
            latents=latents,
            target_count=local_target,
            transform=transform,
            vae=vae,
            dataset=dataset,
            args=args,
            device=device,
            initial_state=local_state,
            progress_callback=save_local_progress,
        )

    def save_global_progress(state: dict[str, Any]) -> None:
        nonlocal global_state
        global_state = state
        save_gaussian_progress_cache(
            cache_path=cache_path,
            signature=signature,
            local_state=local_state,
            global_state=global_state,
        )

    if global_state is not None and bool(global_state.get("complete", False)):
        global_data = global_state["data"]
        print("Génération globale déjà complète dans le checkpoint.")
    else:
        global_data = mine_global_gaussian_balanced_pairs(
            latents=latents,
            target_count=global_target,
            transform=transform,
            vae=vae,
            dataset=dataset,
            args=args,
            device=device,
            initial_state=global_state,
            progress_callback=save_global_progress,
        )

    positive_all = torch.cat((local_data["positive"],global_data["positive"],),dim=0,)[:target_pairs].float()
    negative_all = torch.cat((local_data["negative"],global_data["negative"],),dim=0,)[:target_pairs].float()

    source_indices_all = torch.cat((local_data["source_indices"],global_data["source_indices"],),dim=0,)[:target_pairs].long()
    negative_distances_all = torch.cat((local_data["negative_distances"],global_data["negative_distances"],),dim=0,)[:target_pairs].float()

    negative_type_ids = torch.cat(
        (
            torch.zeros(len(local_data["negative"]),dtype=torch.long,),
            torch.ones(len(global_data["negative"]),dtype=torch.long,),
        ),dim=0,)[:target_pairs].long()

    shuffle_generator = torch.Generator().manual_seed(args.seed + 9_999_991)

    permutation = torch.randperm(target_pairs,generator=shuffle_generator,)
    positive_all = positive_all[permutation]
    negative_all = negative_all[permutation]
    source_indices_all = source_indices_all[permutation]
    negative_distances_all = negative_distances_all[permutation]
    negative_type_ids = negative_type_ids[permutation]

    final_data = {
        "positive_endpoints_normalized": (
            positive_all.float()
        ),
        "hard_negative_endpoints_normalized": (
            negative_all.float()
        ),
        "source_indices": (
            source_indices_all.long()
        ),
        "negative_distances": (
            negative_distances_all.float()
        ),
        "negative_type_ids": (
            negative_type_ids.long()
        ),
        "negative_type_names": {
            0: "local_gaussian_sigma",
            1: "global_gaussian_latent",
        },
        "latent_transform": transform.checkpoint_payload(),
        "metadata": {
            "dataset": args.dataset,
            "dataset_size": args.dataset_size,
            "positive_count": target_pairs,
            "negative_count": target_pairs,
            "negative_dataset_mode": (
                args.negative_dataset_mode
            ),
            "negative_counts": {
                "local_gaussian_sigma": local_target,
                "global_gaussian_latent": global_target,
            },
            "local_gaussian_fraction": (
                args.local_gaussian_fraction
            ),
            "local_gaussian_sigma": (
                args.local_gaussian_sigma
            ),
            "local_gaussian_pair_mode": (
                args.local_gaussian_pair_mode
            ),
            "local_gaussian_samples_per_positive": (
                args.local_gaussian_samples_per_positive
            ),
            "global_gaussian_fraction": (
                1.0 - args.local_gaussian_fraction
            ),
            "global_gaussian_std_scale": (
                args.global_gaussian_std_scale
            ),
            "negative_definition": (
                "Balanced dataset built from 60% local Gaussian "
                "sampling around valid PCA latents and 40% global "
                "Gaussian sampling in transformed latent space. "
                "Both valid and invalid decoded candidates are used."
            ),
            "uses_previous_fm": False,
            "uses_classifier_guidance": False,
            "vae_path": str(
                Path(args.vae_path)
            ),
            "latents_path": str(
                Path(args.latents_path)
            ),
            "training_smiles_path": (
                str(Path(args.training_smiles_path))
                if args.training_smiles_path is not None
                else None
            ),
            "seed": args.seed,
        },
    }

    atomic_torch_save(
        final_data,
        dataset_path,
    )

    try:
        cache_path.unlink(missing_ok=True)
    except OSError as error:
        print(
            f"Dataset sauvegardé, mais checkpoint Gaussian non supprimé : {error}",
            flush=True,
        )

    print(
        f"Dataset sauvegardé : {dataset_path}"
    )

    print(
        "Composition finale : "
        f"local_gaussian_sigma={(negative_type_ids == 0).sum().item():,}, "
        f"global_gaussian_latent={(negative_type_ids == 1).sum().item():,}"
    )

    del (
        vae,
        transform,
        latents,
        local_data,
        global_data,
        final_data,
    )

    clear_memory()
    return dataset_path


def compute_gaussian_balanced_counts(
    target_pairs: int,
    args: argparse.Namespace,
) -> tuple[int, int]:
    local_count = int(
        round(
            target_pairs
            * args.local_gaussian_fraction
        )
    )

    global_count = target_pairs - local_count
    return local_count, global_count


@torch.no_grad()
def mine_local_gaussian_balanced_pairs(
    latents: torch.Tensor,
    target_count: int,
    transform: LatentSpaceTransform,
    vae,
    dataset,
    args: argparse.Namespace,
    device: torch.device,
    initial_state: dict[str, Any] | None = None,
    progress_callback: GaussianProgressCallback | None = None,
) -> dict[str, torch.Tensor]:
    """
    Local Gaussian pair generation.

    Two modes are supported.

    1) one_negative_per_source:
        For each valid source z+, sample local Gaussian candidates:

            z_candidate = z+ + sigma * epsilon

        If at least one invalid candidate is found, keep only one pair:

            positive = z+
            negative = one invalid local candidate

        This maximizes source diversity.

    2) all_balanced:
        For each valid source z+, sample local Gaussian candidates.
        Decode all candidates.
        Keep m valid candidates and m invalid candidates, where:

            m = min(num_valid, num_invalid)

        This is the previous behavior and maximizes the number of local
        pairs extracted per source.
    """

    dim = transform.output_dim

    if target_count <= 0:
        data = empty_gaussian_component(dim)
        if progress_callback is not None:
            progress_callback(
                {
                    "complete": True,
                    "current_count": 0,
                    "attempted_sources": 0,
                    "attempted_candidates": 0,
                    "cursor": 0,
                    "generator_state": None,
                    "data": data,
                }
            )
        return data

    positives: list[torch.Tensor] = []
    negatives: list[torch.Tensor] = []
    sources: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []

    current_count = 0
    attempted_sources = 0
    attempted_candidates = 0
    cursor = 0

    if initial_state is not None:
        cached_data = initial_state["data"]
        current_count = len(cached_data["positive"])
        attempted_sources = int(initial_state.get("attempted_sources", 0))
        attempted_candidates = int(initial_state.get("attempted_candidates", 0))
        cursor = int(initial_state.get("cursor", 0))
        if current_count:
            positives.append(cached_data["positive"].float())
            negatives.append(cached_data["negative"].float())
            sources.append(cached_data["source_indices"].long())
            distances.append(cached_data["negative_distances"].float())

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(args.seed + 202_607)
    if (
        initial_state is not None
        and isinstance(initial_state.get("generator_state"), torch.Tensor)
    ):
        generator.set_state(initial_state["generator_state"])

    save_every = max(1, int(args.dataset_save_every))
    next_save_count = ((current_count // save_every) + 1) * save_every

    def make_progress_state(complete: bool) -> dict[str, Any]:
        data = gaussian_component_from_chunks(
            positives=positives,
            negatives=negatives,
            sources=sources,
            distances=distances,
            target_count=target_count,
            dim=dim,
        )
        return {
            "complete": complete,
            "current_count": len(data["positive"]),
            "attempted_sources": attempted_sources,
            "attempted_candidates": attempted_candidates,
            "cursor": cursor,
            "generator_state": generator.get_state(),
            "data": data,
        }

    print("\nConstruction local Gaussian pairs "f": {current_count:,}/{target_count:,}")
    print(f"  sigma = {args.local_gaussian_sigma}")
    print(f"  samples/source = {args.local_gaussian_samples_per_positive}")
    print(f"  pair mode = {args.local_gaussian_pair_mode}")

    while current_count < target_count:
        if cursor >= len(latents):
            raise RuntimeError(
            "Tous les latents ont été parcourus, mais seulement "
            f"{current_count:,}/{target_count:,} paires locales "
            "ont été trouvées."
        )
        remaining_positions = len(latents) - cursor

        scan_batch_size = min(
            args.positive_candidate_batch_size,
            remaining_positions,
        )

        start_index = cursor
        end_index = cursor + scan_batch_size
        raw = latents[start_index:end_index].float()
        indices = torch.arange(start_index,end_index,dtype=torch.long,)
        cursor = end_index
        attempted_sources += len(raw)

        raw = transform(raw).cpu()
        valid_mask,_ = decode_transformed_latents_to_validity_mask(
            vae=vae,
            transformed_latents=raw,
            transform=transform,
            dataset=dataset,
            batch_size=args.decode_batch_size,
            device=device,
        )
        source_positive = raw[valid_mask]
        source_indices = indices[valid_mask]

        if len(source_positive) == 0:
            continue

        for start in range(0,len(source_positive),args.local_gaussian_positive_batch_size,):
            if current_count >= target_count:
                break
            end = min(start + args.local_gaussian_positive_batch_size,len(source_positive),)

            batch_positive = source_positive[start:end].float()
            batch_sources = source_indices[start:end].long()
            batch_size = len(batch_positive)

            noise = torch.randn(batch_size,args.local_gaussian_samples_per_positive,dim,generator=generator,)

            candidates = (batch_positive[:, None, :] + float(args.local_gaussian_sigma) * noise)

            flat_candidates = candidates.reshape(-1,dim,)

            valid_mask_candidates,_ = decode_transformed_latents_to_validity_mask(
                vae=vae,
                transformed_latents=flat_candidates,
                transform=transform,
                dataset=dataset,
                batch_size=args.decode_batch_size,
                device=device,
            )
            valid = valid_mask_candidates.reshape(batch_size,args.local_gaussian_samples_per_positive,)
            attempted_candidates += len(flat_candidates)

            for row in range(batch_size):
                if current_count >= target_count:
                    break

                center = batch_positive[row].cpu()

                valid_candidates = candidates[
                    row,
                    valid[row],
                ]

                invalid_candidates = candidates[
                    row,
                    ~valid[row],
                ]

                if len(invalid_candidates) == 0:
                    continue

                if args.local_gaussian_pair_mode == "one_negative_per_source":
                    invalid_order = torch.randperm(
                        len(invalid_candidates),
                        generator=generator,
                    )

                    chosen_negative = invalid_candidates[
                        invalid_order[:1]
                    ].cpu()

                    chosen_positive = center.unsqueeze(0)

                    take = 1

                elif args.local_gaussian_pair_mode == "all_balanced":
                    take = min(
                        len(valid_candidates),
                        len(invalid_candidates),
                        target_count - current_count,
                    )

                    if take <= 0:
                        continue

                    valid_order = torch.randperm(
                        len(valid_candidates),
                        generator=generator,
                    )

                    invalid_order = torch.randperm(
                        len(invalid_candidates),
                        generator=generator,
                    )

                    chosen_positive = valid_candidates[
                        valid_order[:take]
                    ].cpu()

                    chosen_negative = invalid_candidates[
                        invalid_order[:take]
                    ].cpu()

                else:
                    raise ValueError(
                        "Mode local_gaussian_pair_mode inconnu : "
                        f"{args.local_gaussian_pair_mode}"
                    )

                positives.append(
                    chosen_positive.float()
                )

                negatives.append(
                    chosen_negative.float()
                )

                sources.append(
                    torch.full(
                        (take,),
                        int(batch_sources[row].item()),
                        dtype=torch.long,
                    )
                )

                distances.append(
                    (
                        chosen_negative
                        - center[None, :]
                    ).norm(dim=1).float()
                )

                current_count += take

        print(
            "Local Gaussian pairs : "
            f"{current_count:,}/{target_count:,} | "
            f"sources testées={attempted_sources:,} | "
            f"candidats décodés={attempted_candidates:,}",
            flush=True,
        )

        if (
            progress_callback is not None
            and current_count >= next_save_count
        ):
            progress_callback(make_progress_state(complete=False))
            while next_save_count <= current_count:
                next_save_count += save_every

        clear_memory()

    completed_state = make_progress_state(complete=True)
    if progress_callback is not None:
        progress_callback(completed_state)
    return completed_state["data"]


@torch.no_grad()
def mine_global_gaussian_balanced_pairs(
    latents: torch.Tensor,
    target_count: int,
    transform: LatentSpaceTransform,
    vae,
    dataset,
    args: argparse.Namespace,
    device: torch.device,
    initial_state: dict[str, Any] | None = None,
    progress_callback: GaussianProgressCallback | None = None,
) -> dict[str, torch.Tensor]:
    """
    Global Gaussian balanced generation.

    Sample candidates directly from an empirical Gaussian fitted in the
    transformed latent space:

        z_candidate = mean + std * epsilon

    Decode candidates.
    Keep the same number of valid and invalid decoded samples.
    """

    dim = transform.output_dim

    if target_count <= 0:
        data = empty_gaussian_component(dim)
        if progress_callback is not None:
            progress_callback(
                {
                    "complete": True,
                    "current_count": 0,
                    "attempted_candidates": 0,
                    "synthetic_source_cursor": len(latents) + 10_000_000,
                    "generator_state": None,
                    "data": data,
                }
            )
        return data

    if args.latent_transform == "none":
        global_std, global_mean = torch.std_mean(latents.float(),dim=0,correction=0,)
        global_std = global_std.clamp_min(1e-8)

    generator = torch.Generator(device="cpu").manual_seed(args.seed + 303_607)

    positives: list[torch.Tensor] = []
    negatives: list[torch.Tensor] = []
    sources: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []

    current_count = 0
    attempted_candidates = 0
    synthetic_source_cursor = len(latents) + 10_000_000

    if initial_state is not None:
        cached_data = initial_state["data"]
        current_count = len(cached_data["positive"])
        attempted_candidates = int(initial_state.get("attempted_candidates", 0))
        synthetic_source_cursor = int(
            initial_state.get(
                "synthetic_source_cursor",
                len(latents) + 10_000_000 + current_count,
            )
        )
        if current_count:
            positives.append(cached_data["positive"].float())
            negatives.append(cached_data["negative"].float())
            sources.append(cached_data["source_indices"].long())
            distances.append(cached_data["negative_distances"].float())

    if (
        initial_state is not None
        and isinstance(initial_state.get("generator_state"), torch.Tensor)
    ):
        generator.set_state(initial_state["generator_state"])

    save_every = max(1, int(args.dataset_save_every))
    next_save_count = ((current_count // save_every) + 1) * save_every

    def make_progress_state(complete: bool) -> dict[str, Any]:
        data = gaussian_component_from_chunks(
            positives=positives,
            negatives=negatives,
            sources=sources,
            distances=distances,
            target_count=target_count,
            dim=dim,
        )
        return {
            "complete": complete,
            "current_count": len(data["positive"]),
            "attempted_candidates": attempted_candidates,
            "synthetic_source_cursor": synthetic_source_cursor,
            "generator_state": generator.get_state(),
            "data": data,
        }

    print("\nConstruction global Gaussian balanced pairs "f": {current_count:,}/{target_count:,}")
    print(f"  std scale = {args.global_gaussian_std_scale}")

    while current_count < target_count:
        remaining_budget = (args.max_noise_negative_candidates - attempted_candidates)
        if remaining_budget <= 0:
            raise RuntimeError(
        "Nombre maximal de candidats globaux atteint, "
        f"mais seulement {current_count:,}/{target_count:,} "
        "paires ont été trouvées."
    )
        batch_size = min(args.global_gaussian_batch_size,args.max_noise_negative_candidates - attempted_candidates,)
        if args.latent_transform == "standardize":
            candidates = (float(args.global_gaussian_std_scale) * torch.randn(batch_size,dim,generator=generator,dtype=torch.float32,)
    )
        if args.latent_transform == "none":
            noise = torch.randn(batch_size,dim,generator=generator,)
            candidates = (global_mean + float(args.global_gaussian_std_scale) * global_std * noise)
            

        valid_mask,_ = decode_transformed_latents_to_validity_mask(
            vae=vae,
            transformed_latents=candidates,
            transform=transform,
            dataset=dataset,
            batch_size=args.decode_batch_size,
            device=device,
        )

        valid_candidates = candidates[valid_mask]
        invalid_candidates = candidates[~valid_mask]

        take = min(len(valid_candidates),
            len(invalid_candidates),
            target_count - current_count,
        )

        if take > 0:
            chosen_positive = valid_candidates[:take].cpu()
            chosen_negative = invalid_candidates[:take].cpu()

            positives.append(chosen_positive.float())
            negatives.append(chosen_negative.float())

            pair_sources = torch.arange(synthetic_source_cursor,synthetic_source_cursor + take,dtype=torch.long,)
            synthetic_source_cursor += take
            sources.append(pair_sources)

            distances.append((chosen_negative - chosen_positive).norm(dim=1).float())
            current_count += take

        attempted_candidates += batch_size

        print(
            "Global Gaussian balanced : "
            f"{current_count:,}/{target_count:,} | "
            f"candidats décodés={attempted_candidates:,} | "
            f"valid={len(valid_candidates):,} | "
            f"invalid={len(invalid_candidates):,}",
            flush=True,
        )

        if (
            progress_callback is not None
            and current_count >= next_save_count
        ):
            progress_callback(make_progress_state(complete=False))
            while next_save_count <= current_count:
                next_save_count += save_every

        clear_memory()

    completed_state = make_progress_state(complete=True)
    if progress_callback is not None:
        progress_callback(completed_state)
    return completed_state["data"]


def make_paired_stratified_splits(
    positive: torch.Tensor,
    negative: torch.Tensor,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, TensorDataset]:
    """
    Construit des splits equilibres sans separer les paires.

    ``positive[i]`` et ``negative[i]`` sont toujours affectes au meme
    split. Les indices des paires sont d'abord melanges, puis decoupes
    en train/validation/test. Chaque split contient donc exactement
    autant de positifs que de negatifs. Chaque TensorDataset contient
    ``(x, label, pair_index)`` ; les deux membres d'un couple partagent
    le meme ``pair_index``.
    """

    pair_count = len(positive)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Une seule permutation est appliquee aux indices des paires.
    # Les deux membres d'une paire restent donc dans le meme split.
    pair_indices = torch.randperm(pair_count, generator=generator)

    n_validation = int(round(pair_count * validation_fraction))
    n_test = int(round(pair_count * test_fraction))
    n_train = pair_count - n_validation - n_test

    split_indices = {
        "train": pair_indices[:n_train],
        "validation": pair_indices[n_train:n_train + n_validation],
        "test": pair_indices[n_train + n_validation:],
    }

    def build_split(indices: torch.Tensor) -> TensorDataset:
        split_positive = positive.index_select(0, indices).float()
        split_negative = negative.index_select(0, indices).float()

        x = torch.cat((split_positive, split_negative), dim=0)
        y = torch.cat(
            (
                torch.ones(len(split_positive), 1),
                torch.zeros(len(split_negative), 1),
            ),
            dim=0,
        )
        pair_indices = torch.cat((indices, indices), dim=0).long()

        # Le melange final ne change pas l'affectation des paires.
        example_indices = torch.randperm(len(x), generator=generator)
        return TensorDataset(
            x[example_indices],
            y[example_indices],
            pair_indices[example_indices],
        )

    return {
        name: build_split(indices)
        for name, indices in split_indices.items()
    }


class PairedBatchSampler(Sampler):
    """Construit des batches contenant toujours des couples complets."""

    def __init__(
        self,
        pair_indices: torch.Tensor,
        batch_size: int,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:

        self.batch_size = batch_size
        self.pairs_per_batch = batch_size // 2
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        pair_indices = pair_indices.detach().cpu().reshape(-1).long()
        pair_order = torch.argsort(pair_indices)
        sorted_pair_indices = pair_indices[pair_order]

        _, pair_counts = torch.unique_consecutive(
            sorted_pair_indices,
            return_counts=True,
        )


        # Chaque ligne contient les deux positions, positive et negative,
        # correspondant a un meme couple dans le TensorDataset.
        self.pair_positions = pair_order.reshape(-1, 2)

    def set_epoch(self, epoch: int) -> None:
        """Permet de reproduire exactement le melange d'une epoque."""
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator(device="cpu").manual_seed(
            self.seed + self.epoch
        )

        if self.shuffle:
            permutation = torch.randperm(
                len(self.pair_positions),
                generator=generator,
            )
            pair_positions = self.pair_positions[permutation]
        else:
            pair_positions = self.pair_positions

        for start in range(0, len(pair_positions), self.pairs_per_batch):
            batch_pairs = pair_positions[start:start + self.pairs_per_batch]

            if self.drop_last and len(batch_pairs) < self.pairs_per_batch:
                continue

            batch_positions = batch_pairs.reshape(-1)

            # Les membres des couples restent dans le batch, mais ne sont pas
            # necessairement adjacents dans l'ordre donne au modele.
            if self.shuffle:
                example_permutation = torch.randperm(
                    len(batch_positions),
                    generator=generator,
                )
                batch_positions = batch_positions[example_permutation]

            yield batch_positions.tolist()

        if self.shuffle:
            self.epoch += 1

    def __len__(self) -> int:
        pair_count = len(self.pair_positions)

        if self.drop_last:
            return pair_count // self.pairs_per_batch

        return (
            pair_count + self.pairs_per_batch - 1
        ) // self.pairs_per_batch


def collate_paired_examples(batch):
    """Empile les exemples et retire pair_index avant l'entrainement."""
    endpoints, labels, _ = default_collate(batch)
    return endpoints, labels


def make_paired_loader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    drop_last: bool = False,
) -> DataLoader:
    """
    Cree un DataLoader qui conserve les couples dans le meme batch.

    ``batch_size`` designe le nombre total d'exemples. Un batch de taille
    1024 est donc construit a partir de 512 couples positif/negatif.
    La boucle d'entrainement recoit seulement ``(endpoints, labels)``.
    """

    batch_sampler = PairedBatchSampler(
        pair_indices=dataset.tensors[2],
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "collate_fn": collate_paired_examples,
    }

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(**loader_kwargs)
