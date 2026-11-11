from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from fcd import get_predictions, load_ref_model

from evaluations import load_canonical_smiles


@dataclass
class FCDReference:
    model: nn.Module
    mean: np.ndarray
    covariance: np.ndarray
    count: int


def empty_fcd_result() -> dict[str, object]:
    return {
        "fcd": None,
        "fcd_protocol": "categorical_flow_maps",
        "fcd_reference_smiles_path": None,
        "fcd_largest_fragment": True,
        "fcd_unique_generated": True,
        "fcd_covariance_epsilon": 1e-6,
        "fcd_reference_count": 0,
        "fcd_generated_count": 0,
        "fcd_total_valid_generated": 0,
        "fcd_total_unique_generated": 0,
    }


def compute_fcd_activation_statistics(
    model: nn.Module,
    smiles: list[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    activations = get_predictions(
        model=model,
        smiles_list=smiles,
        batch_size=batch_size,
        n_jobs=num_workers,
        device=str(device),
    )
    return (
        np.mean(activations, axis=0),
        np.cov(activations, rowvar=False),
    )


def calculate_categorical_flow_maps_frechet_distance(
    mean1: np.ndarray,
    covariance1: np.ndarray,
    mean2: np.ndarray,
    covariance2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Frechet distance with the stabilization used by Categorical Flow Maps."""
    from scipy.linalg import sqrtm

    mean_difference = mean1 - mean2
    identity = np.eye(covariance1.shape[0])
    stabilized_covariance1 = covariance1 + identity * eps
    stabilized_covariance2 = covariance2 + identity * eps
    covariance_mean, _ = sqrtm(
        stabilized_covariance1.dot(stabilized_covariance2),
        disp=False,
    )

    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real

    trace_term = np.trace(
        stabilized_covariance1
        + stabilized_covariance2
        - 2.0 * covariance_mean
    )
    return float(
        (mean_difference.dot(mean_difference) + trace_term).real
    )


def prepare_fcd_reference(
    args: argparse.Namespace,
    device: torch.device,
) -> FCDReference:
    if not args.fcd_reference_smiles_path:
        raise ValueError(
            "--fcd_reference_smiles_path is required when FCD is enabled. "
            "Use validation SMILES, not training SMILES."
        )

    reference_path = Path(args.fcd_reference_smiles_path)
    reference_smiles = load_canonical_smiles(
        reference_path,
        label="FCD validation",
    )
    if len(reference_smiles) < 2:
        raise ValueError(
            "At least two valid validation SMILES are required for FCD, "
            f"found {len(reference_smiles):,}."
        )

    if (
        args.fcd_num_reference > 0
        and len(reference_smiles) > args.fcd_num_reference
    ):
        generator = np.random.default_rng(args.seed + 90_000)
        reference_indices = generator.choice(
            len(reference_smiles),
            size=args.fcd_num_reference,
            replace=False,
        )
        sampled_reference_smiles = [
            reference_smiles[int(index)]
            for index in reference_indices
        ]
    else:
        sampled_reference_smiles = reference_smiles

    print("\nPreparing FCD reference statistics...")
    print(f"  Reference path      : {reference_path}")
    print(
        f"  Reference molecules : "
        f"{len(sampled_reference_smiles):,}"
    )

    model = load_ref_model()
    mean, covariance = compute_fcd_activation_statistics(
        model=model,
        smiles=sampled_reference_smiles,
        batch_size=args.fcd_batch_size,
        num_workers=args.fcd_num_workers,
        device=device,
    )
    return FCDReference(
        model=model,
        mean=mean,
        covariance=covariance,
        count=len(sampled_reference_smiles),
    )


def compute_fcd_result(
    args: argparse.Namespace,
    device: torch.device,
    reference: FCDReference,
    generated_valid_smiles: list[str],
) -> dict[str, object]:
    unique_generated_smiles = sorted(set(generated_valid_smiles))
    generated_count = (
        min(args.fcd_num_generated, len(unique_generated_smiles))
        if args.fcd_num_generated > 0
        else len(unique_generated_smiles)
    )
    score: Optional[float] = None

    if generated_count < 2:
        print(
            "\nFCD could not be computed: fewer than two "
            "valid generated SMILES are available."
        )
    else:
        generator = np.random.default_rng(args.seed + 100_000)
        if len(unique_generated_smiles) > generated_count:
            selected_indices = generator.choice(
                len(unique_generated_smiles),
                size=generated_count,
                replace=False,
            )
            selected_generated_smiles = [
                unique_generated_smiles[int(index)]
                for index in selected_indices
            ]
        else:
            selected_generated_smiles = unique_generated_smiles

        print("\nComputing final FCD...")
        print(
            f"  Valid SMILES collected : "
            f"{len(generated_valid_smiles):,}"
        )
        print(
            f"  Unique valid SMILES    : "
            f"{len(unique_generated_smiles):,}"
        )
        print(
            f"  Generated SMILES used  : "
            f"{len(selected_generated_smiles):,}"
        )
        print(f"  Reference SMILES used  : {reference.count:,}")

        generated_mean, generated_covariance = (
            compute_fcd_activation_statistics(
                model=reference.model,
                smiles=selected_generated_smiles,
                batch_size=args.fcd_batch_size,
                num_workers=args.fcd_num_workers,
                device=device,
            )
        )
        score = calculate_categorical_flow_maps_frechet_distance(
            mean1=generated_mean,
            covariance1=generated_covariance,
            mean2=reference.mean,
            covariance2=reference.covariance,
        )
        print(f"  Final FCD              : {score:.6f}")

    return {
        **empty_fcd_result(),
        "fcd": score,
        "fcd_reference_smiles_path": str(
            args.fcd_reference_smiles_path
        ),
        "fcd_reference_count": reference.count,
        "fcd_generated_count": generated_count,
        "fcd_total_valid_generated": len(generated_valid_smiles),
        "fcd_total_unique_generated": len(unique_generated_smiles),
    }
