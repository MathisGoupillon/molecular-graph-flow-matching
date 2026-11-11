from typing import Optional, Any
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from src.utils.evaluations import molecules_from_outputs



@torch.no_grad()
def decode_latents_to_validity_mask(
    vae: nn.Module,
    latents: torch.Tensor,
    dataset: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[Optional[str]]]:

    mask_chunks: list[torch.Tensor] = []
    canonical_smiles: list[Optional[str]] = []

    for start in range(0, len(latents), batch_size):
        end = min(
            start + batch_size,
            len(latents),
        )

        latent_batch = latents[start:end].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        outputs = vae.decoder(latent_batch)

        batch_molecules = molecules_from_outputs(
            outputs,
            dataset,
        )

        batch_smiles = [
            canonical_smiles_or_none(molecule)
            for molecule in batch_molecules
        ]

        batch_mask = torch.tensor(
            [
                smiles is not None
                for smiles in batch_smiles
            ],
            dtype=torch.bool,
        )

        mask_chunks.append(batch_mask)
        canonical_smiles.extend(batch_smiles)

        del latent_batch
        del outputs
        del batch_molecules
        del batch_smiles
        del batch_mask

    if not mask_chunks:
        return (
            torch.empty(0, dtype=torch.bool),
            [],
        )

    return (
        torch.cat(mask_chunks, dim=0),
        canonical_smiles,
    )

def canonical_smiles_or_none(
    molecule: Any,
) -> Optional[str]:
    if molecule is None:
        return None

    try:
        from rdkit import Chem

        # Match Categorical Flow Maps: the complete molecule must sanitize,
        # then metrics are computed on its largest connected fragment.
        Chem.SanitizeMol(molecule)
        fragments = Chem.rdmolops.GetMolFrags(
            molecule,
            asMols=True,
            sanitizeFrags=True,
        )
        largest_fragment = max(
            fragments,
            default=molecule,
            key=lambda fragment: fragment.GetNumAtoms(),
        )
        smiles = Chem.MolToSmiles(
            largest_fragment,
            canonical=True,
        )

        return smiles if smiles else None

    except Exception:
        return None


def canonicalize_smiles_or_none(smiles: str) -> Optional[str]:
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None

        canonical = Chem.MolToSmiles(
            molecule,
            canonical=True,
        )
        return canonical if canonical else None

    except Exception:
        return None

@torch.no_grad()
def compute_generation_metrics_from_latents(
    vae: nn.Module,
    latents: torch.Tensor,
    dataset: str,
    batch_size: int,
    device: torch.device,
    training_smiles: Optional[set[str]] = None,
    return_valid_smiles: bool = False,
) -> dict[str, float | int | None | list[str]]:

    valid_mask, canonical_smiles = (
        decode_latents_to_validity_mask(
            vae=vae,
            latents=latents,
            dataset=dataset,
            batch_size=batch_size,
            device=device,
        )
    )

    total_count = len(valid_mask)
    valid_count = int(valid_mask.sum().item())
    valid_smiles = [smiles for smiles in canonical_smiles if smiles is not None]

    unique_smiles = set(valid_smiles)
    unique_count = len(unique_smiles)

    validity = (valid_count / total_count if total_count > 0 else 0.0)

    uniqueness = (unique_count / valid_count if valid_count > 0 else 0.0)

    if training_smiles is None:
        novel_count = None
        novelty = None
    else:
        novel_smiles = unique_smiles.difference(
            training_smiles
        )

        novel_count = len(novel_smiles)

        novelty = (novel_count / unique_count if unique_count > 0 else 0.0)

    results: dict[
        str,
        float | int | None | list[str],
    ] = {
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "total_count": total_count,
        "valid_count": valid_count,
        "unique_count": unique_count,
        "novel_count": novel_count,
    }

    if return_valid_smiles:
        results["valid_smiles"] = valid_smiles

    return results

def load_training_smiles(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"SMILES file not found: {path}")

    print(f"Training smiles loading : {path}")
    with open(path, "r", encoding="utf-8") as file:
        smiles = {line.strip() for line in file if line.strip()}
    print(f"Unique training smiles : {len(smiles):,}")
    return smiles


def load_canonical_smiles(path: Path, label: str) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} SMILES file not found: {path}")

    print(f"{label} SMILES loading : {path}")
    canonical_smiles: list[str] = []
    invalid_count = 0

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            raw_smiles = line.strip()
            if not raw_smiles:
                continue

            canonical = canonicalize_smiles_or_none(raw_smiles)
            if canonical is None:
                invalid_count += 1
            else:
                canonical_smiles.append(canonical)

    print(f"Canonical {label} SMILES : {len(canonical_smiles):,}")
    if invalid_count:
        print(f"Invalid {label} SMILES ignored : {invalid_count:,}")

    return canonical_smiles

def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    for metric_name in ("validity", "uniqueness", "novelty"):
        values = [trial[metric_name] for trial in trials]
        values = [value for value in values if value is not None]

        if not values:
            summary[metric_name] = None
            continue

        array = np.asarray(values, dtype=float)
        summary[metric_name] = {
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
        }

    return summary
