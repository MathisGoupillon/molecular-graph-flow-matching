from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import set_seed, LatentSpaceTransform, clear_memory,  atomic_json_dump,load_vae, float_tag,load_latent_transform_from_checkpoint
from evaluations import compute_generation_metrics_from_latents, summarize_trials, load_training_smiles
from predictor import ValidityPredictor,load_validity_predictor
from flow_matching import FlowMatchingInference,load_fm_model
from fcd_evaluation import (
    FCDReference,
    compute_fcd_result,
    prepare_fcd_reference,
)



# ============================================================
# Guided sampling 
# ============================================================


def validity_gradient(
    predictor: ValidityPredictor,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
Per-sample gradient of log p(valid | x_t, t).

The sum is intentional: unlike averaging over the batch, it does not divide
each sample's gradient by the batch size. Therefore, the value of lambda does
not depend on eval_batch_size.
    """
    with torch.enable_grad():
        x_guided = x.detach().requires_grad_(True)
        logits = predictor(x_guided, t.detach())
        log_probability_valid = F.logsigmoid(logits).sum()
        gradient = torch.autograd.grad(
            log_probability_valid,
            x_guided,
            create_graph=False,
            retain_graph=False,
        )[0]
    return gradient.detach()


def sample_guided_ode(
    fm_model: FlowMatchingInference,
    predictor: ValidityPredictor,
    num_samples: int,
    latent_dim: int,
    num_steps: int,
    device: torch.device,
    temperature: float,
    guidance_scale: float,
    guidance_start: float,
    solver: str,
) -> torch.Tensor:
    """
Euler or midpoint integration of the guided FM vector field.

v_guided = v_FM + lambda * grad_x log p(valid | x_t, t).
    """

    if solver not in {"euler", "midpoint"}:
        raise ValueError(f"Unsupported guidance solver: {solver}")

    x = torch.randn(num_samples, latent_dim, device=device) * temperature
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t0_value = step / num_steps

        if solver == "euler":
            t0 = torch.full(
                (num_samples, 1),
                t0_value,
                device=device,
                dtype=x.dtype,
            )

            with torch.no_grad():
                velocity = fm_model.vector_field(x, t0)

            if guidance_scale > 0.0 and t0_value >= guidance_start:
                gradient = validity_gradient(predictor, x, t0)
                velocity = velocity + guidance_scale * gradient

            with torch.no_grad():
                x = x + dt * velocity
            continue

        t_mid_value = min(t0_value + 0.5 * dt, 1.0)
        t0 = torch.full(
            (num_samples, 1),
            t0_value,
            device=device,
            dtype=x.dtype,
        )
        t_mid = torch.full(
            (num_samples, 1),
            t_mid_value,
            device=device,
            dtype=x.dtype,
        )

        with torch.no_grad():
            k1 = fm_model.vector_field(x, t0)
            x_mid = x + 0.5 * dt * k1
            velocity = fm_model.vector_field(x_mid, t_mid)


        if guidance_scale > 0.0 and t_mid_value >= guidance_start:
            gradient = validity_gradient(predictor, x_mid, t_mid)
            correction = guidance_scale * gradient
            velocity = velocity + correction

        with torch.no_grad():
            x = x + dt * velocity
    return x.detach()


def guidance_config_key(scale: float, start: float) -> str:
    return f"lambda_{float_tag(scale)}_start_{float_tag(start)}"

def evaluate_one_guidance_config(
    args: argparse.Namespace,
    fm_model: FlowMatchingInference,
    predictor: ValidityPredictor,
    vae: nn.Module,
    transform: LatentSpaceTransform,
    training_smiles: Optional[set[str]],
    device: torch.device,
    guidance_scale: float,
    guidance_start: float,
    fcd_reference: Optional[FCDReference] = None,
) -> dict[str, Any]:
    latent_dim = transform.output_dim
    trials: list[dict[str, Any]] = []
    all_generated_valid_smiles: list[str] = []

    for trial in range(args.num_trials):
        set_seed(args.seed + trial)
        raw_latent_chunks: list[torch.Tensor] = []

        remaining = args.num_molecules

        while remaining > 0:
            batch_count = min(
                args.eval_batch_size,
                remaining,
            )
            model_space_latents = sample_guided_ode(
                fm_model=fm_model,
                predictor=predictor,
                num_samples=batch_count,
                latent_dim=latent_dim,
                num_steps=args.eval_steps,
                device=device,
                temperature=args.temperature,
                guidance_scale=guidance_scale,
                guidance_start=guidance_start,
                solver=args.guidance_solver,
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
        generated_valid_smiles = metrics.pop("valid_smiles", [])
        if fcd_reference is not None:
            all_generated_valid_smiles.extend(generated_valid_smiles)

        trials.append(
            {
                "trial": trial + 1,
                "seed": args.seed + trial,
                **metrics,
            }
        )

        novelty_text = (
            "N/A"
            if metrics["novelty"] is None
            else f"{metrics['novelty']:.4f}"
        )

        print(
    f"  Trial {trial + 1:02d}/{args.num_trials:02d} | "
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

    result = {
        "guidance_scale": guidance_scale,
        "guidance_start": guidance_start,
        "num_trials": args.num_trials,
        "num_molecules": args.num_molecules,
        "eval_steps": args.eval_steps,
        "temperature": args.temperature,
        "solver": args.guidance_solver,
        "latent_transform": (
            transform
            .checkpoint_payload()["config"]
        ),
        "trials": trials,
        "summary": summarize_trials(trials),
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


def write_guidance_report(
    results: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "Flow matching evaluation with validity guidance",
        "=" * 78,
        "",
    ]

    for key, result in results.items():
        summary = result["summary"]
        lines.extend(
            [
                f"{key}",
                "-" * 50,
                f"lambda  : {result['guidance_scale']}",
                f"t_start : {result['guidance_start']}",
            ]
        )

        for metric_name in ("validity", "uniqueness", "novelty"):
            values = summary.get(metric_name)

            if values is None:
                lines.append(f"{metric_name:<11}: N/A")
            else:
                lines.append(
                    f"{metric_name:<11}: "
                    f"{values['mean']:.4f} ± {values['std']:.4f} "
                    f"(min={values['min']:.4f}, max={values['max']:.4f})"
                )

        fcd_value = result.get("fcd")
        if "fcd" in result:
            if fcd_value is None:
                lines.append(f"{'fcd':<11}: N/A")
            else:
                lines.append(
                    f"{'fcd':<11}: {fcd_value:.6f} "
                    f"(generated={result['fcd_generated_count']:,}, "
                    f"reference={result['fcd_reference_count']:,})"
                )

        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")



def evaluate_guidance_grid(
    args: argparse.Namespace,
    predictor_path: Path,
    output_dir: Path,
    device: torch.device,
) -> None:
    evaluation_dir = output_dir / args.guidance_evaluation_subdir
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    fm_model, latent_mean, latent_std, fm_metadata = load_fm_model(
        Path(args.fm_checkpoint),
        device,
    )
    predictor, predictor_metadata = load_validity_predictor(
        predictor_path,
        device,
    )
    transform = load_latent_transform_from_checkpoint(
    predictor_path,
    device,
)

    vae = load_vae(Path(args.vae_path), device)
    fcd_reference = (
        prepare_fcd_reference(args, device)
        if not args.skip_fcd
        else None
    )
    training_smiles = (
        None
        if args.skip_novelty
        else load_training_smiles(Path(args.training_smiles_path))
    )

    configurations = [(0.0, 0.0)]
    configurations.extend(
        (float(scale), float(start))
        for start in args.guidance_starts
        for scale in args.guidance_scales
        if float(scale) > 0.0
    )

    all_results: dict[str, Any] = {}

    for scale, start in configurations:
        key = guidance_config_key(scale, start)
        result_path = evaluation_dir / f"{key}.json"

        if result_path.is_file() and not args.overwrite_evaluation:
            cached_result = json.loads(
                result_path.read_text(encoding="utf-8")
            )
            if fcd_reference is None or "fcd" in cached_result:
                print(
                    "Result already present; skipping configuration: "
                    f"{key}"
                )
                all_results[key] = cached_result
                continue
            print(
                "Existing result has no FCD; recomputing configuration: "
                f"{key}"
            )

        print("\n" + "=" * 80)
        print(f"Evaluation: lambda={scale:g} | t_start={start:g}")
        print("=" * 80)
        result = evaluate_one_guidance_config(
            args=args,
            fm_model=fm_model,
            predictor=predictor,
            vae=vae,
            transform=transform,
            training_smiles=training_smiles,
            device=device,
            guidance_scale=scale,
            guidance_start=start,
            fcd_reference=fcd_reference,
        )
        atomic_json_dump(result, result_path)
        all_results[key] = result

        atomic_json_dump(
            {
                "fm_checkpoint": str(Path(args.fm_checkpoint)),
                "fm_metadata": fm_metadata,
                "predictor_path": str(predictor_path),
                "predictor_metadata": predictor_metadata,
                "guidance_results": all_results,
            },
            evaluation_dir / "guided_evaluation_summary.json",
        )
        write_guidance_report(
            all_results,
            evaluation_dir / "guided_evaluation_summary.txt",
        )

    print("\nGuided evaluation finished.")
    print(f"Results: {evaluation_dir}")

    del fm_model, predictor, vae, training_smiles, transform
    if fcd_reference is not None:
        del fcd_reference
    clear_memory()
