"""
Modular pipeline for generative models in a VAE latent space.

Available stages:

* build_dataset       : builds a balanced validity dataset;
* train_predictor     : trains the p(validity | x_t, t) predictor;
* evaluate_guidance   : evaluates a standard FM model with gradient-based guidance;
* train_vfm           : trains an endpoint-based VFM with a diagonal Gaussian
  distribution q_theta(x_1 | x_t, t);
* evaluate_vfm        : evaluates the integrated VFM, with or without
  predictor-based endpoint reweighting;
* validity_all, vfm_all, and all combine multiple stages.

The script does not hard-code any path, dataset, or latent dimension. It can
operate in the raw VAE latent space, a standardized latent space, or a PCA
space. The exact transformation is stored in the predictor and VFM checkpoints.
It is reused consistently for dataset construction, guidance, VFM training,
and inverse transformation before VAE decoding.
"""


from __future__ import annotations

import argparse
from pathlib import Path

import torch
from utils import set_seed,  clear_memory, atomic_json_dump
from build_dataset_predictor_from_fm import build_balanced_dataset_from_fm
from predictor import  train_validity_predictor
from flow_matching import load_fm_model, train_flow_matching,evaluate_flow_matching
from naive_guidance_evaluation import evaluate_guidance_grid
from vfm import train_vfm, evaluate_vfm
from build_dataset_predictor_from_zero import build_predictor_dataset_gaussian

# ============================================================
# Arguments
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Modular pipeline for validity prediction, guided Flow Matching, "
            "and endpoint-based Variational Flow Matching in a VAE latent space."
        )
    )

    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "all_no_fm",
            "fm_all",
            "train_fm",
            "evaluate_fm",
            "validity_all",
            "build_dataset",
            "train_predictor",
            "evaluate_guidance",
            "vfm_all",
            "train_vfm",
            "evaluate_vfm",
        ),
        default="all",
        help=(
            "all runs the complete pipeline; validity_all runs the original "
            "guidance pipeline; vfm_all trains and evaluates only the VFM."
        ),
    )

    # Paths and general context
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fm_checkpoint", default=None)
    parser.add_argument("--fm_last_checkpoint",default=None,)
    parser.add_argument("--vae_path", default=None)
    parser.add_argument("--latents_path", default=None)
    parser.add_argument("--training_smiles_path", default=None)
    parser.add_argument(
        "--fcd_reference_smiles_path",
        default=None,
        help=(
            "Validation SMILES used as the real distribution for FCD. "
            "This is intentionally separate from --training_smiles_path, "
            "which is used for novelty."
        ),
    )
    parser.add_argument("--predictor_dataset_path", default=None)
    parser.add_argument("--positive_cache_path", default=None)
    parser.add_argument("--negative_cache_path", default=None)
    parser.add_argument(
        "--gaussian_cache_path",
        default=None,
        help=(
            "Checkpoint de progression du dataset Gaussian. Par défaut : "
            "<output_dir>/data_from_zero.pt."
        ),
    )
    parser.add_argument("--predictor_path", default=None)
    parser.add_argument("--vfm_checkpoint", default=None)
    parser.add_argument("--vfm_last_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )


# ============================================================
# Flow Matching architecture
# ============================================================

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--time_embedding_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--use_film",
        action=argparse.BooleanOptionalAction,
        default=True,
)
    
# ============================================================
# Flow Matching Batch OT
# ============================================================

    parser.add_argument(
        "--use_batch_ot",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--batch_ot_size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--batch_ot_reg",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--batch_ot_max_iterations",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--batch_ot_stop_threshold",
        type=float,
        default=1e-6,
    )
# ============================================================
# Flow Matching data
# ============================================================

    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="0 utilise tous les latents disponibles.",
    )

    parser.add_argument(
        "--val_size",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--stats_chunk_size",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

# ============================================================
# Flow Matching training
# ============================================================

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--amp_dtype",
        choices=(
            "auto",
            "bf16",
            "fp16",
            "none",
        ),
        default="auto",
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument("--fm_lr_restart_epoch", type=int, default=None,
                        help="Completed epoch at which a new cosine LR schedule starts (requires --resume).")
    parser.add_argument("--fm_lr_end_epoch", type=int, default=None,
                        help="Final epoch of the continuation LR schedule, independent of evaluation segments.")
    parser.add_argument("--fm_min_learning_rate", type=float, default=0.0)

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    # Predictor dataset

    parser.add_argument(
        "--predictor_dataset_source",
        choices=("fm", "gaussian"),
        default="fm",
    )

    parser.add_argument(
        "--dataset_size",
        type=int,
        default=200_000,
        help=(
            "Total dataset size, evenly split between positive and "
            "negative samples."
        ),
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        nargs="+",
        default=[1, 2, 5, 20],
        help="Numbers of evaluation steps.",
    )
    parser.add_argument(
        "--fm_paired_seeds",
        action="store_true",
        help="Reuse the same initial Gaussian noise per trial across FM step counts.",
    )
    parser.add_argument(
        "--fm_eval_solver",
        choices=("euler", "heun"),
        default="heun",
        help="ODE solver used for standard FM evaluation.",
    )
    parser.add_argument(
        "--fm_evaluation_subdir",
        default="flow_matching_evaluation",
        help=(
            "Subdirectory created inside the experiment directory for FM "
            "evaluation results. Use a distinct name to preserve earlier runs."
        ),
    )

    #####################################
    # Gaussian dataset predictor 
    ####################################

    parser.add_argument(
    "--negative_dataset_mode",
    choices=("local_global_gaussian",),
    default="local_global_gaussian",
    )

    parser.add_argument(
        "--local_gaussian_fraction",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--local_gaussian_sigma",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--local_gaussian_samples_per_positive",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--local_gaussian_pair_mode",
        choices=(
            "one_negative_per_source",
            "all_balanced",
        ),
        default="one_negative_per_source",
    )

    parser.add_argument(
        "--local_gaussian_positive_batch_size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--global_gaussian_batch_size",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--global_gaussian_std_scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max_noise_negative_candidates",
        type=int,
        default=10_000_000,
    )
    ##########################################
    # Predictor dataset from fm
    ##########################################

    parser.add_argument(
        "--positive_candidate_batch_size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--hard_negative_generation_batch_size",
        type=int,
        default=1024,
    )
    parser.add_argument("--decode_batch_size", type=int, default=256)
    parser.add_argument("--hard_negative_steps", type=int, default=20)
    parser.add_argument(
        "--sample_method",
        choices=("euler", "heun", "midpoint"),
        default="midpoint",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--dataset_save_every",
        type=int,
        default=10_000,
        help=(
            "Nombre de paires supplémentaires entre deux sauvegardes "
            "intermédiaires du dataset."
        ),
    )
    parser.add_argument(
        "--max_negative_candidates",
        type=int,
        default=10_000_000,
    )
    parser.add_argument("--overwrite_dataset", action="store_true")

    # Predictor training
    parser.add_argument("--predictor_hidden_dim", type=int, default=512)
    parser.add_argument("--predictor_num_hidden_layers", type=int, default=3)
    parser.add_argument("--predictor_dropout", type=float, default=0.0)
    parser.add_argument("--predictor_epochs", type=int, default=100)
    parser.add_argument("--predictor_batch_size", type=int, default=1024)
    parser.add_argument("--predictor_learning_rate", type=float, default=3e-4)
    parser.add_argument("--predictor_weight_decay", type=float, default=1e-5)
    parser.add_argument("--predictor_grad_clip", type=float, default=1.0)
    parser.add_argument("--predictor_patience", type=int, default=15)
    parser.add_argument("--validation_fraction", type=float, default=0.10)
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction reserved for a separate predictor test set. The default "
            "is zero to reproduce the original 90/10 train/validation split."
        ),
    )
    parser.add_argument(
        "--predictor_test_times",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6, 0.8, 1.0],
    )
    parser.add_argument("--overwrite_predictor", action="store_true")

    # Evaluation of the standard FM with guidance
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--num_molecules", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument(
        "--guidance_scales",
        type=float,
        nargs="+",
        default=[1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0],
    )
    parser.add_argument(
        "--guidance_starts",
        type=float,
        nargs="+",
        default=[0.4, 0.6, 0.8],
    )
    parser.add_argument(
        "--guidance_solver",
        choices=("euler", "midpoint"),
        default="midpoint",
        help="ODE solver used for guided FM evaluation.",
    )
    parser.add_argument(
        "--guidance_evaluation_subdir",
        default="guidance_evaluation",
        help=(
            "Subdirectory created inside the experiment directory for guided "
            "FM results. Use a distinct name to preserve earlier runs."
        ),
    )
    parser.add_argument("--skip_novelty", action="store_true")
    parser.add_argument("--overwrite_evaluation", action="store_true")

    # Latent-space transformation for the VFM
    parser.add_argument(
        "--latent_transform",
        choices=("none", "standardize"),
        default="none",
        help=(
            "Transformation applied before the VFM. Use standardize for "
            "normalized PC16 latent vectors."
        ),
    )
    parser.add_argument(
        "--transform_path",
        default=None,
        help=(
            "Checkpoint containing mean/std for standardize mode, or a .pkl "
            "file containing scaler and pca for PCA mode."
        ),
    )
    parser.add_argument(
        "--transform_statistics_batch_size",
        type=int,
        default=8192,
    )

    # VFM training
    parser.add_argument("--vfm_hidden_dim", type=int, default=512)
    parser.add_argument("--vfm_num_layers", type=int, default=4)
    parser.add_argument("--vfm_dropout", type=float, default=0.0)
    parser.add_argument("--vfm_min_logvar", type=float, default=-6.0)
    parser.add_argument("--vfm_max_logvar", type=float, default=2.0)
    parser.add_argument("--vfm_epochs", type=int, default=500)
    parser.add_argument("--vfm_batch_size", type=int, default=512)
    parser.add_argument("--vfm_eval_batch_size", type=int, default=512)
    parser.add_argument("--vfm_learning_rate",type=float,default=3e-4,)
    parser.add_argument("--vfm_min_learning_rate",type=float,default=1e-5,)
    parser.add_argument("--vfm_weight_decay", type=float, default=1e-5)
    parser.add_argument("--vfm_grad_clip", type=float, default=1.0)
    parser.add_argument("--vfm_patience", type=int, default=20)
    parser.add_argument(
        "--vfm_validation_fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--vfm_max_samples",
        type=int,
        default=0,
        help="0 means that all available latent vectors are used.",
    )
    parser.add_argument(
        "--vfm_split_mode",
        choices=("contiguous", "random"),
        default="random",
    )
    parser.add_argument(
        "--vfm_shuffle_train",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--vfm_drop_last",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--vfm_validity_weight",
        type=float,
        default=0.1,
        help=(
            "Weight of the -log p(valid) term added to the Gaussian NLL. "
            "Set it to 0 to train a VFM without integrated validity."
        ),
    )
    parser.add_argument(
        "--vfm_validity_samples",
        type=int,
        default=4,
        help=(
            "Number of reparameterized endpoints used for each training sample."
        ),
    )
    parser.add_argument("--vfm_eps_time", type=float, default=1e-4)
    parser.add_argument("--vfm_resume_checkpoint", default=None)
    parser.add_argument("--overwrite_vfm", action="store_true")

    # VFM sampling and evaluation
    parser.add_argument("--vfm_num_trials", type=int, default=20)
    parser.add_argument("--vfm_num_molecules", type=int, default=1024)
    parser.add_argument("--vfm_eval_steps", type=int, default=20)
    parser.add_argument(
        "--vfm_evaluation_subdir",
        default="vfm_evaluation",
        help=(
            "Subdirectory created inside the experiment directory for VFM "
            "evaluation results. Use a distinct name to preserve earlier runs."
        ),
    )
    parser.add_argument("--vfm_temperature", type=float, default=1.0)
    parser.add_argument(
        "--vfm_eval_solver",
        choices=("euler", "heun", "midpoint"),
        default="euler",
        help=(
            "ODE solver used for VFM evaluation. The Euler implementation "
            "exactly reproduces the original endpoint-VFM evaluation loop."
        ),
    )
 
    parser.add_argument(
        "--vfm_eval_include_mean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Evaluates the integrated VFM without using the predictor "
            "during sampling."
        ),
    )
    parser.add_argument(
        "--vfm_eval_reweight",
        action="store_true",
        help=(
            "Adds configurations that reweight multiple endpoint proposals "
            "using the predictor during sampling."
        ),
    )
    parser.add_argument(
        "--vfm_endpoint_counts",
        type=int,
        nargs="+",
        default=[8, 16, 32],
    )
    parser.add_argument(
        "--vfm_betas",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 5.0],
    )
    parser.add_argument(
        "--skip_fcd",
        action="store_true",
        help="Skip Fréchet ChemNet Distance during model evaluations.",
    )

    parser.add_argument(
        "--fcd_num_reference",
        type=int,
        default=0,
        help="Maximum FCD reference count; 0 uses the full validation split.",
    )
    parser.add_argument(
        "--fcd_num_generated",
        type=int,
        default=0,
        help="Maximum unique generated SMILES for FCD; 0 uses all of them.",
    )

    parser.add_argument(
        "--fcd_batch_size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--fcd_num_workers",
        type=int,
        default=1,
    )

    parser.add_argument("--overwrite_vfm_evaluation", action="store_true")

    return parser.parse_args()


def stage_is_requested(args: argparse.Namespace, stage: str) -> bool:
    if args.stage == "all":
        return True
    
    if args.stage == "all_no_fm":
        return stage in {
            "build_dataset",
            "train_predictor",
            "vfm_all",
            "train_vfm",
            "evaluate_vfm",
        }
    
    if args.stage == "fm_all":
        return stage in {
            "train_fm",
            "evaluate_fm",
        }

    if args.stage == "validity_all":
        return stage in {
            "build_dataset",
            "train_predictor",
            "evaluate_guidance",
        }

    if args.stage == "vfm_all":
        return stage in {"train_vfm", "evaluate_vfm"}

    return args.stage == stage


def resolve_device(device_argument: str) -> torch.device:
    if device_argument == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if device_argument == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but CUDA/ROCm is unavailable."
        )

    return torch.device(device_argument)


# ============================================================
# Main program
# ============================================================


def main() -> None:
    args = parse_args()

    if stage_is_requested(args, "evaluate_fm"):
        fm_evaluation_subdir = Path(args.fm_evaluation_subdir)
        if (
            fm_evaluation_subdir.is_absolute()
            or len(fm_evaluation_subdir.parts) != 1
            or fm_evaluation_subdir.name in {"", ".", ".."}
        ):
            raise ValueError(
                "--fm_evaluation_subdir must be a single relative directory "
                "name."
            )

    if stage_is_requested(args, "evaluate_guidance"):
        guidance_evaluation_subdir = Path(args.guidance_evaluation_subdir)
        if (
            guidance_evaluation_subdir.is_absolute()
            or len(guidance_evaluation_subdir.parts) != 1
            or guidance_evaluation_subdir.name in {"", ".", ".."}
        ):
            raise ValueError(
                "--guidance_evaluation_subdir must be a single relative "
                "directory name."
            )

    if stage_is_requested(args, "evaluate_vfm"):
        evaluation_subdir = Path(args.vfm_evaluation_subdir)
        if (
            evaluation_subdir.is_absolute()
            or len(evaluation_subdir.parts) != 1
            or evaluation_subdir.name in {"", ".", ".."}
        ):
            raise ValueError(
                "--vfm_evaluation_subdir must be a single relative "
                "directory name."
            )

        if not args.skip_fcd:
            if not args.fcd_reference_smiles_path:
                raise ValueError(
                    "VFM FCD evaluation requires "
                    "--fcd_reference_smiles_path pointing to validation SMILES."
                )
            fcd_reference_path = Path(args.fcd_reference_smiles_path)
            if not fcd_reference_path.is_file():
                raise FileNotFoundError(
                    f"FCD validation SMILES not found: {fcd_reference_path}"
                )

        if not args.skip_novelty:
            if not args.training_smiles_path:
                raise ValueError(
                    "VFM novelty evaluation requires "
                    "--training_smiles_path pointing to training SMILES."
                )
            training_smiles_path = Path(args.training_smiles_path)
            if not training_smiles_path.is_file():
                raise FileNotFoundError(
                    f"Training SMILES not found: {training_smiles_path}"
                )

    set_seed(args.seed)

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    
    fm_best_path = (
        Path(args.fm_checkpoint)
        if args.fm_checkpoint
        else output_dir / "flow_matching_best.pt"
    )
    args.fm_checkpoint = str(fm_best_path)

    fm_last_path = (
        Path(args.fm_last_checkpoint)
        if args.fm_last_checkpoint
        else output_dir / "flow_matching_last.pt"
    )

    dataset_path = (
        Path(args.predictor_dataset_path)
        if args.predictor_dataset_path
        else output_dir / f"validity_endpoint_dataset_{args.dataset_size}.pt"
    )
    positive_cache_path = (
        Path(args.positive_cache_path)
        if args.positive_cache_path
        else output_dir / "valid_positive_endpoints_partial.pt"
    )
    negative_cache_path = (
        Path(args.negative_cache_path)
        if args.negative_cache_path
        else output_dir / "hard_negative_endpoints_partial.pt"
    )
    cache_path = (
        Path(args.gaussian_cache_path)
        if args.gaussian_cache_path
        else output_dir / "data_from_zero.pt"
    )
    predictor_path = (
        Path(args.predictor_path)
        if args.predictor_path
        else output_dir / "validity_predictor_best.pt"
    )
    vfm_checkpoint_path = (
        Path(args.vfm_checkpoint)
        if args.vfm_checkpoint
        else output_dir / "vfm_endpoint_best.pt"
    )
    vfm_last_checkpoint_path = (
        Path(args.vfm_last_checkpoint)
        if args.vfm_last_checkpoint
        else output_dir / "vfm_endpoint_last.pt"
    )

    print(f"Device          : {device}")
    print(f"Stage           : {args.stage}")
    print(f"Dataset         : {args.dataset}")
    print(f"Output dir      : {output_dir}")
    print(f"FM checkpoint   : {args.fm_checkpoint}")
    print(f"Predictor       : {predictor_path}")
    print(f"VFM checkpoint  : {vfm_checkpoint_path}")

    atomic_json_dump(
        {
            **vars(args),
            "resolved_predictor_dataset_path": str(dataset_path),
            "resolved_positive_cache_path": str(positive_cache_path),
            "resolved_negative_cache_path": str(negative_cache_path),
            "resolved_gaussian_cache_path": str(cache_path),
            "resolved_predictor_path": str(predictor_path),
            "resolved_vfm_checkpoint": str(vfm_checkpoint_path),
            "resolved_vfm_last_checkpoint": str(vfm_last_checkpoint_path),
        },
        output_dir / "pipeline_config.json",
    )
    if stage_is_requested(args, "train_fm"):
        fm_best_path = train_flow_matching(
        args=args,
        output_dir=output_dir,
        best_path=fm_best_path,
        last_path=fm_last_path,
        device=device,
        )

    if stage_is_requested(args,"evaluate_fm",):

        evaluate_flow_matching(
            args=args,
            checkpoint_path=fm_best_path,
            output_dir=output_dir,
            device=device,
        )

    if stage_is_requested(args, "build_dataset"):
        if args.predictor_dataset_source == "fm":
            fm_model, fm_mean, fm_std, fm_metadata = (load_fm_model(fm_best_path,torch.device("cpu"),))

            fm_train_count = (
                fm_metadata
                .get("training_config", {})
                .get("train_count")
            )

            predictor_dataset_path=build_balanced_dataset_from_fm(
                args=args,
                dataset_path=dataset_path,
                positive_cache_path=positive_cache_path,
                negative_cache_path=negative_cache_path,
                device=device,
                fm_model=fm_model,
                fm_mean=fm_mean,
                fm_std=fm_std,
                fm_train_count=fm_train_count,
            )

            del fm_model
            del fm_mean
            del fm_std
            del fm_metadata
            clear_memory()

        elif args.predictor_dataset_source == "gaussian":
            predictor_dataset_path=build_predictor_dataset_gaussian(
                args=args,
                dataset_path=dataset_path,
                dataset=args.dataset,
                cache_path=cache_path,
                device=device,
            )

    if stage_is_requested(args, "train_predictor"):
        train_validity_predictor(
            args=args,
            dataset_path=predictor_dataset_path,
            predictor_path=predictor_path,
            output_dir=output_dir,
            device=device,
        )

    if stage_is_requested(args, "evaluate_guidance"):
        evaluate_guidance_grid(
            args=args,
            predictor_path=predictor_path,
            output_dir=output_dir,
            device=device,
        )

    if stage_is_requested(args, "train_vfm"):
        train_vfm(
            args=args,
            predictor_path=predictor_path,
            vfm_checkpoint_path=vfm_checkpoint_path,
            vfm_last_checkpoint_path=vfm_last_checkpoint_path,
            output_dir=output_dir,
            device=device,
        )

    if stage_is_requested(args, "evaluate_vfm"):
        evaluate_vfm(
            args=args,
            predictor_path=(
                predictor_path
                if predictor_path.is_file()
                else None
            ),
            vfm_checkpoint_path=vfm_checkpoint_path,
            output_dir=output_dir,
            device=device,
        )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
