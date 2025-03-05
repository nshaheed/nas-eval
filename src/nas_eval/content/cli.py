"""
Content Preservation Evaluation
"""

import json
from pathlib import Path

import numpy as np
from loguru import logger

from ..common import (
    assemble_targets_list_single,
    infer_sample_rate,
    load_audio_files,
    validate_input_dirs,
)
from .core import ContentEvalBase, LoudnessEval, PitchEval


def run(
    folder_1: Path,
    folder_2: Path,
    eval: ContentEvalBase,
    num_samples: int = None,
    cache: bool = False,
):
    # Try to load saved features
    features_1 = load_saved_features(folder_1, eval.name) if cache else None
    audio_1 = None
    if features_1 is not None:
        logger.info(f"Loaded {len(features_1)} cached features from {folder_1}")
    else:
        # load the entire folder of audio files
        audio_1 = load_audio_files(folder_1, eval.sample_rate, num_samples)
        logger.info(f"Loaded {len(audio_1)} audio files from {folder_1}")

    # Try to load saved features
    features_2 = load_saved_features(folder_2, eval.name) if cache else None
    audio_2 = None
    if features_2 is not None:
        logger.info(f"Loaded {len(features_2)} cached features from {folder_2}")
    else:
        # load the entire folder of audio
        audio_2 = load_audio_files(folder_2, eval.sample_rate, num_samples)
        logger.info(f"Loaded {len(audio_2)} audio files from {folder_2}")

    # Run evaluations
    reults = eval(audio_1, audio_2, features_1, features_2)

    # Cache features
    if features_1 is None and cache:
        logger.info(f"Caching input features to {folder_1}")
        eval.save_input_features(folder_1, eval.name)

    if features_2 is None and cache:
        logger.info(f"Caching output features to {folder_2}")
        eval.save_output_features(folder_2, eval.name)

    return reults


def load_saved_features(folder: Path, suffix: str):
    features = {}
    for file in folder.glob(f"*{suffix}.npy"):
        key = file.name.replace(f"_{suffix}.npy", "")
        features[key] = np.load(file)
    return features if len(features) > 0 else None


def main(parser, arguments):
    parser.add_argument(
        "references",
        help="Path to reference audios",
        type=str,
    )
    parser.add_argument(
        "reconstructions",
        help="(For matrix test) Path to audio reconstructions",
        type=str,
    )
    parser.add_argument(
        "--matrix",
        help="Compute MMD matrix - specify input-target",
        type=str,
        default=None,
    )
    parser.add_argument("--sr", help="Sample rate", type=int, default=None)
    parser.add_argument(
        "--activation",
        help="Create an activation signal by rectifying the difference of the loudness",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--pitch",
        help="Pitch extraction method (crepe, pyin)",
        type=str,
        default="crepe",
    )
    parser.add_argument(
        "--device",
        help="Device to run pitch extraction on with crepe (cpu, cuda)",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--maxlen",
        help="Maximum length of audio samples per file to load (for testing)",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--cache",
        help="Cache features to disk",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--self",
        help="Compute self-comparison",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--output",
        help="Output file for results (json)",
        type=str,
        default=None,
    )
    args = parser.parse_args(arguments)

    # Load directories for testing
    if args.matrix is None:
        validate_input_dirs(args.references, args.reconstructions)
        inputs = [Path(args.references)]
        targets = [Path(args.reconstructions)]
        logger.info(
            f"Running {args.eval} evaluation between {args.references} and {args.reconstructions}"
        )
    else:
        inputs, targets = assemble_targets_list_single(
            args.matrix,
            test_audios_dir=args.references,
            reconstructions_dir=args.reconstructions,
        )

    # Use the sample rate from audio of first input if not specified
    if args.sr is None:
        sample_rate = infer_sample_rate(inputs[0])
    else:
        sample_rate = args.sr

    # Setup eval
    if args.eval == "loudness":
        eval = LoudnessEval(sample_rate=sample_rate, activation=args.activation)
    elif args.eval == "pitch":
        eval = PitchEval(
            sample_rate=sample_rate,
            method=args.pitch,
            device=args.device,
        )
    else:
        raise ValueError("Unknown eval type")

    # Add inputs to targets for self-comparison
    if args.self:
        targets = inputs + targets

    # Add inputs to targets for self-comparison
    logger.info(f"Running evaluation for {len(targets)} pairs")
    matrix = {}
    for input_dir in inputs:
        for target_dir in targets:
            if input_dir == target_dir:
                logger.info(f"Self comparison for {input_dir}")
            else:
                logger.info(f"Computing error between {input_dir} and {target_dir}")

            result, eval_order = run(
                Path(input_dir), Path(target_dir), eval, args.maxlen, args.cache
            )
            logger.info(f"Results: {result}")
            key = f"{Path(input_dir).name}->{Path(target_dir).parent.name}"
            matrix[key] = result
            matrix[key]["eval_order"] = eval_order

    # Save results to file
    if args.output is not None:
        outfile = Path(args.output).with_suffix(".json")
    elif args.matrix is not None:
        outfile = Path(f"{args.eval}_results_{args.matrix}.json")
    else:
        outfile = Path(f"{args.eval}_results.json")

    logger.info(f"Saving results to {outfile}")
    with open(outfile, "w") as outfile:
        json.dump(matrix, outfile, indent=2)
