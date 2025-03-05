"""
Command line interface for timbre evaluation
"""

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..common import assemble_targets_list_single, infer_sample_rate, load_audio_files
from .mmd import compute_mfcc_features, MMD


def run(folder_1: Path, folder_2: Path, sample_rate: int):
    # load the entire folder of audio files
    audio_1 = load_audio_files(folder_1, sample_rate).values()
    logger.info(f"Loaded {len(audio_1)} audio files from {folder_1}")

    audio_2 = load_audio_files(folder_2, sample_rate).values()
    logger.info(f"Loaded {len(audio_2)} audio files from {folder_2}")

    # compute audio features for each audio file
    features_1 = compute_mfcc_features(audio_1, sample_rate)
    features_2 = compute_mfcc_features(audio_2, sample_rate)

    logger.info(f"Extracted {features_1.shape[0]} texture windows from {folder_1}")
    logger.info(f"Extracted {features_2.shape[0]} texture windows from {folder_2}")

    mmd = MMD()
    distance = mmd(features_1, features_2)
    logger.info(f"MMD: {distance}")
    return [
        distance,
    ]


def main(parser, arguments):
    parser.add_argument(
        "references",
        help="Path to reference audios",
        type=str,
    )
    parser.add_argument(
        "reconstructions",
        help="Path to audio reconstructions",
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
        "--plot",
        help="Plot the MMD matrix. Provide a filename for output PNG file of plot.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save",
        help="Save the MMD matrix. Provide a filename for output NPY file of matrix.",
        type=str,
        default=None,
    )
    args = parser.parse_args(arguments)

    # Load directories for testing
    if args.matrix is None:
        inputs = [Path(args.references)]
        targets = [Path(args.reconstructions)]
        logger.info(
            f"Running timbre evaluation between {args.references} and {args.reconstructions}"
        )
    else:
        inputs, targets = assemble_targets_list_single(
            args.matrix,
            test_audios_dir=args.references,
            reconstructions_dir=args.reconstructions,
            include_output=True,
        )

    # Use the sample rate from audio of first input if not specified
    if args.sr is None:
        sample_rate = infer_sample_rate(inputs[0])
    else:
        sample_rate = args.sr

    # Add inputs to targets for self-comparison
    if args.matrix is not None:
        targets = inputs + targets

    logger.info(f"Running timbre evaluation for {len(targets)} pairs")
    matrix = np.zeros((len(inputs), len(targets)))
    for input_dir in inputs:
        for target_dir in targets:
            if input_dir == target_dir:
                logger.info(f"Self comparison for {input_dir}")
            else:
                logger.info(f"Computing MMD loss between {input_dir} and {target_dir}")

            result = run(Path(input_dir), Path(target_dir), sample_rate)
            matrix[inputs.index(input_dir), targets.index(target_dir)] = np.mean(result)

    logger.info(f"MMD matrix: \n{matrix}")

    # Plot matrix
    if args.plot is not None:
        fig, ax1 = plt.subplots(figsize=(1.7 * len(targets), 8))
        xticks = [f"{Path(t).parent.name}.{Path(t).name}" for t in targets]
        yticks = [f"{Path(t).name}" for t in inputs]
        sns.heatmap(matrix, annot=True, xticklabels=xticks, yticklabels=yticks, ax=ax1)

        for ind, col in enumerate(matrix.T):
            min_row = np.argmin(col)
            ax1.add_patch(
                plt.Rectangle(
                    (ind, min_row), 1, 1, fc="none", ec="skyblue", lw=5, clip_on=False
                )
            )

        fig.tight_layout()

        outfile = Path(args.plot).with_suffix(".png")
        fig.savefig(outfile, dpi=300)

    # Cached matrix
    if args.save is not None:
        outfile = Path(args.save).with_suffix(".npy")
        np.save(outfile, matrix)
