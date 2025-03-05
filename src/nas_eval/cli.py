import argparse
import sys

from loguru import logger

from .timbre.cli import main as timbre_main
from .content.cli import main as content_main


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "eval",
        help="Evaluation type (loudness, pitch, timbre)",
        type=str,
    )

    if len(sys.argv) < 2:
        parser.print_help()
        logger.error("Please provide an evaluation type.")
        sys.exit(1)

    if sys.argv[1] == "timbre":
        return timbre_main(parser, sys.argv[1:])
    elif sys.argv[1] in ["loudness", "pitch"]:
        return content_main(parser, sys.argv[1:])
    else:
        _ = parser.parse_args(arguments)
        raise ValueError(
            f"Unknown evaluation type: {sys.argv[1]}. Must be one of ['loudness', 'pitch', 'timbre']."
        )


def cli_main():
    sys.exit(main(sys.argv[1:]))
