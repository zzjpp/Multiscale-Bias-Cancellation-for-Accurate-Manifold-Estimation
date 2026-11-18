"""Shared delegation for experiment-local fresh reproduction entrypoints."""

import argparse
import sys

from .catalog import BY_CODE
from .cli import main


def reproduce_experiment(code: str, argv: list[str] | None = None) -> None:
    """Run one fixed selection, preserving the root CLI's scientific protocol.

    No-argument invocation displays help rather than guessing an output path,
    loading removed historical data, or starting an expensive production run.
    """
    experiment = BY_CODE[code]
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(arg.split("=", 1)[0] in ("--experiment", "--list") for arg in arguments):
        parser = argparse.ArgumentParser()
        parser.error(
            "This entrypoint selects one experiment; use python -m manifold_experiments to select others."
        )
    if not arguments:
        print(f"{experiment.title} ({code}): fresh fitting, evaluation and plotting.\n")
        arguments = ["--help"]
    main(["--experiment", experiment.name, *arguments])
