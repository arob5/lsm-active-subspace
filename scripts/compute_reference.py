"""Compute a reference posterior and store it in ``baselines/``.

On this machine:

    uv run python scripts/compute_reference.py nuts toy fixed
    uv run python scripts/compute_reference.py rwmh harvard_forest_small fixed

On a cluster, the RWMH steps run as separate jobs: one pilot, then each segment of each chain as an
array task (segment k of a chain needs segment k - 1 of that chain), then the diagnostics and the
finish:

    uv run python scripts/compute_reference.py pilot harvard_forest inferred RUN_DIR --chains 64
    uv run python scripts/compute_reference.py segment RUN_DIR CHAIN SEGMENT --steps 2000
    uv run python scripts/compute_reference.py diagnostics RUN_DIR
    uv run python scripts/compute_reference.py finish RUN_DIR

The run's files live in RUN_DIR (``runs/reference/<problem>/<scales>`` by default for
``rwmh``), so an interrupted run continues where it stopped.
"""

import argparse
from pathlib import Path

from lsmas.experiments.reference import (
    finish_rwmh_reference,
    nuts_reference,
    run_rwmh_segment,
    rwmh_diagnostics,
    rwmh_reference,
    start_rwmh_reference,
)
from lsmas.problems import load

DEFAULT_RUN_ROOT = Path(__file__).resolve().parents[1] / "runs" / "reference"


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "nuts":
        reference = nuts_reference(
            load(arguments.problem, scales=arguments.scales), seed=arguments.seed
        )
        print(f"saved {reference.save()}")
    elif arguments.command == "rwmh":
        run_directory = (
            arguments.run_directory or DEFAULT_RUN_ROOT / arguments.problem / arguments.scales
        )
        reference = rwmh_reference(
            load(arguments.problem, scales=arguments.scales),
            run_directory,
            n_chains=arguments.chains,
            segment_steps=arguments.steps,
            max_segments=arguments.max_segments,
            seed=arguments.seed,
        )
        print(reference.diagnostics.describe().loc[["min", "max"]])
        print(f"saved {reference.save()}")
    elif arguments.command == "pilot":
        start_rwmh_reference(
            load(arguments.problem, scales=arguments.scales),
            arguments.run_directory,
            n_chains=arguments.chains,
            seed=arguments.seed,
        )
    elif arguments.command == "segment":
        print(
            run_rwmh_segment(
                arguments.run_directory, arguments.chain, arguments.segment, arguments.steps
            )
        )
    elif arguments.command == "diagnostics":
        print(rwmh_diagnostics(arguments.run_directory).to_string())
    elif arguments.command == "finish":
        print(f"saved {finish_rwmh_reference(arguments.run_directory).save()}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    nuts = commands.add_parser("nuts", help="a NUTS reference, for a differentiable problem")
    rwmh = commands.add_parser(
        "rwmh", help="an RWMH reference on this machine, one process per chain"
    )
    pilot = commands.add_parser("pilot", help="the RWMH run's preliminary EKI run")
    for command in (nuts, rwmh, pilot):
        command.add_argument("problem")
        command.add_argument("scales", choices=["fixed", "inferred"])
        command.add_argument("--seed", type=int, default=0)
    for command in (rwmh, pilot):
        command.add_argument("--chains", type=int, default=8)
    rwmh.add_argument("--run-directory", type=Path, default=None)
    rwmh.add_argument("--steps", type=int, default=2000, help="steps per segment")
    rwmh.add_argument("--max-segments", type=int, default=50)
    pilot.add_argument("run_directory", type=Path)
    segment = commands.add_parser("segment", help="one segment of one RWMH chain")
    segment.add_argument("run_directory", type=Path)
    segment.add_argument("chain", type=int)
    segment.add_argument("segment", type=int)
    segment.add_argument("--steps", type=int, default=2000)
    for name, help_text in (
        ("diagnostics", "the run's convergence so far"),
        ("finish", "store the reference"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("run_directory", type=Path)
    return parser


if __name__ == "__main__":
    main()
