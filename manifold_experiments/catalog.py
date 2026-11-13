"""Public names and manuscript locations for the seven experiment selections.

E1--E5 are stable wire identifiers in task packets. Human-facing names follow
the paper; aliases are resolved before planning so they cannot change seeds,
job IDs, iteration order, or the numerical protocol.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
    """An experiment's public name, paper heading, and implementation directory."""

    code: str
    name: str
    title: str
    section: str
    directory: str
    figures: str


EXPERIMENT_CATALOG = (
    Experiment(
        "E1",
        "population",
        "Population Verification",
        "5.1; E.1; E.7",
        "experiments/population_verification",
        "2, 4, E.1–E.3",
    ),
    Experiment(
        "E2",
        "fixed-scales",
        "Finite-Sample Behavior at Fixed Scales",
        "5.2; E.2",
        "experiments/finite_sample_fixed_scales",
        "5, E.4–E.6",
    ),
    Experiment(
        "E3",
        "bandwidth",
        "Bandwidth Robustness and Reverse Coverage",
        "5.3; E.3",
        "experiments/bandwidth_robustness",
        "6, E.7–E.12",
    ),
    Experiment(
        "E4",
        "correspondence",
        "Correspondence Accuracy and Computational Stability",
        "5.4; E.3.1; E.6.3",
        "experiments/correspondence_accuracy",
        "7, E.13",
    ),
    Experiment(
        "E5",
        "directional-ablation",
        "Directional Ablation of the YY Estimator",
        "5.5; E.4; E.6.3",
        "experiments/yy_directional_ablation",
        "8, E.14; Table 2",
    ),
    Experiment(
        "nonuniform_E1",
        "nonuniform-population",
        "Multiscale Bias Cancellation for STSP under Nonuniform Sampling — Population",
        "E.5",
        "experiments/nonuniform/population",
        "E.15–E.16",
    ),
    Experiment(
        "nonuniform_E2",
        "nonuniform-fixed-scales",
        "Multiscale Bias Cancellation for STSP under Nonuniform Sampling — Finite Sample",
        "E.5",
        "experiments/nonuniform/fixed_scales",
        "E.17–E.18",
    ),
)

BY_CODE = {experiment.code: experiment for experiment in EXPERIMENT_CATALOG}
BY_NAME = {experiment.name: experiment for experiment in EXPERIMENT_CATALOG}
SELECTIONS = tuple(BY_NAME) + tuple(BY_CODE) + ("all",)

# Display names are deliberately separate from native packet fields such as
# "Multi" or "ENF_Multi_metrics". Those fields retain their scientific meaning.
METHOD_LABELS = {
    "STSP": "Successive tangent-space projection (STSP)",
    "DR": "Gaussian density-ridge fitting (DR)",
    "YY": "YY estimator",
    "GBC-STSP": "GBC-STSP",
    "GBC-DR": "GBC-DR",
    "GBC-YY": "GBC-YY",
    "MS-STSP": "MS-STSP (Xia et al., 2026)",
}


def resolve_selections(selections: list[str]) -> tuple[str, ...]:
    """Resolve public names to the original ordered, duplicate-free job codes."""
    if "all" in selections:
        return tuple(BY_CODE)
    result = []
    for selection in selections:
        experiment = BY_NAME.get(selection) or BY_CODE.get(selection)
        if experiment is None:
            raise ValueError(f"Unknown experiment: {selection}")
        if experiment.code not in result:
            result.append(experiment.code)
    return tuple(result)


def experiment_directory(code: str, root: Path) -> Path:
    """Resolve one current implementation path without legacy folder aliases."""
    return root / BY_CODE[code].directory


def describe_experiments() -> str:
    """Return the command-line experiment guide without importing any fitter."""
    return "\n\n".join(
        f"{item.name} ({item.code}) — {item.title}\n"
        f"  Sections {item.section}; Figures {item.figures}\n"
        f"  {item.directory}/"
        for item in EXPERIMENT_CATALOG
    )
