"""Every public reproduce.py must reach the fresh pipeline without archives."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manifold_experiments.catalog import EXPERIMENT_CATALOG
from manifold_experiments.protocol import plan


@pytest.mark.parametrize("experiment", EXPERIMENT_CATALOG, ids=lambda item: item.name)
def test_standalone_no_arguments_is_safe_help(experiment, tmp_path):
    done = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            str(ROOT / experiment.directory / "reproduce.py"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf8",
    )
    assert done.returncode == 0, done.stderr
    assert "--output" in done.stdout and experiment.title in done.stdout
    assert "FileNotFoundError" not in done.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("experiment", EXPERIMENT_CATALOG, ids=lambda item: item.name)
def test_standalone_writes_the_exact_fresh_plan(experiment, tmp_path):
    destination = tmp_path / "new_run"
    done = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            str(ROOT / experiment.directory / "reproduce.py"),
            "--output",
            str(destination),
            "--plan-only",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf8",
    )
    assert done.returncode == 0, done.stderr
    actual = json.loads((destination / "PLAN.json").read_bytes())
    expected = plan((experiment.code,), "paper", "cpu")
    actual.pop("source_manifest_sha256")
    assert actual == expected
    assert {item.name for item in destination.iterdir()} == {"PLAN.json"}


@pytest.mark.parametrize(
    "directory",
    [
        "experiments/finite_sample_fixed_scales",
        "experiments/bandwidth_robustness",
        "experiments/nonuniform/fixed_scales",
    ],
)
def test_plotting_only_cli_requires_explicit_table(directory, tmp_path):
    done = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            str(ROOT / directory / "render_figures.py"),
            "--output",
            str(tmp_path / "plots"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf8",
    )
    assert done.returncode == 2
    assert "--table" in done.stderr and "FileNotFoundError" not in done.stderr
    assert not (tmp_path / "plots").exists()


def test_workflow_temp_directory_has_existing_checkout_parent():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text("utf8")
    assert "--basetemp ./pytest_tmp" in workflow
    assert "--basetemp ./tmp/pytest" not in workflow
    assert "windows-latest" in workflow and "ubuntu-latest" in workflow


def test_test_outputs_are_not_distributed_runtime_inputs():
    for filename in ("RUN_SOURCE_MANIFEST.json", "SOURCE_MANIFEST.json"):
        members = json.loads((ROOT / filename).read_bytes())["files"]
        assert not any("pytest_tmp" in Path(name).parts for name in members)
        assert "pytest-results.xml" not in members
