"""Names/layout must not change the scientific jobs or hide old source changes."""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manifold_experiments.catalog import (
    EXPERIMENT_CATALOG,
    describe_experiments,
    resolve_selections,
)
from manifold_experiments.protocol import EXPERIMENTS, jobs, plan


@pytest.mark.parametrize("experiment", EXPERIMENT_CATALOG, ids=lambda item: item.name)
def test_paper_name_keeps_ordered_jobs_and_ids(experiment):
    named = resolve_selections([experiment.name])
    coded = resolve_selections([experiment.code])
    assert named == coded == (experiment.code,)
    assert plan(named, "paper", "cpu") == plan(coded, "paper", "cpu")
    assert jobs(named, "smoke") == jobs(coded, "smoke")
    assert (ROOT / experiment.directory / "README.md").is_file()


def test_mixed_aliases_deduplicate_without_reordering():
    assert resolve_selections(["E3", "population", "bandwidth", "E1"]) == ("E3", "E1")
    assert resolve_selections(["all"]) == EXPERIMENTS
    with pytest.raises(ValueError):
        resolve_selections(["unknown-experiment"])


def test_module_and_compatibility_cli_emit_identical_plans(tmp_path):
    outputs = [tmp_path / "named", tmp_path / "coded"]
    commands = [
        ["-m", "manifold_experiments", "--experiment", "population"],
        [str(ROOT / "run_paper.py"), "--experiment", "E1"],
    ]
    for arguments, output in zip(commands, outputs):
        done = subprocess.run(
            [
                sys.executable,
                "-X",
                "utf8",
                "-B",
                *arguments,
                "--output",
                str(output),
                "--plan-only",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf8",
        )
        assert done.returncode == 0, done.stderr
        assert {path.name for path in output.iterdir()} == {"PLAN.json"}
    assert json.loads((outputs[0] / "PLAN.json").read_bytes()) == json.loads(
        (outputs[1] / "PLAN.json").read_bytes()
    )


def test_list_is_read_only_and_needs_no_numerical_imports():
    script = (
        "import sys; from manifold_experiments.cli import main; "
        "sys.argv=['manifold_experiments','--list']; main(); "
        "assert 'numpy' not in sys.modules; "
        "assert 'manifold_experiments.worker' not in sys.modules"
    )
    done = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf8",
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == describe_experiments()


def test_original_scientific_sources_are_bound_at_their_new_locations():
    preservation = json.loads((ROOT / "docs/SCIENTIFIC_PRESERVATION.json").read_bytes())
    assert len(preservation["files"]) == 329
    for member in preservation["files"]:
        current = ROOT / member["current_path"]
        assert current.is_file(), member
        assert (
            hashlib.sha256(current.read_bytes()).hexdigest() == member["sha256"]
        ), member


def test_figure_map_binds_current_paper_and_separates_include_paths():
    paper = json.loads((ROOT / "PAPER_REFERENCE.json").read_bytes())
    figures = json.loads((ROOT / "docs/FIGURE_MAP.json").read_bytes())
    assert figures["manuscript_sha256"] == paper["manuscript_pdf"]["sha256"]
    assert len(figures["figures"]) == 26
    for item in figures["figures"]:
        assert 1 <= item["pdf_page"] <= paper["manuscript_pdf"]["pages"]
        assert item["manuscript_include_paths"]
        if item["experiment"] == "schematic":
            assert all((ROOT / source).is_file() for source in item["source_files"])
        else:
            assert (ROOT / item["source_directory"]).is_dir()
            assert all(name.startswith("RUN/") for name in item["generated_files"])
            assert " -m manifold_experiments " in item["command"]


@pytest.mark.parametrize(
    "name",
    [
        "README.md",
        "README.zh-CN.md",
        "CONTRIBUTING.md",
        "docs/CODE_STRUCTURE.md",
        "docs/EXPERIMENTS.md",
        "docs/REPRODUCIBILITY.md",
    ],
)
def test_public_documentation_links_resolve(name):
    document = ROOT / name
    for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", document.read_text("utf8")):
        if "://" in link or link.startswith("#"):
            continue
        target = document.parent / unquote(link.split("#")[0])
        assert target.exists(), (name, link)
