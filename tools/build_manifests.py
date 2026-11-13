"""Maintainer command: bind runtime dependencies separately from optional evidence."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".paper-memory",
    "tmp",
    "paper_run",
    "smoke_run",
    "new_raw",
    "new_figures",
    "replay_output",
}
ARCHIVED = {
    "data",
    "results",
    "analysis",
    "outputs",
    "imports",
    "figures",
    "paper_figures",
    "finite_sample",
    "finite_summary",
    "finite_pilot",
    "counterexample",
    "circle_check",
    "strict_circle_validation",
    "packets",
    "baseline",
    "ms",
    "reference_values",
}


def member(p):
    return dict(
        size=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()
    )


def main():
    files = [
        p
        for p in sorted(ROOT.rglob("*"))
        if p.is_file() and not EXCLUDE.intersection(p.relative_to(ROOT).parts)
    ]
    runtime = {}
    for p in files:
        rel = p.relative_to(ROOT)
        name = rel.as_posix()
        if p.name in (
            "SOURCE_MANIFEST.json",
            "RUN_SOURCE_MANIFEST.json",
            "PUBLICATION_QA.json",
        ):
            continue
        if ARCHIVED.intersection(rel.parts):
            continue
        if "tests" in rel.parts or name.startswith(
            ("illustrations/", ".github/", "docs/")
        ):
            continue
        if (
            p.suffix in (".py", ".json", ".txt", ".md")
            or p.name.upper().startswith("LICENSE")
            or name.startswith("experiments/bandwidth_robustness/assets/")
        ):
            runtime[name] = member(p)
    record = dict(
        schema="PAPER_RUNTIME_SOURCE_AND_DESIGN_V1",
        manuscript_sha256=json.loads((ROOT / "PAPER_REFERENCE.json").read_bytes())[
            "manuscript_pdf"
        ]["sha256"],
        description="Only source/configuration/registered E3 query-reference coordinates; no saved fitted points, tables or exports.",
        files=runtime,
    )
    # Backwards compatibility of the paper-reference field is explicit.
    (ROOT / "RUN_SOURCE_MANIFEST.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf8"
    )
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_bytes())
    files = [
        p
        for p in sorted(ROOT.rglob("*"))
        if p.is_file()
        and not EXCLUDE.intersection(p.relative_to(ROOT).parts)
        and p.name != "SOURCE_MANIFEST.json"
    ]
    manifest["files"] = {p.relative_to(ROOT).as_posix(): member(p) for p in files}
    (ROOT / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf8"
    )
    print(
        json.dumps(
            dict(
                runtime_files=len(runtime),
                runtime_bytes=sum(v["size"] for v in runtime.values()),
                distribution_files=len(files),
            )
        )
    )


if __name__ == "__main__":
    main()
