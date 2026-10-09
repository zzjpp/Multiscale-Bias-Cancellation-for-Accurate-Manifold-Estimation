# Manifold-fitting simulation experiments (paper 62)

This release contains only experiments **E1–E6** of `manifold_fitting (62).pdf`.
The former standalone E7 experiment is not distributed. Corrugated torus is
still the sixth geometry in E1–E3. Results are not filtered to remove failures.
No server accounts, authentication files, remote schedulers or private research
archives are required by the public entry points.

## Start here

Use Python 3.12.9 in a separate environment for **saved-result figure replay**:

```bash
python -m pip install -r requirements.txt
python validate_release.py
python -m pytest -q tests
python reproduce_all.py --output ./paper_figures
```

The last command uses the bundled numerical tables (and E5 selected exports),
not new fits. Output must be a fresh directory. It produces the paper figure
set, including all **12 E3 figures** with six panels each, without green
backgrounds or small footer text. YY's green `Single(1.6r1)` curve is retained.
See [docs/FIGURE_MAP.md](docs/FIGURE_MAP.md) for figure numbers.
The existing `E3/paper_figures/` include paths are retained with the exact
selected 24 PNG/PDF bytes; they are not replaced by the diagnostic 0.8 version.

Small **actual CPU computations**, separate from paper reproduction:

```bash
python smoke_all.py
```

Smoke tests use explicitly reduced sample/query/quadrature budgets; they do
not verify full-grid accuracy, GPU equivalence or identical scientific reruns.

## Numerical reruns

Each `E*/README.md` documents the original computation and a local entry point.
E2/E3 production GPU runs require **Linux Python 3.12.7, NumPy 2.1.3, SciPy
1.18.1, CuPy 14.2.0, CUDA runtime 12.9, NVIDIA L40**, float64 and one CPU
thread per worker. No untested CPU, true-normal, basis-repair or enlarged-radius
fallback is enabled. Use `requirements-gpu.txt` in that separate environment.
E6's CPU fitting environment has its own original lock. Figure requirements
must not be substituted for fitting requirements.

Rerun outputs are fresh files/directories. Public E2/E3 runners perform new
same-input CPU/GPU gates, preserve every estimator outcome, and write an
explicit new-run schema; they do **not** fabricate the historical queue,
archival receipts or pre-existing scientific PASS records. Frozen solvers and
registered coordinates are supplied. Full historical E2/E3 raw packets and
private composite execution histories are not included: saved-table replay is
complete, but this is **not** a closed offline replay of every historical fit.

## Important experimental details

- E3 sample sizes, in panel order: **12,000 / 24,000 / 24,000 / 60,000 /
  24,000 / 36,000**. Reverse query counts: **1,024 / 8,192 / 8,192 /
  16,384 / 8,192 / 1,024**. All reverse evaluations use **262,144 reference
  points**, including corrugated torus. This deliberately corrects the older
  65,536 statement on paper page 110; see [errata](docs/PAPER_ERRATA.md).
- E3's selected YY radii are **(0.4, 1, 1.6) r1**, not the later 0.8 diagnostic
  copy. DR's S3 failures at n=60,000 remain in the data and plots.
- STSP/DR **inter-manifold matching** uses the original centered compact
  h1-PCA. This does not change DR's internal Gaussian fitting operator.
- Original known-geometry query generation, controlled branch/initialization
  and evaluation are disclosed per experiment. Oracle in E4 uses true
  geometry by design. These are not blind real-data experiments.
- E4 uses the exact archived **ENF h1 fitted coordinates for all three methods**
  at every NN candidate count. Candidate density and full matching cost are
  reported; discrete NN is not identified with continuous nearest projection.
- Reliability is unchanged: E2/E3 main at least 18/20 complete repeats and
  overall query failure rate <=1%; E3 reverse failure rate <=1%. Partial finite
  repeat statistics, NaN gaps, hollow unreliable points and all failure classes
  are retained. Finite summaries are not evidence of all-query success.
- Method-specific sampling streams, original platform differences, YY
  main-V2/reverse-V1/corrugated source versions and adaptive sample-budget
  selection are preserved. Cross-method samples are not asserted to be paired.

## Scope and provenance

[docs/PROVENANCE.md](docs/PROVENANCE.md) distinguishes byte-identical numerical
modules, mechanical extraction, relative-path interfaces and presentation-only
changes. `SOURCE_MANIFEST.json` binds all distributed source, configurations,
inputs and tables; `validate_release.py` verifies it without network access.
The paper itself and private archival paths are not distributed.
See [docs/RELEASE_QA.md](docs/RELEASE_QA.md) for actual local acceptance checks
and the distinction between saved-figure replay and full scientific reruns.

This local preparation does not upload anything to GitHub, select a new
license, claim artifact badges, or independently re-fit all reported results.
Existing notices are preserved: E1 is **all rights reserved**, while E2–E6
retain their existing MIT notices. There is no blanket root license overriding
E1; read [LICENSES.md](LICENSES.md) before redistribution.
