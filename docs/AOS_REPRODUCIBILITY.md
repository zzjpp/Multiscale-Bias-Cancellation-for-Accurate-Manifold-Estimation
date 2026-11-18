# AoS code and reproducibility checklist

Checked against the official IMS pages on 2026-11-19. This document separates journal policy from this repository's engineering practices; it is not a certification by the journal.

## Journal requirements

The [2025–27 Annals of Statistics co-editor statement](https://imstat.org/2025/02/17/from-the-annals-of-statistics-co-editors-2025-27/) requires well-documented code and data sources, when applicable, to be publicly available upon acceptance.

The [AoS supplement instructions](https://imstat.org/journals-and-publications/annals-of-statistics/annals-of-statistics-supplement-instructions/) state that supplemental code/data files must also accompany the paper for publication on Euclid, even when they are hosted on GitHub. Each supplemental file needs a title and short description. The production DOI is assigned by the publisher; no DOI or journal approval is claimed here.

Therefore, submit a fixed source/configuration/design-input snapshot with the paper, identified by its Git commit and manifests. A changing GitHub URL alone does not replace that snapshot. A particular Python directory layout, formatter, CI provider, or test count is not stated as an AoS requirement in these sources.

Suggested code-supplement title: **Python code for “Beyond Curvature Bias: Higher-Order Manifold Fitting via Multiscale Cancellation”.**

Suggested description: **Source, dependency specifications, configurations, registered query/reference design inputs, and scripts for generating observations, fitting estimators, evaluating errors, and reproducing the numerical figures and tables. Large historical output archives are not required. The README describes full and reduced runs, hardware requirements, and validation limits.**

## Reproduction path

1. Record the exact commit and manuscript checksum in `PAPER_REFERENCE.json`.
2. Install the pinned CPU environment with Python 3.12.9, or the separate original GPU environment. See `requirements.txt`, `requirements-gpu.txt`, and the root README.
3. Run `python validate_release.py`, `python verify_paper.py`, and `python -m pytest -q tests --basetemp ./pytest_tmp` from the checkout root.
4. Run the actual reduced fitting/evaluation/plotting pipeline. GitHub Actions does this on Windows and Ubuntu; read the run's actual conclusion, not merely the workflow definition.
5. For the full study, use `python -m manifold_experiments --experiment all --backend cpu --output NEW --plan-only`, inspect the 1,356-task plan, then execute that same plan with `--resume`. GPU requires the documented original environment and same-input equivalence gates. Full runs are expensive; no hardware-independent runtime is promised.
6. Inspect `RUN/EVALUATION.json`, per-experiment tables and precision audits, all failure records, and generated figures. A failed numerical gate must be reported, not replaced by cached values. The figure-to-command/output mapping is `docs/FIGURE_MAP.json`.

All seven experiment-local `reproduce.py` scripts invoke the same fresh pipeline. With no arguments they show help; with `--output NEW` they compute their selected experiment. `render_figures.py` is plotting-only and requires an explicit complete table. `legacy_replay.py` is an archive-dependent historical interface, not the supported public reproduction path.

## Scientific transparency

- Geometry definitions, input generation and original seed rules are distributed. No external dataset or private GPU-server directory is needed by the supported pipeline. Registered bandwidth-study query/reference arrays are design inputs, not saved fits.
- Population and empirical operators remain distinct. STSP/DR use the original estimated-normal correspondence; YY's prescribed preliminary manifold and data-axis construction remain disclosed. The local CP benchmark retains geometry-assisted common query construction and its stated timing scope.
- Numerical quadrature, residual/root gates, fixed grids, reliability conditions, NaN breaks and complete failure accounting are retained. Failed engineering tasks are not automatically retried and relabeled successful.
- The directional finite-sample experiment retains its pilot-selection/conditioning caveats. Finite-query maxima and finite-range slopes are not continuous supremum certificates or asymptotic proofs.
- The code uses MIT with existing notices retained. Neither an artifact badge nor acceptance/compliance certification by AoS is claimed.

## Evidence levels

A source checksum check, a configuration/protocol audit, a reduced real run, an archived-output comparison, and a complete fresh production rerun establish different things. Successful CI demonstrates the first three for the declared reduced CPU profile; it does not independently establish all paper numbers. The 2026-11-19 checks and remaining unverified work are recorded in `docs/CI_AND_ENTRYPOINT_AUDIT.md`. Do not describe full production reproduction as complete until its actual run and numerical comparison exist.
