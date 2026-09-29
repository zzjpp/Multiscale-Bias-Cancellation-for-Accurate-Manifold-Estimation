# E5: directional extrapolation ablation

This repository reproduces exactly two E5 figures and contains no E6 code or
unrelated E5 experiment branch:

| Figure | Design | Evidence class | Output |
|---|---|---|---|
| Population directional ablation | seven noise levels, three component metrics | formal population calculation | `figure_e5_population_directional_ablation` |
| Finite-sample dual metric comparison | `n=240k,480k,960k`, 20 repeats | paper protocol with a revised numerical gate | `figure_e5_finite_sample_dual_metrics` |

Both figures compare `Single`, `Scalar-A`, `Scalar-C`, and `Directional` on the
same nonconstant curve in ambient dimension four. No external data are used.

## Finite-sample protocol and gate provenance

The paper's finite-sample setup uses `r0=3 sigma`, `C2=12`, transverse ratios
`(5,6,12)`, 256 queries, 20 paired repetitions, and sample sizes 240k, 480k,
and 960k. The historical configuration declared `c0=5`, but the executed cell
used the empirical routine's default `c0=3`. The release passes `c0=3`
explicitly to reproduce the paper's figure and retains the old declaration in
the provenance block.

The original complementary-moment condition-number limit of 50 failed: the
observed condition number is approximately 244.94. A revised limit of 256 now
accepts these weights; the audit recomputes the condition number and also
requires a weight-constraint residual at most `1e-10`. The original failed gate
remains in `config/finite_sample.json` and in every finite-sample packet. This
revision was made **after** the original failure. A passing full run therefore
establishes reproducibility under the disclosed revised gate, not that the
original predeclared gate passed. The `formal_evidence` field denotes a
complete, audited run of this stated paper protocol; a smoke run alone does not
establish the reported numerical results.

The population figure uses its frozen setup: `c0=3`, `C2=12`, transverse
ratios `(1.25,5,8)`, 64 queries, and
the predeclared slope window `(0.008,0.01,0.0125,0.016)`.

## Environment

Python 3.12 and the following pinned environment were used:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

No external dataset is required; all samples and quadrature nodes are generated
from the checked-in configurations. See `data/GENERATED_DATA.md`.

## Reproduce the two figures

First run the lightweight end-to-end check:

```bash
python reproduce.py --mode smoke
```

To reproduce both reported figures, including all 60 finite-sample parent
packets, their audit, population slopes, and PNG/PDF outputs, run:

```bash
python reproduce.py --mode formal --workers 4
```

The full finite-sample calculation is CPU- and memory-intensive; runtime is
hardware dependent. Existing packets are resumable, and the audit rejects
wrong configurations, incomplete query grids, or altered source hashes. After
a completed run, figures alone can be rebuilt with:

```bash
python reproduce.py --mode plot
```

Generated packets, CSV summaries, audits, and exactly two figure pairs are
written under `results/`, which is excluded from Git.

## Layout

```text
config/                    the two frozen figure designs
src/e56v2/                 only the E5 numerical dependencies
scripts/plot_two_figures.py
tests/                     release-contract tests
reproduce.py               single entry point
```

## License

MIT; see `LICENSE`.
