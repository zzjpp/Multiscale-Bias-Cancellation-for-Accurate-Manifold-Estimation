# Correspondence Accuracy and Computational Stability

Paper sections 5.4; E.3.1; E.6.3; Figures 7, E.13. Compatibility code: `E4`.

From the repository root:

```bash
python -m manifold_experiments --experiment correspondence --backend cpu --output NEW
```

Space curve γ=(1.5cosθ,sinθ,0.3sin2θ), σ=0.06, h1=0.12, h2=0.24, n=200000, 32 common queries and ten paired repetitions. Continuous CP is a local empirical implicit-branch benchmark. Common-query construction has geometry assistance; true geometry is not CP search input. Fresh total timing includes common reference construction and target-branch/search costs, not a universal 26-second cost. The oscillatory normal-graph example is constructed, not fitted by STSP.

Configuration/source locations: `correspondence/frozen/E4/fitting/original/config.json`; `correspondence/continuous_cp.py`. Original version-specific `src/` and `frozen/` packages retain numerical behavior and provenance names. Use the public pipeline, not historical deployment/replay scripts, for a complete fresh run.

No external dataset is required; observations are generated from the original geometry and seed definitions. Add `--profile smoke` for a reduced real fit that is not paper evidence. See the root README, `docs/EXPERIMENTS.md`, and `docs/FIGURE_MAP.json` for environments, output paths and validation limits.
