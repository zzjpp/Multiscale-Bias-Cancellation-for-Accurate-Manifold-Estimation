# Multiscale Bias Cancellation for STSP under Nonuniform Sampling — Population

Paper sections E.5; Figures E.15–E.16. Compatibility code: `nonuniform_E1`.

From the repository root:

```bash
python -m manifold_experiments --experiment nonuniform-population --backend cpu --output NEW
```

STSP only; seven noise levels, original E1 quadrature and roots with positive area-relative densities of amplitude 0.5 (max/min=3). Circle/Clifford use 128 phases, spheres 33 polar angles, corrugated torus 128 equal-area meridian locations. Maximum is over these finite queries; average is their unweighted mean.

Configuration/source locations: `config.json`, `density.py`. Original version-specific `src/` and `frozen/` packages retain numerical behavior and provenance names. Use the public pipeline, not historical deployment/replay scripts, for a complete fresh run.

No external dataset is required; observations are generated from the original geometry and seed definitions. Add `--profile smoke` for a reduced real fit that is not paper evidence. See the root README, `docs/EXPERIMENTS.md`, and `docs/FIGURE_MAP.json` for environments, output paths and validation limits.
