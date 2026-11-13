# Bandwidth Robustness and Reverse Coverage

Paper sections 5.3; E.3; Figures 6, E.7–E.12. Compatibility code: `E3`.

From the repository root:

```bash
python -m manifold_experiments --experiment bandwidth --backend cpu --output NEW
```

Six-manifold sample sizes 12000/24000/24000/60000/24000/36000; σ=0.015 and all 28 scales. Main uses 20 repetitions and 256 queries (corrugated:512); reverse uses repetition zero and 1024/8192/8192/16384/8192/1024 fitted queries with 262144 reference points on every geometry. YY uses (0.4,1,1.6)r1. Reverse evaluation uses complete fitted clouds. All independent baselines, failures, NaN gaps and hollow markers remain; green backgrounds are absent.

Configuration/source locations: `science_support.py`, `frozen/` configs and `assets/` registered coordinates. Original version-specific `src/` and `frozen/` packages retain numerical behavior and provenance names. Use the public pipeline, not historical deployment/replay scripts, for a complete fresh run.

No external dataset is required; observations are generated from the original geometry and seed definitions. Add `--profile smoke` for a reduced real fit that is not paper evidence. See the root README, `docs/EXPERIMENTS.md`, and `docs/FIGURE_MAP.json` for environments, output paths and validation limits.
