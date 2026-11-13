# Directional Ablation of the YY Estimator

Paper sections 5.5; E.4; E.6.3; Figures 8, E.14; Table 2. Compatibility code: `E5`.

From the repository root:

```bash
python -m manifold_experiments --experiment directional-ablation --backend cpu --output NEW
```

Population and finite directional ablation preserve original shared starts, data-axis construction and geometric/component metrics. Population uses 64 queries and three transverse multipliers (1.25,5,8); finite uses 256 queries, n=240000/480000/960000, 20 repetitions and (5,6,12). Actual c0=3, pilot selection and the historical rejected conditioning gate remain explicit. Table 2 slopes use the first four noise levels. Native Single, Scalar-A, Scalar-C and Directional labels are retained.

Configuration/source locations: `config/population.json`, `config/finite_sample.json`. Original version-specific `src/` and `frozen/` packages retain numerical behavior and provenance names. Use the public pipeline, not historical deployment/replay scripts, for a complete fresh run.

No external dataset is required; observations are generated from the original geometry and seed definitions. Add `--profile smoke` for a reduced real fit that is not paper evidence. See the root README, `docs/EXPERIMENTS.md`, and `docs/FIGURE_MAP.json` for environments, output paths and validation limits.
