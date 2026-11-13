# Population Verification

Paper sections 5.1; E.1; E.7; Figures 2, 4, E.1–E.3. Compatibility code: `E1`.

From the repository root:

```bash
python -m manifold_experiments --experiment population --backend cpu --output NEW
```

Seven noise levels; h1=3σ, h2=6σ for STSP/DR. YY uses (1.25,5,8)σ with r0=3σ. The controlled Single/naive/GBC comparison shares fitted points and ENF correspondence. MS-STSP retains its separate hard-ball moments and three-radius OLS intercept. Population precision gates precede full-profile plotting.

Configuration/source locations: `theory_path_population_*/config.json`, `ms_stsp/config.json`, `weight_control/PLAN.json`. Original version-specific `src/` and `frozen/` packages retain numerical behavior and provenance names. Use the public pipeline, not historical deployment/replay scripts, for a complete fresh run.

No external dataset is required; observations are generated from the original geometry and seed definitions. Add `--profile smoke` for a reduced real fit that is not paper evidence. See the root README, `docs/EXPERIMENTS.md`, and `docs/FIGURE_MAP.json` for environments, output paths and validation limits.
