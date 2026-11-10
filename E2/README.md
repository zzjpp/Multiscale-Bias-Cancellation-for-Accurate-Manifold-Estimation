# E2 — finite sample size

```bash
python E2/reproduce.py --output ./new_figures/E2
python E2/run_fits.py --method STSP --smoke
```

Six figures (PNG/PDF) use the complete 768-row saved table. The paper displays
coarse Single/Multi; independent fine Single for STSP/DR is retained in the table.
Finite partial repeat medians/IQR, unreliable hollow markers and NaN are kept.

Full GPU new parent run, using the strict original environment in the root README:

```bash
python E2/run_fits.py --method STSP --geometry circle_r2 --repeat 0 --output ./new_raw/STSP_circle_r2_r00.json.gz
```

Repeat for STSP/DR/YY × six keys × repeats0–19. Each **500,000-sample parent**
produces eight nested prefixes: 2,000/5,000/10,000/20,000/50,000/100,000/
200,000/500,000. Original native RNG/seed lineage and 144 queries are used,
not a new global seed. The new exact-parent CPU/GPU gate checks all independent
estimators on four queries at both n endpoints, with original tolerance
atol2e-8/rtol2e-7, before fitting. The exact nonorthogonal-PCA-basis failure
adapter is retained, not a basis repair. Every failure and native diagnostic
is saved. These are new runs, not historical receipt replay.

After all 360 parents are complete:

```bash
python E2/evaluate_fits.py --input ./new_raw --output ./new_summary.csv
python E2/reproduce.py --table ./new_summary.csv --output ./new_figures/E2_rerun
```

sigma=.005, STSP/DR h1=.1,h2=.2; YY radii=(.8,1,1.6)*40sigma,
first stage=32sigma. Original YY data-axis construction and controlled
known-geometry preliminary initialization are retained. Matching normals for
STSP/DR are centered compact h1-PCA. Method-specific sample streams and
original platform differences remain; do not assume paired samples.
CPU smoke uses 4,096 samples/two queries and does not pass the production gate.
The corrugated STSP/YY dependency tests explicitly use `--smoke-n 24000`:
their 4,096-sample trial had insufficient support. This test-only override
does not change production sample sizes, and partial failures remain visible.
New production runs validate and record `SOURCE_MANIFEST.json`; evaluation
rejects packets from a different or missing release source contract.
