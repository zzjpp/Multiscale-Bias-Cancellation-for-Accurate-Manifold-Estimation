# Experiment guide

Titles and numbering follow the supplied 2026-11-12 manuscript. Compatibility codes identify packets; public names identify paper topics. All commands run from the repository root. No external dataset or historical fitted outputs are required.

The common six-geometry order is circle S1/R2, S2/R3, Clifford torus/R4, S3/R4, S2/R10, corrugated torus/R3. STSP/DR matching uses compact centered reference-scale PCA without true-normal input. YY retains its prescribed preliminary manifold and original data axis. Reliability rules and failed queries are never discarded.

## Population Verification

Sections 5.1; E.1; E.7. Figures 2, 4, E.1–E.3. Compatibility code: `E1`.

```bash
python -m manifold_experiments --experiment population --backend cpu --output NEW
```

Seven noise levels; h1=3σ, h2=6σ for STSP/DR. YY uses (1.25,5,8)σ with r0=3σ. The controlled Single/naive/GBC comparison shares fitted points and ENF correspondence. MS-STSP retains its separate hard-ball moments and three-radius OLS intercept. Population precision gates precede full-profile plotting.

Configurations and implementation: [experiments/population_verification](../experiments/population_verification/); `theory_path_population_*/config.json`, `ms_stsp/config.json`, `weight_control/PLAN.json`.

## Finite-Sample Behavior at Fixed Scales

Sections 5.2; E.2. Figures 5, E.4–E.6. Compatibility code: `E2`.

```bash
python -m manifold_experiments --experiment fixed-scales --backend cpu --output NEW
```

σ=0.005; h1=0.1, h2=0.2 for STSP/DR; YY uses (0.8,1,1.6)×40σ and r0=32σ. Eight nested sample sizes from 2,000 to 500,000, 144 fixed queries, 20 repetitions. Original medians, IQRs, partial-query metrics and reliability rules remain. No population-reference overlay is added.

Configurations and implementation: [experiments/finite_sample_fixed_scales](../experiments/finite_sample_fixed_scales/); `frozen/{STSP,DR,YY}/` configs and `corrugated/frozen/` variants.

## Bandwidth Robustness and Reverse Coverage

Sections 5.3; E.3. Figures 6, E.7–E.12. Compatibility code: `E3`.

```bash
python -m manifold_experiments --experiment bandwidth --backend cpu --output NEW
```

Six-manifold sample sizes 12000/24000/24000/60000/24000/36000; σ=0.015 and all 28 scales. Main uses 20 repetitions and 256 queries (corrugated:512); reverse uses repetition zero and 1024/8192/8192/16384/8192/1024 fitted queries with 262144 reference points on every geometry. YY uses (0.4,1,1.6)r1. Reverse evaluation uses complete fitted clouds. All independent baselines, failures, NaN gaps and hollow markers remain; green backgrounds are absent.

Configurations and implementation: [experiments/bandwidth_robustness](../experiments/bandwidth_robustness/); `science_support.py`, `frozen/` configs and `assets/` registered coordinates.

## Correspondence Accuracy and Computational Stability

Sections 5.4; E.3.1; E.6.3. Figures 7, E.13. Compatibility code: `E4`.

```bash
python -m manifold_experiments --experiment correspondence --backend cpu --output NEW
```

Space curve γ=(1.5cosθ,sinθ,0.3sin2θ), σ=0.06, h1=0.12, h2=0.24, n=200000, 32 common queries and ten paired repetitions. Continuous CP is a local empirical implicit-branch benchmark. Common-query construction has geometry assistance; true geometry is not CP search input. Fresh total timing includes common reference construction and target-branch/search costs, not a universal 26-second cost. The oscillatory normal-graph example is constructed, not fitted by STSP.

Configurations and implementation: [experiments/correspondence_accuracy](../experiments/correspondence_accuracy/); `correspondence/frozen/E4/fitting/original/config.json`; `correspondence/continuous_cp.py`.

## Directional Ablation of the YY Estimator

Sections 5.5; E.4; E.6.3. Figures 8, E.14; Table 2. Compatibility code: `E5`.

```bash
python -m manifold_experiments --experiment directional-ablation --backend cpu --output NEW
```

Population and finite directional ablation preserve original shared starts, data-axis construction and geometric/component metrics. Population uses 64 queries and three transverse multipliers (1.25,5,8); finite uses 256 queries, n=240000/480000/960000, 20 repetitions and (5,6,12). Actual c0=3, pilot selection and the historical rejected conditioning gate remain explicit. Table 2 slopes use the first four noise levels. Native Single, Scalar-A, Scalar-C and Directional labels are retained.

Configurations and implementation: [experiments/yy_directional_ablation](../experiments/yy_directional_ablation/); `config/population.json`, `config/finite_sample.json`.

## Multiscale Bias Cancellation for STSP under Nonuniform Sampling — Population

Sections E.5. Figures E.15–E.16. Compatibility code: `nonuniform_E1`.

```bash
python -m manifold_experiments --experiment nonuniform-population --backend cpu --output NEW
```

STSP only; seven noise levels, original E1 quadrature and roots with positive area-relative densities of amplitude 0.5 (max/min=3). Circle/Clifford use 128 phases, spheres 33 polar angles, corrugated torus 128 equal-area meridian locations. Maximum is over these finite queries; average is their unweighted mean.

Configurations and implementation: [experiments/nonuniform/population](../experiments/nonuniform/population/); `config.json`, `density.py`.

## Multiscale Bias Cancellation for STSP under Nonuniform Sampling — Finite Sample

Sections E.5. Figures E.17–E.18. Compatibility code: `nonuniform_E2`.

```bash
python -m manifold_experiments --experiment nonuniform-fixed-scales --backend cpu --output NEW
```

STSP only; same nonuniform densities as population, clean area-uniform rejection before Gaussian noise, with the original separate acceptance stream. The fixed-scale finite-sample protocol otherwise follows §5.2. Unreported nonuniform DR/YY studies are not public experiment selections.

Configurations and implementation: [experiments/nonuniform/fixed_scales](../experiments/nonuniform/fixed_scales/); `sampling.py` and the fixed-scale fitter configurations.
