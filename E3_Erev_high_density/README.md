# E3 high-density reverse-coverage extension

This experiment increases the number of fitted query points on the four
higher-dimensional E3 manifolds so that the reverse-coverage metric is not
dominated by the finite query-grid covering radius.  It compares Single and
Multi separately for STSP, DR, and Yao--Yau (YY).

The design is frozen in `config.json`.  It is adaptive only in the explicitly
predeclared sense: every method/geometry pair first runs stage 1, and stage 2
is run only if the frozen separation rule fails.  The scale grid, observation,
noise, kernels, weights, solvers, failure taxonomy, and reliability threshold
remain those of the released E3 experiment.  Failed scales are retained.

For DR and YY, the audited 1024-query packets are immutable inputs and only
the complementary query points are evaluated.  The server contains no STSP
E3 per-query packet that can be reused, so STSP stage 1 must reconstruct the
frozen repeat-0 observation and evaluate the complete stage-1 grid.  This
difference in provenance must remain explicit in the final audit.

This is a repeat-0 query-density diagnostic.  It is not a replacement for the
20-repeat E3 uncertainty experiment.

## Frozen design and gates

The frozen configuration SHA-256 is recorded in
`FROZEN_CONFIG_SHA256.txt`.  Run

```powershell
python gate_query_grids.py
```

before production.  The gate verifies the configuration hash, exact STSP vs
DR/YY query-coordinate equality, nested stages, complement sizes, uniqueness,
and the deterministic query covering radii.  Its audited output is
`QUERY_GRID_GATE.json`.

The initial grid gate passed.  Stage 1 reduces the deterministic covering
radius from 0.10--0.19 to 0.05--0.10 on the two-dimensional manifolds and from
0.398 to 0.185 on the three-sphere.  Stage 2 reduces these values further to
0.025--0.056 and 0.123, respectively.

`run_dryy_increment.py` evaluates only the complement relative to the previous
audited grid.  It refuses to run when the regenerated or immutable observation
hash does not match the base formal packet.  `run_stsp_stage.py` evaluates the
complete stage-1 STSP grid because no reusable STSP per-query packet exists;
its stage 2 evaluates only the complement.

The local STSP single-scale benchmark is recorded in
`STSP_CPU_BENCHMARK.json`.  The observed cost was 0.13--0.22 seconds per query
per scale on the local CPU; this benchmark is for resource planning only and
is not scientific evidence.

## Execution acceleration

`RUNTIME_PLAN.json` freezes execution-only acceleration.  Query fits are
independent, so every method/geometry task is divided into eight deterministic
query shards and merged in global query-index order.  STSP may use at most
eight CPU workers.  DR and YY may use at most four workers per GPU after E7
releases its GPU.  Production cannot start until the sharded and unsharded
implementations agree on coordinates, fitted points, success/failure classes,
failure reasons, weights, digests, and final row order under the stated
tolerances.

No scientific parameter is changed by this acceleration.  In particular, the
28-scale grid and all failed outputs remain in the released packets.
