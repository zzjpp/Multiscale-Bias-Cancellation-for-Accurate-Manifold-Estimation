# Final-paper E4: local CP, ENF, NN and Oracle

Selected experiments: Figure 7 (three panels, ten paired repetitions) and Figure E.13 (constructed nonuniqueness example). The unselected P1 population-scaling experiment and additional plots/reports are excluded from this release.

```bash
python E4/reproduce.py --output NEW
python E4/correspondence/replay_counterexample.py --output NEW_COUNTEREXAMPLE
python E4/correspondence/finite_experiment.py --help
```

`finite_sample/` and `finite_summary/` hold all ten packets and retained failures/diagnostics. `frozen/` binds the original empirical STSP implementation, ten observation arrays and common h1 query coordinates. `PLAN.json` is the historical study plan; only its finite comparison and constructed example are selected by this paper. Frozen evidence is not edited to pretend the original study had a narrower history.

The actual CP branch is constructed from empirical h2 PCA and implicit projected-residual solves. The scalar search receives no true curve parameterization, normals or NN candidate library. Common query initialization and validation are geometry-assisted. It is a checked local target sheet, not a certificate of global closest points across arbitrary additional empirical components. Median total time 25.56218 seconds includes the recorded common setup/validation and local construction/search for 32 queries; the CP correspondence-stage median is 24.61533 seconds. These are implementation-specific elapsed times, not a complexity guarantee. Details: `CP_IMPLEMENTATION_AUDIT_20261104.md`.

The counterexample uses a prescribed oscillatory normal graph rather than STSP output. It demonstrates nonuniqueness of global CP and uniqueness of the local ENF intersection, not failure of fourth-order localization. The recorded roots are replayed without numerical refitting. The immutable original combined counterexample generator remains available for regenerating its numerical certificate; the public replay emits only the selected two-panel figure.
