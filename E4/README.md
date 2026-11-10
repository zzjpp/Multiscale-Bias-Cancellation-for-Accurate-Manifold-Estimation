# E4: ENF, local continuous CP, discrete NN and Oracle

Current Figure 7 is reproduced by `python E4/reproduce.py --output NEW`. The underlying actual measurements and solver are in `correspondence/`. The three panels are final maximum geometric error, direct discrepancy from Oracle and full measured CPU cost. All methods share the same 32 archived h1 fitted reference coordinates and ten n=200000 parent samples.

`python E4/reproduce.py --mode verify` checks the original common-coordinate asset contract. Fresh actual calculations: `python E4/correspondence/finite_experiment.py --help`. Full provenance and geometric scope: `correspondence/CP_IMPLEMENTATION_AUDIT_20261104.md`. Figure E.13 is replayed with `python E4/correspondence/replay_counterexample.py --output NEW`.

CP builds an empirical target-PCA local chart, solves the true empirical STSP residual at every scalar search location, and optimizes continuous distance on the checked local sheet. The search itself does not receive true manifold coordinates, but common-query construction and validation are geometry assisted. Its roughly 25.6-second total includes the reported common setup and local representation/search for 32 queries; it is implementation-specific. The artificial oscillatory graph illustrates global CP nonuniqueness while local ENF remains unique, not failure of fourth-order localization.
