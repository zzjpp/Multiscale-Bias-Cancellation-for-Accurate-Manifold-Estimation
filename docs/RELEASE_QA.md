# Local release acceptance (2026-10-09)

This is engineering evidence for the selected E1–E6 release, not a new
scientific audit, a full independent fit reproduction or an artifact badge.

- **44 actual local tests passed**, zero failures/errors/skips. These include
  reduced CPU fitting, corrugated dependencies, original YY reverse libraries,
  complete saved tables, changed-source and incomplete-run rejection, plus
  explicitly synthetic aggregation fixtures. Synthetic fixtures are not
  experimental outputs or CPU/GPU equivalence evidence.
- E5's original eight tests and its reduced population/finite computations
  also ran successfully. E4's actual CPU test evaluated four fixed h1 queries
  from an original 200,000-point parent, with the same input coordinates for
  all correspondences. E6's test reused fixed fits across changed weights.
- Original-to-release checks found all **358 originally selected source/data
  members unchanged in the research tree**, nine extracted-function records
  preserved, and exact original-versus-extracted CPU-cell output agreement on
  the same reduced data for STSP, DR and YY. This is scoped interface evidence,
  not a claim that all historical estimators were independently rerun.
- **31 logical figures / 62 exports** were produced: 31 PNG, 30 PDF and one
  retained E5 SVG. All PNGs and all 30 PDF pages were actually viewed; the
  unchanged SVG was byte-bound and its corresponding PNG viewed. E3 includes
  all 12 six-panel figures with full curves, NaN gaps and unreliable markers,
  without background shading or footer text.
- A separate source copy cold-started the saved-figure commands: all **62
  exports were byte-identical**. This establishes local saved-output replay,
  not cross-platform bit-exact numerical fitting. Later new-run metadata and
  tests did not change the renderer, registered inputs or these export bytes.
- `repro_check.py` reports all six essentials OK. The bounded release audit
  scanned all 263 Python sources without truncation: zero SAFE and identity
  findings, **123 consciously retained behavior-risk warnings**. These identify
  frozen numerical literals, explicit environment/thread controls and required
  large coordinate arrays. They are not automatically removed to get a clean
  heuristic score. The separate plaintext privacy check had no findings.

GPU production interfaces are supplied and source-bound, but the new public
entrypoints were **not run on a new GPU here**. Their original CPU/GPU gate
requirements and numerical locks must be satisfied when new runs are requested.
Historical raw packets and composite deployment histories are not distributed.
Licensing remains mixed; see `LICENSES.md`. Nothing was uploaded publicly.
