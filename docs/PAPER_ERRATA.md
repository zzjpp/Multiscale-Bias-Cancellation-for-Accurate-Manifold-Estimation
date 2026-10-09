# Paper/code alignment

Authority: `manifold_fitting (62).pdf`, SHA256
`127f2fa740ae507e20604deb2d620491d75655dee8aea91cf219519877bb03fa`.

1. Page 110's E3 reverse reference count for corrugated torus is outdated.
   **All six geometries now use 262,144 reverse reference points**, explicitly
   confirmed by the author. The selected latest result table is retained.
   The old corrugated forward-evaluation auxiliary array of 65,536 points is
   still retained; it is not the reverse reference cloud.
2. Historical E5 configuration labels refer to earlier figure numbers 6/22.
   The numerical configurations remain unchanged. In this paper the finite
   ablation is **Figure 7** and the population ablation **Figure 20**.
3. E4's additional frozen-coordinate residual validation is performed after
   the original reference stopwatch. That extra validation is **not included**
   in the reported matching times. The retained numerical timing columns are
   not retrospectively altered.
4. The historical STSP E3 redundant header labeled a YY axis. Native STSP
   numerical source/traces use centered compact h1-PCA, not a YY axis. The
   public interface declares the actual method; historical data are not
   silently rewritten. YY known-geometry initialization is disclosed separately.
