# Release audit

The final static release audit was run with:

    python release_audit.py simulation_experiments/E7 --blind none --json

Result: **0 findings** (0 safe, 0 behavior-risk, 0 identity).

The public package contains only the files needed to regenerate the three E7
figures. The complete production workspace, raw packets, logs, diagnostic
plots, GPU launchers, exploratory runs, and reverse-error analysis are not part
of this release.

Numerical behavior was protected as follows:

- all three aggregate CSV files were copied byte-for-byte from formally audited
  outputs;
- the complete 28-point scale grid and every unreliable cell are validated at
  load time;
- the publication colors, markers, sizes, labels, and reliability encoding are
  unchanged;
- regenerated PNG files matched the three formal PNG files byte-for-byte.

PDF byte hashes are not used as a gate because Matplotlib embeds generation
timestamps in PDF metadata. PNG equality and the frozen CSV hashes are the
release invariants.
