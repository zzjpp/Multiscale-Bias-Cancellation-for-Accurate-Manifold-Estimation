# MS-STSP population baseline

This is the original hard-ball population adaptation used as the fourth curve in Figure 2. Radii are {3,4.5,6} sigma. All branches start at the same original query, retain a fixed noisy observation distribution and use the actual tangent-projection iteration. The final point is the OLS h-squared intercept, with weights (237,92,-111)/218. It is not the controlled smooth-kernel Naive baseline.

Use `python E1/ms_stsp/study.py cell --help` for a fresh single-cell CPU/GPU calculation. The distributed frozen/ inputs and FROZEN_INPUTS.json are complete; the historical `freeze` import command is not needed for public runs. Use fresh output paths. Production/reference quadrature, residual tolerances, common terminal horizon and extra eight-step check are retained in config.json and the original source. The saved baseline values are under data/ and the actual shared-fit comparison is in ../weight_control/.
