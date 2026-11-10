# Final manuscript figure map

Authority: `arxiv_final.pdf` (SHA-256 in `PAPER_REFERENCE.json`). Page numbers are physical PDF pages. `NEW` is a nonexistent output directory. Existing selected filenames are preserved.

| Paper item | PDF page | Experiment | Selected files | Replay |
|---|---:|---|---|---|
| Figure 1 | 3 | schematic | `illustrations/Figure1/normal_correspondence.png`<br>`illustrations/Figure1/normal_correspondence.pdf` | `Compile illustrations/Figure1/normal_correspondence.tex with TikZ` |
| Figure 2 | 4 | E1 controlled weights | `E1/weight_control/outputs/e1_stsp_population_infinity_with_ms_stsp.png`<br>`E1/weight_control/outputs/e1_stsp_population_infinity_with_ms_stsp.pdf` | `python E1/weight_control/render_and_report.py --output NEW --figure-only` |
| Figure 3 | 24 | schematic | `illustrations/Figure3/yy_normal_decomposition.png`<br>`illustrations/Figure3/yy_normal_decomposition.pdf` | `Compile illustrations/Figure3/yy_normal_decomposition.tex with TikZ` |
| Figure 4 | 30 | E1 normalized | `E1/figures/E1_circle_r2_normalized_E_infinity_STSP_DR_YY.png`<br>`E1/figures/E1_circle_normalized_E_infinity_STSP_DR_YY_clear.pdf` | `python E1/reproduce.py --output NEW` |
| Figure 5 | 31 | E2 | `E2/figures/E2_STSP_E_infty.png`<br>`E2/figures/E2_STSP_E_infty.pdf` | `python E2/reproduce.py --output NEW` |
| Figure 6 | 32 | E3 | `E3/paper_figures/figure_e3_stsp_all_single_scales_max_error.png`<br>`E3/paper_figures/figure_e3_stsp_all_single_scales_max_error.pdf` | `python E3/reproduce.py --output NEW` |
| Figure 7 | 33 | E4 | `E4/correspondence/figures/e4_nn_density_accuracy.png`<br>`E4/correspondence/figures/e4_nn_density_accuracy.pdf` | `python E4/reproduce.py --output NEW` |
| Figure 8 | 34 | E5 | `E5/paper_figures/figure_e5_finite_sample_dual_metrics.png`<br>`E5/paper_figures/figure_e5_finite_sample_dual_metrics.svg` | `Selected-export replay; see E5/README.md` |
| Figure E.1 | 97 | E1 | `E1/figures/E1_STSP_average_sigma.png`<br>`E1/figures/E1_STSP_average_sigma.pdf` | `python E1/reproduce.py --output NEW` |
| Figure E.2 | 98 | E1 | `E1/figures/E1_DR_infinity_sigma.png`<br>`E1/figures/E1_DR_infinity_sigma.pdf`<br>`E1/figures/E1_DR_average_sigma.png`<br>`E1/figures/E1_DR_average_sigma.pdf` | `python E1/reproduce.py --output NEW` |
| Figure E.3 | 99 | E1 | `E1/figures/E1_YY_infinity_sigma.png`<br>`E1/figures/E1_YY_infinity_sigma.pdf`<br>`E1/figures/E1_YY_average_sigma.png`<br>`E1/figures/E1_YY_average_sigma.pdf` | `python E1/reproduce.py --output NEW` |
| Figure E.4 | 100 | E2 | `E2/figures/E2_STSP_E_avg.png`<br>`E2/figures/E2_STSP_E_avg.pdf` | `python E2/reproduce.py --output NEW` |
| Figure E.5 | 101 | E2 | `E2/figures/E2_DR_E_infty.png`<br>`E2/figures/E2_DR_E_infty.pdf`<br>`E2/figures/E2_DR_E_avg.png`<br>`E2/figures/E2_DR_E_avg.pdf` | `python E2/reproduce.py --output NEW` |
| Figure E.6 | 102 | E2 | `E2/figures/E2_YY_E_infty.png`<br>`E2/figures/E2_YY_E_infty.pdf`<br>`E2/figures/E2_YY_E_avg.png`<br>`E2/figures/E2_YY_E_avg.pdf` | `python E2/reproduce.py --output NEW` |
| Figure E.7 | 104 | E3 | `E3/paper_figures/figure_e3_stsp_all_single_scales_average_error.png`<br>`E3/paper_figures/figure_e3_stsp_all_single_scales_average_error.pdf` | `python E3/reproduce.py --output NEW` |
| Figure E.8 | 105 | E3 | `E3/paper_figures/figure_e3_dr_all_single_scales_average_error.png`<br>`E3/paper_figures/figure_e3_dr_all_single_scales_average_error.pdf`<br>`E3/paper_figures/figure_e3_dr_all_single_scales_max_error.png`<br>`E3/paper_figures/figure_e3_dr_all_single_scales_max_error.pdf` | `python E3/reproduce.py --output NEW` |
| Figure E.9 | 107 | E3 | `E3/paper_figures/figure_e3_yy_all_single_scales_average_error.png`<br>`E3/paper_figures/figure_e3_yy_all_single_scales_average_error.pdf`<br>`E3/paper_figures/figure_e3_yy_all_single_scales_max_error.png`<br>`E3/paper_figures/figure_e3_yy_all_single_scales_max_error.pdf` | `python E3/reproduce.py --output NEW` |
| Figure E.10 | 109 | E3 | `E3/paper_figures/figure_e3_stsp_E_reverse_mean.png`<br>`E3/paper_figures/figure_e3_stsp_E_reverse_mean.pdf`<br>`E3/paper_figures/figure_e3_stsp_E_reverse.png`<br>`E3/paper_figures/figure_e3_stsp_E_reverse.pdf` | `python E3/reproduce.py --output NEW` |
| Figure E.11 | 110 | E3 | `E3/paper_figures/figure_e3_dr_E_reverse_mean.png`<br>`E3/paper_figures/figure_e3_dr_E_reverse_mean.pdf`<br>`E3/paper_figures/figure_e3_dr_E_reverse.png`<br>`E3/paper_figures/figure_e3_dr_E_reverse.pdf` | `python E3/reproduce.py --output NEW` |
| Figure E.12 | 111 | E3 | `E3/paper_figures/figure_e3_yy_E_reverse_mean.png`<br>`E3/paper_figures/figure_e3_yy_E_reverse_mean.pdf`<br>`E3/paper_figures/figure_e3_yy_E_reverse.png`<br>`E3/paper_figures/figure_e3_yy_E_reverse.pdf` | `python E3/reproduce.py --output NEW` |
| Figure E.13 | 114 | E4 deterministic counterexample | `E4/correspondence/counterexample/P1_oscillatory_CP_nonuniqueness.png`<br>`E4/correspondence/counterexample/P1_oscillatory_CP_nonuniqueness.pdf` | `python E4/correspondence/replay_counterexample.py --output NEW` |
| Figure E.14 | 115 | E5 | `E5/paper_figures/figure_e5_population_directional_ablation.png`<br>`E5/paper_figures/e5_population_directional_ablation.pdf` | `Selected-export replay; see E5/README.md` |
| Figure E.15 | 116 | nonuniform E1 | `nonuniform/E1/outputs/nonuniform/e1_stsp_nonuniform_density_infinity.png`<br>`nonuniform/E1/outputs/nonuniform/e1_stsp_nonuniform_density_infinity.pdf` | `python nonuniform/E1/reproduce.py --output NEW` |
| Figure E.16 | 116 | nonuniform E1 | `nonuniform/E1/outputs/nonuniform_average/e1_stsp_nonuniform_density_average.png`<br>`nonuniform/E1/outputs/nonuniform_average/e1_stsp_nonuniform_density_average.pdf` | `python nonuniform/E1/reproduce.py --output NEW` |
| Figure E.17 | 117 | nonuniform E2 | `nonuniform/E2/figures/E2_nonuniform_STSP_E_avg.png`<br>`nonuniform/E2/figures/E2_nonuniform_STSP_E_avg.pdf` | `python nonuniform/E2/reproduce.py --output NEW` |
| Figure E.18 | 118 | nonuniform E2 | `nonuniform/E2/figures/E2_nonuniform_STSP_E_infty.png`<br>`nonuniform/E2/figures/E2_nonuniform_STSP_E_infty.pdf` | `python nonuniform/E2/reproduce.py --output NEW` |

The final paper contains 8 main figures and 18 appendix figures. These use 33 logical experimental plots (66 raster/vector exports) and two schematic illustrations.
Only selected paper plots are emitted by the default replay. Other-geometry normalized triptychs, the obsolete two-curve STSP maximum, P1 population scaling, extra timing/error plots, and standalone branch-switch plots are not distributed as experiments. Required frozen numerical implementations and audit inputs remain for source verification.
