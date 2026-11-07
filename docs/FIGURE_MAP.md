# Current manuscript figure map

The supplied main PDF and supplement have separate figure numbering. Filenames below are preserved. `NEW` means a nonexistent output directory.

| Paper item | Experiment | Selected files | Replay |
|---|---|---|---|
| Figure 1 | schematic | `illustrations/Figure1/figure1_clear.png`<br>`illustrations/Figure1/figure1_clear.pdf` | `python illustrations/Figure1/draw_figure1.py` |
| Figure 2 | E1 controlled weights | `E1/weight_control/outputs/e1_stsp_population_infinity_with_ms_stsp.png`<br>`E1/weight_control/outputs/e1_stsp_population_infinity_with_ms_stsp.pdf` | `python E1/weight_control/render_and_report.py --output NEW --figure-only` |
| Figure 3 | schematic | `illustrations/Figure3/yy_normal_decomposition.png`<br>`illustrations/Figure3/yy_normal_decomposition.pdf` | `Compile illustrations/Figure3/yy_normal_decomposition.tex with TikZ` |
| Figure 4 | E1 normalized | `E1/figures/E1_circle_normalized_E_infinity_STSP_DR_YY_clear.png`<br>`E1/figures/E1_circle_normalized_E_infinity_STSP_DR_YY_clear.pdf` | `python E1/reproduce.py --output NEW` |
| Figure 5 | E2 | `E2/figures/E2_STSP_E_infty.png`<br>`E2/figures/E2_STSP_E_infty.pdf` | `python E2/reproduce.py --output NEW` |
| Figure 6 | E3 | `E3/paper_figures/figure_e3_stsp_all_single_scales_max_error.png`<br>`E3/paper_figures/figure_e3_stsp_all_single_scales_max_error.pdf` | `python E3/reproduce.py --output NEW` |
| Figure 7 | E4 | `E4/correspondence/figures/e4_nn_density_accuracy.png`<br>`E4/correspondence/figures/e4_nn_density_accuracy.pdf` | `python E4/reproduce.py --output NEW` |
| Figure 8 | E5 | `E5/paper_figures/figure_e5_finite_sample_dual_metrics.png`<br>`E5/paper_figures/figure_e5_finite_sample_dual_metrics.svg` | `Selected-export replay; see E5/README.md` |
| Figure S1 | E1 | `E1/figures/E1_STSP_average_sigma.png`<br>`E1/figures/E1_STSP_average_sigma.pdf` | `python E1/reproduce.py --output NEW` |
| Figure S2 | E1 | `E1/figures/E1_DR_infinity_sigma.png`<br>`E1/figures/E1_DR_infinity_sigma.pdf`<br>`E1/figures/E1_DR_average_sigma.png`<br>`E1/figures/E1_DR_average_sigma.pdf` | `python E1/reproduce.py --output NEW` |
| Figure S3 | E1 | `E1/figures/E1_YY_infinity_sigma.png`<br>`E1/figures/E1_YY_infinity_sigma.pdf`<br>`E1/figures/E1_YY_average_sigma.png`<br>`E1/figures/E1_YY_average_sigma.pdf` | `python E1/reproduce.py --output NEW` |
| Figure S4 | E2 | `E2/figures/E2_STSP_E_avg.png`<br>`E2/figures/E2_STSP_E_avg.pdf` | `python E2/reproduce.py --output NEW` |
| Figure S5 | E2 | `E2/figures/E2_DR_E_infty.png`<br>`E2/figures/E2_DR_E_infty.pdf`<br>`E2/figures/E2_DR_E_avg.png`<br>`E2/figures/E2_DR_E_avg.pdf` | `python E2/reproduce.py --output NEW` |
| Figure S6 | E2 | `E2/figures/E2_YY_E_infty.png`<br>`E2/figures/E2_YY_E_infty.pdf`<br>`E2/figures/E2_YY_E_avg.png`<br>`E2/figures/E2_YY_E_avg.pdf` | `python E2/reproduce.py --output NEW` |
| Figure S7 | E3 | `E3/paper_figures/figure_e3_stsp_all_single_scales_average_error.png`<br>`E3/paper_figures/figure_e3_stsp_all_single_scales_average_error.pdf` | `python E3/reproduce.py --output NEW` |
| Figure S8 | E3 | `E3/paper_figures/figure_e3_dr_all_single_scales_average_error.png`<br>`E3/paper_figures/figure_e3_dr_all_single_scales_average_error.pdf`<br>`E3/paper_figures/figure_e3_dr_all_single_scales_max_error.png`<br>`E3/paper_figures/figure_e3_dr_all_single_scales_max_error.pdf` | `python E3/reproduce.py --output NEW` |
| Figure S9 | E3 | `E3/paper_figures/figure_e3_yy_all_single_scales_average_error.png`<br>`E3/paper_figures/figure_e3_yy_all_single_scales_average_error.pdf`<br>`E3/paper_figures/figure_e3_yy_all_single_scales_max_error.png`<br>`E3/paper_figures/figure_e3_yy_all_single_scales_max_error.pdf` | `python E3/reproduce.py --output NEW` |
| Figure S10 | E3 | `E3/paper_figures/figure_e3_stsp_E_reverse_mean.png`<br>`E3/paper_figures/figure_e3_stsp_E_reverse_mean.pdf`<br>`E3/paper_figures/figure_e3_stsp_E_reverse.png`<br>`E3/paper_figures/figure_e3_stsp_E_reverse.pdf` | `python E3/reproduce.py --output NEW` |
| Figure S11 | E3 | `E3/paper_figures/figure_e3_dr_E_reverse_mean.png`<br>`E3/paper_figures/figure_e3_dr_E_reverse_mean.pdf`<br>`E3/paper_figures/figure_e3_dr_E_reverse.png`<br>`E3/paper_figures/figure_e3_dr_E_reverse.pdf` | `python E3/reproduce.py --output NEW` |
| Figure S12 | E3 | `E3/paper_figures/figure_e3_yy_E_reverse_mean.png`<br>`E3/paper_figures/figure_e3_yy_E_reverse_mean.pdf`<br>`E3/paper_figures/figure_e3_yy_E_reverse.png`<br>`E3/paper_figures/figure_e3_yy_E_reverse.pdf` | `python E3/reproduce.py --output NEW` |
| Figure S13 | E4 deterministic counterexample | `E4/correspondence/counterexample/P1_oscillatory_CP_nonuniqueness.png`<br>`E4/correspondence/counterexample/P1_oscillatory_CP_nonuniqueness.pdf` | `python E4/correspondence/replay_counterexample.py --output NEW` |
| Figure S14 | E5 | `E5/paper_figures/figure_e5_population_directional_ablation.png`<br>`E5/paper_figures/figure_e5_population_directional_ablation.pdf` | `Selected-export replay; see E5/README.md` |
| Figure S15 | nonuniform E1 | `nonuniform/E1/outputs/nonuniform/e1_stsp_nonuniform_density_infinity.png`<br>`nonuniform/E1/outputs/nonuniform/e1_stsp_nonuniform_density_infinity.pdf` | `python nonuniform/E1/reproduce.py --output NEW` |
| Figure S16 | nonuniform E1 | `nonuniform/E1/outputs/nonuniform_average/e1_stsp_nonuniform_density_average.png`<br>`nonuniform/E1/outputs/nonuniform_average/e1_stsp_nonuniform_density_average.pdf` | `python nonuniform/E1/reproduce.py --output NEW` |
| Figure S17 | nonuniform E2 | `nonuniform/E2/figures/E2_nonuniform_STSP_E_avg.png`<br>`nonuniform/E2/figures/E2_nonuniform_STSP_E_avg.pdf` | `python nonuniform/E2/reproduce.py --output NEW` |
| Figure S18 | nonuniform E2 | `nonuniform/E2/figures/E2_nonuniform_STSP_E_infty.png`<br>`nonuniform/E2/figures/E2_nonuniform_STSP_E_infty.pdf` | `python nonuniform/E2/reproduce.py --output NEW` |

There are 33 logical experimental plots (66 PNG/vector exports), plus two schematic illustrations. Multi-panel paper figures may use more than one logical plot.
The separate three-curve historical MS-STSP comparison is not a new paper figure; the four-curve controlled Figure 2 is selected. The unselected two-curve STSP maximum and P1 scaling/extra timing diagnostics are excluded from default paper replay. Their source/data remain available for verification.
