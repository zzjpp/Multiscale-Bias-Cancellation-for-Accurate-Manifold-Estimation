# Final manuscript figure map

The filenames identify the final manuscript graphics; historical experimental exports are not distributed. Commands below compute new results. Pages are physical PDF pages. `NEW` is a fresh output directory.

| Paper item | Page | Experiment | Calculation | Original filenames |
|---|---:|---|---|---|
| Figure 1 | 3 | schematic | `Compile illustrations/Figure1/normal_correspondence.tex with TikZ` | `normal_correspondence.png`, `normal_correspondence.pdf` |
| Figure 2 | 4 | E1 controlled weights | `python run_paper.py --experiment E1 --output NEW` | `e1_stsp_population_infinity_with_ms_stsp.png`, `e1_stsp_population_infinity_with_ms_stsp.pdf` |
| Figure 3 | 24 | schematic | `Compile illustrations/Figure3/yy_normal_decomposition.tex with TikZ` | `yy_normal_decomposition.png`, `yy_normal_decomposition.pdf` |
| Figure 4 | 30 | E1 normalized | `python run_paper.py --experiment E1 --output NEW` | `E1_circle_r2_normalized_E_infinity_STSP_DR_YY.png`, `E1_circle_normalized_E_infinity_STSP_DR_YY_clear.pdf` |
| Figure 5 | 31 | E2 | `python run_paper.py --experiment E2 --output NEW` | `E2_STSP_E_infty.png`, `E2_STSP_E_infty.pdf` |
| Figure 6 | 32 | E3 | `python run_paper.py --experiment E3 --output NEW` | `figure_e3_stsp_all_single_scales_max_error.png`, `figure_e3_stsp_all_single_scales_max_error.pdf` |
| Figure 7 | 33 | E4 | `python run_paper.py --experiment E4 --output NEW` | `e4_nn_density_accuracy.png`, `e4_nn_density_accuracy.pdf` |
| Figure 8 | 34 | E5 | `python run_paper.py --experiment E5 --output NEW` | `figure_e5_finite_sample_dual_metrics.png`, `figure_e5_finite_sample_dual_metrics.svg` |
| Figure E.1 | 97 | E1 | `python run_paper.py --experiment E1 --output NEW` | `E1_STSP_average_sigma.png`, `E1_STSP_average_sigma.pdf` |
| Figure E.2 | 98 | E1 | `python run_paper.py --experiment E1 --output NEW` | `E1_DR_infinity_sigma.png`, `E1_DR_infinity_sigma.pdf`, `E1_DR_average_sigma.png`, `E1_DR_average_sigma.pdf` |
| Figure E.3 | 99 | E1 | `python run_paper.py --experiment E1 --output NEW` | `E1_YY_infinity_sigma.png`, `E1_YY_infinity_sigma.pdf`, `E1_YY_average_sigma.png`, `E1_YY_average_sigma.pdf` |
| Figure E.4 | 100 | E2 | `python run_paper.py --experiment E2 --output NEW` | `E2_STSP_E_avg.png`, `E2_STSP_E_avg.pdf` |
| Figure E.5 | 101 | E2 | `python run_paper.py --experiment E2 --output NEW` | `E2_DR_E_infty.png`, `E2_DR_E_infty.pdf`, `E2_DR_E_avg.png`, `E2_DR_E_avg.pdf` |
| Figure E.6 | 102 | E2 | `python run_paper.py --experiment E2 --output NEW` | `E2_YY_E_infty.png`, `E2_YY_E_infty.pdf`, `E2_YY_E_avg.png`, `E2_YY_E_avg.pdf` |
| Figure E.7 | 104 | E3 | `python run_paper.py --experiment E3 --output NEW` | `figure_e3_stsp_all_single_scales_average_error.png`, `figure_e3_stsp_all_single_scales_average_error.pdf` |
| Figure E.8 | 105 | E3 | `python run_paper.py --experiment E3 --output NEW` | `figure_e3_dr_all_single_scales_average_error.png`, `figure_e3_dr_all_single_scales_average_error.pdf`, `figure_e3_dr_all_single_scales_max_error.png`, `figure_e3_dr_all_single_scales_max_error.pdf` |
| Figure E.9 | 107 | E3 | `python run_paper.py --experiment E3 --output NEW` | `figure_e3_yy_all_single_scales_average_error.png`, `figure_e3_yy_all_single_scales_average_error.pdf`, `figure_e3_yy_all_single_scales_max_error.png`, `figure_e3_yy_all_single_scales_max_error.pdf` |
| Figure E.10 | 109 | E3 | `python run_paper.py --experiment E3 --output NEW` | `figure_e3_stsp_E_reverse_mean.png`, `figure_e3_stsp_E_reverse_mean.pdf`, `figure_e3_stsp_E_reverse.png`, `figure_e3_stsp_E_reverse.pdf` |
| Figure E.11 | 110 | E3 | `python run_paper.py --experiment E3 --output NEW` | `figure_e3_dr_E_reverse_mean.png`, `figure_e3_dr_E_reverse_mean.pdf`, `figure_e3_dr_E_reverse.png`, `figure_e3_dr_E_reverse.pdf` |
| Figure E.12 | 111 | E3 | `python run_paper.py --experiment E3 --output NEW` | `figure_e3_yy_E_reverse_mean.png`, `figure_e3_yy_E_reverse_mean.pdf`, `figure_e3_yy_E_reverse.png`, `figure_e3_yy_E_reverse.pdf` |
| Figure E.13 | 114 | E4 deterministic counterexample | `python run_paper.py --experiment E4 --output NEW` | `P1_oscillatory_CP_nonuniqueness.png`, `P1_oscillatory_CP_nonuniqueness.pdf` |
| Figure E.14 | 115 | E5 | `python run_paper.py --experiment E5 --output NEW` | `figure_e5_population_directional_ablation.png`, `e5_population_directional_ablation.pdf` |
| Figure E.15 | 116 | nonuniform E1 | `python run_paper.py --experiment nonuniform_E1 --output NEW` | `e1_stsp_nonuniform_density_infinity.png`, `e1_stsp_nonuniform_density_infinity.pdf` |
| Figure E.16 | 116 | nonuniform E1 | `python run_paper.py --experiment nonuniform_E1 --output NEW` | `e1_stsp_nonuniform_density_average.png`, `e1_stsp_nonuniform_density_average.pdf` |
| Figure E.17 | 117 | nonuniform E2 | `python run_paper.py --experiment nonuniform_E2 --output NEW` | `E2_nonuniform_STSP_E_avg.png`, `E2_nonuniform_STSP_E_avg.pdf` |
| Figure E.18 | 118 | nonuniform E2 | `python run_paper.py --experiment nonuniform_E2 --output NEW` | `E2_nonuniform_STSP_E_infty.png`, `E2_nonuniform_STSP_E_infty.pdf` |
