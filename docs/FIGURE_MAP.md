# Paper figures and local entry points

| Experiment | Main text | Appendix | Saved-result command |
|---|---|---|---|
| E1 population | 2 STSP maximum; 3 normalized circle | 8 STSP average; 9 DR maximum/average; 10 YY maximum/average | `python E1/reproduce.py` |
| E2 sample size | 4 STSP maximum | 11 STSP average; 12 DR maximum/average; 13 YY maximum/average | `python E2/reproduce.py` |
| E3 bandwidth | 5 STSP forward maximum | 14 STSP forward average; 15 DR forward; 16 YY forward; 17 STSP reverse; 18 DR reverse; 19 YY reverse | `python E3/reproduce.py` |
| E4 correspondence density | 6 average / maximum / complete computation time | — | `python E4/reproduce.py` |
| E5 directional ablation | 7 finite sample | 20 population | `python E5/reproduce.py --mode paper` |
| E6 weight sensitivity | — | 21 DR; 22 STSP / YY | `python E6/reproduce.py --mode figures` |

E3 has 12 logical figures, each PNG + PDF, each 3 columns × 2 rows. E1 has
six error figures plus the circle triptych, E2 six error figures, E4 one
triptych, E5 two selected figures, and E6 three method figures: **31 logical
paper figures**. E5's retained finite vector export is SVG; its population
vector export is PDF. Saved-export publishing in E5 is not estimator replay.
