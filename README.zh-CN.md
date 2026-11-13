# Beyond Curvature Bias: Higher-Order Manifold Fitting via Multiscale Cancellation

本仓库对应作者提供的 2026-11-12 版论文，文件名及 SHA-256 见 `PAPER_REFERENCE.json`。公开名称、目录说明和图件对应关系按该版论文整理。原 E1–E5 任务编号及图名保留兼容。

## 安装和检查

```bash
python -m venv .venv
# PowerShell：.venv\Scripts\Activate.ps1
# Linux/macOS：source .venv/bin/activate
python -m pip install -r requirements.txt
python validate_release.py
python verify_paper.py
python -m pytest -q tests
python -m manifold_experiments --list
python -m manifold_experiments --profile smoke --backend cpu --output ./smoke_run --workers 2
```

缩小检查有 24 个真实拟合任务，包含评估和绘图，但修改了样本、查询和积分预算，**不能作为论文误差、收敛阶数或完整实验复现的证据**。

## 论文与代码对应

| 选项 / 兼容编号 | 论文标题 | 章节 | 代码 |
|---|---|---|---|
| `population` / `E1` | Population Verification | 5.1; E.1; E.7 | [population_verification](experiments/population_verification/) |
| `fixed-scales` / `E2` | Finite-Sample Behavior at Fixed Scales | 5.2; E.2 | [finite_sample_fixed_scales](experiments/finite_sample_fixed_scales/) |
| `bandwidth` / `E3` | Bandwidth Robustness and Reverse Coverage | 5.3; E.3 | [bandwidth_robustness](experiments/bandwidth_robustness/) |
| `correspondence` / `E4` | Correspondence Accuracy and Computational Stability | 5.4; E.3.1; E.6.3 | [correspondence_accuracy](experiments/correspondence_accuracy/) |
| `directional-ablation` / `E5` | Directional Ablation of the YY Estimator | 5.5; E.4; E.6.3 | [yy_directional_ablation](experiments/yy_directional_ablation/) |
| `nonuniform-population` / `nonuniform_E1` | Multiscale Bias Cancellation for STSP under Nonuniform Sampling — Population | E.5 | [nonuniform/population](experiments/nonuniform/population/) |
| `nonuniform-fixed-scales` / `nonuniform_E2` | Multiscale Bias Cancellation for STSP under Nonuniform Sampling — Finite Sample | E.5 | [nonuniform/fixed_scales](experiments/nonuniform/fixed_scales/) |

```bash
python -m manifold_experiments --experiment population --output ./population_run --plan-only
python -m manifold_experiments --experiment population --output ./population_run --resume
```

选择 `all` 计算全部 1,356 项正式任务，开销较大。新名称与旧编号生成相同有序任务及任务 ID；`run_paper.py` 仍兼容。论文的 GBC-STSP、GBC-DR、GBC-YY 对应原生 `Multi` 字段。已有图名保留，兼容论文 LaTeX。

每项任务采用独立解释器、单 BLAS/OpenMP 线程。CPU 使用 Python 3.12.9 及 `requirements.txt`；GPU 原环境单列于 `requirements-gpu.txt`，保留原环境和精度检查，不自动切换 CPU。有效任务只在相同来源计划下显式续算，工程失败不自动重试，科学失败、NaN 和空心不可靠点均保留。

无需外部数据。观测由原几何和种子生成，带宽实验的固定查询/参考数组属于实验设计输入。公开流程从头生成、拟合、评估和绘图，不依赖历史拟合输出。旧的大型输出不随本次版本上传，浅克隆可避免下载旧历史。

详细设定、图件位置及限制见 [实验说明](docs/EXPERIMENTS.md)、[代码结构](docs/CODE_STRUCTURE.md)、[图件映射](docs/FIGURE_MAP.json) 和 [复现范围](docs/REPRODUCIBILITY.md)。整理检查不代表独立重跑全部生产实验；不宣称 artifact badge。授权为 [MIT](LICENSE)，保留原有版权声明。
