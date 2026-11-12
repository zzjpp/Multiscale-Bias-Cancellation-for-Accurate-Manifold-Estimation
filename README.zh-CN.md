# 论文实验复现

以最终 `arxiv_final.pdf` 为准，提供从**生成输入、拟合、评价到绘图**的新运行入口，不读取历史拟合点或复制旧图作为复现结果。完整说明见 [README.md](README.md)。

```bash
python -m pip install -r requirements.txt
python validate_release.py
python verify_paper.py
python -m pytest -q tests
python run_paper.py --profile smoke --backend cpu --output ./smoke_run --workers 2
```

最后一条命令实际运行 24 个缩小规模任务，覆盖 E1–E5 和两组非均匀实验，生成诊断图。它只验证计算链路，**不能用于报告论文数值或阶数**。

正式配置：

```bash
python run_paper.py --experiment all --backend cpu --output ./paper_run --plan-only
python run_paper.py --experiment all --backend cpu --output ./paper_run --resume --workers 1
```

也可只选择 `E1`、`E2`、`E3`、`E4`、`E5`、`nonuniform_E1`、`nonuniform_E2`。全套有 1,356 个任务，计算量很大，本次整理不自动重跑。GPU 使用原 Linux 环境及 CPU/GPU 门禁；E4、E5 保留原 CPU 实现。CPU 正式模式使用相同预算的原 CPU 拟合器，不是静默回退。

`packets/` 保存全部结果与失败，`evaluation/` 保存汇总与精度诊断；图在 `figures/`、`evaluation/E5/figures/`、`counterexample/`。已有目录必须显式 `--resume` 并匹配原计划、源码；工程失败不自动重试，不覆盖论文目录。公共入口统一为 `run_paper.py`；保留的旧模块作为科学计算库使用，历史部署/重放 CLI 不作为本版运行流程。

E3 保持最新样本量、六流形统一 262144 个反向参考点、全部尺度与可靠性规则。非均匀 E2 仅运行文中 STSP。E2 不加已撤销的 population 参考线；E7 不作为独立实验发布。E4 公共查询的几何辅助、CP 局部分支及计时范围，E5 的 pilot 选择与旧失败门禁，均保留披露。

大型历史输出、重复观测数组和旧图不随新版发布，已保留在本机归档。少量 `reference_values/` 历史对照表**不进入新实验计算**。E3 固定 query/reference 坐标属于设计输入，仍包含在仓库；其余观测由原种子生成。建议 `git clone --depth 1` 避免下载旧提交的大文件历史。

本次验证包括干净安装、不含历史输出的实际小流程；没有声称重跑全部正式实验、获得复现徽章或满足某个尚未指定期刊的全部提交细则。仓库统一采用 MIT 许可证，保留原作者署名；E1 已按仓库所有者于 2026-11-12 的明确授权改为 MIT，详见 `LICENSE` 和 `LICENSES.md`。
