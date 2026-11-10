# 最终稿实验代码

本目录以 `arxiv_final.pdf` 为唯一论文依据，题目为 Beyond Curvature Bias: Higher-Order Manifold Fitting via Multiscale Cancellation。论文 SHA-256 与 128 页信息见 `PAPER_REFERENCE.json`。仓库名称维持原样。

安装 `requirements.txt` 后运行：

```bash
python validate_release.py
python verify_paper.py
python -m pytest -q tests --basetemp ./tmp/pytest
python reproduce_all.py --output ./replay_output
```

输出目录必须不存在。默认重放最终稿选用的 33 张独立实验图（66 个 PNG/矢量文件）；正文图 1–8、附录图 E.1–E.18 的文件和入口见 `docs/FIGURE_MAP.md`。两个示意图的可编辑源文件在 `illustrations/`。

只保留论文使用的 E1–E5、MS-STSP/权重控制、CP 非唯一示例及非均匀密度 STSP 补充实验。未采用的额外归一化图、P1 population scaling 和附加图表已移出发布目录；所需冻结求解器、数据和来源校验依赖仍保留。原始数据字段中的 Single/Multi 名称保留，与图中基线/GBC 一一对应。

E2 不含已撤销的 population 参考线。E3 维持原样本量、YY 的 .4/1/1.6 半径和六个流形统一 262144 个反向参考点，保留失败、空心点和全部 28 档尺度。代码整理不改变拟合器、随机种子、数值精度或任何实验结论。

E5 有限样本的 60 个历史 held-out 原始包缺失；图件可从有校验值的存档导出恢复，但其原始中位数/IQR 不能据此声称已完整复核。E5 的七个 population 包可用于核验 Table 2。详见 `docs/DATA_AVAILABILITY.md`。

服务器连接、私钥、活跃任务状态、缓存、E7 和无关实验不上传。完整实验计算与图件重放分开，烟雾测试不等于全量 GPU 复算。
