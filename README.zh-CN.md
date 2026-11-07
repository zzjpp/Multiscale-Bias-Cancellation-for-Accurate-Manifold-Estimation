# 2026-11-07 论文实验代码

本目录对应本次提供的正文和补充材料，题目为 Beyond Curvature Bias: Higher-Order Manifold Fitting via Multiscale Cancellation。仓库名称暂时维持原样。

安装 `requirements.txt` 后运行 `python validate_release.py`、`python verify_paper.py` 和 `python reproduce_all.py --output ./replay_output`。输出目录必须不存在。默认复现 33 张独立实验图（66 个 PNG/矢量导出），图号与代码的完整对应表在 `docs/FIGURE_MAP.md`。

包括 E1 权重控制、E4 Continuous CP、CP 非唯一反例以及非均匀密度 E1/E2。E2 不包含已撤销的 population 参考线；E3 保留原 .4/1/1.6 半径、全部尺度和失败。图件复现与重新运行拟合严格区分。

原始 E5 有限样本的 60 个历史 held-out 包缺失，故该图可以从绑定的选定导出恢复，但不能据此声称已独立核验其原始中位数。服务器连接、私钥、任务日志、缓存、E7 和无关实验不上传。
