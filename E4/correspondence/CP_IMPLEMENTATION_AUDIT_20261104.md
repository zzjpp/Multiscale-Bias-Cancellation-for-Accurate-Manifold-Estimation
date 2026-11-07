# Continuous CP 的分支来源与计时范围核查

本次只核查源代码、调用路径和十轮已保存的结果，不重新运行拟合或计时。

当前图中约 26 秒的结果来自独立的 `continuous_cp.py:continuous_cp`。
`finite_experiment.py:83` 传入经验 h2 STSP 算子、公共参考 y1、原残差求根器和数值控制参数；
没有传入真实 gamma(theta)、theta、真实法向、Oracle 点或 NN 候选库。
十轮保存包记录的 CP/driver 源码 SHA 与当前实际调用源码一致，共 320 个 CP 查询。

## 连续分支如何构造

没有预先重建一条完整、全局的目标曲线。对每个 y1，在 h2 经验加权协方差上做 PCA，
取估计切向 T 和法向基 B。按需构建局部隐式图

    z(s)=y1+sT+B u(s),   Bᵀ R_h2(z(s))=0.

每次标量搜索求值都从该纤维的零法向坐标开始，用原 Newton/回溯求经验残差根；
协方差、法空间和导数均由观测点及 K6 权重计算。检查投影/完整残差、support、
eigengap、transversality 与 Jacobian；用隐函数导数做连续距离驻点搜索。
有限样本实际保存结果全部使用初始 9 个标量节点，最终每查询 11–12 个不同图求值，
并在选中位置做四个法向初值检查。没有使用真实参数化来生成 CP 的 target chart。

这些检查支持局部图的数值有效性，不能冒充对全残差零集的全局分支重建，
也不能保证任意未检出的分支不存在。图上的 Continuous CP 应理解为该局部 fitted sheet 上的连续搜索。

## 必须区分冻结旧代码中的另一条路径

冻结的 `frozen/E4/fitting/original/src/e4/pipeline.py:199` 中还存在
`continuous_nn`：用真实 gamma(theta)、normal_basis(theta) 生成目标曲线节点，
构造周期样条、连续优化，再做真实参数对应的残差根校正。
这一实现应称为 geometry-assisted continuous CP benchmark。
当前有限样本 driver 没有调用它；导入 pipeline 模块本身不等于调用该函数。
不能把这两条实现路径混为一谈。

## 当前实验中仍然使用真实几何的部分

- 所有方法共享的 archived h1 查询，来自原 E4 的真实法向图构造；当前 driver 还用真实基底重建并核验这些根。
- 原 NN 库继续用真实几何初始化其离散目标点。
- Oracle 与误差评价使用真实曲线和真实投影。
- Population 期望积分按定义使用已知曲线计算 latent 积分；这与有限样本经验算子的计时不是一回事。

因此：CP 的 target matching/search 部分在给定 y1 后依赖经验算子；
整个有限样本 benchmark 仍有 geometry-assisted reference-query design，
不能描述成完全 data-driven 的端到端重建。

## 计时的准确含义

- 25.5621838 秒：十轮各 32 个查询的总耗时中位数，不是单查询耗时。
- 24.6153344 秒：CP correspondence 阶段耗时的中位数。
- 0.9022916 秒：共享 setup/reference 重建核验阶段耗时的中位数。
- 总耗时按每轮实际分项之和再取中位数，分项中位数不要求相加恰好等于总中位数。
- 由 `perf_counter` 测得，是本机 CPU 后端的 elapsed wall-clock time，而不是进程 CPU-time 计数。
- 包含：观测数组、两尺度索引和权重；共享参考根重建/核验；CP 经验 PCA、图求根、导数、连续搜索、法向多初值检查及外推组合。
- 不包含：输入文件读取；`finite_experiment.py:52` 中位于计时器之外的真实弧长/query-grid 生成；真实投影及误差计算；结果写盘与绘图。

因此 26 秒只能用来描述这一具体实现和协议。不能据此宣称一般 CP 都需要该成本，
也不能将其表述为一个完全 data-driven、完整端到端流程的成本。

建议用名：**empirical local-chart continuous CP benchmark with geometry-assisted common reference queries**。
可直接用于论文的段落在 `manuscript/CP_scope_clarification.tex`。
可机器核对的源码 SHA、十轮结果 SHA 及计时证据在 `CP_IMPLEMENTATION_AUDIT_20261104.json`。
