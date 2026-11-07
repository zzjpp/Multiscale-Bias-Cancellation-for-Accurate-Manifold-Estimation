"""Create the P1 tables and manuscript text from sealed actual summaries only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_csv(p):return list(csv.DictReader(p.open(encoding='utf8')))
def csv_write(p,rows):
    with p.open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def number(x):return f'{float(x):.6g}'

def table_figure(out,finite,pop,slopes,counter):
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig=plt.figure(figsize=(11.7,8.3))
    fig.text(.04,.96,'P1: ENF and continuous closest-point correspondence',fontsize=17,weight='bold')
    fig.text(.04,.925,'Actual finite STSP data, deterministic population quadrature, and a constructed counterexample',fontsize=11)
    fig.text(.04,.875,'A. Finite STSP: medians over ten original parents, 32 exact shared references each',weight='bold')
    ax=fig.add_axes([.04,.57,.92,.28]);ax.axis('off')
    cols=['Method',r'$E_\infty$',r'$E_{\mathrm{avg}}$',r'$E_{\mathrm{match}}$',r'$E_{\mathrm{base}}$','Total CPU (s)','Failed queries']
    rr=[]
    for r in finite:
        label=r['method']+(f" ({r['candidate_count']})" if r['method']=='NN' else '')
        rr.append([label]+[number(r[k+'_median']) for k in ['E_infinity','E_average','E_match','E_base','total_seconds']]+[r['total_query_failures']+'/320'])
    t=ax.table(cellText=rr,colLabels=cols,cellLoc='center',loc='upper center',colWidths=[.18,.13,.13,.15,.15,.13,.13])
    t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,1.65)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor('#CCCCCC');cell.set_linewidth(.5)
        if r==0:cell.set_facecolor('#E9EEF5');cell.set_text_props(weight='bold')
    fig.text(.04,.545,'Timing includes empirical target representation and search. It is implementation-specific.',fontsize=10)
    fig.text(.04,.495,'B. Population matching accuracy: seven noise levels, no Monte Carlo',weight='bold')
    ax=fig.add_axes([.04,.35,.92,.13]);ax.axis('off')
    rr=[]
    for m in ['ENF','CP']:
        s=next(x for x in slopes if x['method']==m and x['metric']=='E_match')
        normalized=[float(r[m+'_match_over_sigma4']) for r in pop]
        rr.append([m,f"{float(s['slope_all7']):.6f}",f"{float(s['slope_smallest4']):.6f}",f'{min(normalized):.6g} to {max(normalized):.6g}'])
    t=ax.table(cellText=rr,colLabels=['Method','Slope, all seven','Slope, smallest four',r'$E_{\mathrm{match}}/\sigma^4$ range'],cellLoc='center',loc='upper center')
    t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,1.6)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor('#CCCCCC');cell.set_linewidth(.5)
        if r==0:cell.set_facecolor('#E9EEF5');cell.set_text_props(weight='bold')
    fig.text(.04,.325,'Finite-range slopes and resolution checks are numerical evidence, not an asymptotic proof.',fontsize=10)
    fig.text(.04,.275,'C. Oscillatory normal graph: local ENF uniqueness versus nonunique global CP',weight='bold')
    lines=[
        f"N = {counter['N']}; a = {counter['a']}; reference radius = 1.01; target mean radius = 1.04.",
        f"Two CP angles: +/-{abs(float(counter['CP_angles'][1])):.12g} rad; local ENF point: (1.04000256, 0).",
        f"g(0) = {float(counter['ENF_g']):.15g}; g(CP) = {float(counter['CP_g']):.15g}.",
        f"Limiting CP branch jump = {float(counter['limiting_CP_point_jump']):.12g}; ENF varies continuously.",
        'The graph is constructed, not STSP-generated. Nonuniqueness does not disprove fourth-order position accuracy.',
        'Continuous CP in the empirical test searches the verified local fitted sheet; no theorem certifies all spurious components.'
    ]
    for i,line in enumerate(lines):fig.text(.04,.24-i*.032,line,fontsize=10)
    fig.savefig(out/'P1_SUMMARY_TABLE.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(out/'P1_SUMMARY_TABLE.png',dpi=220)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'manuscript');a=p.parse_args()
    out=a.output.resolve()
    if not out.is_relative_to(ROOT) or out.exists():raise ValueError('Fresh isolated output required')
    for phase in ['finite','population']:
        assert json.loads((ROOT/f'{phase}_summary/AUDIT.json').read_bytes())['status']=='PASS'
    all_finite=read_csv(ROOT/'finite_summary/SUMMARY.csv')
    finite=[next(r for r in all_finite if r['method']==m and (m!='NN' or int(r['candidate_count'])==c)) for m,c in [('Oracle',0),('ENF',0),('CP',0),('NN',32),('NN',128),('NN',512)]]
    pop=read_csv(ROOT/'population_summary/VALUES.csv');slopes=read_csv(ROOT/'population_summary/SLOPES.csv')
    counter=json.loads((ROOT/'counterexample/RESULT.json').read_bytes())
    pa=json.loads((ROOT/'population_summary/AUDIT.json').read_bytes())
    out.mkdir();csv_write(out/'FINITE_SELECTED_TABLE.csv',finite)
    combined=[]
    for s in slopes:
        vals=[float(r[s['method']+'_'+s['metric']])/float(r['sigma'])**4 for r in pop]
        combined.append(dict(**s,normalized_min=min(vals),normalized_max=max(vals)))
    csv_write(out/'POPULATION_ORDER_TABLE.csv',combined)
    e=next(r for r in finite if r['method']=='ENF');c=next(r for r in finite if r['method']=='CP')
    es=next(r for r in slopes if r['method']=='ENF' and r['metric']=='E_match')
    cs=next(r for r in slopes if r['method']=='CP' and r['metric']=='E_match')
    speed=float(c['total_seconds_median'])/float(e['total_seconds_median'])
    md='''# P1: ENF 与连续 CP 的受控比较

本实验分别检验两个问题：正常光滑 STSP 分支上的精度和计算成本；仅有弱位置/一阶控制时 CP 对应的唯一性。两组结果不能混为一个精度优劣结论。

## 1. 真实有限样本 STSP

曲线为 γ(t)=(1.5 cos t, sin t, 0.3 sin 2t)。沿用原 E4 的十个 n=200000 父样本（repeat 10–19）、σ=0.06、h1=0.12、h2=0.24 和每轮 32 个原始 h1 查询坐标。所有方法逐位共享原查询数组；没有将 Oracle 与 ENF 放在不同参考位置。

|方法|最终 E∞|最终 Eavg|E_match|E_base|总 CPU 秒|查询失败|
|---|---:|---:|---:|---:|---:|---:|
'''
    for r in finite:
        label=r['method']+(f" ({r['candidate_count']})" if r['method']=='NN' else '')
        md+='|'+label+'|'+'|'.join(number(r[k+'_median']) for k in ['E_infinity','E_average','E_match','E_base','total_seconds'])+'|'+r['total_query_failures']+'/320|\n'
    md+=f'''
表中是各轮指标的中位数；图中的带状区域为四分位区间。每种方法/每档 NN 候选数均完成 320/320 查询；所有候选点也成功，因此没有隐藏失败分类或仅筛选成功轮。

ENF 与 CP 的最终几何误差非常接近。CP 的直接匹配误差更小，不能据此宣称 ENF 在精度上支配 CP。此次实现中，CP/ENF 总 CPU 时间中位数之比为 {speed:.4f}；这是实现相关的实测值，不是所有 CP 算法的成本下界。NN 在部分有限候选数下的最终误差可能略低于 Oracle；这不意味着离散对应具有更高渐近精度。

### 连续 CP 的具体含义与搜索范围

在 y1 处用目标 h2 的经验加权 PCA 得到切向 T 和法向基 B，构建 z(s)=y1+sT+B u(s)，每次求值均用真实 STSP 残差求 BᵀR_h2(z)=0，并检查完整残差。用隐函数导数连续求距离驻点，而不是对 512 个候选点取 NN，也不是拟合一条未经残差校正的 spline。CP 输入接口不接收真实曲线参数、真实法向、Oracle 点或 NN 库。

中央有效拟合点给出距离上界 d0，所有 |Tᵀ(z−y1)|>d0 的点都不可能更近。在该切向区间内初始 9 点搜索，梯度不单调时增密至 41/81 点，连续 Brent 求根后比较驻点目标值；还用四个法向初值核验所得分支。所有 320 个 CP 点均与 512 候选库不同，并且到参考点的距离不大于 ENF 或 NN512（容差 2e−8）。

**限制：这是对数值核验的局部 STSP 拟合分支进行连续最近点搜索，不是对任意可能的伪残差分支作形式化全局最优证明。** 多初值、网格加密和切向排除不能冒充对任意残差集合的数学全局证书。

STSP 接受阈值沿用原 finite 实现：投影残差 ≤1e−11、完整残差 ≤1e−9，且 support、eigengap、transversality、Jacobian 检查均通过。CP 另需距离驻点残差 ≤1e−10。没有改变核、原 STSP 拟合器或失败门槛。

### 指标与计时

令 x_j=π_M(y1_j)，X_h2(x_j) 为该真实法纤维上的目标 STSP 根。E_match=max_j||Z_j−X_h2(x_j)||；E_base=max_j||π_M(Z_j)−x_j||。外推点按原精确系数权重计算，最终误差使用到真实曲线的距离。真实几何仅用于 Oracle、评价及原公共参考/NN 分支初始化；公共参考构造本身仍是受控几何设置，不能称为完全未知几何端到端方法。

所有计时是新运行的同一 CPU/float64/单线程计时：包含数据数组构造、两尺度索引、系数、公共参考根构造和核验、目标表示/搜索与线性组合。CP 的经验 PCA、局部图构建、每次残差求根、连续搜索及法向核验均计入；NN 复用原嵌套候选位置，但每档收费包括实际累计构建到 M 的时间和该档查找。输入文件读取、真值指标计算、绘图和结果写盘不计入。不混用旧 GPU 时间。各方法顺序随 repeat 循环轮换。

## 2. Population scaling

同一空间曲线、同一 32 个固定弧长查询，七个 σ=0.008,0.01,0.0125,0.016,0.02,0.025,0.03；此处按用户要求 h1=3σ、h2=6σ，区别于上面的有限样本 2σ/4σ 路径。

使用均匀弧长 latent 确定性积分、Gaussian reduction 和 K6 径向 Gauss–Legendre 积分，没有 Monte Carlo。生产/加密分辨率为 8192/160 和 16384/224，tail padding=12。期望算子解析导数通过原残差/有限差分核验；CPU 加速使用等价半整数 Bessel 闭式及精确中心缓存，不改变积分节点、核或科学公式。投影/完整残差阈值分别为 1e−13/1e−11，CP 距离驻点阈值 1e−13。此 population 扫描未执行有限样本的四个法向重初值；已做实际 pilot 标量网格加密及全部双分辨率比较。

|方法|E_match 全七档斜率|最小四档斜率|E_match/σ⁴ 范围|
|---|---:|---:|---:|
'''
    for m in ['ENF','CP']:
        s=next(r for r in combined if r['method']==m and r['metric']=='E_match')
        md+=f"|{m}|{float(s['slope_all7']):.6f}|{float(s['slope_smallest4']):.6f}|{float(s['normalized_min']):.6g}–{float(s['normalized_max']):.6g}|\n"
    md+=f'''
14 个完整 cell、448 个查询/分辨率工作流全部成功（其中 4 个 query0 按相同源/参数复用有效 pilot，不重复拟合）。生产/加密目标点最大差异 {pa['maximum_point_difference']:.6g}，匹配最大误差差异 {pa['maximum_match_metric_difference']:.6g}，门槛 {pa['precision_budget']:.6g}（最小匹配最大误差的 0.1%）。全部通过。有限区间斜率和分辨率差是数值证据，不是渐近证明或严格数值误差界。

## 3. 高频 normal graph 反例

Rref=1.01、Rtarget=1.04、N=15625、a=2.56e−6。该目标曲线为人工构造，未声称由 STSP 算法产生。

- 局部向外径向 ENF 点为 (1.04000256,0)，在该局部法纤维上唯一；不声称整条法向直线只有一个交点。
- CP 两个对称角度为 ±{abs(float(counter['CP_angles'][1])):.15g}。
- g(0)={float(counter['ENF_g']):.16g}，g(CP)={float(counter['CP_g']):.16g}，下降 {float(counter['objective_gap']):.12g}。
- g''(0)={float(counter['g_second_derivative_at_zero']):.12g}；目标曲线中心曲率约 {float(counter['target_curvature_at_zero']):.10g}。
- 参考点圆周角 φ 从负侧到正侧趋于 0 时，最近点分支发生极限跳跃 {float(counter['limiting_CP_point_jump']):.12g}；ENF 随 φ 连续变化。φ=0 时 CP 是二值集合。

非唯一性还有独立于数值根枚举的解析核验：在 θ=π/N 处，利用 sin x≤x 与 π²<10，可得 g(π/N)−g(0)≤−2.64175721906176e−7<0。g 为偶函数，且 g(π)>g(0)，紧致性因此保证至少两个非零的对称全局最小点。数值根位置使用 80 位算术，并用角向距离下界排除远处圆弧。

这说明在该弱正则构造下 CP 无法提供单值且连续的最近点选择，而局部 ENF 对应仍然良定义；**不能把它解释为 CP 的四阶定位误差失效，也没有据此计算完整外推像的嵌入性**。

## 4. 理论与实验的对应

正常 STSP 分支：两者都展示四阶匹配精度，最终几何精度近似相同；ENF 此次直接求法纤维交点，避免了 CP 的表示与连续搜索成本。反例：ENF 的独立价值在于可控局部对应的唯一性与连续性，而非垄断四阶位置精度。两项观察支持不同的理论陈述，不能互相替代。

## 5. 文件与复现

- 主三面板图：../figures/e4_nn_density_accuracy.png / .pdf
- 平均误差与 E_base：../figures/P1_E4_correspondence_supplement.png / .pdf
- Population：../figures/P1_population_correspondence_scaling.png / .pdf
- 反例：../counterexample/P1_oscillatory_CP_nonuniqueness.png / .pdf
- 分支切换：../counterexample/P1_CP_branch_switch.png / .pdf
- 汇总表：P1_SUMMARY_TABLE.pdf，及两张机器可读 CSV 表
- 完整数据：../finite_sample、../finite_summary、../population、../population_summary、../counterexample
- 冷启动绘图入口：../reproduce_plots.py；科学重算入口见 ../README.md。无须重新拟合即可重画。

全部工作位于独立 P1_ENF_CP_20261103。原 E1/E4、正式图和发布包不覆盖、不上传。当前源与原数据绑定、双分辨率审计、实际视觉检查、冷重放各自记录，任何一种工程 PASS 都不能替代科学结果。
'''
    (out/'REPORT.md').write_text(md,encoding='utf8')
    impl=r'''\paragraph{Implementation of continuous CP.}
We retain the original STSP operator, sample parents, and the exact archived fine-scale reference coordinates of E4. At each reference point $y_1$, weighted PCA for the empirical $h_2$-scale operator supplies a tangent vector $T$ and a normal basis $B$. We construct the local fitted curve as $z(s)=y_1+sT+Bu(s)$, where every evaluation solves $B^\top\widehat R_{h_2}(z(s))=0$ and checks the full STSP residual. The derivative follows from the implicit-function equation; continuous stationary points of $\|z(s)-y_1\|^2$ are bracketed and refined by Brent's method, and their objective values are compared. A valid central fit gives an incumbent distance $d_0$, excluding points with $|T^\top(z-y_1)|>d_0$. We start with nine scalar nodes and refine to 41 or 81 when the sampled derivative is nonmonotone; four normal-direction starts check the selected empirical branch. The projected and full residual tolerances are $10^{-11}$ and $10^{-9}$, respectively, in addition to the original support, eigengap, transversality, and Jacobian checks; the normalized distance-stationarity tolerance is $10^{-10}$. This is continuous optimization on a numerically checked local STSP sheet, not nearest-neighbour selection from a candidate library or a formal global certificate for arbitrary spurious residual components. CP receives no true curve parameter, normal, Oracle point, or candidate coordinates. Its time includes PCA, chart construction, all residual solves, and continuous search. All four methods are measured afresh on the same single-thread CPU backend. The original controlled geometric information used to construct the common reference branch and NN candidates is retained and disclosed; true geometry is also used for Oracle and evaluation.

\paragraph{Population scaling.}
On the same space curve we use 32 fixed arclength queries, $h_1=3\sigma$, $h_2=6\sigma$, and $\sigma\in\{0.008,0.01,0.0125,0.016,0.02,0.025,0.03\}$. Expectations are evaluated by deterministic uniform-arclength latent quadrature and the Gaussian-reduced $K_6$ moments, without Monte Carlo. Production and refined latent/radial resolutions are $8192/160$ and $16384/224$, with tail padding 12. The projected/full residual tolerances are $10^{-13}/10^{-11}$ and the CP stationarity tolerance is $10^{-13}$. This population scan omits the finite-sample normal multistarts; actual pilot scalar-grid checks and full two-resolution comparisons are recorded. Resolution differences must remain below $10^{-3}$ times the smallest observed maximum matching error. Slopes are finite-range log--log regressions rather than asymptotic proofs.
'''
    (out/'implementation_details.tex').write_text(impl,encoding='utf8')
    discuss=rf'''\paragraph{{What ENF contributes beyond fourth-order accuracy.}}
On the original finite-sample STSP curve experiment, ENF and continuous CP have almost identical final geometric errors: their median maximum errors are ${float(e['E_infinity_median']):.6f}$ and ${float(c['E_infinity_median']):.6f}$, respectively, and both have zero failed matches in 320 queries. CP has the smaller direct correspondence error in this example. The median total CPU costs are ${float(e['total_seconds_median']):.3f}$ s for ENF and ${float(c['total_seconds_median']):.3f}$ s for CP, including target representation and search. This implementation-specific comparison does not provide a universal complexity lower bound for CP. In the deterministic population scan, the fitted slopes of $E_{{\mathrm{{match}}}}$ are ${float(es['slope_all7']):.4f}$ and ${float(cs['slope_all7']):.4f}$ for ENF and CP. Thus fourth-order correspondence accuracy is not exclusive to ENF. Its separate advantage is a directly computable local correspondence with controllable uniqueness and continuity. The oscillatory normal-graph example in the supplement isolates this distinction: global CP is nonunique and switches branches under tangential perturbations, whereas the local outward ENF intersection remains unique and continuous. This nonuniqueness does not contradict fourth-order position accuracy of the individual nearest points.
'''
    (out/'section_5_4_discussion.tex').write_text(discuss,encoding='utf8')
    supplementary=r'''\paragraph{A nonunique closest-point correspondence under weak first-order control.}
This constructed example is separate from the STSP-generated curves. Let $X_1(\theta)=1.01(\cos\theta,\sin\theta)$ and $X_2(\theta)=[1.04+a\cos(N\theta)](\cos\theta,\sin\theta)$, with $N=15625$ and $a=2.56\times10^{-6}$. For $y_1=(1.01,0)$ the local outward radial ENF intersection is uniquely $(1.04000256,0)$. Closest points minimize $g(\theta)=r(\theta)^2+1.01^2-2(1.01)r(\theta)\cos\theta$. High-precision evaluation locates two equal minima at $\theta\simeq\pm1.9034756261746264\times10^{-4}$, with $g=0.0008998866120100563<g(0)=0.0009001536065536$. Nonuniqueness also follows without trusting a stationary-point search: writing $R=1.04$ and $\rho=1.01$, the elementary bounds $\sin x\le x$ and $\pi^2<10$ give
\[
g(\pi/N)-g(0)\le -4a(R-\rho)+\frac{10\rho(R-a)}{N^2}
=-2.64175721906176\times10^{-7}<0.
\]
Since $g$ is even and $g(\pi)>g(0)$, compactness implies at least two symmetric global minimizers. Under reference-angle perturbations $\phi\to0^\pm$, the selected CP branch has a limiting point jump of approximately $3.9592196689924\times10^{-4}$, while the local ENF angle equals $\phi$ and varies continuously. Uniqueness here is local to the outward normal fibre, not the entire normal line. The example diagnoses a lack of a single-valued continuous CP selection; it does not refute fourth-order position accuracy or numerically establish loss of embeddedness of a complete extrapolated image.
'''
    (out/'supplementary_counterexample.tex').write_text(supplementary,encoding='utf8')
    table_figure(out,finite,pop,slopes,counter)
    manifest=dict(status='ACTUAL_ALL_PHASES',finite_audit_sha256=sha(ROOT/'finite_summary/AUDIT.json'),
        population_audit_sha256=sha(ROOT/'population_summary/AUDIT.json'),counter_sha256=sha(ROOT/'counterexample/RESULT.json'),
        generator_sha256=sha(__file__),outputs={x.name:sha(x) for x in out.iterdir() if x.is_file()})
    (out/'REPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(out)

if __name__=='__main__':main()
