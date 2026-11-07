#!/usr/bin/env python3
"""AoS-style two-panel multiscale manifold correspondence schematic.

Regenerate with: python draw_figure1.py
Produces editable vector SVG, embedded-font vector PDF,
and a 600 dpi PNG. All correspondences use analytic curves and a continuous
root solve, not discretized nearest-neighbour matching.

The drawing is a schematic and the offsets/angles are illustration
parameters, not actual estimator error bounds.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parent

# ---- Changeable design parameters -----------------------------------------
A = 0.145                   # curvature of the true local curve
D1 = 0.320                  # fine-scale normal offset
D2 = 0.550                  # coarse-scale normal offset
W1, W2 = 2.0, -1.0          # illustrative affine extrapolation weights
T_LEFT, T_RIGHT = 0.0, -0.24
TILT = 0.48                 # visibly tilted estimated normal (schematic)
# Professional muted colors, no saturated hues
COLORS = {
    "true": "#33383C",      # charcoal
    "fine": "#748A95",      # desaturated blue grey
    "coarse": "#AF8579",    # muted warm clay
    "corrected": "#738E83", # muted sage
    "normal": "#969DA1",    # grey true normal
    "estimated": "#60696F", # slate estimated normal
    "leader": "#B2B5B7",    # pale annotations
    "text": "#303539",      # main text
}
# If preferred for a particular manuscript revision, change False to True
# to display hats on scale-specific fitted manifolds.
USE_HATS = False

mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 11.5,
    "svg.fonttype": "none",  # SVG master retains text objects
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
    "savefig.facecolor": "white",
    "lines.dash_capstyle": "butt",
    "lines.solid_capstyle": "round",
})


def P(t):
    t = np.asarray(t)
    return np.stack((t, A*t*t), axis=-1)


def T(t):
    t = np.asarray(t)
    v = np.stack((np.ones_like(t), 2*A*t), axis=-1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def N(t):
    t = np.asarray(t)
    v = np.stack((-2*A*t, np.ones_like(t)), axis=-1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def F(t, d):
    return P(t) + d*N(t)


def N_est(t):
    v = N(t) + TILT*T(t)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def cross(a, b):
    return a[0]*b[1] - a[1]*b[0]


def enf_points(t):
    """First point, target intersection, extrapolated point, root parameter.

    Return the unique *local* intersection of the estimated normal line
    through the reference point with the coarse target curve.
    """
    y1 = F(t, D1)
    normal = N_est(t)
    fun = lambda u: cross(normal, F(u, D2)-y1)
    span = 0.24
    lo, hi = t-span, t+span
    assert fun(lo)*fun(hi) < 0, f"No local bracket at t={t}"
    u = brentq(fun, lo, hi, xtol=5e-15, rtol=1e-14)
    z2 = F(u, D2)
    s = np.dot(z2-y1, normal)
    assert s > 0
    assert abs(fun(u)) < 1e-11
    zcorrected = W1*y1 + W2*z2
    return y1, z2, zcorrected, u


def dist_to_true(z):
    o = minimize_scalar(lambda t: np.sum((P(t)-z)**2), bounds=(-1.0, 1.0), method="bounded",
                        options={"xatol": 1e-14})
    return float(np.sqrt(o.fun))


def assert_geometry():
    xL = P(T_LEFT)
    y1L = F(T_LEFT, D1)
    y2L = F(T_LEFT, D2)
    zL = W1*y1L + W2*y2L
    zL_expected = F(T_LEFT, W1*D1 + W2*D2)
    assert abs(W1+W2-1) < 1e-14
    assert np.linalg.norm(zL-zL_expected) < 1e-12
    assert abs(cross(y1L-xL,N(T_LEFT))) < 1e-12
    assert abs(cross(y2L-xL,N(T_LEFT))) < 1e-12
    assert abs(cross(zL-xL,N(T_LEFT))) < 1e-12
    xR = P(T_RIGHT)
    y1R,z2R,zR,u = enf_points(T_RIGHT)
    ntrue = N(T_RIGHT)
    nhat = N_est(T_RIGHT)
    oracle2 = F(T_RIGHT,D2)
    assert abs(cross(y1R-xR,ntrue)) < 1e-12
    assert abs(cross(oracle2-xR,ntrue)) < 1e-12
    assert abs(cross(z2R-y1R,nhat)) < 1e-11
    assert abs(cross(zR-y1R,nhat)) < 1e-11
    assert abs(cross(ntrue,nhat)) > 0.20
    assert np.linalg.norm(zR-(W1*y1R+W2*z2R)) < 1e-12
    distances = [dist_to_true(q) for q in (y1R, z2R, zR)]
    assert distances[2] < min(distances[:2])
    return {"left corrected error": float(np.linalg.norm(zL-zL_expected)),
            "right intersection root u": float(u),
            "right estimated-normal residual": float(abs(cross(z2R-y1R,nhat))),
            "right extrapolated-normal residual": float(abs(cross(zR-y1R,nhat))),
            "right distance (fine, coarse, corrected)": distances}


def point(ax, pt, kind, color, size=None, zorder=9):
    symbols = {"true":"o", "fine":"o", "coarse":"s", "corrected":"D", "oracle":"o"}
    sz = size or {"true":29,"fine":36,"coarse":32,"corrected":39,"oracle":25}[kind]
    if kind in ("fine", "corrected", "oracle"):
        face = "white"
        edge = color
        width = 1.1 if kind != "oracle" else 0.85
    else:
        face = color
        edge = "white"
        width = 0.55
    ax.scatter([pt[0]],[pt[1]],marker=symbols[kind],s=sz,facecolor=face,
               edgecolor=edge,linewidth=width,zorder=zorder)


LABELS = []


def textlabel(ax, label, pt, target, ha="left", va="center", size=11.5, color=None):
    artist = ax.annotate(label, xy=pt, xytext=target, xycoords="data",textcoords="data",
                ha=ha,va=va,fontsize=size,color=color or COLORS["text"],
                arrowprops=dict(arrowstyle="-", color=COLORS["leader"], lw=0.65,
                                shrinkA=3.0, shrinkB=4.0, connectionstyle="arc3,rad=0"),
                zorder=12,annotation_clip=False)
    LABELS.append((ax, artist))
    return artist


def graph_m_label(ax, label, pt, target):
    # Exact curve endpoints, including the horizontal normal offset.
    return textlabel(ax, label, pt, target, size=12.0)


def draw_base(ax, corrected_points, panel):
    t = np.linspace(-1.46,1.42,680)
    ax.plot(*P(t).T, color=COLORS["true"],lw=1.25,ls="-",zorder=3)
    ax.plot(*F(t,D1).T,color=COLORS["fine"],lw=1.25,
            linestyle=(0,(2.6,2.0)),zorder=4)
    ax.plot(*F(t,D2).T,color=COLORS["coarse"],lw=1.25,
            linestyle=(0,(7.0,2.6)),zorder=4)
    ax.plot(corrected_points[:,0],corrected_points[:,1],
            color=COLORS["corrected"],lw=1.40,linestyle=(0,(5.0,1.6,1.0,1.6)),zorder=5)
    endpoints={"true":P(t[-1]),"fine":F(t[-1],D1),
               "coarse":F(t[-1],D2),"corrected":corrected_points[-1]}
    mhat = r"\widehat{\mathcal{M}}" if USE_HATS else r"\mathcal{M}"
    labels={"true":r"$\mathcal{M}$", "fine":rf"${mhat}_{{h_1}}$",
            "coarse":rf"${mhat}_{{h_2}}$", "corrected":r"$\widetilde{\mathcal{M}}$"}
    # Separate rows, with identical label locations in both panels.
    rows={"coarse":.89,"fine":.64,"corrected":.42,"true":.19}
    for key in ("coarse","fine","corrected","true"):
        graph_m_label(ax, labels[key],endpoints[key],(1.65,rows[key]))
    ax.set_xlim(-1.68,2.17)
    ax.set_ylim(-.23,1.11)
    ax.set_aspect("auto")
    ax.axis("off")
    artist=ax.text(0.50,-0.10, panel,transform=ax.transAxes,ha="center",va="top",fontsize=11.0,
                   color=COLORS["text"])
    LABELS.append((ax, artist))


def validate_layout(fig):
    """Check text bounds and pairwise text overlap at the final physical size."""
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    records=[]
    overlaps=[]
    clipped=[]
    curve_collisions=[]
    for i,(ax,artist) in enumerate(LABELS):
        # Annotation.get_window_extent also includes the leader; use text only.
        bbox=Text.get_window_extent(artist,renderer).expanded(1.04,1.08)
        records.append((ax,artist,bbox))
        if not fig.bbox.contains(bbox.x0,bbox.y0) or not fig.bbox.contains(bbox.x1,bbox.y1):
            clipped.append(artist.get_text())
        for other_ax,other,other_bbox in records[:i]:
            if ax is other_ax and bbox.overlaps(other_bbox):
                overlaps.append([other.get_text(),artist.get_text()])
        for line in ax.lines:
            path=line.get_transform().transform_path(line.get_path())
            if path.intersects_bbox(bbox,filled=False):
                curve_collisions.append(artist.get_text())
    assert not clipped, f"Text outside the figure: {clipped}"
    assert not overlaps, f"Overlapping labels: {overlaps}"
    assert not curve_collisions, f"Text touching a curve or normal: {curve_collisions}"
    return {"text_labels":len(records),"overlapping_text_pairs":overlaps,
            "clipped_text":clipped,"text_curve_collisions":curve_collisions,
            "label_font_size_pt":[11.0,12.0],
            "figure_width_cm":18.0,"figure_height_cm":7.6}


def main():
    stats = assert_geometry()
    LABELS.clear()
    t = np.linspace(-1.46,1.42,680)
    ideal_corr = F(t,W1*D1+W2*D2)
    empirical_corr = np.stack([enf_points(float(u))[2] for u in t])
    fig = plt.figure(figsize=(18.0/2.54,7.6/2.54),facecolor="white")
    aL=fig.add_axes([.015,.185,.470,.785])
    aR=fig.add_axes([.515,.185,.470,.785])
    draw_base(aL,ideal_corr,r"(a) True-normal correspondence")
    draw_base(aR,empirical_corr,r"(b) Estimated-normal correspondence")

    # Left: one line, four exactly aligned points.
    x=P(T_LEFT); y1=F(T_LEFT,D1); y2=F(T_LEFT,D2); z=W1*y1+W2*y2
    ns=np.linspace(-.11,.85,40)
    line=x[None,:]+ns[:,None]*N(T_LEFT)[None,:]
    aL.plot(line[:,0],line[:,1],color=COLORS["normal"],lw=.80,ls=(0,(3,2)),zorder=6)
    point(aL,x,"true",COLORS["true"])
    point(aL,y1,"fine",COLORS["fine"])
    point(aL,y2,"coarse",COLORS["coarse"])
    point(aL,z,"corrected",COLORS["corrected"])
    normal_label=aL.text(x[0],.92,r"$n(x)$",fontsize=11.5,color=COLORS["text"],ha="center")
    LABELS.append((aL,normal_label))
    textlabel(aL,r"$X_{h_2}(x)$",y2,(-.76,.79),ha="center")
    textlabel(aL,r"$X_{h_1}(x)$",y1,(.32,.49))
    textlabel(aL,r"$\widetilde{X}(x)$",z,(-1.08,.35))
    x_label=aL.text(x[0]+.11,x[1]-.14,r"$x$",color=COLORS["text"],fontsize=12.0,ha="left")
    LABELS.append((aL,x_label))

    # Right: true normal through x and y1; estimated normal through y1 and z2.
    x=P(T_RIGHT); y1,z2,zcorr,u=enf_points(T_RIGHT)
    oracle2=F(T_RIGHT,D2)
    norm=N(T_RIGHT); e_norm=N_est(T_RIGHT)
    ss=np.linspace(-.14,.90,60)
    Ltrue=x[None,:]+ss[:,None]*norm[None,:]
    aR.plot(Ltrue[:,0],Ltrue[:,1],color=COLORS["normal"],lw=.83,ls="-",zorder=6)
    ss2=np.linspace(-.40,.56,60)
    Lhat=y1[None,:]+ss2[:,None]*e_norm[None,:]
    aR.plot(Lhat[:,0],Lhat[:,1],color=COLORS["estimated"],lw=.85,
            ls=(0,(3,2)),zorder=7)
    point(aR,x,"true",COLORS["true"])
    point(aR,y1,"fine",COLORS["fine"])
    point(aR,z2,"coarse",COLORS["coarse"])
    point(aR,zcorr,"corrected",COLORS["corrected"])
    point(aR,oracle2,"oracle",COLORS["coarse"],size=14,zorder=8)
    # True-normal / ENF labels positioned in the top open space.
    textlabel(aR,r"$n(x)$",Ltrue[-1],(-.27,1.03),ha="center")
    textlabel(aR,r"$\widetilde{n}(y_1)$",Lhat[-1],(.61,1.00),ha="center")
    textlabel(aR,r"$X_{h_2}(x)$",oracle2,(-1.12,.88),ha="center")
    textlabel(aR,r"$Z_{h_2}(x)$",z2,(.44,.75))
    textlabel(aR,r"$y_1=X_{h_1}(x)$",y1,(.02,.47),size=11.0)
    textlabel(aR,r"$\widetilde{X}(x)$",zcorr,(-1.12,.34))
    x_label=aR.text(x[0]+.12,x[1]-.15,r"$x$",color=COLORS["text"],fontsize=12.0)
    LABELS.append((aR,x_label))

    # The minimalist diagram intentionally omits background scatter and axes.
    layout=validate_layout(fig)
    svg=ROOT/"figure1_clear.svg"
    png=ROOT/"figure1_clear.png"
    pdf=ROOT/"figure1_clear.pdf"
    fig.savefig(svg,format="svg",transparent=False,facecolor="white")
    fig.savefig(pdf,format="pdf",transparent=False,facecolor="white",
                metadata={"Title":"Multiscale manifold correspondence",
                          "Subject":"Analytic schematic; illustrative offsets and angles"})
    fig.savefig(png,format="png",dpi=600,transparent=False,facecolor="white")
    fig.savefig(ROOT/"figure1_preview.png",format="png",dpi=240,facecolor="white")
    plt.close(fig)
    pdf_result="Embedded-font vector PDF exported directly by Matplotlib"
    (ROOT/"validation.txt").write_text(
        "Mathematical validation passed\n" +
        "\n".join(f"{key}: {value}" for key,value in stats.items()) +
        "\n"+pdf_result+"\n",encoding="utf-8")
    (ROOT/"validation.json").write_text(json.dumps({"geometry":stats,"layout":layout},indent=2),encoding="utf-8")
    print("\n".join(f"{key}: {value}" for key,value in stats.items()))
    print(svg, pdf, png)

if __name__ == "__main__":
    main()
