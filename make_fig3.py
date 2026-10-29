"""Fig. 3: structure of the framework, drawn as a four-stage concept
diagram in which every panel carries the quantity it represents."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

from model_core import channels, couplings, gate
from robustness import channels_partition, blocks

BLUE, ORANGE, GREEN, BLACK = "#0072B2", "#E69F00", "#009E73", "#000000"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
})

IMAX, DI = 3.0, 0.4
g = couplings()
t = np.linspace(0.0, 0.45, 20001)
x = t / channels(np.array([1e-9]), g)["T2"] * 0 + t / channels(t, g)["T2"]
d1 = channels(t, g, kappa=1.0)
x = t / d1["T2"]
base = d1["coherence"] / d1["coherence"][0]
g1 = gate(d1["I_total"], IMAX, DI)
c1 = base * g1 / g1[0]
t_sat = x[int(np.argmax(d1["I_total"] >= IMAX))]

# matched-T2 coupling structures for panel (c)
N = 20
jj = np.arange(N)
fA = 1.0 + 0.03 * (2 * (jj + 0.5) / N - 1.0)
fB = 10.0 ** ((jj / (N - 1.0)) - 0.5)
fB *= np.sqrt((fA ** 2).sum() / (fB ** 2).sum())
tm = np.linspace(0.0, 0.5, 20001)
resid = {}
for name, f in (("A", fA), ("B", fB)):
    dm = channels_partition(tm, f, blocks(N, 1))
    xm = tm / dm["T2"]
    bm = dm["coherence"] / dm["coherence"][0]
    qm = gate(dm["I_total"], IMAX, DI)
    qm = qm / qm[0]
    resid[name] = (xm, bm - bm * qm)

fig = plt.figure(figsize=(7.6, 3.7))
gs = fig.add_gridspec(1, 2, left=0.055, right=0.985, bottom=0.275, top=0.965,
                      wspace=0.28)
ax = [fig.add_subplot(gs[0, k]) for k in range(2)]

# ---- (a) region of support ------------------------------------------------
a = ax[0]
a.set_aspect("equal")
a.set_xlim(-1.40, 1.55)
a.set_ylim(-1.30, 1.30)
a.axis("off")
a.add_patch(Circle((0, 0), 1.0, facecolor="#E8F1F8", edgecolor=BLUE,
                   linewidth=1.3, zorder=1))
th = np.linspace(0, 2 * np.pi, 13)[:-1]
rr = 0.56 + 0.24 * np.cos(3.0 * th)
a.scatter(rr * np.cos(th), rr * np.sin(th), s=15, color=BLUE, zorder=3,
          edgecolor="white", linewidth=0.4)
a.scatter([0], [0], s=150, color=ORANGE, zorder=4, edgecolor=BLACK,
          linewidth=0.8)
a.annotate("system", xy=(0.07, -0.07), xytext=(0.78, -1.10),
           fontsize=13, ha="left", va="center",
           arrowprops=dict(arrowstyle="-", lw=0.9, color=BLACK,
                           shrinkA=2, shrinkB=0))
a.annotate("record fragments", xy=(rr[4] * np.cos(th[4]), rr[4] * np.sin(th[4])),
           xytext=(-1.38, 1.16), fontsize=13, ha="left", va="center",
           arrowprops=dict(arrowstyle="-", lw=0.9, color=BLUE,
                           shrinkA=2, shrinkB=2))
a.add_patch(FancyArrowPatch((0, 0), (0.94 * np.cos(np.deg2rad(48)),
                                     0.94 * np.sin(np.deg2rad(48))),
                            arrowstyle="-|>", mutation_scale=10,
                            linewidth=1.1, color=BLACK, zorder=5))
a.text(0.30, 0.70, r"$R$", fontsize=16, zorder=6)

# ---- (b) budget column ----------------------------------------------------
b = ax[1]
i = int(np.argmin(np.abs(x - 0.62)))
vals = [d1["I_past"][i], d1["I_present"][i], d1["I_future"][i]]
bottom = 0.0
LAB = [r"$I_{\rm past}$", r"$I_{\rm present}$", r"$I_{\rm future}$"]
for v, col, hh, lb in zip(vals, [BLUE, ORANGE, GREEN], ["", "//", "xx"], LAB):
    b.bar([0], [v], bottom=[bottom], width=0.75, color=col, hatch=hh,
          edgecolor="white", linewidth=0.6, zorder=2, label=lb)
    bottom += v
b.axhline(IMAX, color=BLACK, lw=1.6, zorder=3)
b.annotate("", xy=(0.44, bottom), xytext=(0.44, IMAX),
           arrowprops=dict(arrowstyle="|-|", color="0.35", lw=0.9,
                           mutation_scale=3), zorder=4)
# direct labels beside the column, one per segment, in place of a legend
NAMES = ["records", "coherence", "future"]
lo = 0.0
for v, lb, nm in zip(vals, LAB, NAMES):
    b.text(0.56, lo + v / 2, rf"{lb} ({nm}): ${v:.2f}$ bits", fontsize=11.5,
           ha="left", va="center")
    lo += v
b.text(0.56, (bottom + IMAX) / 2, rf"still open: ${IMAX - bottom:.2f}$ bits",
       fontsize=11.5, ha="left", va="center", color="0.25")
b.text(-0.58, IMAX + 0.10, r"$I_{\rm cap}$", fontsize=14, va="bottom")
b.set_xlim(-0.62, 2.10)
b.set_ylim(0, 3.75)
b.set_yticks([0, 1, 2, 3])
b.set_xticks([])
b.set_ylabel("information (bits)", fontsize=13.5, labelpad=2)
b.tick_params(labelsize=12)
for sp in ("top", "right"):
    b.spines[sp].set_visible(False)

# ---- panel labels ------------------------------------
for k, axk in enumerate(ax):
    pos = axk.get_position()
    off = 0.135   # common baseline for all three panel labels
    fig.text((pos.x0 + pos.x1) / 2, pos.y0 - off, f"({chr(97 + k)})",
             ha="center", va="top", fontsize=17)

fig.savefig("fig_structure.png", dpi=400, bbox_inches="tight")

print("t_sat/T2 =", round(t_sat, 3))
print("budget at stage shown:", [round(v, 2) for v in vals])
for k in ("A", "B"):
    xx, dd = resid[k]
    m = xx <= 3
    print(f"model {k}: max residual {dd[m].max():.3f} at t/T2 {xx[m][dd[m].argmax()]:.2f}")
