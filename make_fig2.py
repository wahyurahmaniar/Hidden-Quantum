"""Fig. 2: four stages of a measurement interaction, with each budget bar
evaluated from the solved central-spin model."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyArrowPatch

from model_core import channels, couplings

BLUE, ORANGE, GREEN, BLACK = "#0072B2", "#E69F00", "#009E73", "#000000"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 13,
    "axes.linewidth": 1.0,
    "xtick.direction": "out",
    "pdf.fonttype": 42,
})

IMAX = 3.0
g = couplings()
t = np.linspace(0.0, 0.6, 40001)
d = channels(t, g)
x = t / d["T2"]
stages = [0.0, 0.35, 0.62, 1.05]
idx = [int(np.argmin(np.abs(x - v))) for v in stages]

states = [r"$\frac{1}{\sqrt{2}}(|0\rangle+|1\rangle)\otimes|+\rangle^{\otimes N}$",
          r"$\frac{1}{\sqrt{2}}\sum_s|s\rangle\,|E^{s}(t)\rangle$",
          r"$|r(t)|<1$",
          r"$C_{\rm RCC}<C_{\rm env},\;\; q(t)<1$"]
stamps = [r"$t=0,\;\;|r|=1$",
          r"$t=0.35\,T_2,\;\;|r|=0.88$",
          r"$t=0.62\,T_2,\;\;|r|=0.68$",
          r"$t=1.05\,T_2,\;\;|r|=0.32$"]

fig, ax = plt.subplots(2, 4, figsize=(11.0, 3.15),
                       gridspec_kw=dict(height_ratios=[0.36, 1.0],
                                        hspace=0.30, wspace=0.30,
                                        top=0.90, bottom=0.26))

for k, i in enumerate(idx):
    a = ax[0, k]
    a.axis("off")
    a.text(0.5, 0.02, states[k], ha="center", va="center", fontsize=15)

    b = ax[1, k]
    vals = [d["I_past"][i], d["I_present"][i], d["I_future"][i]]
    left = 0.0
    for v, c, hh in zip(vals, [BLUE, ORANGE, GREEN], ["", "//", "xx"]):
        b.barh([0], [v], left=[left], height=0.55, color=c, hatch=hh,
               edgecolor="white", linewidth=0.6)
        left += v
    b.axvline(IMAX, color=BLACK, lw=2.4)
    b.set_xlim(0, 4.6)
    b.set_ylim(-0.55, 0.55)
    b.set_yticks([])
    b.set_xticks([0, 1, 2, 3, 4])
    b.text(0.5, -0.34, stamps[k], transform=b.transAxes,
           ha="center", va="top", fontsize=12.5)
    b.text(0.5, -0.60, f"({chr(97 + k)})", transform=b.transAxes,
           ha="center", va="top", fontsize=15)

handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
           plt.Rectangle((0, 0), 1, 1, color=ORANGE, hatch="//"),
           plt.Rectangle((0, 0), 1, 1, color=GREEN, hatch="xx"),
           plt.Line2D([0], [0], color=BLACK, lw=1.8)]
labels = [r"$I_{\rm past}$", r"$I_{\rm present}$", r"$I_{\rm future}$",
          r"$I_{\rm cap}$"]
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 0.995), handlelength=1.9, columnspacing=2.4,
           fontsize=13)

# temporal connectors, placed in the gutter between the budget panels
for k in range(3):
    p0 = ax[1, k].get_position()
    p1 = ax[1, k + 1].get_position()
    y = (p0.y0 + p0.y1) / 2
    fig.add_artist(FancyArrowPatch((p0.x1 + 0.009, y), (p1.x0 - 0.009, y),
                                   transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=12,
                                   linewidth=1.1, color="#7F7F7F"))

# temporal connectors, placed in the gaps between the budget axes
fig.canvas.draw()
for k in range(3):
    p0 = ax[1, k].get_position()
    p1 = ax[1, k + 1].get_position()
    gap = p1.x0 - p0.x1
    y = (p0.y0 + p0.y1) / 2
    fig.add_artist(FancyArrowPatch((p0.x1 + 0.22 * gap, y),
                                   (p1.x0 - 0.22 * gap, y),
                                   transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=13,
                                   linewidth=1.2, color="#5A5A5A",
                                   zorder=10))

fig.savefig("fig_mechanism.png", dpi=400, bbox_inches="tight")

for v, i in zip(stages, idx):
    print(f"t/T2={v:.2f}: past={d['I_past'][i]:.2f} pres={d['I_present'][i]:.2f}"
          f" fut={d['I_future'][i]:.2f} total={d['I_total'][i]:.2f}")
