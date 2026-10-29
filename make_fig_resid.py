"""Matched-T2 coupling-structure comparison, as a standalone figure placed
where Sec. 6 discusses it."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model_core import gate
from robustness import channels_partition, blocks

BLUE, ORANGE, BLACK = "#0072B2", "#E69F00", "#000000"
IMAX, DI = 3.0, 0.4

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 12, "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
})

N = 20
jj = np.arange(N)
fA = 1.0 + 0.03 * (2 * (jj + 0.5) / N - 1.0)          # near-uniform
fB = 10.0 ** ((jj / (N - 1.0)) - 0.5)                  # heterogeneous
fB *= np.sqrt((fA ** 2).sum() / (fB ** 2).sum())       # matched second moment

tm = np.linspace(0.0, 0.5, 20001)
resid = {}
for name, f in (("A", fA), ("B", fB)):
    dm = channels_partition(tm, f, blocks(N, 1))
    xm = tm / dm["T2"]
    bm = dm["coherence"] / dm["coherence"][0]
    qm = gate(dm["I_total"], IMAX, DI)
    qm = qm / qm[0]
    resid[name] = (xm, bm - bm * qm)

fig, ax = plt.subplots(figsize=(5.0, 3.6))
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.155, top=0.965)

for key, col, ls, lab in (("A", BLUE, "-", "uniform couplings"),
                          ("B", ORANGE, (0, (5, 1.8)),
                           "heterogeneous couplings")):
    xx, dd = resid[key]
    ax.plot(xx, dd, color=col, ls=ls, lw=2.0, label=lab)
    m = xx <= 3
    k = int(dd[m].argmax())
    ax.plot(xx[m][k], dd[m][k], "o", color=col, ms=6, zorder=6)
    ax.annotate(rf"${dd[m][k]:.2f}$", xy=(xx[m][k], dd[m][k]),
                xytext=(22, -20), textcoords="offset points",
                fontsize=12, color=col)

ax.set_xlim(0, 3)
ax.set_ylim(-0.012, 0.50)
ax.set_xlabel(r"$t/T_2$", fontsize=13)
ax.set_ylabel(r"$\Delta C_{\rm RCC}(t)$", fontsize=13)
ax.tick_params(labelsize=11.5)
# legend inside the axes, in the empty upper-right region beside the
# curves, clear of the data and of the peak labels
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False,
          fontsize=11, handlelength=1.9, labelspacing=0.3,
          borderaxespad=0.1)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

fig.savefig("fig_residual.png", dpi=600)
print("wrote fig_residual.png")
for key in ("A", "B"):
    xx, dd = resid[key]
    m = xx <= 3
    print(f"  model {key}: peak {dd[m].max():.3f} at t/T2 {xx[m][dd[m].argmax()]:.2f}")
