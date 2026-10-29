r"""Fig. 1: the mechanism.  The system splits into two conditional
alternatives, each dragging its own environmental record and its own
inaccessible state; the two overlaps multiply into the observable
coherence, and the energy of the inaccessible sector bounds its own
factor."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, ORANGE, GREY, BLACK = "#0072B2", "#E69F00", "#8C8C8C", "#000000"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 11, "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(8.3, 4.0))
ax.set_xlim(-0.01, 1.20); ax.set_ylim(0.145, 1.075); ax.axis("off")


def box(x, y, w, h, txt, edge, face, fs=12, z=4):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.012", linewidth=1.3,
                                edgecolor=edge, facecolor=face, zorder=z))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs, zorder=z + 1)


def arrow(p, q, col=BLACK, lw=1.2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=lw, color=col, zorder=3,
                                 shrinkA=0, shrinkB=0))


# ---- the system and its two conditional alternatives ---------------------
box(0.085, 0.62, 0.115, 0.115, r"$S$", BLACK, "white", 14)
ax.text(0.085, 0.548, "system", ha="center", va="top", fontsize=9.5,
        color=GREY)

for y, ket in ((0.82, r"$|0\rangle$"), (0.42, r"$|1\rangle$")):
    arrow((0.156, 0.62), (0.2185, y))
    box(0.275, y, 0.085, 0.10, ket, BLACK, "white", 12.5)

# ---- what each alternative carries ---------------------------------------
for y, e, x in ((0.82, r"$|E_0\rangle$", r"$|X_0\rangle$"),
                (0.42, r"$|E_1\rangle$", r"$|X_1\rangle$")):
    arrow((0.332, y), (0.3905, y + 0.078), BLUE)
    arrow((0.332, y), (0.3905, y - 0.078), ORANGE)
    box(0.470, y + 0.105, 0.135, 0.10, e, BLUE, "#EAF3FA", 12.5)
    box(0.470, y - 0.105, 0.135, 0.10, x, ORANGE, "#FDF4E6", 12.5)

ax.text(0.470, 1.062, "environmental records", ha="center", va="top",
        fontsize=9.5, color=BLUE)
ax.text(0.470, 0.196, "inaccessible sector", ha="center", va="top",
        fontsize=9.5, color=ORANGE)

# ---- the two overlaps ----------------------------------------------------
EDGE = 0.5495                     # visible right edge of the |E>, |X> boxes
JOIN = {}
for y0, y1, xv, ylab, lab, col in (
        (0.715, 0.315, 0.582, 0.470, r"$r_X=\langle X_1|X_0\rangle$", ORANGE),
        (0.925, 0.525, 0.630, 0.775, r"$r_E=\langle E_1|E_0\rangle$", BLUE)):
    ax.plot([xv, xv], [y0, y1], color=col, lw=1.3, zorder=3)
    for y in (y0, y1):
        ax.add_patch(FancyArrowPatch((xv, y), (EDGE, y), arrowstyle="-|>",
                                     mutation_scale=10, linewidth=1.3,
                                     color=col, zorder=3,
                                     shrinkA=0, shrinkB=0))
    ax.plot([xv, 0.660], [ylab, ylab], color=col, lw=1.3, zorder=3)
    ax.text(0.670, ylab, lab, ha="left", va="center", fontsize=11.5,
            color=col)
    JOIN[col] = ylab

# ---- the observable coherence -------------------------------------------
box(1.050, 0.620, 0.245, 0.120,
    r"$C_{\rm obs}=C_{0}|r_E||r_X|$", BLACK, "white", 11.5, z=6)

# the two overlaps join and feed the observable coherence
for col in (BLUE, ORANGE):
    y = JOIN[col]
    ax.plot([0.836, 0.858], [y, y], color=BLACK, lw=1.2, zorder=2)
ax.plot([0.858, 0.858], [JOIN[BLUE], JOIN[ORANGE]], color=BLACK, lw=1.2,
        zorder=2)
ax.add_patch(FancyArrowPatch((0.858, 0.620), (0.9155, 0.620),
                             arrowstyle="-|>", mutation_scale=12,
                             linewidth=1.2, color=BLACK, zorder=4,
                             shrinkA=0, shrinkB=0))

# ---- the energy constraint on the hidden factor --------------------------
ax.add_patch(FancyBboxPatch((0.925, 0.185), 0.250, 0.185,
                            boxstyle="round,pad=0.012", linewidth=1.1,
                            edgecolor=ORANGE, facecolor="#FDF4E6", zorder=4))
ax.text(1.050, 0.316, r"$E_X\rightarrow|r_X|\rightarrow\chi_X$",
        ha="center", va="center", fontsize=11, zorder=5)
ax.text(1.050, 0.238, r"$\chi_X\leq C_X(E_X)$", ha="center",
        va="center", fontsize=11.5, zorder=5)
ax.add_patch(FancyArrowPatch((1.050, 0.382), (1.050, 0.548),
                             arrowstyle="-|>", mutation_scale=12,
                             linewidth=1.3, color=ORANGE, zorder=3,
                             shrinkA=0, shrinkB=0))

# ---- everything above is supported in one bounded region ---------------
ax.add_patch(FancyBboxPatch((0.020, 0.235), 0.535, 0.755,
                            boxstyle="round,pad=0.010", linewidth=1.0,
                            linestyle=(0, (5, 3)), edgecolor=GREY,
                            facecolor="none", zorder=1))
ax.text(0.028, 0.196, r"bounded region $\mathcal{R}$", ha="left",
        va="top", fontsize=10, color=GREY)

fig.savefig("fig_region.png", dpi=600, bbox_inches="tight")
print("wrote fig_region.png")
