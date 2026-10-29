"""Shared style for the figures added or redrawn in the revision: the same
serif STIX type, inward ticks and Okabe-Ito colours as the other figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
VERM, PURPLE, SKY, BLACK = "#D55E00", "#CC79A7", "#56B4E9", "#000000"
GREY = "0.45"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "pdf.fonttype": 42,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "legend.handlelength": 2.2,
    "legend.labelspacing": 0.35,
    "legend.borderaxespad": 0.5,
})


def panel_label(ax, k, y=-0.155):
    """Panel letter centred below the axes, as in the other figures."""
    ax.text(0.5, y, f"({chr(97 + k)})", transform=ax.transAxes,
            ha="center", va="top", fontsize=12)


def finish(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=11.5)
    ax.set_ylabel(ylabel, fontsize=11.5)
