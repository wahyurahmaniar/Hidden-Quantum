"""Hidden-sector dynamics and its effect on the observable coherence."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model_core import couplings
from model_xosc import r_X, E_X, chi_X, C_X

BLUE, ORANGE, GREEN, BLACK = "#0072B2", "#E69F00", "#009E73", "#000000"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10.5, "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
})

OM = 1.0
tau = np.linspace(0.0, 2.0 * np.pi / OM, 2001)          # one hidden period
G = (0.1, 0.2, 0.35, 0.6)
STY = ((BLUE, "-"), (ORANGE, (0, (5, 1.8))), (GREEN, (0, (1.5, 1.5))),
       (BLACK, (0, (4, 1.5, 1, 1.5))))

fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.1),
                       gridspec_kw=dict(wspace=0.32, bottom=0.30, top=0.94,
                                        left=0.06, right=0.985))

a, b, c = ax
for gX, (col, ls) in zip(G, STY):
    a.plot(tau / np.pi, r_X(tau, gX, OM), color=col, ls=ls, lw=1.7,
           label=rf"$g_X/\hbar\Omega={gX}$")
a.set_xlabel(r"$\Omega t/\pi$")
a.set_ylabel(r"$|r_X(t)|$")
a.set_xlim(0, 2); a.set_ylim(-0.03, 1.05)
a.legend(frameon=False, fontsize=8.5, handlelength=1.7,
         loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
         columnspacing=1.2, labelspacing=0.3, borderaxespad=0.15)

gX = 0.35
n = E_X(tau, gX, OM) / (1.0 * OM)
b.plot(tau / np.pi, chi_X(tau, gX, OM), color=BLUE, lw=1.8,
       label=r"$\chi_X(t)$")
b.plot(tau / np.pi, C_X(n), color=ORANGE, lw=1.8, ls=(0, (5, 1.8)),
       label=r"$C_X(\bar n_X)$")
b.fill_between(tau / np.pi, chi_X(tau, gX, OM), C_X(n), color=ORANGE,
               alpha=0.16, linewidth=0)
b.axhline(1.0, color="0.35", lw=0.9, ls=(0, (1.2, 1.8)), zorder=1)
b.set_xlabel(r"$\Omega t/\pi$")
b.set_ylabel("information (bits)")
b.set_xlim(0, 2)
b.text(0.03, 0.93, rf"$g_X/\hbar\Omega={gX}$", transform=b.transAxes,
       fontsize=9, va="top")
b.legend(frameon=False, fontsize=8.5, handlelength=1.9, loc="upper right",
         borderaxespad=0.2)

nb = np.logspace(-3, 1, 600)
rr = np.exp(-2.0 * nb)
from model_xosc import h2
c.semilogx(nb, h2((1 + rr) / 2) / C_X(nb), color=BLUE, lw=1.8)
c.axhline(1.0, color=BLACK, lw=0.9, ls=(0, (2, 2)))
# informative regime of the floor: C_X < 1 bit, nbar_X < 0.294
from scipy.optimize import brentq
NSTAR = brentq(lambda v: float(C_X(v)) - 1.0, 1e-6, 5.0)
c.axvspan(nb[0], NSTAR, color=ORANGE, alpha=0.14, linewidth=0, zorder=0)
c.axvline(NSTAR, color="0.35", lw=1.0, ls=(0, (4, 2)), zorder=1)
c.set_xlim(nb[0], nb[-1])
c.set_xlabel(r"$\bar n_X$")
c.set_ylabel(r"$\chi_X/C_X$")
c.set_ylim(0, 1.08)

for k, axk in enumerate(ax):
    axk.tick_params(labelsize=9)
    for sp in ("top", "right"):
        axk.spines[sp].set_visible(False)
    axk.text(0.5, -0.30, f"({chr(97+k)})", transform=axk.transAxes,
             ha="center", va="top", fontsize=16)

fig.savefig("fig_hidden.png", dpi=600, bbox_inches="tight")
print("wrote fig_hidden.png", "nstar", round(NSTAR, 4))
for gX in G:
    print(f"  g_X={gX}: min |r_X| = {r_X(tau,gX,OM).min():.4f}, "
          f"max nbar = {(E_X(tau,gX,OM)/OM).max():.4f}")
