"""Observable coherence with the hidden sector present."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model_core import couplings
from model_xosc import r_X, chi_X

BLUE, ORANGE, GREEN, BLACK = "#0072B2", "#E69F00", "#009E73", "#000000"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10.5, "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
})

g = couplings()
T2 = 1.0 / (np.sqrt(2.0) * np.sqrt((g ** 2).sum()))
OM = 0.5 / T2                       # slow hidden mode, half a radian per T2
t = np.linspace(0.0, 3.0 * T2, 3001)
x = t / T2
rE = np.abs(np.prod(np.cos(2.0 * np.outer(t, g)), axis=1))

# dimensionless couplings g_X/hbar Omega; r_X takes g_X itself, so multiply by OM
G = (0.0, 0.45, 0.90, 1.50)
STY = ((BLACK, "-"), (BLUE, (0, (5, 1.8))), (GREEN, (0, (1.5, 1.5))),
       (ORANGE, (0, (4, 1.5, 1, 1.5))))

fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.1),
                       gridspec_kw=dict(wspace=0.30, bottom=0.30, top=0.94,
                                        left=0.09, right=0.985))
a, b = ax
half = lambda y: x[int(np.argmax(y / y[0] <= 0.5))]
rows = []
for gX, (col, ls) in zip(G, STY):
    rX = r_X(t, gX * OM, OM) if gX else np.ones_like(t)
    C = rE * rX
    lab = r"$C_{\rm env}$" if gX == 0 else rf"$g_X/\hbar\Omega={gX:.2f}$"
    a.plot(x, C / C[0], color=col, ls=ls, lw=1.8, label=lab)
    rows.append((gX, half(C), float(chi_X(half(C) * T2, gX * OM, OM)) if gX else 0.0))
a.axhline(0.5, color="0.5", lw=0.8)
a.set_xlabel(r"$t/T_2$"); a.set_ylabel(r"$C_{\rm obs}(t)/C_{\rm obs}(0)$")
a.set_xlim(0, 3); a.set_ylim(-0.03, 1.05)
a.legend(frameon=False, fontsize=8.5, handlelength=2.0, borderaxespad=0.2)

gg = np.linspace(0.0, 1.8, 241)
hs, cs = [], []
for gX in gg:
    rX = r_X(t, gX * OM, OM) if gX else np.ones_like(t)
    C = rE * rX
    hs.append(half(C))
    cs.append(float(chi_X(hs[-1] * T2, gX * OM, OM)) if gX else 0.0)
l1, = b.plot(gg, np.array(hs), color=BLUE, lw=1.8,
             label=r"$t_{1/2}/T_2$")
b2 = b.twinx()
l2, = b2.plot(gg, cs, color=ORANGE, lw=1.8, ls=(0, (5, 1.8)),
              label=r"$\chi_X$ at $t_{1/2}$")
b2.set_ylabel(r"$\chi_X$ at $t_{1/2}$ (bits)", color=ORANGE)
b.legend(handles=[l1, l2], frameon=False, fontsize=8.5, handlelength=2.0,
         loc="center left", bbox_to_anchor=(0.03, 0.62), borderaxespad=0.0)
b2.tick_params(axis="y", colors=ORANGE, labelsize=9)
b.set_xlabel(r"$g_X/\hbar\Omega$"); b.set_ylabel(r"$t_{1/2}/T_2$", color=BLUE)
b.tick_params(axis="y", colors=BLUE)
b.set_xlim(0, 1.8)

for k, axk in enumerate(ax):
    axk.tick_params(labelsize=9)
    axk.text(0.5, -0.30, f"({chr(97+k)})", transform=axk.transAxes,
             ha="center", va="top", fontsize=12)
for sp in ("top",):
    a.spines[sp].set_visible(False)

fig.savefig("fig_cobs.png", dpi=600, bbox_inches="tight")
print("wrote fig_cobs.png")
from model_xosc import E_X
for gX, h, c in rows:
    if gX:
        print(f"  nbar at t_1/2 for g_X/hbarOmega={gX}: {float(E_X(h*T2, gX*OM, OM)/OM):.4f}")
for gX, h, c in rows:
    print(f"  g_X/hbarOmega={gX:4.2f}: t_1/2 = {h:.3f} T2, chi_X there = {c:.3f} bits")
