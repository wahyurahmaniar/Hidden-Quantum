"""Fig. 7 (redrawn): visibility of the capacity contribution in the solved
model.  (a) The peak residual is the largest gap between the ungated and the
gated coherence.  (b) That peak residual against the regional capacity, for
generation factors spanning a factor of 24."""
import numpy as np
from scipy.optimize import brentq

from figstyle import plt, BLUE, ORANGE, GREEN, BLACK, panel_label, finish
from model_core import couplings, channels, gate

DI = 0.4              # gate width in bits
KAPPA_DYN = 0.85      # value returned by the interacting model
CAPS = (3.0, 5.0, 8.0)            # capacities shown in panel (a)
CAP_COL = (BLUE, ORANGE, GREEN)
CAP_LS = ("-", (0, (5, 2)), (0, (1.2, 1.6)))
KAPPAS = (0.25, 0.85, 2.5, 6.0)

g = couplings()
t = np.linspace(0.0, 0.60, 16001)          # 0 .. 3.9 T2
d = channels(t, g, kappa=1.0)
T2, dH = d["T2"], d["dH"]
x = t / T2
base = d["coherence"] / d["coherence"][0]
I_rec = d["I_past"] + d["I_present"]      # channels from the exact state
cap = float(len(g) + 1)                   # log2 d_alt


def gated(kappa, Ic):
    I_fut = np.minimum(kappa * (4.0 * dH / np.pi) * t, cap)
    q = gate(I_rec + I_fut, Ic, DI)
    return base * q / q[0]


def peak(kappa, Ic):
    return float((base - gated(kappa, Ic)).max())


fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.2),
                       gridspec_kw=dict(wspace=0.26))

# ---- (a) what the peak residual is ---------------------------------------
a = ax[0]
a.plot(x, base, color=BLACK, lw=2.0, label=r"$C_{\rm env}$, no gate")
for Ic, col, ls in zip(CAPS, CAP_COL, CAP_LS):
    a.plot(x, gated(KAPPA_DYN, Ic), color=col, lw=1.9, ls=ls,
           label=rf"$C_{{\rm RCC}}$, $I_{{\rm cap}}={Ic:g}$ bits")
gap = base - gated(KAPPA_DYN, CAPS[0])
k = int(np.argmax(gap))
a.annotate("", xy=(x[k], base[k] - gap[k]), xytext=(x[k], base[k]),
           arrowprops=dict(arrowstyle="<|-|>", color=BLACK, lw=1.0,
                           mutation_scale=9, shrinkA=0, shrinkB=0),
           zorder=6)
a.set_xlim(0, 2.5)
a.set_ylim(-0.03, 1.05)
a.legend(loc="upper right")
finish(a, r"$t/T_2$", r"$C(t)/C(0)$")
panel_label(a, 0)

# ---- (b) dependence on capacity and rate ---------------------------------
b = ax[1]
Ic_grid = np.logspace(0.0, np.log10(25.0), 240)
STYLE = {0.25: dict(color="0.55", lw=1.3, ls=(0, (1.2, 1.6))),
         0.85: dict(color=BLACK, lw=2.0, ls="-"),
         2.5: dict(color="0.40", lw=1.3, ls=(0, (5, 2))),
         6.0: dict(color="0.25", lw=1.3, ls=(0, (6, 2, 1.2, 2)))}
curves = {}
for kap in KAPPAS:
    pk = np.array([peak(kap, c) for c in Ic_grid])
    curves[kap] = pk
    lab = (rf"$\kappa_{{\rm alt}}={kap:g}$ (interacting model)"
           if kap == KAPPA_DYN else rf"$\kappa_{{\rm alt}}={kap:g}$")
    b.plot(Ic_grid, 100 * pk, label=lab, **STYLE[kap])
for lev in (10.0, 1.0):
    b.axhline(lev, color="0.75", lw=0.8, zorder=0)
for Ic, col in zip(CAPS, CAP_COL):
    b.plot([Ic], [100 * peak(KAPPA_DYN, Ic)], "o", ms=7, color=col,
           markeredgecolor="white", markeredgewidth=1.2, zorder=6)
b.set_xscale("log")
b.set_yscale("log")
b.set_xlim(1.0, 25.0)
b.set_ylim(1e-2, 150)
b.set_xticks([1, 2, 5, 10, 20])
b.set_xticklabels(["1", "2", "5", "10", "20"])
b.set_yticks([1e-2, 1e-1, 1, 10, 100])
b.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
b.legend(loc="lower left")
finish(b, r"regional capacity $I_{\rm cap}$ (bits)", r"peak residual (%)")
panel_label(b, 1)

fig.savefig("fig_map.png", dpi=400, bbox_inches="tight")


def crossing(kap, lev):
    return brentq(lambda c: peak(kap, c) - lev, 1.0, 40.0, xtol=1e-4)


print(f"peak residual at I_cap=3, kappa=0.85: {100*peak(KAPPA_DYN, 3.0):.1f}%"
      f" at t = {x[k]:.2f} T2")
for Ic in CAPS:
    print(f"  I_cap = {Ic:g}: peak residual {100*peak(KAPPA_DYN, Ic):.2f}%")
for lev in (0.10, 0.05, 0.01):
    row = ", ".join(f"kappa {kap:g}: {crossing(kap, lev):.2f}" for kap in KAPPAS)
    print(f"  residual {100*lev:g}% reached at I_cap = {row} bits")
