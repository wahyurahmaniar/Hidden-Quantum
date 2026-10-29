"""New figure (Comments 1.1 and 2.6): budget combination in the solved model.

(a) The additive budget and the complete-overlap end of Eq. (18), the
    running maximum of the largest single channel, with the class between
    them shaded and the three channels shown thin.
(b) The ungated coherence and the gated coherence at the two ends.
"""
import numpy as np

from figstyle import (plt, BLUE, ORANGE, GREEN, BLACK, panel_label, finish)
from model_core import channels, couplings
from analyses import gate_form, half_time, IMAX, DI

g = couplings()
t = np.linspace(0.0, 0.60, 48001)             # 0 .. 3.9 T2
d = channels(t, g, kappa=1.0)
x = t / d["T2"]
base = d["coherence"] / d["coherence"][0]
P, C, F = d["I_past"], d["I_present"], d["I_future"]
S_add = P + C + F
S_max = np.maximum.accumulate(np.maximum.reduce([P, C, F]))


def gated(I):
    G = gate_form(I, IMAX, DI, "logistic")
    return base * G / G[0]


g_add, g_max = gated(S_add), gated(S_max)
k_add = int(np.argmax(S_add >= IMAX))
k_max = int(np.argmax(S_max >= IMAX))
h_env, h_add, h_max = half_time(x, base), half_time(x, g_add), half_time(x, g_max)

fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.2),
                       gridspec_kw=dict(wspace=0.26))

# ---- (a) budgets -----------------------------------------------------------
a = ax[0]
a.fill_between(x, S_max, S_add, facecolor="0.55", alpha=0.20, linewidth=0,
               label="class of Eq. (18)")
a.plot(x, S_add, color=BLACK, lw=2.0, label="additive budget, Eq. (14)")
a.plot(x, S_max, color=BLACK, lw=2.0, ls=(0, (5, 2)),
       label="complete overlap")
a.plot(x, P, color=BLUE, lw=1.1, label=r"$I_{\rm past}$")
a.plot(x, C, color=ORANGE, lw=1.1, label=r"$I_{\rm present}$")
a.plot(x, F, color=GREEN, lw=1.1, label=r"$I_{\rm future}$")
a.axhline(IMAX, color="0.35", lw=1.0, zorder=1)
a.plot([x[k_add]], [IMAX], "o", color=BLACK, ms=6.5, zorder=6)
a.plot([x[k_max]], [IMAX], "o", color="white", markeredgecolor=BLACK,
       markeredgewidth=1.4, ms=6.5, zorder=6)
a.text(0.02, IMAX + 0.08, r"$I_{\rm cap}$", fontsize=11, va="bottom")
a.set_xlim(0, 1.25)
a.set_ylim(0, 7.0)
a.legend(loc="upper left")
finish(a, r"$t/T_2$", "information (bits)")
panel_label(a, 0)

# ---- (b) coherence ---------------------------------------------------------
b = ax[1]
b.fill_between(x, g_add, g_max, facecolor=GREEN, alpha=0.18, linewidth=0,
               label="range of the class")
b.plot(x, base, color=BLACK, lw=2.0, label=r"$C_{\rm env}(t)/C(0)$")
b.plot(x, g_add, color=GREEN, lw=2.0, label="gated, additive")
b.plot(x, g_max, color=GREEN, lw=2.0, ls=(0, (5, 2)),
       label="gated, complete overlap")
b.axhline(0.5, color="0.45", lw=0.9, zorder=1)
b.plot([h_add, h_max, h_env], [0.5] * 3, "o", color=BLACK, ms=5.5, zorder=6)
b.set_xlim(0, 2.5)
b.set_ylim(-0.03, 1.05)
b.legend(loc="upper right")
finish(b, r"$t/T_2$", r"$C(t)/C(0)$")
panel_label(b, 1)

fig.savefig("fig_combination.png", dpi=400, bbox_inches="tight")

print(f"first crossing: additive {x[k_add]:.3f}, overlap end {x[k_max]:.3f} T2")
print(f"half-coherence: env {h_env:.3f}, additive {h_add:.3f}, overlap {h_max:.3f} T2")
print(f"budget class ordered pointwise: {bool((S_max <= S_add + 1e-12).all())};"
      f" gated ordered: {bool((g_add <= g_max + 1e-12).all() and (g_max <= base + 1e-12).all())}")
