"""New figure (Comments 1.4 and 2.7): dependence on the gate form.

(a) The four gates of Table 2 against the budget excess in units of dI.
(b) Their tails 1 - G below the threshold on a log scale.
(c) The gated coherence of the solved model at I_cap = 3 bits for each gate.
"""
import numpy as np

from figstyle import (plt, BLUE, ORANGE, GREEN, BLACK, panel_label, finish)
from model_core import channels, couplings
from analyses import gate_form, half_time, IMAX, DI

STYLE = {"logistic": dict(color=BLACK, lw=2.0, ls="-", label="logistic"),
         "erfc": dict(color=BLUE, lw=1.8, ls=(0, (5, 2)), label="Gaussian"),
         "arctan": dict(color=ORANGE, lw=1.8, ls=(0, (6, 2, 1.2, 2)),
                        label="Cauchy"),
         "step": dict(color=GREEN, lw=1.8, ls=(0, (1.2, 1.6)), label="step")}

fig, ax = plt.subplots(1, 3, figsize=(10.6, 3.7),
                       gridspec_kw=dict(wspace=0.34))

# ---- (a) gate shapes -------------------------------------------------------
a = ax[0]
u = np.linspace(-6, 6, 2401)
for f in ("step", "arctan", "erfc", "logistic"):
    a.plot(u, gate_form(IMAX + u * DI, IMAX, DI, f), **STYLE[f])
a.set_xlim(-6, 6)
a.set_ylim(-0.03, 1.08)
a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
h, l = a.get_legend_handles_labels()
a.legend(h[::-1], l[::-1], loc="lower left", fontsize=9.5)
finish(a, r"$(I_{\rm total}-I_{\rm cap})/\delta I$", r"gate $G$")
panel_label(a, 0, y=-0.17)

# ---- (b) tails -------------------------------------------------------------
b = ax[1]
D = np.linspace(0.0, 14.0, 1401)
for f in ("arctan", "erfc", "logistic"):
    tail = 1.0 - gate_form(IMAX - D * DI, IMAX, DI, f)
    b.plot(D, tail, **STYLE[f])
b.set_yscale("log")
b.set_xlim(0, 14)
b.set_ylim(1e-12, 5)
b.set_yticks([1e-12, 1e-9, 1e-6, 1e-3, 1])
finish(b, r"remaining capacity $\Delta/\delta I$", r"tail $1-G$")
panel_label(b, 1, y=-0.17)

# ---- (c) gated coherence ---------------------------------------------------
c = ax[2]
g = couplings()
t = np.linspace(0.0, 0.60, 48001)
d = channels(t, g, kappa=1.0)
x = t / d["T2"]
base = d["coherence"] / d["coherence"][0]
c.plot(x, base, color="0.55", lw=2.4, label=r"$C_{\rm env}$, no gate", zorder=1)
halves = {}
for f in ("step", "arctan", "erfc", "logistic"):
    G = gate_form(d["I_total"], IMAX, DI, f)
    y = base * G / G[0]
    halves[f] = half_time(x, y)
    st = dict(STYLE[f]); st.pop("label")
    c.plot(x, y, **st)
k = int(np.argmax(d["I_total"] >= IMAX))
c.axvline(x[k], color=BLACK, lw=1.0, ls=(0, (2, 2)), zorder=0)
c.set_xlim(0, 2.0)
c.set_ylim(-0.03, 1.32)
c.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
c.legend(loc="upper right", fontsize=9)
finish(c, r"$t/T_2$", r"$C(t)/C(0)$")
panel_label(c, 2, y=-0.17)

fig.savefig("fig_gates.png", dpi=400, bbox_inches="tight")

print(f"first crossing {x[k]:.3f} T2;  env half {half_time(x, base):.3f}")
for f, h_ in halves.items():
    print(f"  {f:>9}: t_1/2 = {h_:.3f} T2")
for f in ("logistic", "erfc", "arctan"):
    tail = 1.0 - gate_form(IMAX - D * DI, IMAX, DI, f)
    print(f"  {f:>9}: 1-G at Delta = 10 dI: {tail[np.argmin(abs(D-10))]:.2e};"
          f" max ratio to e^-D: {np.max(tail / np.exp(-D)):.3f}")
