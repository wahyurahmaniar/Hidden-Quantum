"""Fig. 1: normalized temporal information budget and the capacity-gated
coherence, side by side, both computed from the solved dephasing model."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model_core import channels, couplings, gate

BLUE, ORANGE, GREEN, BLACK = "#0072B2", "#E69F00", "#009E73", "#000000"

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
})

IMAX, DI, KAPPA = 3.0, 0.4, 1.0
g = couplings()
t = np.linspace(0.0, 0.45, 40001)
d = channels(t, g, kappa=KAPPA)
x = t / d["T2"]

past, pres, fut = (d[k] / IMAX for k in ("I_past", "I_present", "I_future"))
tot = past + pres + fut
k_sat = int(np.argmax(tot >= 1.0))
x_sat = x[k_sat]

fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.2),
                       gridspec_kw=dict(wspace=0.26,
                                        width_ratios=[0.45, 0.55]))

a = ax[0]
a.stackplot(x, past, pres, fut, colors=[BLUE, ORANGE, GREEN],
            edgecolor="0.35", linewidth=0.6,
            labels=[r"$I_{\rm past}$ (records)", r"$I_{\rm present}$ (coherence)",
                    r"$I_{\rm future}$ (alternatives)"])
total = past + pres + fut
a.plot(x, total, color=BLACK, lw=2.0, zorder=7, label=r"$I_{\rm total}$")

# the post-crossing regime, shaded but unlabelled in the legend
a.axvspan(x_sat, 1.25, facecolor="0.35", alpha=0.10, linewidth=0,
          zorder=6, label="post-crossing regime")
a.axhline(1.0, color="0.25", lw=1.2, ls=(0, (5, 3)), zorder=7,
          label="capacity")
a.axvline(x_sat, color=BLACK, lw=1.0, ls=(0, (2, 2)), zorder=7)
a.plot([x_sat], [1.0], "o", color=BLACK, ms=6, zorder=8)

a.text(x_sat + 0.025, 1.62, r"$t_{\rm sat}$", ha="left", va="center",
       fontsize=10.5, zorder=9)

a.set_xlabel(r"$t/T_2$", fontsize=11.5)
a.set_ylabel(r"information $/\,I_{\rm cap}$", fontsize=11.5)
a.set_ylim(0, 1.85)
a.set_xlim(0, 1.25)
a.legend(loc="upper left", frameon=False, handlelength=1.5,
         labelspacing=0.35, fontsize=9.5, borderaxespad=0.4)
a.text(0.5, -0.155, "(a)", transform=a.transAxes, ha="center", va="top",
       fontsize=12)

b = ax[1]
base = d["coherence"] / d["coherence"][0]
gt = gate(d["I_total"], IMAX, DI)
b.plot(x, base, color=BLACK, lw=2.0, ls="-", label=r"$C_{\rm env}(t)/C(0)$")
b.plot(x, base * gt / gt[0], color=GREEN, lw=2.0, ls=(0, (5, 1.8)),
       label=r"$C_{\rm RCC}(t)/C(0)$")
b.fill_between(x, base * gt / gt[0], base, facecolor=GREEN, alpha=0.16,
               linewidth=0, label="capacity correction")
b.text(x_sat - 0.04, 0.97, r"$t_{\rm sat}$", ha="right", va="center",
       fontsize=10.5)
b.axvline(x_sat, color=BLACK, lw=1.3, ls=(0, (2, 2)))
b.axhline(0.5, color="0.45", lw=0.9, ls="-", zorder=1)
gm = base * gt / gt[0]
h_env = x[int(np.argmax(base <= 0.5))]
h_mod = x[int(np.argmax(gm <= 0.5))]
b.plot([h_mod, h_env], [0.5, 0.5], "o", color=BLACK, ms=5.5, zorder=6)
b.annotate(rf"${h_mod:.2f}\,T_2$", xy=(h_mod, 0.5), xytext=(-9, -6),
           textcoords="offset points", ha="right", va="top", fontsize=10)
b.annotate(rf"${h_env:.2f}\,T_2$", xy=(h_env, 0.5), xytext=(12, 22),
           textcoords="offset points", ha="left", fontsize=10)
b.set_xlabel(r"$t/T_2$", fontsize=11.5)
b.set_ylabel(r"$C(t)/C(0)$", fontsize=11.5)
b.set_ylim(-0.04, 1.06)
b.set_xlim(0, x[-1])
b.legend(loc="upper right", frameon=False, handlelength=2.0,
         labelspacing=0.35, fontsize=9.5, borderaxespad=0.4)
b.text(0.5, -0.155, "(b)", transform=b.transAxes, ha="center", va="top",
       fontsize=12)

fig.savefig("fig_budget.png", dpi=400, bbox_inches="tight")

half_env = x[int(np.argmax(base <= 0.5))]
half_mod = x[int(np.argmax(base * gt / gt[0] <= 0.5))]
print(f"t_sat/T2 = {x_sat:.2f}")
print(f"half-coherence: env {half_env:.2f} T2, model {half_mod:.2f} T2")
