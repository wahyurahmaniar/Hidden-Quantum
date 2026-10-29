"""New figure (Comments 1.3, 2.4 and 2.5): what the speed limit proves and
what the branching assumptions add, against the future channel computed for
the interacting baths of Sec. 7.

x axis: t / tau_MT, with tau_MT = pi hbar / (2 Delta H_rel) and
Delta H_rel = 2 Delta H_alt for the relative generator 2V.
"""
import numpy as np

from figstyle import plt, VERM, BLACK, finish

D = __import__("model2_data").load()
t, T2, dH = D["t"], float(D["T2"]), float(D["dH"])
tau = np.pi / (4.0 * dH)                 # relative generator 2V
x = t / tau
m = t <= 4.0 * T2 + 1e-12

xx = np.linspace(0, x[m].max(), 600)
count = np.log2(1 + xx)

fig, a = plt.subplots(figsize=(5.0, 3.7))
a.plot(xx, 2.0 * xx, color="0.45", lw=1.3, ls=(0, (6, 2, 1.2, 2)),
       label=r"ceiling, $\kappa_{\max}=2$")
a.plot(xx, 1.0 * xx, color="0.25", lw=1.5, ls=(0, (5, 2)),
       label=r"ceiling, $\kappa_{\max}=1$")
a.plot(xx, count, color=BLACK, lw=2.0,
       label=r"count $\log_2(1+t/\tau_{\rm MT})$")
a.plot(x[m], D["XX_I_future"][m], color=VERM, lw=2.0,
       label="computed, Sec. 7")
a.set_xlim(0, x[m].max())
a.set_ylim(0, 5.2)
a.legend(loc="upper left")
finish(a, r"$t/\tau_{\rm MT}$", "information (bits)")

fig.savefig("fig_qsl.png", dpi=400, bbox_inches="tight")

# ---------------------------------------------------------------- checks
lim = np.log2(1 + x[m])
ratio = np.where(lim > 0, D["XX_I_future"][m] / np.where(lim > 0, lim, 1), 0)
for bath in ("XX", "XXZ", "XXZ2"):
    I = D[f"{bath}_I_future"][m]
    r = np.where(lim > 0, I / np.where(lim > 0, lim, 1), 0)
    print(f"{bath:>4}: I_future(4 T2) = {I[-1]:.3f} bits;"
          f" max I/count = {r.max():.3f}; below ramp: {bool((I <= x[m] + 1e-9).all())}")
dev = max(np.max(np.abs(D[f"{b}_I_future"][m] - D["XX_I_future"][m]))
          for b in ("XXZ", "XXZ2"))
print(f"t/tau_MT at 4 T2 = {x[m].max():.3f}; count there = {lim[-1]:.3f} bits;"
      f" max |XXZ - XX| = {dev:.4f} bits")
