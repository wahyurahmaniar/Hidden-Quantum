"""New figure (Comments 1.5 and 2.8): the coherence floor fixed by the
hidden-sector modular energy, for a mode at the centre of the ball.

(a) Displacement occupation nbar_disp = dK / (pi R Omega / c) of Eq. (78).
(b) The floor 2 h^{-1}(C_X(n_th + n_disp) + S_0) - 1 of Corollary 3, with
    the thermal reduced vacuum of condition (ii) supplying both the reference
    occupation n_th and the reference entropy S_0 = C_X(n_th).
"""
import numpy as np

from figstyle import (plt, BLUE, ORANGE, GREEN, BLACK, panel_label, finish)
from analyses import C_X, h2_inv_upper

NSTAR = 0.29382
DKS = (0.1, 1.0, 10.0)
COLS = (BLUE, BLACK, ORANGE)
LSS = ((0, (5, 2)), "-", (0, (6, 2, 1.2, 2)))
X = np.logspace(0.0, 4.0, 801)                  # R Omega / c
c, R, f = 2.99792458e8, 0.30, 1.0e10
X_EX = R * 2 * np.pi * f / c


def n_th(x):
    b = np.pi * x                                 # beta hbar Omega, centred mode
    return np.where(b < 700, 1.0 / np.expm1(np.minimum(b, 700)), 0.0)


def floor(dk, x):
    nt = n_th(x)
    n = dk / (np.pi * x)
    S0 = np.where(nt > 1e-300, C_X(np.maximum(nt, 1e-300)), 0.0)
    return np.array([2 * h2_inv_upper(float(C_X(n_ + t_) + s_)) - 1
                     for n_, t_, s_ in zip(n, nt, S0)])


fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.2),
                       gridspec_kw=dict(wspace=0.26))

# ---- (a) occupation --------------------------------------------------------
a = ax[0]
a.axvspan(1.0, 10.0, facecolor="0.55", alpha=0.16, linewidth=0,
          label=r"$R\Omega/c<10$: condition (i) fails")
for dk, col, ls in zip(DKS, COLS, LSS):
    a.plot(X, dk / (np.pi * X), color=col, lw=2.0 if dk == 1 else 1.8, ls=ls,
           label=rf"$\Delta\langle K_X\rangle={dk:g}$")
a.axhline(NSTAR, color="0.35", lw=1.0, ls=(0, (1.5, 1.5)),
          label=r"$C_X=1$ bit at $\bar n_X=0.294$")
a.plot([X_EX], [1.0 / (np.pi * X_EX)], "o", ms=7, color=BLACK,
       markeredgecolor="white", markeredgewidth=1.2, zorder=6)
a.set_xscale("log")
a.set_yscale("log")
a.set_xlim(1, 1e4)
a.set_ylim(1e-5, 300)
a.legend(loc="upper right")
finish(a, r"$R\Omega/c$", r"displacement occupation $\bar n_{\rm disp}$")
panel_label(a, 0)

# ---- (b) floor -------------------------------------------------------------
b = ax[1]
b.axvspan(1.0, 10.0, facecolor="0.55", alpha=0.16, linewidth=0)
for dk, col, ls in zip(DKS, COLS, LSS):
    b.plot(X, floor(dk, X), color=col, lw=2.0 if dk == 1 else 1.8, ls=ls,
           label=rf"$\Delta\langle K_X\rangle={dk:g}$")
fl_ex = float(floor(1.0, np.array([X_EX]))[0])
b.plot([X_EX], [fl_ex], "o", ms=7, color=BLACK, markeredgecolor="white",
       markeredgewidth=1.2, zorder=6, label=r"10 GHz, $R=30$ cm")
b.set_xscale("log")
b.set_xlim(1, 1e4)
b.set_ylim(-0.02, 1.02)
b.legend(loc="lower right")
finish(b, r"$R\Omega/c$", r"floor on $|r_X|$")
panel_label(b, 1)

fig.savefig("fig_modular.png", dpi=400, bbox_inches="tight")

print(f"example: R Omega/c = {X_EX:.2f}, nbar = {1/(np.pi*X_EX):.3e}, floor = {fl_ex:.5f}")
for dk in DKS:
    fl = floor(dk, X)
    k = int(np.argmax(fl > 0))
    print(f"  dK = {dk:g}: floor becomes nontrivial at R Omega/c = {X[k]:.2f}"
          f" (expected {dk/(np.pi*NSTAR):.2f} without S_0)")
print(f"  S_0 at R Omega/c = 1: {float(C_X(n_th(1.0))):.3f} bits; at 10: {float(C_X(max(n_th(10.0), 1e-300))):.2e}")
