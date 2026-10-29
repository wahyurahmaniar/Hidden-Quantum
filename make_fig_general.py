"""New figure (Comment 1.2 and 2.2): scope of the energy-coherence relation
of Eq. (10) and of the general floor of Corollary 3.

(a) Branch overlap against the mean branch occupation for coherent pairs
    (opposite and a_j = 0), a displaced squeezed vacuum, and a driven
    thermal mode, with the pure-state floor of Corollary 2 shaded.
(b) The random draws behind Corollary 3: h2[(1+|r_X|)/2] against
    C_X(nbar) + S_0, all below the diagonal.
"""
import numpy as np
from scipy.linalg import expm

from figstyle import (plt, BLUE, ORANGE, GREEN, VERM, BLACK, panel_label,
                      finish)
from analyses import C_X, vn_bits, h2_inv_upper
from model_core import h2

R_SQ, N_TH = 0.2, 0.2
nb = np.linspace(0.0, 0.8, 1601)

# ------------------------------------------------------------ families
eq10 = np.exp(-2 * nb)                              # opposite coherent pair
aj0 = np.exp(-nb)                                   # a_j = 0: theta = 1/2
s2 = np.sinh(R_SQ) ** 2
gsq = np.linspace(0.0, 0.8 - s2, 1601)              # |gamma|^2
n_sq = s2 + gsq
r_sq = np.exp(-2 * gsq * np.exp(2 * R_SQ))          # squeezed-quadrature drive
nd = np.linspace(0.0, 0.8 - N_TH, 1601)             # deposited occupation
n_th = N_TH + nd
r_th = np.exp(-2 * nd * (2 * N_TH + 1))             # driven thermal mode
floor = np.array([2 * h2_inv_upper(float(C_X(n))) - 1 if n > 0 else 1.0
                  for n in nb])


def fock_check(r=R_SQ, gam=0.3, d=60):
    """Overlap of D(+-gam) S(r)|0> in a Fock truncation, against the
    closed form used for the squeezed curve."""
    a = np.diag(np.sqrt(np.arange(1, d)), 1)
    ad = a.T
    S = expm(0.5 * r * (a @ a - ad @ ad))
    vac = np.zeros(d); vac[0] = 1.0
    psi = S @ vac
    D = lambda z: expm(z * ad - np.conj(z) * a)
    p1, p2 = D(gam) @ psi, D(-gam) @ psi
    ov = abs(np.vdot(p2, p1))
    nbar = float(np.real(np.vdot(p1, ad @ a @ p1)))
    return ov, np.exp(-2 * gam ** 2 * np.exp(2 * r)), nbar, s2 + gam ** 2


# ------------------------------------------------------------ random draws
def draws(seed=7, trials=4000, d=14):
    """Exactly the loop of analyses.section_general_states."""
    rng = np.random.default_rng(seed)
    n_op = np.diag(np.arange(d, dtype=float))
    out = []
    for _ in range(trials):
        rank = rng.integers(1, 4)
        A = (rng.normal(size=(d, rank)) + 1j * rng.normal(size=(d, rank)))
        A *= np.exp(-0.9 * np.arange(d))[:, None] * rng.uniform(0.2, 2.0)
        rho0 = A @ A.conj().T
        rho0 /= np.trace(rho0).real

        def U(scale):
            Hm = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
            Hm = (Hm + Hm.conj().T) / 2
            Hm *= np.exp(-0.45 * np.add.outer(np.arange(d), np.arange(d)))
            w, v = np.linalg.eigh(scale * Hm)
            return (v * np.exp(-1j * w)) @ v.conj().T
        s = rng.uniform(0.02, 0.6)
        U0, U1 = U(s), U(s)
        rx = abs(np.trace(U1.conj().T @ U0 @ rho0))
        rbar = 0.5 * (U0 @ rho0 @ U0.conj().T + U1 @ rho0 @ U1.conj().T)
        nbar = float(np.trace(rbar @ n_op).real)
        S0 = vn_bits(rho0)
        out.append((float(C_X(nbar)) + S0, float(h2((1 + rx) / 2)), rank))
    return np.array(out)


D = draws()
rhs, lhs, rank = D[:, 0], D[:, 1], D[:, 2]

fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.2),
                       gridspec_kw=dict(wspace=0.26))

# ---- (a) -------------------------------------------------------------------
a = ax[0]
a.fill_between(nb, 0, floor, facecolor="0.55", alpha=0.18, linewidth=0,
               label="below the pure-state floor (Corollary 2)")
a.plot(nb, eq10, color=BLACK, lw=2.0, label=r"coherent, $a_j=-a_i$: Eq. (10)")
a.plot(nb, aj0, color=BLUE, lw=1.8, ls=(0, (6, 2, 1.2, 2)),
       label=r"coherent, $a_j=0$")
a.plot(n_sq, r_sq, color=ORANGE, lw=1.9, ls=(0, (5, 2)),
       label=rf"squeezed vacuum, $r={R_SQ:g}$")
a.plot(n_th, r_th, color=GREEN, lw=1.9, ls=(0, (1.2, 1.6)),
       label=rf"thermal, $n_{{\rm th}}={N_TH:g}$")
a.set_xlim(0, 0.8)
a.set_ylim(0, 1.50)
a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
a.legend(loc="upper right")
finish(a, r"mean branch occupation $\bar n_X$", r"branch overlap $|r_X|$")
panel_label(a, 0)

# ---- (b) -------------------------------------------------------------------
b = ax[1]
XMAX = 2.0
xx = np.linspace(0, XMAX, 400)
xt = np.linspace(0, 1, 200)
b.fill_between(xt, xt, 1.0, facecolor="0.55", alpha=0.18,
               linewidth=0, label="excluded by Corollary 3")
pure = rank == 1
b.scatter(rhs[~pure], lhs[~pure], s=7, color=ORANGE, alpha=0.55,
          linewidths=0, label=r"mixed initial states, $S_0>0$", zorder=3)
b.scatter(rhs[pure], lhs[pure], s=7, color=BLUE, alpha=0.7, linewidths=0,
          label="pure initial states", zorder=4)
b.plot([0, 1], [0, 1], color=BLACK, lw=1.3, zorder=5)
b.set_xlim(0, XMAX)
b.set_ylim(0, 1.40)
b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
b.legend(loc="upper right", markerscale=2.2)
finish(b, r"$C_X(\bar n_X)+S_0$ (bits)", r"$h_2[(1+|r_X|)/2]$ (bits)")
panel_label(b, 1)

fig.savefig("fig_general.png", dpi=400, bbox_inches="tight")

# ------------------------------------------------------------ checks
ov, ov_cf, nf, nf_cf = fock_check()
print(f"Fock check, squeezed pair: overlap {ov:.6f} vs closed form {ov_cf:.6f};"
      f" nbar {nf:.6f} vs {nf_cf:.6f}")
x_sq = n_sq[np.argmax(r_sq < np.exp(-2 * n_sq) - 1e-12)]
x_th = n_th[np.argmax(r_th < np.exp(-2 * n_th) - 1e-12)]
print(f"squeezed curve crosses below Eq. (10) at nbar = {x_sq:.3f};"
      f" thermal at nbar = {x_th:.3f}")
fl_sq = np.interp(n_sq, nb, floor)
print(f"squeezed curve minus pure-state floor: min {np.min(r_sq - fl_sq):.3f}")
print(f"coherent a_j=0 minus Eq. (10): min {np.min(aj0 - eq10):.3e}")
slack = rhs - lhs
print(f"{len(D)} draws: min slack {slack.min():.4f} bits (pure {slack[pure].min():.4f});"
      f" nontrivial (rhs<1): {(rhs < 1).sum()}; pure draws {pure.sum()};"
      f" draws beyond x={XMAX}: {(rhs > XMAX).sum()}")
