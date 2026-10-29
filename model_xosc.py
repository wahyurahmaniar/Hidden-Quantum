r"""Hidden harmonic mode: an energy-costed inaccessible sector.

    H_X   = hbar Omega a^dag a,
    H_SX  = g_X A_S (x) (a + a^dag),   A_S|s> = a_s|s>,  a_s = +-1.

Completing the square, each branch drives the mode into a coherent state

    alpha_s(t) = (g_X a_s / hbar Omega) (e^{-i Omega t} - 1),

so the branch separation, the overlap, and the branch excitation energy are

    |alpha_+ - alpha_-|^2 = 16 (g_X/hbar Omega)^2 sin^2(Omega t/2),
    |r_X(t)|              = exp[-8 (g_X/hbar Omega)^2 sin^2(Omega t/2)],
    E_X(t)                = (4 g_X^2 / hbar Omega) sin^2(Omega t/2),

whence the bridge that the free-translation model lacks,

    |r_X(t)| = exp[-2 E_X(t) / hbar Omega].

Distinguishability in the hidden sector now costs energy, and the maximum
information a bosonic mode can carry at fixed mean occupation is the
thermal value C_X(nbar) = (nbar+1)log2(nbar+1) - nbar log2 nbar.
"""
import numpy as np

HBAR = 1.0


def alpha(t, gX, Omega, a_s):
    return (gX * a_s / (HBAR * Omega)) * (np.exp(-1j * Omega * np.asarray(t)) - 1.0)


def r_X(t, gX, Omega):
    d2 = np.abs(alpha(t, gX, Omega, +1) - alpha(t, gX, Omega, -1)) ** 2
    return np.exp(-d2 / 2.0)


def E_X(t, gX, Omega):
    return HBAR * Omega * np.abs(alpha(t, gX, Omega, +1)) ** 2


def h2(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def chi_X(t, gX, Omega):
    return h2((1.0 + r_X(t, gX, Omega)) / 2.0)


def C_X(nbar):
    """Maximum entropy of a bosonic mode at fixed mean occupation, in bits."""
    n = np.clip(np.asarray(nbar, float), 1e-15, None)
    return (n + 1) * np.log2(n + 1) - n * np.log2(n)


if __name__ == "__main__":
    Omega = 1.0
    t = np.linspace(0.0, 2.0 * np.pi / Omega, 2001)

    print("closed forms verified against the coherent-state construction")
    for gX in (0.1, 0.3, 0.7):
        d2 = np.abs(alpha(t, gX, Omega, +1) - alpha(t, gX, Omega, -1)) ** 2
        pred = 16 * (gX / (HBAR * Omega)) ** 2 * np.sin(Omega * t / 2) ** 2
        rp = np.exp(-8 * (gX / (HBAR * Omega)) ** 2 * np.sin(Omega * t / 2) ** 2)
        Ep = 4 * gX ** 2 / (HBAR * Omega) * np.sin(Omega * t / 2) ** 2
        print(f"  g_X={gX}:  |dalpha|^2 err {np.abs(d2-pred).max():.2e}"
              f"   r_X err {np.abs(r_X(t,gX,Omega)-rp).max():.2e}"
              f"   E_X err {np.abs(E_X(t,gX,Omega)-Ep).max():.2e}"
              f"   r_X=exp(-2E/hbarOmega) err "
              f"{np.abs(r_X(t,gX,Omega)-np.exp(-2*E_X(t,gX,Omega)/(HBAR*Omega))).max():.2e}")

    print("\nis the Holevo information within the energy-constrained capacity?")
    print(f"{'g_X':>6} {'max nbar':>9} {'max chi_X':>10} {'C_X at that nbar':>17} {'chi<=C':>8}")
    worst = 1e9
    for gX in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0):
        n = E_X(t, gX, Omega) / (HBAR * Omega)
        ch, cap = chi_X(t, gX, Omega), C_X(n)
        ok = bool((ch <= cap + 1e-12).all())
        worst = min(worst, float((cap - ch).min()))
        k = int(np.argmax(ch))
        print(f"{gX:6.2f} {n.max():9.4f} {ch.max():10.4f} {cap[k]:17.4f} {str(ok):>8}")
    print(f"\nsmallest slack C_X - chi_X over every point tested: {worst:.4f} bits")
