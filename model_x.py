"""Minimal S + E + X model.

    H_tot = H_SE + H_SX,      H_SX = lambda A_S (x) P_X,

with A_S = sigma_z^S the coarse-grained alternative observable and P_X the
generator of translation in an inaccessible coordinate xi.  Because both
terms are diagonal in A_S they commute, so the branch structure is exact:

    |Psi(t)> = sum_s c_s |s> |E_s(t)> |X_s(t)>,
    rho_01   = c_0 c_1^* <E_1|E_0> <X_1|X_0> = c_0 c_1^* r_E r_X.

For a Gaussian hidden packet of width sigma the conditional states are
rigid translations, xi_s(t) = xi_0 + lambda a_s t, so

    |r_X(t)| = exp[ -lambda^2 (a_0 - a_1)^2 t^2 / (8 sigma^2) ].
"""
import numpy as np

HBAR = 1.0


def r_X_gaussian(t, lam, sigma, da=2.0):
    """Hidden-sector overlap for a Gaussian packet, analytic."""
    return np.exp(-(lam * da * np.asarray(t)) ** 2 / (8.0 * sigma ** 2))


def r_X_numeric(t, lam, sigma, da=2.0, L=40.0, M=1024):
    """Same overlap by explicit translation on a xi grid."""
    xi = np.linspace(-L / 2, L / 2, M, endpoint=False)
    psi = np.exp(-xi ** 2 / (4.0 * sigma ** 2))
    psi /= np.sqrt(np.trapezoid(np.abs(psi) ** 2, xi))
    out = np.empty(np.size(t))
    for k, tk in enumerate(np.atleast_1d(t)):
        d = lam * da * tk                      # separation of the two packets
        a = np.interp(xi - d / 2, xi, psi, left=0.0, right=0.0)
        b = np.interp(xi + d / 2, xi, psi, left=0.0, right=0.0)
        out[k] = abs(np.trapezoid(a * b, xi))
    return out


def coherence_SEX(t, g, lam, sigma, da=2.0):
    """Observable coherence of the central spin with both sectors traced out."""
    r_E = np.prod(np.cos(2.0 * np.outer(t, g) / HBAR), axis=1)
    r_X = r_X_gaussian(t, lam, sigma, da)
    return 0.5 * np.abs(r_E) * r_X, r_E, r_X


def h2(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def chi_X(t, lam, sigma, da=2.0):
    """Holevo information the hidden sector carries about the alternative.

    For two equally weighted pure conditional states the Holevo quantity
    reduces to  chi = h2[(1+|r_X|)/2],  bounded by one bit.
    """
    r = r_X_gaussian(t, lam, sigma, da)
    return h2((1.0 + np.abs(r)) / 2.0)


def I_SX(t, lam, sigma, r_E, da=2.0):
    """S:X mutual information once the environment is present as well."""
    rX = np.abs(r_X_gaussian(t, lam, sigma, da))
    rE = np.abs(r_E)
    return (h2((1 + rE * rX) / 2) + h2((1 + rX) / 2) - h2((1 + rE) / 2))


if __name__ == "__main__":
    from model_core import couplings
    g = couplings()
    T2 = HBAR / (np.sqrt(2.0) * np.sqrt((g ** 2).sum()))
    t = np.linspace(0.0, 3.0 * T2, 601)
    lam, sigma = 2.0, 1.0

    an = r_X_gaussian(t, lam, sigma)
    nu = r_X_numeric(t, lam, sigma)
    print(f"analytic vs grid r_X : max |diff| = {np.abs(an - nu).max():.3e}")

    C, rE, rX = coherence_SEX(t, g, lam, sigma)
    print(f"factorisation check  : max |C - 0.5|r_E||r_X|| = "
          f"{np.abs(C - 0.5 * np.abs(rE) * rX).max():.3e}")

    C0, _, rX0 = coherence_SEX(t, g, 0.0, sigma)
    print(f"lambda -> 0 recovers Theorem 2: max |r_X - 1| = "
          f"{np.abs(rX0 - 1.0).max():.3e}")

    x = t / T2
    half = lambda y: x[int(np.argmax(y / y[0] <= 0.5))]
    print()
    print(f"{'lambda':>7} {'T_X/T2':>8} {'t_1/2 env':>10} {'t_1/2 obs':>10}"
          f" {'chi_X(T2)':>9}")
    for lam in (0.0, 0.5, 1.0, 2.0, 4.0):
        C, rE, rX = coherence_SEX(t, g, lam, sigma)
        TX = np.sqrt(2.0) * sigma / (lam * 2.0) if lam else np.inf
        print(f"{lam:7.1f} {TX / T2:8.2f} {half(np.abs(rE)):10.2f}"
              f" {half(C):10.2f} {chi_X(T2, lam, sigma):11.3f}")
