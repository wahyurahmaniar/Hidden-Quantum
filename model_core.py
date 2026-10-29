"""Exact central-spin dephasing model used in the manuscript.

H = (omega/2) sigma_z^S + sum_j (omega_j/2) sigma_z^j
    + sum_j g_j sigma_z^S (x) sigma_z^j,      hbar = 1.

Initial state: (|0>+|1>)/sqrt(2) tensor product of |+> environment qubits.
The free terms contribute overall phases only, so |rho_01(t)| depends on the
couplings g_j alone.  All information quantities are returned in bits.
"""
import numpy as np


def h2(p):
    """Binary entropy in bits, safe at the endpoints."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    out = np.zeros_like(p)
    m = (p > 0) & (p < 1)
    out[m] = -p[m] * np.log2(p[m]) - (1 - p[m]) * np.log2(1 - p[m])
    return out


def couplings(N=20, g0=1.0, spread=0.6):
    """Deterministic coupling sequence g_j = g0 * f_j (no RNG).

    The shape factors f_j are normalized so that their mean is unity,
    making g0 the mean coupling of the sequence.
    """
    j = np.arange(N)
    f = (1.0 + spread * (2.0 * (j + 0.5) / N - 1.0))
    f = f * (1.0 + 0.07 * np.cos(3.0 * np.pi * (j + 0.5) / N))
    f = f / f.mean()
    return g0 * f


def channels(t, g, kappa=1.0):
    """Exact channel populations at times t (array), in bits."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    C = np.cos(2.0 * np.outer(t, g))                # (T, N) conditional overlaps
    r = np.prod(C, axis=1)
    absr = np.abs(r)

    r_hat = np.empty_like(C)
    for k in range(len(g)):
        idx = [m for m in range(len(g)) if m != k]
        r_hat[:, k] = np.prod(C[:, idx], axis=1)

    S_S = h2((1.0 + absr) / 2.0)
    S_Ej = h2((1.0 + np.abs(C)) / 2.0)
    S_SEj = h2((1.0 + np.abs(r_hat)) / 2.0)

    I_past = np.sum(S_S[:, None] + S_Ej - S_SEj, axis=1)
    I_pres = 1.0 - S_S
    I_SE = 2.0 * S_S
    dH = np.sqrt(np.sum(g ** 2))                    # spread of H_int
    T2 = 1.0 / (np.sqrt(2.0) * dH)
    # branch generator is H_+ - H_- = 2V, so Delta H_rel = 2 dH and
    # tau_MT = pi hbar / (2 Delta H_rel) = pi hbar / (4 dH)
    rate = 4.0 * kappa * dH / np.pi                 # QSL ceiling, bits per time
    cap = float(len(g) + 1)                         # log2 d_alt for N+1 qubits
    I_fut = np.minimum(rate * t, cap)
    coh = 0.5 * absr

    return dict(t=t, r=r, coherence=coh, I_past=I_past, I_present=I_pres,
                I_SE=I_SE, I_future=I_fut, I_total=I_past + I_pres + I_fut,
                rate=rate, dH=dH, T2=T2)


def gate(I_total, I_max, dI):
    return 1.0 / (1.0 + np.exp((I_total - I_max) / dI))


if __name__ == "__main__":
    g = couplings()
    t = np.linspace(0.0, 8.0, 80001)
    d = channels(t, g)
    print("N =", len(g), " mean g =", round(g.mean(), 4))
    print("Delta H_alt =", round(d["dH"], 4))
    print("T2 =", round(d["T2"], 5))
    print("r_QSL (kappa=1) =", round(d["rate"], 4), " r_QSL*T2 =",
          round(d["rate"] * d["T2"], 4), " (= sqrt(2)/pi =",
          round(np.sqrt(2) / np.pi, 4), ")")
    print("max I_past =", round(d["I_past"].max(), 3),
          " plateau =", round(d["I_past"][t / d["T2"] > 12].mean(), 3))
    Ibar = d["I_past"][t / d["T2"] > 12].mean()
    for Imax in (2.0, 4.0, 8.0, 14.0, 20.0):
        k = int(np.argmax(d["I_total"] >= Imax))
        pred = np.pi * (Imax - Ibar) / np.sqrt(2.0)
        print(f"  I_max={Imax:5.1f}: t_sat/T2={t[k] / d['T2']:6.2f}"
              f"   ramp prediction={pred:6.2f}")
