"""Analyses behind Table 2 and Figs. 3, 7, 9, 11 and 13.

1. Dependence of the predictions on the gate form (Table 2, Fig. 11).
2. The budget-combination class and double counting (Fig. 7).
3. The rate-based against the dynamical future channel (Fig. 9).
4. The coherence floor for arbitrary initial states and conditional
   unitaries (Fig. 3).
5. The regime of the floor and the prefactors quoted in the text (Fig. 13).

hbar = 1 throughout; information in bits.
"""
import numpy as np
from scipy.special import erfc
from scipy.optimize import brentq

from model_core import h2, couplings, channels
from robustness import channels_partition, blocks

IMAX, DI = 3.0, 0.4


# ---------------------------------------------------------------- gate forms
def gate_form(I, Ic, dI, form):
    """Monotone gates sharing midpoint Ic and the logistic midpoint slope
    -1/(4 dI)."""
    u = (I - Ic)
    if form == "logistic":
        return 1.0 / (1.0 + np.exp(u / dI))
    if form == "erfc":                      # Gaussian-smoothed step
        s = 4.0 * dI / np.sqrt(2.0 * np.pi)
        return 0.5 * erfc(u / (np.sqrt(2.0) * s))
    if form == "arctan":                    # Cauchy-smoothed step
        w = 4.0 * dI / np.pi
        return 0.5 - np.arctan(u / w) / np.pi
    if form == "step":                      # zero-width limit
        return (u < 0).astype(float)
    raise ValueError(form)


FORMS = ("logistic", "erfc", "arctan", "step")


def half_time(x, y):
    k = int(np.argmax(y <= 0.5))
    return x[k] if y[k] <= 0.5 else np.nan


def gated(d, Itot, form, Ic=IMAX, dI=DI):
    G = gate_form(Itot, Ic, dI, form)
    q = G / G[0]
    base = d["coherence"] / d["coherence"][0]
    return base, base * q


def section_gate_forms():
    print("=== 1. gate-form dependence, solved model, I_cap=3, dI=0.4 ===")
    g = couplings()
    t = np.linspace(0.0, 0.60, 48001)        # 0 .. 4.0 T2
    d = channels(t, g, kappa=1.0)
    x = t / d["T2"]
    k = int(np.argmax(d["I_total"] >= IMAX))
    print(f"  t_sat/T2 = {x[k]:.3f} for every form (first passage)")
    base = d["coherence"] / d["coherence"][0]
    print(f"  environmental t_1/2 = {half_time(x, base):.3f} T2")
    res = {}
    for f in FORMS:
        b, gtd = gated(d, d["I_total"], f)
        res[f] = (half_time(x, gtd), float((b - gtd).max()))
        print(f"  {f:>9}: t_1/2^RCC = {res[f][0]:.3f} T2,"
              f"  peak residual (0..4 T2) = {res[f][1]:.3f}")

    # matched-T2 pair
    N = 20
    jj = np.arange(N)
    fA = 1.0 + 0.03 * (2 * (jj + 0.5) / N - 1.0)
    fB = 10.0 ** ((jj / (N - 1.0)) - 0.5)
    fB *= np.sqrt((fA ** 2).sum() / (fB ** 2).sum())
    tm = np.linspace(0.0, 0.5, 40001)
    print("  matched-T2 pair, peak residual on 0..3 T2 (A uniform, B heterog.)")
    pair = {}
    for f in FORMS:
        row = []
        for nm, ff in (("A", fA), ("B", fB)):
            dm = channels_partition(tm, ff, blocks(N, 1))
            xm = tm / dm["T2"]
            b, gtd = gated(dm, dm["I_total"], f)
            msk = xm <= 3.0
            row.append(float((b - gtd)[msk].max()))
        pair[f] = row
        print(f"  {f:>9}: A {row[0]:.3f}   B {row[1]:.3f}   "
              f"A-B {row[0]-row[1]:+.3f}")

    # capacity at which the peak residual falls to 1 percent, kappa = 0.85
    print("  capacity giving a 1% peak residual at kappa_alt = 0.85")
    I_rec = d["I_past"] + d["I_present"]
    I_fut = np.minimum(0.85 * (4.0 * d["dH"] / np.pi) * t, float(len(g) + 1))
    Itot = I_rec + I_fut
    thr = {}
    for f in FORMS:
        if f == "step":
            # residual is base after the crossing: 1% needs a crossing at
            # which base has fallen to 0.01, outside the window
            def peak(Ic):
                G = gate_form(Itot, Ic, DI, f)
                return float((base * (1 - G / G[0])).max())
        else:
            def peak(Ic, f=f):
                G = gate_form(Itot, Ic, DI, f)
                return float((base * (1 - G / G[0])).max())
        Ics = np.linspace(3.0, 60.0, 2281)
        pk = np.array([peak(c) for c in Ics])
        idx = np.nonzero(pk >= 0.01)[0]
        thr[f] = Ics[idx[-1]] if idx.size else np.nan
        print(f"  {f:>9}: I_cap < {thr[f]:.1f} bits")
    return res, pair, thr


# ------------------------------------------------------------- budget rules
def section_budget_rules():
    print("\n=== 2. budget rule: disjoint support (sum) vs complete overlap"
          " (max) ===")
    g = couplings()
    t = np.linspace(0.0, 0.60, 48001)
    d = channels(t, g, kappa=1.0)
    x = t / d["T2"]
    base = d["coherence"] / d["coherence"][0]
    F = len(g)
    S_S = h2((1.0 + np.abs(d["r"])) / 2.0)
    rules = {
        "sum": d["I_past"] + d["I_present"] + d["I_future"],
        "dedup": d["I_past"] - (F - 1) * S_S + d["I_present"] + d["I_future"],
        "max": np.maximum.reduce([d["I_past"], d["I_present"],
                                  d["I_future"]]),
    }
    out = {}
    for nm, It in rules.items():
        k = int(np.argmax(It >= IMAX))
        G = gate_form(It, IMAX, DI, "logistic")
        th = half_time(x, base * G / G[0])
        out[nm] = (x[k], th)
        print(f"  {nm:>6}: t_sat = {x[k]:.3f} T2,  t_1/2^RCC = {th:.3f} T2")
    # check ordering max <= dedup <= sum pointwise
    print(f"  pointwise max <= dedup: {bool((rules['max'] <= rules['dedup'] + 1e-12).all())}"
          f"   dedup <= sum: {bool((rules['dedup'] <= rules['sum'] + 1e-12).all())}")
    # where the margin assumption of the insensitivity bound sits
    for Ic in (8.0, 20.0):
        ks = int(np.argmax(rules["sum"] >= Ic))
        km = int(np.argmax(rules["max"] >= Ic))
        print(f"  I_cap = {Ic:4.1f}: sum crosses at {x[ks]:.2f} T2,"
              f" max crosses at {x[km]:.2f} T2")
    return out


# --------------------------------------------------- future-channel choices
def section_future_channel():
    print("\n=== 3. dynamical (Eq. 51) vs ceiling-rate future channel ===")
    D = __import__("model2_data").load()
    t, T2, dH = D["t"], float(D["T2"]), float(D["dH"])
    x = t / T2
    rQSL = 4.0 * dH / np.pi                   # kappa_max = 1
    capd = 11.0                               # log2 d_alt, N = 10 bath + S
    out = {}
    for bath in ("XX", "XXZ", "XXZ2"):
        Ip, Ipr, Ifd = (D[f"{bath}_{k}"] for k in ("I_past", "I_present",
                                                  "I_future"))
        Ifb = np.minimum(rQSL * t, capd)
        row = []
        for Ic in (2.0, 3.0, 4.0):
            kd = np.argmax(Ip + Ipr + Ifd >= Ic)
            kb = np.argmax(Ip + Ipr + Ifb >= Ic)
            td = x[kd] if (Ip + Ipr + Ifd)[kd] >= Ic else np.nan
            tb = x[kb] if (Ip + Ipr + Ifb)[kb] >= Ic else np.nan
            row.append((Ic, tb, td))
        out[bath] = row
        print(f"  {bath:>4}: " + "  ".join(
            f"I_cap={Ic:.0f}: ceiling {tb:.2f}, dynamical {td:.2f} T2"
            for Ic, tb, td in row))
        dI = np.gradient(Ifd, t)
        msk = x <= 4.0
        print(f"        I_future(Eq.51) <= ramp on 0..4 T2: "
              f"{bool((Ifd[msk] <= Ifb[msk] + 1e-9).all())};"
              f"  min dI/dt = {dI[msk].min():.3f};"
              f"  I_future(4 T2) = {Ifd[np.argmin(abs(x-4))]:.2f} bits"
              f" vs ramp {Ifb[np.argmin(abs(x-4))]:.2f}")
    return out


# ------------------------------------------------ general-state floor check
def C_X(n):
    n = np.clip(np.asarray(n, float), 1e-300, None)
    return (n + 1) * np.log2(n + 1) - n * np.log2(n)


def vn_bits(rho):
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-14]
    return float(-(w * np.log2(w)).sum())


def section_general_states(seed=7, trials=4000, d=14):
    print("\n=== 4. floor for arbitrary initial states and unitaries ===")
    rng = np.random.default_rng(seed)
    n_op = np.diag(np.arange(d, dtype=float))
    worst = np.inf
    worst_pure = np.inf
    nontriv = 0
    for k in range(trials):
        # low-energy random initial state, possibly mixed
        rank = rng.integers(1, 4)
        A = (rng.normal(size=(d, rank)) + 1j * rng.normal(size=(d, rank)))
        A *= np.exp(-0.9 * np.arange(d))[:, None] * rng.uniform(0.2, 2.0)
        rho0 = A @ A.conj().T
        rho0 /= np.trace(rho0).real
        # random conditional unitaries generated by weak random Hamiltonians
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
        lhs = float(h2((1 + rx) / 2))
        rhs = float(C_X(nbar)) + S0
        worst = min(worst, rhs - lhs)
        if rank == 1:
            worst_pure = min(worst_pure, rhs - lhs)
        if rhs < 1:
            nontriv += 1
    print(f"  {trials} draws in a {d}-level truncation:"
          f" min slack C_X + S_0 - h2 = {worst:.4f} bits"
          f" (pure inputs: {worst_pure:.4f});"
          f" floor non-trivial in {nontriv} draws")

    # displaced pairs at fixed mean energy: symmetric pair is extremal
    a0 = rng.normal(size=2000) + 1j * rng.normal(size=2000)
    a1 = rng.normal(size=2000) + 1j * rng.normal(size=2000)
    rr = np.exp(-np.abs(a0 - a1) ** 2 / 2)
    Eb = (np.abs(a0) ** 2 + np.abs(a1) ** 2) / 2
    print(f"  coherent pairs: min(|r_X| - exp(-2 nbar)) = "
          f"{(rr - np.exp(-2 * Eb)).min():.2e} (>= 0)")

    # thermal hidden mode driven as in Eq. (8): |r_X| = exp(-2 nbar_d (2n_th+1))
    print("  thermal initial mode, symmetric drive:")
    for nth in (0.0, 0.01, 0.05):
        S0 = float(C_X(nth)) if nth > 0 else 0.0
        for nd in (0.01, 0.05):
            rx = np.exp(-2 * nd * (2 * nth + 1))
            lhs = float(h2((1 + rx) / 2))
            rhs = float(C_X(nd + nth)) + S0
            print(f"    n_th={nth:5.2f} n_drive={nd:5.2f}: h2={lhs:.4f}"
                  f"  C_X+S_0={rhs:.4f}  holds={lhs <= rhs + 1e-12}")
    return worst, worst_pure


def h2_inv_upper(y):
    return 0.5 if y >= 1 else brentq(lambda p: float(h2(p)) - y, 0.5, 1 - 1e-15)


def section_regime():
    print("\n=== 5. regime of the floor, prefactors ===")
    n_star = brentq(lambda n: float(C_X(n)) - 1.0, 1e-6, 5.0)
    print(f"  C_X = 1 bit at nbar = {n_star:.4f};"
          f"  coupling bound g_X/hbar Omega < {np.sqrt(n_star/4):.4f}")
    for n in (1e-3, 1e-2, 0.1, 0.2, 4.0, 10.0):
        r = np.exp(-2 * n)
        print(f"  nbar={n:6.3f}: chi/C = {float(h2((1+r)/2)/C_X(n)):.3f}"
              f"   1/C_X = {1/float(C_X(n)):.3f}")
    print(f"  pi^2/(2 ln 2) = {np.pi**2/(2*np.log(2)):.3f}   "
          f"2 ln 2/pi^2 = {2*np.log(2)/np.pi**2:.4f}   "
          f"(old pi^2/ln2 = {np.pi**2/np.log(2):.2f})")
    # modular example inside the localized regime
    c = 2.99792458e8
    for lam_m, R in ((1e-6, 1e-5), (3e-2, 0.3)):
        Om = 2 * np.pi * c / lam_m
        ratio = R * Om / c
        n = 1.0 / (np.pi * ratio)
        fl = 2 * h2_inv_upper(float(C_X(n))) - 1
        print(f"  wavelength {lam_m:g} m, R = {R:g} m: R Omega/c = {ratio:.1f},"
              f" dK=1 -> nbar = {n:.2e}, floor |r_X| >= {fl:.4f},"
              f" threshold dK < {n_star*np.pi*ratio:.1f}")


if __name__ == "__main__":
    section_gate_forms()
    section_budget_rules()
    section_future_channel()
    section_general_states()
    section_regime()
