"""Compute the numbers that appear in the tables of the manuscript.

Table 1  partition dependence in the solved model
Table 2  dependence of the predictions on the gate form
Table 3  regional capacity against radius and capacity energy scale
"""
import numpy as np

from analyses import FORMS, gate_form, half_time, IMAX, DI
from model_core import channels, couplings, gate
from robustness import blocks, channels_partition

HBAR, C, EV, AMU = 1.054571817e-34, 2.99792458e8, 1.602176634e-19, 1.66053907e-27


def table1():
    """max I_past, its plateau, the first crossing and the gated half time."""
    g = couplings()
    t = np.linspace(0.0, 3.0, 60001)
    out = {}
    for size in (1, 2, 4, 5):
        d = channels_partition(t, g, blocks(len(g), size))
        x = t / d["T2"]
        base = d["coherence"] / d["coherence"][0]
        q = gate(d["I_total"], IMAX, DI)
        out[f"{len(g) // size}x{size}"] = dict(
            max_I_past=float(d["I_past"].max()),
            plateau=float(d["I_past"][x > 8].mean()),
            t_sat=float(x[int(np.argmax(d["I_total"] >= IMAX))]),
            t_half=float(half_time(x, base * q / q[0])))
    return out


def table2():
    """Gated half time, peak residual, the matched pair, and the 1% capacity."""
    g = couplings()
    t = np.linspace(0.0, 0.60, 48001)
    d = channels(t, g, kappa=1.0)
    x = t / d["T2"]
    base = d["coherence"] / d["coherence"][0]
    N = 20
    j = np.arange(N)
    fA = 1.0 + 0.03 * (2 * (j + 0.5) / N - 1.0)
    fB = 10.0 ** ((j / (N - 1.0)) - 0.5)
    fB *= np.sqrt((fA ** 2).sum() / (fB ** 2).sum())
    tm = np.linspace(0.0, 0.5, 40001)
    I85 = (d["I_past"] + d["I_present"]
           + np.minimum(0.85 * (4.0 * d["dH"] / np.pi) * t, float(len(g) + 1)))
    out = {}
    for f in FORMS:
        G = gate_form(d["I_total"], IMAX, DI, f)
        gtd = base * G / G[0]
        peaks = []
        for ff in (fA, fB):
            dm = channels_partition(tm, ff, blocks(N, 1))
            xm = tm / dm["T2"]
            Gm = gate_form(dm["I_total"], IMAX, DI, f)
            bm = dm["coherence"] / dm["coherence"][0]
            peaks.append(float((bm - bm * Gm / Gm[0])[xm <= 3].max()))
        Ics = np.linspace(3.0, 20.0, 3401)
        pk = np.array([float((base * (1 - gate_form(I85, c, DI, f)
                                     / gate_form(I85, c, DI, f)[0])).max())
                       for c in Ics])
        out[f] = dict(t_half=float(half_time(x, gtd)),
                      peak_residual=float((base - gtd).max()),
                      matched_A=peaks[0], matched_B=peaks[1],
                      I_cap_1pct=float(Ics[np.nonzero(pk >= 0.01)[0][-1]]))
    return out


def table3():
    """Bekenstein capacity in bits for C60, over radii and energy scales."""
    M = 60 * 12.011 * AMU                        # C60, 720.7 amu
    radii = {"cage_0.5nm": 0.5e-9, "packet_10nm": 10e-9,
             "path_1um": 1e-6, "apparatus_1m": 1.0}
    energies = {"Mc2": M * C ** 2, "13eV": 13 * EV, "3eV": 3 * EV}
    return {r: {e: 2 * np.pi * R * E / (HBAR * C * np.log(2))
                for e, E in energies.items()} for r, R in radii.items()}


if __name__ == "__main__":
    print("Table 1  partition dependence, N = 20")
    print(f"{'partition':>12} {'max I_past':>11} {'plateau':>9} "
          f"{'t_sat/T2':>9} {'t_half/T2':>10}")
    for k, v in table1().items():
        print(f"{k:>12} {v['max_I_past']:11.2f} {v['plateau']:9.2f} "
              f"{v['t_sat']:9.2f} {v['t_half']:10.2f}")

    print("\nTable 2  gate form, I_cap = 3 bits, dI = 0.4 bits")
    print(f"{'gate':>10} {'t_half/T2':>10} {'peak':>7} {'A':>7} {'B':>7} "
          f"{'I_cap 1%':>9}")
    for k, v in table2().items():
        print(f"{k:>10} {v['t_half']:10.2f} {v['peak_residual']:7.2f} "
              f"{v['matched_A']:7.2f} {v['matched_B']:7.2f} "
              f"{v['I_cap_1pct']:9.1f}")

    print("\nTable 3  regional capacity in bits")
    t3 = table3()
    cols = list(next(iter(t3.values())))
    print(f"{'radius':>14}" + "".join(f"{c:>12}" for c in cols))
    for k, v in t3.items():
        print(f"{k:>14}" + "".join(f"{v[c]:12.3g}" for c in cols))
