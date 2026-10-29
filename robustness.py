"""Partition robustness of I_past and matched-T2 coupling-structure comparison."""
import numpy as np
from model_core import h2, couplings, gate

IMAX, DI, KAPPA = 3.0, 0.4, 1.0


def channels_partition(t, g, parts, kappa=KAPPA):
    """Exact channels for a stated partition of the environment."""
    t = np.atleast_1d(np.asarray(t, float))
    C = np.cos(2.0 * np.outer(t, g))
    r = np.prod(C, axis=1)
    S_S = h2((1.0 + np.abs(r)) / 2.0)
    I_past = np.zeros_like(t)
    for F in parts:
        comp = [k for k in range(len(g)) if k not in F]
        cF = np.prod(C[:, F], axis=1)
        rbar = np.prod(C[:, comp], axis=1) if comp else np.ones_like(t)
        I_past += S_S + h2((1 + np.abs(cF)) / 2) - h2((1 + np.abs(rbar)) / 2)
    I_pres = 1.0 - S_S
    dH = np.sqrt(np.sum(g ** 2))
    T2 = 1.0 / (np.sqrt(2.0) * dH)
    rate = 4.0 * kappa * dH / np.pi
    I_fut = np.minimum(rate * t, float(len(g) + 1))
    return dict(t=t, r=r, coherence=0.5 * np.abs(r), I_past=I_past,
                I_present=I_pres, I_future=I_fut,
                I_total=I_past + I_pres + I_fut, T2=T2, dH=dH, rate=rate)


def blocks(N, size):
    return [list(range(i, i + size)) for i in range(0, N, size)]


if __name__ == "__main__":
    g = couplings()
    N = len(g)
    t = np.linspace(0.0, 3.0, 60001)

    print("=== partition robustness (N = 20) ===")
    print(f"{'partition':>18} {'max I_past':>11} {'plateau':>9} "
          f"{'t_sat/T2':>9} {'t_1/2 gated':>12}")
    for size in (1, 2, 4, 5):
        parts = blocks(N, size)
        d = channels_partition(t, g, parts)
        x = t / d["T2"]
        base = d["coherence"] / d["coherence"][0]
        q = gate(d["I_total"], IMAX, DI)
        q = q / q[0]
        k = int(np.argmax(d["I_total"] >= IMAX))
        kh = int(np.argmax(base * q <= 0.5))
        plateau = d["I_past"][x > 8].mean()
        print(f"{len(parts):>4} x {size:<2} qubits  {d['I_past'].max():11.2f} "
              f"{plateau:9.2f} {x[k]:9.2f} {x[kh]:12.2f}")

    print("\n=== matched-T2 coupling structures ===")
    j = np.arange(N)
    fA = 1.0 + 0.03 * (2 * (j + 0.5) / N - 1.0)
    fB = 10.0 ** ((j / (N - 1.0)) - 0.5)
    fB *= np.sqrt((fA ** 2).sum() / (fB ** 2).sum())
    for name, f in (("A uniform", fA), ("B heterogeneous", fB)):
        print(f"{name:>16}: sum f^2 = {(f**2).sum():.4f}  "
              f"min {f.min():.3f}  max {f.max():.3f}  ratio {f.max()/f.min():.1f}")
    dA = channels_partition(t, fA, blocks(N, 1))
    dB = channels_partition(t, fB, blocks(N, 1))
    print(f"T2_A = {dA['T2']:.5f}   T2_B = {dB['T2']:.5f}   "
          f"dH_A = {dA['dH']:.4f}   dH_B = {dB['dH']:.4f}")
    for name, d in (("A", dA), ("B", dB)):
        x = t / d["T2"]
        base = d["coherence"] / d["coherence"][0]
        q = gate(d["I_total"], IMAX, DI)
        q = q / q[0]
        dC = base - base * q
        k = int(np.argmax(d["I_total"] >= IMAX))
        print(f"  model {name}: t_sat/T2 = {x[k]:.2f}, "
              f"max residual = {dC.max():.3f} at t/T2 = {x[dC.argmax()]:.2f}, "
              f"I_past max = {d['I_past'].max():.2f}")
