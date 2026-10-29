"""Model II production run: three bath regimes at matched N, dH, T2."""
import numpy as np
from model2 import channels_model2

N = 10
g = 0.15 * (1 + 0.3 * np.cos(np.arange(N)))
t = np.linspace(0, 6, 241)
REG = {"XX": dict(J1=1.0, delta=0.0, J2=0.0),
       "XXZ": dict(J1=1.0, delta=1.0, J2=0.0),
       "XXZ2": dict(J1=1.0, delta=1.0, J2=0.5)}
out = {"t": t, "g": g}
for k, kw in REG.items():
    d = channels_model2(N, g, t, frag=2, **kw)
    for f in ("I_past", "I_present", "I_future", "coherence"):
        out[f"{k}_{f}"] = d[f]
    out[f"{k}_r"] = np.abs(d["r"])
    out["dH"], out["T2"], out["tau_MT"] = d["dH"], d["T2"], d["tau_MT"]
from model2_data import save
save(out)

tau_echo = np.pi / (4 * out["dH"])
print(f"N={N}  dH={out['dH']:.4f}  T2={out['T2']:.4f}  tau_echo={tau_echo:.4f}")
print(f"{'bath':>6s} {'maxIpast':>9s} {'Ifut(3T2)':>10s} {'maxKalt':>8s} {'Kalt(T2)':>9s}")
x = t / out["T2"]
for k in REG:
    ka = tau_echo * np.gradient(out[f"{k}_I_future"], t)
    k1 = int(np.argmin(abs(x - 1)))
    k3 = int(np.argmin(abs(x - 3)))
    print(f"{k:>6s} {out[f'{k}_I_past'].max():9.2f} "
          f"{out[f'{k}_I_future'][k3]:10.3f} {ka[2:].max():8.3f} {ka[k1]:9.3f}")
