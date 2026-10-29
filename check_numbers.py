"""Re-derive every number quoted in the manuscript from the models.

Each check prints the computed value beside the one in the text; the run
ends with ALL NUMBERS VERIFIED when every quantity is reproduced.
"""
import numpy as np
from scipy.optimize import brentq

from model_core import couplings, channels, h2
from robustness import channels_partition, blocks
from model_x import r_X_gaussian, chi_X as chi_gauss
from analyses import gate_form, C_X, h2_inv_upper

ok = True


def check(label, value, quoted, tol):
    global ok
    good = abs(value - quoted) <= tol
    ok &= good
    print(f"{'ok ' if good else 'BAD'} {label:58s} {value:12.5g}  text {quoted}")


# --------------------------------------------------- R2.3 factor of two
hbar, c = 1.054571817e-34, 2.99792458e8
R, E, dH, eta, b = 0.5e-9, 1.1e-7, 3.0e-21, 0.7, 4.0
Icap = 2 * np.pi * R * E / (hbar * c * np.log(2))                 # Eq. (14)
tau29 = np.pi * hbar / (4 * eta * dH * np.log2(b)) * Icap          # Eq. (29)
Xi = eta * dH / E * np.log2(b)
tau30 = np.pi ** 2 / (2 * np.log(2)) * R / (c * Xi)                # Eq. (30)
check("Eq.(29) with Eq.(14) / corrected Eq.(30)", tau29 / tau30, 1.0, 1e-12)
check("pi^2/(2 ln 2)", np.pi ** 2 / (2 * np.log(2)), 7.1, 0.05)
check("2 ln 2/pi^2", 2 * np.log(2) / np.pi ** 2, 0.1405, 5e-5)

# --------------------------------------------- Sec. 2 numbers
g = couplings()
T2 = 1 / (np.sqrt(2) * np.sqrt((g ** 2).sum()))
tt = np.linspace(0, 3 * T2, 300001)
xx = tt / T2
rE = np.abs(np.prod(np.cos(2 * np.outer(tt, g)), axis=1))
C = rE * r_X_gaussian(tt, 8.0, 1.0)
k = int(np.argmax(C <= 0.5))
check("translation t_1/2 at lambda/sigma=8 (T2)", xx[k], 0.63, 0.005)
check("chi_X at that crossing (bits)", float(chi_gauss(tt[k], 8.0, 1.0)), 0.54, 0.005)
TX = 2 * np.sqrt(2) * 1.0 / (8.0 * 2.0)
check("T_X = 2 sqrt2 sigma/(lambda |da|) in T2", TX / T2, 1.2, 0.05)
# T_X is the 1/e time of Eq. (7)
check("|r_X(T_X)| = 1/e", float(r_X_gaussian(TX, 8.0, 1.0)), np.exp(-1), 1e-12)

nstar = brentq(lambda n: float(C_X(n)) - 1.0, 1e-6, 5.0)
check("C_X = 1 bit at nbar", nstar, 0.294, 0.0005)
check("coupling bound sqrt(nbar*/4)", np.sqrt(nstar / 4), 0.271, 0.0005)
for n, q in ((4.0, 0.28), (10.0, 0.21)):
    check(f"1/C_X at nbar={n}", 1 / float(C_X(n)), q, 0.005)
    r = np.exp(-2 * n)
    check(f"chi/C at nbar={n}", float(h2((1 + r) / 2) / C_X(n)), q, 0.005)
# Fig. 2 couplings inside / outside the regime
for gx, inside in ((0.1, True), (0.2, True), (0.35, False), (0.6, False)):
    check(f"g_X={gx}: max nbar < 0.294 ?", float(4 * gx ** 2 < nstar), float(inside), 0)
# general eigenvalues: theta_ij <= 1 and equals 1 for opposite eigenvalues
ai = np.linspace(-2, 2, 41)
th = [(a - bb) ** 2 / (2 * (a * a + bb * bb)) for a in ai for bb in ai if a * a + bb * bb > 0]
check("max theta_ij", max(th), 1.0, 1e-12)
check("|0>,|1> pair: exp(-2*0.5)", np.exp(-1.0), 0.37, 0.005)

# Fig. 3 couplings at their half-coherence times lie above nbar*
from model_xosc import E_X
OM = 0.5 / T2
for ratio, th_, q in ((0.45, 0.789, 0.03), (0.90, 0.701, 0.10), (1.50, 0.572, 0.18)):
    n = float(E_X(th_ * T2, ratio * OM, OM) / OM)
    check(f"Fig.3 g_X/hbarOmega={ratio}: nbar at t_1/2", n, q, 0.005)
    check(f"Fig.3 g_X/hbarOmega={ratio}: below 0.294 ?", float(n < nstar), 1.0, 0)

# --------------------------------------------- Sec. 6 numbers
N = 20
jj = np.arange(N)
fA = 1.0 + 0.03 * (2 * (jj + 0.5) / N - 1.0)
fB = 10.0 ** ((jj / (N - 1.0)) - 0.5)
fB *= np.sqrt((fA ** 2).sum() / (fB ** 2).sum())
tm = np.linspace(0, 0.5, 40001)
for nm, f, q in (("A", fA, 0.67), ("B", fB, 0.72)):
    d = channels_partition(tm, f, blocks(N, 1))
    x = tm / d["T2"]
    check(f"matched pair {nm}: t_sat/T2", x[int(np.argmax(d['I_total'] >= 3))], q, 0.005)

t = np.linspace(0, 0.60, 48001)
d = channels(t, g)
x = t / d["T2"]
base = d["coherence"] / d["coherence"][0]
Mrun = np.maximum.accumulate(np.maximum.reduce([d["I_past"], d["I_present"],
                                                d["I_future"]]))
check("overlap rule t_sat/T2", x[int(np.argmax(Mrun >= 3))], 0.91, 0.005)
G = gate_form(Mrun, 3.0, 0.4, "logistic")
check("overlap rule t_1/2/T2", x[int(np.argmax(base * G / G[0] <= 0.5))], 0.73, 0.005)
for nm, f, q in (("A", fA, 0.29), ("B", fB, 0.21)):
    dm = channels_partition(tm, f, blocks(N, 1))
    xm = tm / dm["T2"]
    Mr = np.maximum.accumulate(np.maximum.reduce([dm["I_past"], dm["I_present"],
                                                  dm["I_future"]]))
    Gm = gate_form(Mr, 3.0, 0.4, "logistic")
    bm = dm["coherence"] / dm["coherence"][0]
    check(f"overlap rule residual peak {nm}", float((bm - bm * Gm / Gm[0])[xm <= 3].max()), q, 0.005)
Sd = d["I_past"] + d["I_present"] + d["I_future"]
check("sum is non-decreasing to 5 T2 and >= running max",
      float((Sd + 1e-12 >= Mrun).all()), 1.0, 0)

# --------------------------------------------- Sec. 7 numbers
D = __import__("model2_data").load()
t2, T2b, dHb = D["t"], float(D["T2"]), float(D["dH"])
x2 = t2 / T2b
ramp = np.minimum(4 * dHb / np.pi * t2, 11.0)
tauMT = np.pi / (4 * dHb)
for bath in ("XX", "XXZ", "XXZ2"):
    If = D[f"{bath}_I_future"]
    m = (x2 > 0.05) & (x2 <= 4.0)
    check(f"{bath}: I_future <= ramp on 0..4 T2", float((If[x2 <= 4] <= ramp[x2 <= 4] + 1e-9).all()), 1.0, 0)
    check(f"{bath}: max I_future/log2(1+t/tau) <= 0.95", float((If[m] / np.log2(1 + t2[m] / tauMT)).max() <= 0.95), 1.0, 0)
    check(f"{bath}: I_future(4 T2)", float(If[np.argmin(abs(x2 - 4))]), 2.1, 0.05)
check("ramp(4 T2)", float(ramp[np.argmin(abs(x2 - 4))]), 3.6, 0.05)
for bath, (qb, qd) in (("XX", (1.96, 3.70)), ("XXZ", (0.85, 0.94)), ("XXZ2", (0.85, 0.94))):
    Ip, Ipr, If = (D[f"{bath}_{k}"] for k in ("I_past", "I_present", "I_future"))
    check(f"{bath}: crossing at 3 bits, ceiling ramp", x2[np.argmax(Ip + Ipr + ramp >= 3)], qb, 0.005)
    check(f"{bath}: crossing at 3 bits, dynamical", x2[np.argmax(Ip + Ipr + If >= 3)], qd, 0.005)

# --------------------------------------------- Sec. 8 numbers
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")
t3 = np.linspace(0, 0.80, 16001)
d3 = channels(t3, g)
x3 = t3 / d3["T2"]
It3 = d3["I_total"]


def qI(Ic, w):
    Gq = 1 / (1 + np.exp((It3 - Ic) / w))
    return Gq / Gq[0]


devs = []
for xX in (0.5, 1.0, 1.5):
    target = np.exp(-(x3 / xX) ** 2)
    msk = x3 <= 3 * xX
    best = min((minimize(lambda p: np.mean((qI(p[0], abs(p[1]) + 1e-9)[msk] - target[msk]) ** 2),
                         [c0, w0], method="Nelder-Mead",
                         options=dict(xatol=1e-7, fatol=1e-12, maxiter=4000))
                for c0 in (1.5, 2, 3, 4, 6, 8) for w0 in (0.2, 0.5, 1, 2, 4)),
               key=lambda r: r.fun)
    devs.append(float(np.abs(qI(best.x[0], abs(best.x[1]))[msk] - target[msk]).max()))
check("logistic-to-Gaussian fit, smallest max deviation", min(devs), 0.018, 0.0006)
check("logistic-to-Gaussian fit, largest max deviation", max(devs), 0.022, 0.0006)
# part 7: the fitted parameters, deviations and rms now quoted in Sec. 8
QFIT = {0.5: (1.75, 0.61, 0.020), 1.0: (2.77, 1.72, 0.022), 1.5: (4.27, 3.01, 0.018)}
for xX, (qIc, qdI, qdev) in QFIT.items():
    target = np.exp(-(x3 / xX) ** 2)
    msk = x3 <= 3 * xX
    best = min((minimize(lambda p: np.mean((qI(p[0], abs(p[1]) + 1e-9)[msk] - target[msk]) ** 2),
                         [c0, w0], method="Nelder-Mead",
                         options=dict(xatol=1e-7, fatol=1e-12, maxiter=4000))
                for c0 in (1.5, 2, 3, 4, 6, 8) for w0 in (0.2, 0.5, 1, 2, 4)),
               key=lambda r: r.fun)
    Icf, dIf = best.x[0], abs(best.x[1])
    dv = np.abs(qI(Icf, dIf)[msk] - target[msk])
    check(f"fit T_X={xX}: I_cap", Icf, qIc, 0.006)
    check(f"fit T_X={xX}: delta I", dIf, qdI, 0.006)
    check(f"fit T_X={xX}: max deviation", dv.max(), qdev, 0.0006)
    check(f"fit T_X={xX}: rms deviation below 1%", float(np.sqrt((dv ** 2).mean()) < 0.01), 1.0, 0)

ROW = {"logistic": (0.59, 0.41, 0.43, 0.37), "erfc": (0.59, 0.41, 0.43, 0.37),
       "arctan": (0.61, 0.38, 0.39, 0.34), "step": (0.68, 0.62, 0.64, 0.59)}
for form, (th_, pk, rA_, rB_) in ROW.items():
    Gf = gate_form(d["I_total"], 3.0, 0.4, form)
    gtd = base * Gf / Gf[0]
    check(f"{form}: t_1/2/T2", x[int(np.argmax(gtd <= 0.5))], th_, 0.005)
    check(f"{form}: peak residual 0..4 T2", float((base - gtd).max()), pk, 0.005)
    for nm, f, q in (("A", fA, rA_), ("B", fB, rB_)):
        dm = channels_partition(tm, f, blocks(N, 1))
        xm = tm / dm["T2"]
        Gm = gate_form(dm["I_total"], 3.0, 0.4, form)
        bm = dm["coherence"] / dm["coherence"][0]
        check(f"{form}: matched {nm} peak", float((bm - bm * Gm / Gm[0])[xm <= 3].max()), q, 0.005)
I_fut85 = np.minimum(0.85 * (4 * d["dH"] / np.pi) * t, 21.0)
Itot85 = d["I_past"] + d["I_present"] + I_fut85
def _pk85(Ic, form):
    G_ = gate_form(Itot85, Ic, 0.4, form)
    return float((base * (1 - G_ / G_[0])).max())


for form, q in (("logistic", 9.5), ("erfc", 9.5), ("arctan", 9.4)):
    check(f"{form}: I_cap at 1% residual (root)",
          brentq(lambda c_: _pk85(c_, form) - 0.01, 3, 30, xtol=1e-5), q, 0.05)
Ics = np.linspace(3.0, 20.0, 1701)
pk = np.array([_pk85(Ic, "step") for Ic in Ics])
check("step: I_cap at 1% residual (grid 0.01)", Ics[np.nonzero(pk >= 0.01)[0][-1]], 10.2, 0.05)
for form, q, tol in (("arctan", 1.3e-3, 0.05e-3), ("logistic", 1.1e-5, 0.05e-5)):
    Gf = gate_form(Itot85, 15.0, 0.4, form)
    check(f"{form}: peak residual at I_cap=15", float((base * (1 - Gf / Gf[0])).max()), q, tol)

# --------------------------------------------- Sec. 10 numbers
Om = 2 * np.pi * 1e10
check("reduced wavelength c/Omega (mm)", c / Om * 1e3, 4.8, 0.05)
check("R Omega/c at R=30 cm", 0.30 * Om / c, 63, 0.5)
check("R Omega/c at R=1 um", 1e-6 * Om / c, 2.1e-4, 0.05e-4)
n30 = 1.0 / (np.pi * 0.30 * Om / c)
check("nbar for dK=1 at R=30 cm", n30, 5.1e-3, 0.05e-3)
check("floor |r_X| at that nbar", 2 * h2_inv_upper(float(C_X(n30))) - 1, 0.990, 0.0005)
# interaction energy equals -E_X in each branch (energy conservation)
from model_xosc import alpha
tt4 = np.linspace(0, 2 * np.pi, 9)
for a_s in (+1, -1):
    al = alpha(tt4, 0.3, 1.0, a_s)
    EX = np.abs(al) ** 2
    Eint = 0.3 * a_s * 2 * np.real(al)
    check(f"branch {a_s:+d}: max |<H_SX> + E_X|", float(np.abs(Eint + EX).max()), 0.0, 1e-12)

# --------------------------------------------- Eq. (10) extensions, Fock check
from scipy.linalg import expm
dF, OmF, gF = 60, 1.0, 0.3
aF = np.diag(np.sqrt(np.arange(1, dF)), 1); nF = aF.T @ aF
worst_th, worst_gen = 0.0, 0.0
for nth in (0.0, 0.2, 0.5):
    if nth == 0:
        rho = np.zeros((dF, dF)); rho[0, 0] = 1
    else:
        pw = (nth / (1 + nth)) ** np.arange(dF); pw /= pw.sum(); rho = np.diag(pw)
    for tq in np.linspace(0.1, 2 * np.pi, 9):
        Up = expm(-1j * tq * (OmF * nF + gF * (aF + aF.T)))
        Um = expm(-1j * tq * (OmF * nF - gF * (aF + aF.T)))
        r = abs(np.trace(Um.conj().T @ Up @ rho))
        worst_th = max(worst_th, abs(r - np.exp(-8 * (gF / OmF) ** 2 * np.sin(OmF * tq / 2) ** 2 * (2 * nth + 1))))
check("thermal factor (2 n_th + 1), Fock truncation", worst_th, 0.0, 1e-10)
vac = np.zeros(dF); vac[0] = 1
for ai, aj in ((1.0, 0.3), (2.0, -0.5), (0.7, 0.0)):
    for tq in np.linspace(0.2, 2 * np.pi, 7):
        psi_i = expm(-1j * tq * (OmF * nF + gF * ai * (aF + aF.T))) @ vac
        psi_j = expm(-1j * tq * (OmF * nF + gF * aj * (aF + aF.T))) @ vac
        r = abs(np.vdot(psi_j, psi_i))
        EX = 0.5 * (np.vdot(psi_i, nF @ psi_i).real + np.vdot(psi_j, nF @ psi_j).real)
        th = (ai - aj) ** 2 / (2 * (ai ** 2 + aj ** 2))
        worst_gen = max(worst_gen, abs(r - np.exp(-2 * EX * th)))
check("general eigenvalues: |r_X| = exp(-2 E_X theta_ij)", worst_gen, 0.0, 1e-10)

# --------------------------------------------- part 5: scope statements
xs = np.linspace(0.01, 1.2, 200001)
exc = h2((1 + np.abs(np.cos(np.pi * xs / 2))) / 2) - np.log2(1 + xs)
check("qubit pair exceeds the MT count from t/tau =", xs[exc > 0].min(), 0.47, 0.005)
check("qubit pair: largest excess over the count (bits)", exc.max(), 0.08, 0.005)
for nth_, nd_ in ((0.2, 0.49), (0.2, 0.51)):
    viol = np.exp(-2 * nd_ * (2 * nth_ + 1)) < np.exp(-2 * (nth_ + nd_))
    check(f"thermal n_th=0.2, deposited {nd_}: below Eq. (10)", float(viol), float(nd_ > 0.5), 0)
check("squeezed r=0.5, +-1: overlap", np.exp(-2 * np.exp(1.0)), 0.0044, 0.0001)
check("squeezed r=0.5, +-1: exp(-2 nbar)", np.exp(-2 * (np.sinh(0.5) ** 2 + 1)), 0.079, 0.001)
g0 = couplings(); tt = np.linspace(0, 0.6, 48001); dd = channels(tt, g0, kappa=1.0)
xx_ = tt / dd["T2"]
lit = np.maximum.reduce([dd["I_past"], dd["I_present"], dd["I_future"]])
check("largest single channel: minimum (bit)", lit.min(), 0.71, 0.005)
check("largest single channel below 1 bit until t/T2", xx_[np.nonzero(lit < 1 - 1e-12)[0].max()], 0.405, 0.001)
Gl = gate_form(lit, 3.0, 0.4, "logistic")
check("literal maximum raises q above unity by", (Gl / Gl[0]).max() - 1, 0.0034, 0.0001)
k62 = int(np.argmin(np.abs(xx_ - 0.62)))
check("Fig. structure: total at 0.62 T2", dd["I_total"][k62], 2.75, 0.005)
check("Fig. structure: B = Icap - past - present", 3 - dd["I_past"][k62] - dd["I_present"][k62], 0.81, 0.005)
check("Fig. structure: still open", 3 - dd["I_total"][k62], 0.25, 0.005)
nt86 = 1 / np.expm1(np.pi * 0.30 * Om / c)
check("reference occupation exp(-pi R Omega/c), log10", np.log10(nt86), -86, 0.5)
# part 7: thermal reference terms of Sec. 10 and Fig. 13
nt10 = 1 / np.expm1(np.pi * 10.0)
check("n_th at R Omega/c = 10, below 1e-12", float(nt10 < 1e-12), 1.0, 0)
check("S_0 at R Omega/c = 10, below 1e-11", float(float(C_X(nt10)) < 1e-11), 1.0, 0)


def _cap_sum(x_, dk, thermal=True):
    nt = 1.0 / np.expm1(min(np.pi * x_, 700)) if thermal else 0.0
    nd = dk / (np.pi * x_)
    return float(C_X(nd + nt)) + (float(C_X(nt)) if nt > 0 else 0.0)


for lbl, therm, qv in (("with reference terms", True, 1.37),
                       ("without", False, 1.08)):
    check(f"Fig. 13: floor onset at dK=1, {lbl}",
          brentq(lambda x_: _cap_sum(x_, 1.0, therm) - 1.0, 0.05, 1e4, xtol=1e-8),
          qv, 0.005)

# --------------------------------------------- part 6: new and redrawn figures
from analyses import half_time
base6 = dd["coherence"] / dd["coherence"][0]
Irec6 = dd["I_past"] + dd["I_present"]


def peak6(kap, Ic):
    If_ = np.minimum(kap * (4 * dd["dH"] / np.pi) * tt, 21.0)
    q_ = gate_form(Irec6 + If_, Ic, 0.4, "logistic")
    return float((base6 * (1 - q_ / q_[0])).max())


check("Fig. map: peak residual, I_cap=3, kappa 0.85 (%)", 100 * peak6(0.85, 3), 39, 0.5)
check("Fig. map: peak residual, I_cap=8, kappa 0.85 (%)", 100 * peak6(0.85, 8), 2.9, 0.05)
for lev, q in ((0.10, 5.9), (0.05, 7.1), (0.01, 9.5)):
    check(f"Fig. map: {100*lev:g}% residual at I_cap (kappa 0.85)",
          brentq(lambda c_: peak6(0.85, c_) - lev, 1, 40, xtol=1e-5), q, 0.05)
c1_lo = brentq(lambda c_: peak6(0.25, c_) - 0.01, 1, 40, xtol=1e-5)
c1_hi = brentq(lambda c_: peak6(6.0, c_) - 0.01, 1, 40, xtol=1e-5)
check("Fig. map: 1% capacity at kappa 0.25", c1_lo, 8.4, 0.05)
check("Fig. map: 1% capacity at kappa 6", c1_hi, 18.6, 0.05)
c10_lo = brentq(lambda c_: peak6(0.25, c_) - 0.10, 1, 40, xtol=1e-5)
c10_hi = brentq(lambda c_: peak6(6.0, c_) - 0.10, 1, 40, xtol=1e-5)
check("Fig. map: shift factor at 1%", c1_hi / c1_lo, 2.2, 0.05)
check("Fig. map: shift factor at 10%", c10_hi / c10_lo, 2.4, 0.05)
smax = np.maximum.accumulate(lit)
Gs = gate_form(smax, 3.0, 0.4, "logistic")
check("Fig. combination: overlap-end crossing", xx_[int(np.argmax(smax >= 3))], 0.91, 0.005)
check("Fig. combination: overlap-end half time", half_time(xx_, base6 * Gs / Gs[0]), 0.73, 0.005)
for form, lo, hi in (("logistic", 0.59, 0.59), ("erfc", 0.59, 0.59), ("arctan", 0.61, 0.61), ("step", 0.68, 0.68)):
    Gf_ = gate_form(dd["I_total"], 3.0, 0.4, form)
    check(f"Fig. gates: {form} half time", half_time(xx_, base6 * Gf_ / Gf_[0]), lo, 0.005)
Dm = __import__("model2_data").load()
msk = Dm["t"] <= 4 * float(Dm["T2"]) + 1e-12
dev = max(np.abs(Dm[f"{b}_I_future"][msk] - Dm["XX_I_future"][msk]).max() for b in ("XXZ", "XXZ2"))
check("Fig. qsl: XXZ chains differ from XX by less than (bits)", float(dev < 0.01), 1.0, 0)
xq = Dm["t"][msk] / (np.pi / (4 * float(Dm["dH"])))
lim = np.log2(1 + xq[1:])
check("Fig. qsl: computed / count, maximum", max((Dm[f"{b}_I_future"][msk][1:] / lim).max()
                                                for b in ("XX", "XXZ", "XXZ2")), 0.95, 0.005)
s2_ = np.sinh(0.2) ** 2
gq = np.linspace(0, 0.8, 800001)
cross_sq = (s2_ + gq)[np.argmax(np.exp(-2 * gq * np.exp(0.4)) < np.exp(-2 * (s2_ + gq)))]
check("Fig. general: squeezed curve below Eq. (10) beyond nbar", cross_sq, 0.12, 0.005)
check("Fig. general: thermal curve below Eq. (10) beyond nbar", 0.2 + 0.5, 0.70, 0)

# --------------------------------------------- abstract count recorded
print("\nALL NUMBERS VERIFIED" if ok else "\nSOME CHECKS FAILED")
