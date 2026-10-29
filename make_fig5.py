"""Fig. 5: Model II, channels obtained from an interacting bath Hamiltonian."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREEN, BLACK = "#0072B2", "#E69F00", "#009E73", "#000000"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 13,
    "axes.linewidth": 1.0, "pdf.fonttype": 42,
})

D = __import__("model2_data").load()
t, T2, dH = D["t"], float(D["T2"]), float(D["dH"])
x = t / T2
tau_echo = np.pi / (4 * dH)
STY = {"XX": (BLUE, "-", r"XX $(\Delta=0,\,J_2=0)$"),
       "XXZ": (ORANGE, (0, (7, 4)), r"XXZ $(\Delta=1,\,J_2=0)$"),
       "XXZ2": (GREEN, (0, (1.2, 3.2)), r"XXZ $(\Delta=1,\,J_2=0.5J_1)$")}

fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.7),
                       gridspec_kw=dict(wspace=0.30, bottom=0.26, top=0.95,
                                        left=0.055, right=0.99))
LW = {"XX": 3.0, "XXZ": 1.9, "XXZ2": 1.2}   # nested widths: the three
#  curves coincide in (b) and (c), so each is drawn thinner than the last
MK = {"XX": ("o", (0, 26)), "XXZ": ("s", (9, 26)), "XXZ2": ("^", (18, 26))}
m = x <= 4

a = ax[0]
for k, (c, ls, lab) in STY.items():
    a.plot(x[m], D[f"{k}_I_past"][m], color=c, ls=ls, lw=LW[k], label=lab,
           marker=MK[k][0], markevery=MK[k][1], ms=5.5,
           markerfacecolor="white", markeredgewidth=1.2)
a.set_xlabel(r"$t/T_2$", fontsize=14)
a.set_ylabel(r"$I_{\rm past}$ (bits)", fontsize=14)
a.legend(loc="upper left", frameon=False, fontsize=11.5,
         handlelength=2.2, borderaxespad=0.3)
a.set_ylim(-0.2, 7.8)

b = ax[1]
for k, (c, ls, lab) in STY.items():
    b.plot(x[m], D[f"{k}_I_future"][m], color=c, ls=ls, lw=LW[k], label=lab,
           marker=MK[k][0], markevery=MK[k][1], ms=5.5,
           markerfacecolor="white", markeredgewidth=1.2)
b.set_xlabel(r"$t/T_2$", fontsize=14)
b.set_ylabel(r"$I_{\rm future}$ (bits)", fontsize=14)
b.set_ylim(-0.12, 2.75)
b.legend(loc="upper left", frameon=False, fontsize=11.5,
         handlelength=2.2, borderaxespad=0.3)

c_ = ax[2]
for k, (c, ls, lab) in STY.items():
    ka = tau_echo * np.gradient(D[f"{k}_I_future"], t)
    c_.plot(x[m][2:], ka[m][2:], color=c, ls=ls, lw=LW[k], label=lab,
            marker=MK[k][0], markevery=MK[k][1], ms=5.5,
            markerfacecolor="white", markeredgewidth=1.2)
c_.axhline(1.0, color=BLACK, lw=1.8)
c_.text(0.08, 1.03, r"$\kappa_{\max}$", ha="left", va="bottom", fontsize=13)
c_.set_ylim(0, 1.62)
c_.set_xlabel(r"$t/T_2$", fontsize=14)
c_.set_ylabel(r"$\kappa_{\rm alt}(t)$", fontsize=14)
c_.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False,
          fontsize=9, handlelength=1.8, borderaxespad=0.15,
          labelspacing=0.3)

for k, a_ in enumerate(ax):
    a_.tick_params(labelsize=12)
    a_.text(0.5, -0.29, f"({chr(97 + k)})", transform=a_.transAxes,
            ha="center", va="top", fontsize=19)

# ---- insets: deviation from the XX bath ---------------------------------
for host, key, lbl in (
        (b, "I_future",
         r"$\Delta I_{\rm future}=I_{\rm future}^{\rm bath}"
         r"-I_{\rm future}^{\rm XX}$"),
        (c_, "kappa",
         r"$\Delta\kappa_{\rm alt}=\kappa_{\rm alt}^{\rm bath}"
         r"-\kappa_{\rm alt}^{\rm XX}$")):
    pos = ([0.55, 0.08, 0.42, 0.30] if key == "I_future"
           else [0.63, 0.71, 0.34, 0.26])
    ins = host.inset_axes(pos)
    ins.patch.set_facecolor("white"); ins.patch.set_alpha(0.95)
    if key == "I_future":
        ref = D["XX_I_future"][m]
        cur = {k: D[f"{k}_I_future"][m] for k in STY}
        xs = x[m]
    else:
        ref = tau_echo * np.gradient(D["XX_I_future"], t)[m][2:]
        cur = {k: tau_echo * np.gradient(D[f"{k}_I_future"], t)[m][2:]
               for k in STY}
        xs = x[m][2:]
    for k, (col, ls, _) in STY.items():
        if k == "XX":
            continue
        ins.plot(xs, 1e3 * (cur[k] - ref), color=col, ls=ls, lw=1.6)
    ins.axhline(0.0, color=BLACK, lw=0.8)
    ins.set_xlim(0, 4)
    ins.tick_params(labelsize=12, pad=1.5)   # the caption names the quantity

fig.savefig("fig_model2.png", dpi=400, bbox_inches="tight")
for k in STY:
    ka = tau_echo * np.gradient(D[f"{k}_I_future"], t)
    print(f"{k:>5s}: max I_past {D[f'{k}_I_past'].max():5.2f}   "
          f"max kappa_alt {ka[2:][x[2:] <= 4].max():.3f}")
