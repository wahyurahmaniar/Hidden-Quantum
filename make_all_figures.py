"""Regenerate every figure of the manuscript, in figure order.

Each script writes one PNG into the working directory. Running this file
reproduces Figs. 1 to 14 from the models in model_core.py, model_x.py,
model_xosc.py and model2.py. Figs. 8 and 9 read the interacting-bath results,
which run_model2.py regenerates by exact diagonalization.
"""
import subprocess
import sys

FIGURES = [
    ("Fig. 1  region of support", "make_fig0.py", "fig_region.png"),
    ("Fig. 2  hidden harmonic mode", "make_fig_hidden.py", "fig_hidden.png"),
    ("Fig. 3  scope of the floor", "make_fig_general.py", "fig_general.png"),
    ("Fig. 4  observable coherence", "make_fig_cobs.py", "fig_cobs.png"),
    ("Fig. 5  budget and gate", "make_fig1.py", "fig_budget.png"),
    ("Fig. 6  matched-T2 residual", "make_fig_resid.py", "fig_residual.png"),
    ("Fig. 7  budget combination", "make_fig_combination.py", "fig_combination.png"),
    ("Fig. 8  interacting baths", "make_fig5.py", "fig_model2.png"),
    ("Fig. 9  speed-limit count", "make_fig_qsl.py", "fig_qsl.png"),
    ("Fig. 10 visibility of the residual", "make_fig4.py", "fig_map.png"),
    ("Fig. 11 gate forms", "make_fig_gates.py", "fig_gates.png"),
    ("Fig. 12 stages of a measurement", "make_fig2.py", "fig_mechanism.png"),
    ("Fig. 13 modular floor", "make_fig_modular.py", "fig_modular.png"),
    ("Fig. 14 regional variables", "make_fig3.py", "fig_structure.png"),
]

if __name__ == "__main__":
    only = sys.argv[1:]
    for label, script, out in FIGURES:
        if only and script not in only:
            continue
        print(f"[{label}] {script} -> {out}", flush=True)
        # a separate process per script, so that no figure inherits the
        # Matplotlib settings of the one before it
        subprocess.run([sys.executable, script], check=True)
    print("done")
