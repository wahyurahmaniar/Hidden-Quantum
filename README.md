# Energy-constrained inaccessible degrees of freedom and quantum coherence in bounded systems

Code that reproduces every figure, table and quoted number of

> W. Rahmaniar, *Energy-Constrained Inaccessible Degrees of Freedom and Quantum
> Coherence in Bounded Systems*, Chinese Journal of Physics (manuscript
> CJPHY-D-26-02599).

Everything is plain Python. No value is entered by hand: each number in the
paper is recomputed from the models below.

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.11, NumPy 2.x, SciPy 1.14 and Matplotlib 3.9.

## Reproduce the paper

```bash
python run_model2.py        # exact diagonalization of the three baths (~20 s)
python make_all_figures.py  # Figs. 1 to 14
python tables.py            # the numbers in Tables 1 to 3
python check_numbers.py     # every number quoted in the text
```

`check_numbers.py` prints the computed value beside the one in the manuscript
and ends with `ALL NUMBERS VERIFIED`. `make_all_figures.py` runs each figure
script in its own process, so no figure inherits another one's Matplotlib
settings; name a script to rebuild one figure, for example
`python make_all_figures.py make_fig_gates.py`.

## Models

| File | Contents |
|---|---|
| `model_core.py` | Exactly solved central-spin dephasing model: decoherence factor, the record, coherence and future channels, and the logistic gate |
| `model_x.py` | Conditional translation of a Gaussian packet in the inaccessible sector |
| `model_xosc.py` | Hidden harmonic mode: branch overlap, hidden energy, Holevo information, energy-constrained capacity |
| `model2.py`, `run_model2.py` | Interacting baths, one XX and two XXZ chains, by exact diagonalization |
| `model2_data.py` | Stores and loads those results, `model2_results.csv` |
| `robustness.py` | Matched-`T2` coupling sequences and the partitioned record channel |
| `analyses.py` | Gate-form comparison, the budget-combination class, the dynamical future channel, the random-state test of the general floor, and the regime of the floor |

## Figures

| Figure | Script | Output |
|---|---|---|
| 1 | `make_fig0.py` | `fig_region.png` |
| 2 | `make_fig_hidden.py` | `fig_hidden.png` |
| 3 | `make_fig_general.py` | `fig_general.png` |
| 4 | `make_fig_cobs.py` | `fig_cobs.png` |
| 5 | `make_fig1.py` | `fig_budget.png` |
| 6 | `make_fig_resid.py` | `fig_residual.png` |
| 7 | `make_fig_combination.py` | `fig_combination.png` |
| 8 | `make_fig5.py` | `fig_model2.png` |
| 9 | `make_fig_qsl.py` | `fig_qsl.png` |
| 10 | `make_fig4.py` | `fig_map.png` |
| 11 | `make_fig_gates.py` | `fig_gates.png` |
| 12 | `make_fig2.py` | `fig_mechanism.png` |
| 13 | `make_fig_modular.py` | `fig_modular.png` |
| 14 | `make_fig3.py` | `fig_structure.png` |

`figstyle.py` holds the shared plotting style.

## Tables

`tables.py` recomputes the numbers that appear in the tables. Each function
returns them as a dictionary, and running the file prints them.

| Table | Contents | Source |
|---|---|---|
| 1 | Partition dependence in the solved model | `robustness.channels_partition` |
| 2 | Dependence of the predictions on the gate form | `analyses.gate_form` |
| 3 | Regional capacity against radius and capacity energy scale | Bekenstein form, evaluated for C60 |
| 4 | Status of each result and the assumptions it carries | statement about the text, no computation |

## Data

`run_model2.py` writes `model2_results.csv`, the channels of the three bath
Hamiltonians on a grid of 241 times, with the generator spread, `T2` and the
orthogonalization time in its header. It also writes `model2_results.npz` as a
local cache, which is not tracked. The whole run takes about twenty seconds, so
the CSV can be deleted and regenerated at any time.

## License

MIT, see `LICENSE`.
