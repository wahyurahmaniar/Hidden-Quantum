"""Loader for the interacting-bath results of Sec. 7.

`run_model2.py` diagonalizes the three bath Hamiltonians, which takes about
twenty seconds, and stores the channels it computes. The results live in a
text file, `model2_results.csv`, so that they can be read and compared
without NumPy; `.npz` is kept only as a fast local cache. `load()` returns a
dictionary of arrays from whichever file is present, and runs the models if
neither is.
"""
import os
import pathlib
import subprocess
import sys

import numpy as np

CSV = pathlib.Path("model2_results.csv")
NPZ = pathlib.Path("model2_results.npz")


def save(data):
    """Write the results as CSV, one column per array, and as an npz cache."""
    keys = [k for k in data if np.atleast_1d(data[k]).size > 1]
    scal = {k: float(data[k]) for k in data if np.atleast_1d(data[k]).size == 1}
    n = max(np.atleast_1d(data[k]).size for k in keys)
    cols = []
    for k in keys:
        v = np.atleast_1d(data[k]).astype(float)
        cols.append(np.concatenate([v, np.full(n - v.size, np.nan)]))
    header = ("# interacting-bath channels of Sec. 7, from run_model2.py\n"
              "# " + "  ".join(f"{k}={v!r}" for k, v in scal.items()) + "\n"
              + ",".join(keys))
    np.savetxt(CSV, np.column_stack(cols), delimiter=",", header=header,
               comments="", fmt="%.12g")
    np.savez(NPZ, **data)


def load():
    """The results as a dictionary of arrays, computing them if needed."""
    if NPZ.exists():
        return dict(np.load(NPZ))
    if not CSV.exists():
        subprocess.run([sys.executable, "run_model2.py"], check=True)
        return load()
    with open(CSV) as fh:
        lines = [fh.readline() for _ in range(3)]
    scal = dict(kv.split("=") for kv in lines[1][2:].split() if "=" in kv)
    keys = lines[2].strip().split(",")
    arr = np.loadtxt(CSV, delimiter=",", skiprows=3, ndmin=2)
    out = {k: arr[:, j][~np.isnan(arr[:, j])] for j, k in enumerate(keys)}
    out.update({k: np.array(float(v)) for k, v in scal.items()})
    return out


if __name__ == "__main__":
    d = load()
    print(f"{len(d)} arrays, {sum(np.atleast_1d(v).size for v in d.values())} values")
