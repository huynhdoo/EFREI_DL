#!/usr/bin/env python3
"""Build notebooks/03_regularisation_normalisation_optimisation.ipynb.

Single source of truth: the CELLS list defined by the mirror script
``code/03_regularisation_normalisation_optimisation.py`` (imported, not
duplicated). The only documented transformation: the notebook's first code
cell gets the ``!pip install`` line prepended (notebook-only, Colab-ready).

Execution: NotebookClient(timeout=600) then nbformat.write -> baked outputs.
Run with the build venv python:
    /home/huynhdoo/venvs/notebook-adif72/bin/python code/_build_nb_03.py
"""
import importlib.util
import os
import sys

import nbformat as nbf
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MIRROR = os.path.join(HERE, "03_regularisation_normalisation_optimisation.py")
OUT = os.path.join(ROOT, "notebooks", "03_regularisation_normalisation_optimisation.ipynb")
PIP_LINE = "!pip install -q numpy scipy pandas matplotlib scikit-learn"
COLAB_NAME = "DLCyber - T03 Regularisation Normalisation Optimisation"


def load_cells():
    spec = importlib.util.spec_from_file_location("mirror03", MIRROR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CELLS


def make_nb(cells):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3 (ipykernel)",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"name": COLAB_NAME, "provenance": []},
        "accelerator": "GPU",
    }
    nb.cells = cells
    return nb


def main():
    cells_src = load_cells()
    cells, first_code = [], True
    for kind, src in cells_src:
        if kind == "md":
            cells.append(nbf.v4.new_markdown_cell(src))
        else:
            compile(src, "<cell-pre-gate>", "exec")  # pre-gate: every cell compiles
            if first_code:
                src = PIP_LINE + "\n" + src
                first_code = False
            cells.append(nbf.v4.new_code_cell(src))
    nb = make_nb(cells)

    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    nb = client.nb
    nbf.validate(nb)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    nbf.write(nb, OUT)

    n_ex = sum(1 for c in nb.cells
               if c.cell_type == "code" and "# ---- Exercice P" in c.source)
    headers = [ln.strip() for c in nb.cells if c.cell_type == "markdown"
               for ln in c.source.splitlines() if ln.strip().startswith("## ")]
    print("wrote", OUT)
    print("cells:", len(nb.cells), "| Exercice P blocks:", n_ex)
    print("section headers:", " | ".join(headers))
    print("NB 03 BUILD PASS ✅")


if __name__ == "__main__":
    sys.exit(main())
