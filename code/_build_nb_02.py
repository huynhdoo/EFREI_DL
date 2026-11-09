#!/usr/bin/env python3
"""Build notebooks/02_dnn_gradients_surapprentissage.ipynb from the mirror script.

Single source of truth: imports BLOCKS from code/02_dnn_gradients_surapprentissage.py,
converts them to nbformat v4 cells, pre-gates each code cell with compile(),
executes the notebook with nbclient and writes it back with baked outputs.
"""
import importlib.util
import os
import sys

import nbformat as nbf
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MIRROR = os.path.join(HERE, "02_dnn_gradients_surapprentissage.py")
OUT = os.path.join(ROOT, "notebooks", "02_dnn_gradients_surapprentissage.ipynb")

# Load the mirror module (digit-leading name -> importlib by path).
spec = importlib.util.spec_from_file_location("mirror02", MIRROR)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

NB = nbf.v4

nb = NB.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)",
                   "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
    "colab": {"name": "DLCyber - T02 Reseaux Profonds Gradients Surapprentissage", "provenance": []},
    "accelerator": "GPU",
}

cells = []
for blk in mod.BLOCKS:
    if blk["kind"] == "md":
        cells.append(NB.new_markdown_cell(blk["src"]))
    else:
        # Pre-gate: every code cell must compile once !/ magics lines are stripped.
        src = "\n".join(l for l in blk["src"].splitlines()
                        if not l.lstrip().startswith(("!", "%")))
        compile(src, "<cell>", "exec")
        cells.append(NB.new_code_cell(blk["src"]))
nb.cells = cells

# Code cells keep their original source (including the !pip line, kept for Colab).
os.makedirs(os.path.dirname(OUT), exist_ok=True)
client = NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()
nbf.write(nb, OUT)
nbf.validate(nb)

n_ex = sum(1 for c in nb.cells
           if c.cell_type == "code" and "# ---- Exercice P" in c.source)
n_assert = sum(c.source.count("assert ") for c in nb.cells if c.cell_type == "code")
heads = [l[3:] for c in nb.cells if c.cell_type == "markdown"
         for l in c.source.splitlines() if l.startswith("## ") and l[3].isdigit()]
print("wrote", OUT)
print("cells:", len(nb.cells), " exercises:", n_ex, " asserts:", n_assert)
print("sections:", heads)
