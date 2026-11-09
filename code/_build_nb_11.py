#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build notebooks/11_graphiques.ipynb from the single source of truth
(code/11_graphiques.py -> CELLS), execute it with nbclient (inline backend —
the notebook NEVER sets matplotlib.use('Agg'), that lives only in the mirror
script), bake the outputs, and validate. Also verifies the 8 PNGs produced
by the kernel run exist in outputs/.
"""
import importlib.util
import os

import nbformat as nbf
from nbclient import NotebookClient

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_mirror():
    path = os.path.join(BASE, 'code', '11_graphiques.py')
    spec = importlib.util.spec_from_file_location('mirror11', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_nb(cells, title):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3 (ipykernel)",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"name": title, "provenance": []},
        "accelerator": "GPU",
    }
    nb.cells = cells
    return nb


def main():
    mod = load_mirror()
    cells = []
    for i, (kind, src) in enumerate(mod.CELLS):
        # Pre-gate: every code cell must compile; no explicit Agg in notebook.
        if kind == "code":
            compile(src, f"<cell {i}>", "exec")
            if "matplotlib.use" in src:
                raise SystemExit(f"FAIL: matplotlib.use() dans une cellule "
                                 f"du notebook (cell {i})")
            cells.append(nbf.v4.new_code_cell(src))
        else:
            cells.append(nbf.v4.new_markdown_cell(src))
    nb = make_nb(cells, "DLCyber - T11 Graphiques")
    nb_path = os.path.join(BASE, 'notebooks', '11_graphiques.ipynb')
    print(f"building {nb_path} ({len(cells)} cells)")

    client = NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": os.path.dirname(nb_path)}})
    client.execute()
    nbf.write(client.nb, nb_path)
    nbf.validate(client.nb)

    n_figs = sum(1 for k, s in mod.CELLS if k == "md" and s.startswith("## Figure"))
    n_asserts = sum(s.count("assert ") for k, s in mod.CELLS if k == "code")
    out_dir = os.path.join(BASE, 'outputs')
    missing = [n for n in mod.FIG_NAMES
               if not os.path.isfile(os.path.join(out_dir, n))]
    if n_figs != 8:
        raise SystemExit(f"FAIL: {n_figs} sections Figure (attendu 8)")
    if missing:
        raise SystemExit(f"FAIL: PNG manquants apres execution du notebook : {missing}")
    n_pngs = sum(1 for s in mod.CELLS if s[0] == "code" and "savefig" in s[1])
    print(f"PASS — 8 figures ({n_pngs} cellules savefig), {n_asserts} asserts, "
          f"{len(cells)} cells, 8 PNG presents, nbformat validate OK -> {nb_path}")


if __name__ == "__main__":
    main()
