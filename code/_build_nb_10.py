#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build notebooks/10_preparation_projet.ipynb from the single source of truth
(code/10_preparation_projet.py -> CELLS), execute it with nbclient, bake the
outputs, and validate. Re-run until PASS.
"""
import importlib.util
import os
import sys

import nbformat as nbf
from nbclient import NotebookClient

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_mirror():
    path = os.path.join(BASE, 'code', '10_preparation_projet.py')
    spec = importlib.util.spec_from_file_location('mirror10', path)
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
        # Pre-gate: every code cell must compile before we ever build.
        if kind == "code":
            compile(src, f"<cell {i}>", "exec")
            cells.append(nbf.v4.new_code_cell(src))
        else:
            cells.append(nbf.v4.new_markdown_cell(src))
    nb = make_nb(cells, "DLCyber - T10 Preparation Projet")
    nb_path = os.path.join(BASE, 'notebooks', '10_preparation_projet.ipynb')
    print(f"building {nb_path} ({len(cells)} cells)")

    client = NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": os.path.dirname(nb_path)}})
    client.execute()
    nbformat_out = client.nb
    nbf.write(nbformat_out, nb_path)
    nbf.validate(nbformat_out)

    n_drills = sum(1 for k, s in mod.CELLS if k == "md" and s.startswith("## Drill"))
    n_asserts = sum(s.count("assert ") for k, s in mod.CELLS if k == "code")
    final = mod.CELLS[-1][1]
    if n_drills != 15:
        raise SystemExit(f"FAIL: {n_drills} drills (attendu 15)")
    if "NB 10 — les 15 drills sont vérifiés. ✅" not in final:
        raise SystemExit("FAIL: cellule finale manquante")
    print(f"PASS — 15 drills, {n_asserts} asserts, {len(cells)} cells, "
          f"nbformat validate OK -> {nb_path}")


if __name__ == "__main__":
    main()
