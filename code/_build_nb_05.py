#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builder for notebooks/05_rnn_lstm_gru_logs.ipynb (EFREI · DL · T05).

Single source of truth = code/05_rnn_lstm_gru_logs.py (jupytext-style
`# %%` cell markers). This builder parses that file into nbformat cells,
prepends the Colab `!pip install` line to the first code cell, executes
the notebook with nbclient (timeout=600) and writes it back with baked
outputs. No computation diverges from the mirror script.

Run:  /home/huynhdoo/venvs/notebook-adif72/bin/python code/_build_nb_05.py
"""
import os
import re
import sys

import nbformat as nbf
from nbclient import NotebookClient

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "code", "05_rnn_lstm_gru_logs.py")
OUT = os.path.join(BASE, "notebooks", "05_rnn_lstm_gru_logs.ipynb")

MARKER = re.compile(r"^# %%\s*(\[markdown\])?\s*(.*)$")
PIP_LINE = "!pip install -q numpy scipy pandas matplotlib scikit-learn"


def parse_cells(path):
    """Split the mirror script into (kind, source) cells on `# %%` markers."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    cells, kind, buf = [], None, []

    def flush():
        nonlocal buf
        if kind is not None:
            text = "\n".join(buf).strip("\n")
            if text.strip():
                cells.append((kind, text))
        buf = []

    for ln in lines:
        m = MARKER.match(ln)
        if m:
            flush()
            kind = "markdown" if m.group(1) else "code"
            continue
        if kind == "markdown":
            if ln.startswith("# "):
                buf.append(ln[2:])
            elif ln == "#":
                buf.append("")
            else:
                buf.append(ln)
        elif kind == "code":
            buf.append(ln)
    flush()
    return cells


def main():
    cells = parse_cells(SRC)
    # Pre-gate: every code cell must compile standalone.
    for i, (kind, src) in enumerate(cells):
        if kind == "code":
            try:
                compile(src, f"<cell {i}>", "exec")
            except SyntaxError as exc:
                sys.exit(f"cell {i} does not compile: {exc}")
    # Prepend the pip preamble to the FIRST code cell only (notebook-specific).
    for i, (kind, src) in enumerate(cells):
        if kind == "code":
            cells[i] = (kind, PIP_LINE + "\n" + src)
            break

    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3 (ipykernel)",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"name": "DLCyber - T05 Reseaux Recurrents RNN LSTM GRU", "provenance": []},
        "accelerator": "GPU",
    }
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "markdown"
                else nbf.v4.new_code_cell(s) for k, s in cells]

    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    nbf.validate(nb)
    nbf.write(nb, OUT)
    n_ex = sum(1 for k, s in cells if k == "code" and "# ---- Exercice P" in s)
    heads = [s.splitlines()[0] for k, s in cells if k == "markdown"
             and re.match(r"^## \d+ ·|^## 📌", s.splitlines()[0])]
    print(f"wrote {OUT}")
    print("cells:", len(cells), "— code:", sum(1 for k, _ in cells if k == "code"),
          "— markdown:", sum(1 for k, _ in cells if k == "markdown"))
    print("Exercice P cells:", n_ex)
    print("section headers:", heads)
    print("EXECUTION: PASS")


if __name__ == "__main__":
    main()
