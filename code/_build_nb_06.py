#!/usr/bin/env python3
"""Build notebooks/06_autoencodeurs_anomalies.ipynb from the mirror script.

Single source of truth: the mirror code/06_autoencodeurs_anomalies.py uses
percent-format cell markers (`# %%` = code, `# %% [markdown]` = markdown,
comment-prefixed lines). This builder parses those chunks verbatim into
nbformat v4 cells, injects the Colab `!pip` line into the imports cell and a
`## 📌 Bloc 0` heading, pre-gates every code cell with compile(), executes the
notebook with nbclient and writes it back with baked outputs.
"""
import os

import nbformat as nbf
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MIRROR = os.path.join(HERE, "06_autoencodeurs_anomalies.py")
OUT = os.path.join(ROOT, "notebooks", "06_autoencodeurs_anomalies.ipynb")

PIP_LINE = "!pip install -q numpy scipy pandas matplotlib scikit-learn"
BLOC0_MD = ("## 📌 Bloc 0 — Environnement & imports\n"
            "Installez le stack si besoin (déjà pré-installé sur Colab).\n"
            "Le notebook contient des sorties pré-calculées : relisibles sans rien exécuter,\n"
            "ou « Run All » pour reproduire (numpy pur, graines figées, aucune donnée réseau).")


def parse_chunks(path):
    """Split the percent-format mirror into {'kind','src'} chunks."""
    chunks, cur = [], None
    for ln in open(path, encoding="utf-8").read().splitlines():
        s = ln.strip()
        if s == "# %% [markdown]":
            kind = "md"
        elif s == "# %%" or s.startswith("# %% "):
            kind = "code"
        else:
            if cur is not None:
                cur["lines"].append(ln)
            continue
        if cur is not None:
            chunks.append(cur)
        cur = {"kind": kind, "lines": []}
    if cur is not None:
        chunks.append(cur)
    out = []
    for c in chunks:
        if c["kind"] == "md":
            src = "\n".join(l[2:] if l.startswith("# ") else ("" if l == "#" else l)
                            for l in c["lines"]).strip("\n")
        else:
            src = "\n".join(c["lines"]).strip("\n")
        if src.strip():
            out.append({"kind": c["kind"], "src": src})
    return out


chunks = parse_chunks(MIRROR)

# Inject the notebook-only imports preamble: Bloc 0 heading + !pip line.
for i, c in enumerate(chunks):
    if c["kind"] == "code":
        c["src"] = PIP_LINE + "\n" + c["src"]
        chunks.insert(i, {"kind": "md", "src": BLOC0_MD})
        break
else:
    raise SystemExit("no code chunk found in mirror")

NB = nbf.v4
nb = NB.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)",
                   "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
    "colab": {"name": "DLCyber - T06 Autoencodeurs Anomalies", "provenance": []},
    "accelerator": "GPU",
}

cells = []
for blk in chunks:
    if blk["kind"] == "md":
        cells.append(NB.new_markdown_cell(blk["src"]))
    else:
        src = "\n".join(l for l in blk["src"].splitlines()
                        if not l.lstrip().startswith(("!", "%")))
        compile(src, "<cell>", "exec")  # pre-gate
        cells.append(NB.new_code_cell(blk["src"]))
nb.cells = cells

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
