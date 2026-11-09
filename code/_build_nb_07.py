#!/usr/bin/env python3
"""Build notebook 07 (Fondements du Reinforcement Learning) for EFREI/DL.

The notebook code cells are taken VERBATIM from the `# %% TAG` chunks of the
mirror script code/07_rl_fondements_qlearning.py (single source of truth).
Markdown pedagogy lives in this builder; plot cells are notebook-only.
Run:  python code/_build_nb_07.py           (build + execute + bake outputs)
"""
import os
import re
import sys

import nbformat as nbf
from nbclient import NotebookClient

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(BASE, "code", "07_rl_fondements_qlearning.py")
OUT = os.path.join(BASE, "notebooks", "07_rl_fondements_qlearning.ipynb")


def load_chunks(path):
    src = open(path, encoding="utf-8").read()
    parts = re.split(r"^# %% (\S+)\s*$", src, flags=re.M)
    return {parts[i]: parts[i + 1].strip("\n") for i in range(1, len(parts), 2)}


def make_nb(cells, colab_title):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3 (ipykernel)",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"name": colab_title, "provenance": []},
        "accelerator": "GPU",
    }
    nb.cells = cells
    return nb


MD = {}  # tag -> markdown shown right BEFORE that code chunk

MD["__header__"] = """# T07 · Fondements du Reinforcement Learning — DL/RL appliqués à la cybersécurité

**Objectifs du cours :**
- Maîtriser la boucle **agent ↔ environnement** (état, action, récompense) sur un
  environnement « grille de défense réseau » codé à la main en numpy.
- Savoir écrire et **résoudre les équations de Bellman à la main** sur un petit MDP,
  puis entraîner un agent **Q-Learning tabulaire** et vérifier sa convergence.
- Distinguer exploration et exploitation (ε-greedy, UCB, regret) et situer SARSA / DQN.

**Pourquoi c'est fondamental en cyber :**

| Besoin cyber | Outil RL de ce notebook |
|---|---|
| Durcir les défenses sans jeu d'étiquettes « attaque / non attaque » | boucle récompense scalaire (P1, P2) |
| Choisir quelle règle de pare-feu régler pour un effort limité | bandits ε-greedy / UCB, regret (P4) |
| Politique de réponse à incident qui optimise le gain long terme | valeur V, Q, Bellman (P3, P5) |
| Automatiser la réponse sans sur-réagir au cas par cas | politique gloutonne issue de Q (P5, P7) |

**Prérequis :** numpy (tableaux, indexation), probabilités de base (espérance),
notion d'algorithme itératif. Aucun framework RL requis : tout est codé ici.

> **Conseil d'étude :** refaites chaque exercice Pn à la main (papier/crayon) AVANT
> de lire le corrigé — les valeurs sont toutes calculables sans ordinateur.
"""

MD["__bloc0__"] = """## 📌 Bloc 0 — Environnement & imports

numpy + scikit-learn (pour la P2 supervisée) + matplotlib. Les graines sont
**isolées par cellule** : l'ordre d'exécution complet est reproductible.
"""

BLOC0_CODE = """!pip install -q numpy scipy pandas matplotlib scikit-learn
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True, linewidth=100)
print("numpy", np.__version__)"""

MD["p1-env"] = """## 1 · Agent, environnement, état, action, récompense *(réf. Sutton & Barto ch. 3)*

Le RL formalise l'apprentissage par **essai-erreur séquentiel** : à chaque pas
l'agent observe un **état** $s_t$, choisit une **action** $a_t$, l'environnement
répond par une **récompense** $r_{t+1}$ et l'état suivant $s_{t+1}$.

Notre environnement : une **grille de défense réseau**. Les lignes sont des hôtes,
les colonnes des segments réseau. L'agent (défenseur) part de la passerelle (0,0) ;
le joyau couronné est en (3,3) ; deux hôtes sont compromis (cases rouges, −5) ;
un attaquant avance case par case. Récompenses : −1 par pas, −5 hôte compromis,
+10 si le défenseur occupe le joyau avant l'attaquant, −10 sinon.

**Piège classique :** confondre récompense immédiate et **retour** (somme
escomptée des récompenses). Un agent myope maximise $r_{t+1}$ ; un agent RL
maximise $G_t = r_{t+1} + \\gamma r_{t+2} + \\gamma^2 r_{t+3} + \\dots$
"""

MD["p1-exo"] = """### Exercice corrigé"""

MD["p2-sup"] = """## 2 · RL vs apprentissage supervisé

| | Supervisé | RL |
|---|---|---|
| Donnée | $(x, y)$ étiquettée | trajectoire $(s, a, r, s')$ |
| Signal | étiquette exacte | récompense **scalaire**, bruitée |
| Indépendance | exemples i.i.d. | états **corrélés** dans le temps |
| Objectif | minimiser l'erreur de prédiction | maximiser le **retour cumulé** |
| Couverture | distribution des données | l'agent **choisit** ses états (exploration) |

En cyber, le supervisé exige un SOC qui a **déjà étiquetté** les incidents ; le RL
s'attaque aux décisions pour lesquelles l'étiquette n'existe pas (« fallait-il
isoler ce serveur ? ») — seul le résultat final (incident contenu / non) « étiquette »
la séquence, des mois plus tard : c'est précisément le régime de la récompense différée.
"""

MD["p2-exo"] = """### Exercice corrigé"""

MD["p3-mdp"] = """## 3 · Politiques, valeur & équations de Bellman *(réf. Sutton & Barto ch. 4)*

Une **politique** $\\pi(a\\mid s)$ est un plan de décision. La **valeur** d'un état
sous $\\pi$ est le retour espéré depuis cet état :

$$V^{\\pi}(s) = \\mathbb{E}_\\pi\\!\\left[\\sum_{k\\ge 0} \\gamma^k r_{t+k+1} \\mid s_t = s\\right]
\\qquad Q^{\\pi}(s,a) = \\mathbb{E}_\\pi\\!\\left[\\textstyle\\sum_{k\\ge 0}\\gamma^k r_{t+k+1} \\mid s_t{=}s, a_t{=}a\\right]$$

**Équation de Bellman** (à connaître par cœur, calculable à la main) :

$$V^{\\pi}(s) = \\sum_a \\pi(a\\mid s) \\sum_{s'} P(s'\\mid s,a)\\,[R(s,a,s') + \\gamma V^{\\pi}(s')]$$

Pour un MDP à politique fixée, c'est un **système linéaire** $V = R + \\gamma P V$
→ résolution exacte. L'**optimalité** remplace l'espérance sur les actions par un
$\\max$ : $V^{*}(s) = \\max_a \\sum_{s'} P(s'\\mid s,a)[R + \\gamma V^{*}(s')]$.
"""

MD["p3-exo"] = """### Exercice corrigé"""

MD["p4-bandit"] = """## 4 · Exploration / exploitation (ε-greedy, UCB, regret)

Le dilemme du joueur de bandit : tirer la meilleure règle connue (**exploitation**)
ou essayer les autres (**exploration**). Le défi cyber : on dispose de 5 règles de
pare-feu et on ne peut en régler qu'une à la fois.

- **ε-greedy** : exploiter à $1-\\varepsilon$, tirer au hasard à $\\varepsilon$.
  Son regret cumulé croît **linéairement** en $T$.
- **UCB1** : optimisme face à l'incertitude —

$$a_t = \\arg\\max_a \\left[ \\hat{Q}(a) + \\sqrt{\\frac{c \\ln t}{N(a)}} \\right]$$

  Le regret ne croît que **logarithmiquement** (Lai-Robbins). Le **regret** cumulé
  $= \\sum_t (\\mu^{*} - \\mu_{a_t})$ est LA métrique des algorithmes d'exploration.
"""

MD["p4-plot"] = """La courbe de regret ci-dessous montre la signature théorique : ε-greedy monte en
ligne droite, UCB s'aplatit en logarithme."""

PLOT_P4 = """fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(regret_eps, color="#c12526", lw=2, label="epsilon-greedy (eps=0.1)")
ax.plot(regret_ucb, color="#2878b5", lw=2, label="UCB1 (c=2)")
ax.set_xlabel("pas de decision")
ax.set_ylabel("regret cumule")
ax.set_title("Defenseur : reglage de regles pare-feu — regret cumule")
ax.legend()
plt.show()"""

MD["p4-exo"] = """### Exercice corrigé"""

MD["p5-rl"] = """## 5 · Q-Learning tabulaire *(réf. Sutton & Barto ch. 6)*

Itération hors-politique de Watkins (1989) — la seule règle de mise à jour :

$$Q(s_t, a_t) \\leftarrow Q(s_t, a_t) + \\alpha \\left[ r_{t+1} + \\gamma \\max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \\right]$$

Conditions de convergence (Robbins-Monro) : toutes les paires $(s,a)$ visitées
infiniment, $\\sum \\alpha = \\infty$, $\\sum \\alpha^2 < \\infty$, et assez d'exploration
(ε-greedy). Référence de vérification : l'**itération de valeur** (Bellman optimaire
itéré) donne $V^{*}$ exact — la table apprise doit converger dessus.

**Piège classique :** oublier le $\\max$ (→ SARSA, on-policy) ou oublier d'escompter
l'état terminal ($\\max Q(\\text{terminal}) = 0$).
"""

MD["p5-plot"] = """Convergence : le retour par épisode (politique ε-greedy, bruitée) se stabilise
dès que Q est rempli ; la politique **gloutonne**, elle, vaut exactement $V^{*}$."""

PLOT_P5 = """fig, ax = plt.subplots(figsize=(7, 4.5))
k = 100
smooth = np.convolve(ep_returns, np.ones(k) / k, mode="valid")
ax.plot(smooth, color="#31a354", lw=2)
ax.axhline(V_vi[0], color="#c12526", ls="--", lw=1.5,
           label=f"V*(depart) it. valeur = {V_vi[0]:.3f}")
ax.set_xlabel("episode")
ax.set_ylabel("retour (eps-greedy, lisse)")
ax.set_title("Q-Learning tabulaire sur la grille de defense")
ax.legend()
plt.show()"""

MD["p5-exo"] = """### Exercice corrigé"""

MD["p6-sarsa"] = """## 6 · Au-delà : SARSA & DQN (vue conceptuelle)

| | Q-Learning | SARSA |
|---|---|---|
| Cible | $r + \\gamma \\max_{a'} Q(s',a')$ (**off-policy**) | $r + \\gamma Q(s',a'_{\\text{réel}})$ (**on-policy**) |
| Comportement | apprend l'optimum même en errant | apprend ce qu'il fait vraiment |
| Risque | sur-estime si Q bruité | trop prudent en exploration |

Exemple classique : dans un couloir avec une falaise, Q-Learning longe la falaise
(optimum) et y tombe à chaque exploration ; SARSA s'en écarte (prudent).

**DQN** (mnih et al., 2015) : on remplace la table $Q(s,a)$ par un **réseau**
$Q(s,a;\\theta)$ → généralisation aux grands espaces d'états, buffer de **replay**
(casser les corrélations) et **cible retardée** (stabiliser la cible du $\\max$).
Hors programme ici : pas de deep learning nécessaire sur 16 états — le tabulaire
de la P5 est déjà LA méthode exacte (et vérifiable).
"""

MD["p6-exo"] = """### Exercice corrigé"""

MD["p7-integration"] = """## 7 · Synthèse globale — l'essentiel & exemple intégrateur

### Ce qu'il faut retenir (les 6 points)

| # | Point | L'essentiel |
|---|---|---|
| 1 | Agent / environnement | boucle $(s, a, r, s')$ ; le **retour** compte, pas la récompense immédiate |
| 2 | RL vs supervisé | pas d'étiquettes : un signal scalaire différé ; exemples corrélés |
| 3 | Bellman | $V = R + \\gamma P V$ (politique fixée) ; $\\max$ pour l'optimalité |
| 4 | Exploration | ε-greedy : regret linéaire ; UCB : regret logarithmique |
| 5 | Q-Learning | maj $\\alpha[r + \\gamma \\max Q - Q]$ ; convergence vs itération de valeur |
| 6 | SARSA / DQN | on-policy vs off-policy ; DQN = Q réseau + replay + cible retardée |

**Exemple intégrateur :** on redéploie le Q-agent entraîné en P5 (attaquant figé)
contre un **attaquant mobile** : l'agent doit atteindre le joyau (6 pas) avant
l'attaquant (6 pas aussi, vitesse 1/2) — la politique optimale est la *seule* qui
gagne. Le baseline aléatoire se fait exfiltrer quasi sûrement.
"""

MD["p7-exo"] = """### Exercice corrigé"""

ORDER = ["p1-env", "p1-exo", "p2-sup", "p2-exo", "p3-mdp", "p3-exo",
         "p4-bandit", "p4-exo", "p5-rl", "p5-exo", "p6-sarsa", "p6-exo",
         "p7-integration", "p7-exo", "marker"]

EXTRA_PLOTS = {"p4-bandit": (MD["p4-plot"], PLOT_P4),
               "p5-rl": (MD["p5-plot"], PLOT_P5)}


def main():
    chunks = load_chunks(SCRIPT)
    missing = [t for t in ORDER if t not in chunks]
    assert not missing, f"chunks manquants dans le script miroir : {missing}"

    cells = [nbf.v4.new_markdown_cell(MD["__header__"]),
             nbf.v4.new_markdown_cell(MD["__bloc0__"]),
             nbf.v4.new_code_cell(BLOC0_CODE)]
    for tag in ORDER:
        if tag in MD:
            cells.append(nbf.v4.new_markdown_cell(MD[tag]))
        cells.append(nbf.v4.new_code_cell(chunks[tag]))
        if tag in EXTRA_PLOTS:
            plot_md, plot_code = EXTRA_PLOTS[tag]
            cells.append(nbf.v4.new_markdown_cell(plot_md))
            cells.append(nbf.v4.new_code_cell(plot_code))

    # pre-gate: every code cell must compile (conventions: no silent collapse);
    # IPython magics (lines starting with '!') are stripped before compiling.
    for i, c in enumerate(cells):
        if c.cell_type == "code":
            py = "\n".join(l for l in c.source.splitlines() if not l.startswith("!"))
            compile(py, f"<cell {i}>", "exec")

    nb = make_nb(cells, "DLCyber - T07 Fondements du Reinforcement Learning")
    NotebookClient(nb, timeout=600, kernel_name="python3").execute()
    nbf.validate(nb)
    nbf.write(nb, OUT)

    n_ex = sum(c.source.count("# ---- Exercice P") for c in cells if c.cell_type == "code")
    heads = [c.source.splitlines()[0] for c in cells
             if c.cell_type == "markdown" and c.source.startswith("## ")]
    print("PASS", OUT)
    print("cells:", len(cells), "| exercices P:", n_ex)
    for h in heads:
        print("  ", h)


if __name__ == "__main__":
    sys.exit(main())
