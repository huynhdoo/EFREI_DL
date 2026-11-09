# DEVOIR 2 — Deep Learning & RL appliqués à la cybersécurité

**Durée : 2 h — Calculatrice non autorisée — Sans document**
**Barème : 20 points**

*Tables utiles (fournies) : e⁰·⁶⁹³ ≈ 2 ; ln 0.4 ≈ −0.9163, ln 0.2 ≈ −1.6094 ; tanh(1) ≈ 0.762, tanh(−0.619) ≈ −0.549, tanh(0.7255) ≈ 0.620 ; √2 ≈ 1.414 ; 1/√2 ≈ 0.7071 ; Q-Learning : Q(s,a) ← Q(s,a) + α[r + γ·max Q(s′,·) − Q(s,a)] ; UCB(a) = x̄ₐ + c·√(ln t / nₐ).*

---

## PARTIE A — Fondements du deep learning (7 points)

**A1 (2 pts)** Rétropropagation complète d'un MLP `2→2→1` (cachée ReLU, sortie linéaire, `b₁ = (0,0)`, `b₂ = 0`) :
`W₁ = [[1, 0], [−1, 3]]` (une ligne par neurone caché), `w₂ = (1, 2)`, entrée `x = (2, 1)`, cible `y = 2`, perte `L = ½(a₂ − y)²`.
a) Propagation avant : `z₁ = W₁x`, `a₁ = ReLU(z₁)`, `a₂ = w₂·a₁`. Donner `z₁, a₁, a₂`.
b) Calculer `δ₂ = ∂L/∂a₂` puis les gradients `∂L/∂w₂` et `∂L/∂b₂`.
c) Calculer `δ₁ = (w₂ δ₂) ⊙ ReLU′(z₁)` (ReLU′ = 1 si z>0 else 0) puis `∂L/∂W₁ = δ₁ xᵀ` et `∂L/∂b₁`.

**A2 (1.5 pt)** Unités ReLU.
a) Calculer `ReLU(−3)`, `ReLU(0)`, `ReLU(2)` et la dérivée en `−1` et `0.5`.
b) Leaky-ReLU (pente 0.1) : calculer `f(−2)`.
c) Un neurone dont la pré-activation vaut `−1.5` pour **tous** les exemples reçoit un gradient nul : comment s'appelle ce défaut et citer deux remèdes.

**A3 (1.5 pt)** Classifieur e-mail : logits `z = (0, 0.693, 0.693)` sur (phishing, malware, bénin). Softmax : `pᵢ = e^{zᵢ}/Σe^{zⱼ}` ; CE : `L = −ln p_y`.
a) Calculer les trois probabilités (indice : `e⁰·⁶⁹³ ≈ 2`).
b) Calculer la perte CE si la vraie classe est **phishing**, puis si elle est **malware**.
c) Quelle probabilité la prédiction argmax attribue-t-elle à la vraie classe « phishing » ?

**A4 (1 pt)** Réseau de 10 couches de sigmoïdes, chaque poids initialisé pour donner localement `σ′ ≤ 0.25` (maximum de `σ(1−σ)`).
a) Majorer le produit des dérivées sur 4 couches (`0.25⁴`).
b) Majorer le produit sur 10 couches (`0.25¹⁰`).
c) Citer deux parades vues en cours.

**A5 (1 pt)** Sortie sigmoïde `p = σ(z)` avec BCE. Rappel : `∂L/∂z = p − y`. Deux exemples : `(p=0.9, y=1)` et `(p=0.6, y=0)`.
a) Calculer `∂L/∂z` pour chaque exemple.
b) Donner la moyenne des deux gradients. Interpréter le signe pour l'exemple 2.

---

## PARTIE B — Architectures (7 points)

**B1 (2 pts)** CNN de classification d'images de bytes `16×16×3`. Architecture :
`conv 3×3, 8 filtres (p=1, s=1)` → `max-pool 2×2 (s=2)` → `conv 3×3, 16 filtres (p=1, s=1)` → `max-pool 2×2 (s=2)` → `flatten` → `FC 256→10`.
Rappel : paramètres d'une couche conv = `n_filtres·(F·F·C_in + 1)` ; FC = `n_in·n_out + n_out`.
a) Nombre de paramètres de la conv 1 (donner aussi sa taille de sortie).
b) Nombre de paramètres de la conv 2.
c) Taille du flatten, paramètres de la FC, puis **total** du réseau.

**B2 (1.5 pt)** RNN scalaire sur 3 pas : `h_t = tanh(w_xh·x_t + w_hh·h_{t−1})` avec `w_xh = 1`, `w_hh = 0.5`, `h₀ = 0` ; sortie `ŷ₃ = w_hy·h₃`, `w_hy = 2` ; entrée `x = (1, −1, 1)`, cible `y = 1`, perte `L = ½(ŷ₃ − y)²`.
a) Propagation avant sur les 3 pas (utiliser la table fournie en en-tête) : `h₁, h₂, h₃`, puis `ŷ₃`.
b) Calculer `δ₃ = ∂L/∂ŷ₃` et `∂L/∂w_hy = δ₃·h₃`.
c) Remonter d'un pas : `∂L/∂h₂ = δ₃·w_hy·(1 − h₃²)` puis `∂L/∂w_hh = (∂L/∂h₂)·h₂`.

**B3 (1.5 pt)** Auto-encodeur linéaire 2D→1D→2D, sans biais : `W_enc = (1/√2, 1/√2)ᵀ`, `W_dec = W_encᵀ`. Entrée `x = (3, 1)`.
a) Calculer le code `z = W_enc·x` (valeur exacte avec √2, puis décimale).
b) Calculer la reconstruction `x̂ = W_dec·z` et le vecteur d'erreur `x − x̂`.
c) En déduire SSE et MSE `= ½‖x − x̂‖²`. Que calcule cet AE sur la direction `(1,1)/√2` (lien PCA) ?

**B4 (1 pt)** Scores d'erreur de reconstruction (AE) sur le jeu de validation :
bénins `[0.08, 0.10, 0.12, 0.15, 0.20]`, malveillants `[0.25, 0.18, 0.40]`. Seuil choisi : `t = 0.20` (on alerte si `erreur > t`).
a) TPR (fraction des malveillants alertés).
b) FPR.
c) Précision (part des alertes qui sont des malveillants).

**B5 (1 pt)** Un RNN vanilla a `w_hh = 0.5` fixe ; une cellule LSTM a un forget gate moyen `f ≈ 0.95`.
a) Ordre de grandeur de la chaîne de gradient sur 50 pas pour le RNN (`0.5⁵⁰`).
b) Idem pour le LSTM (`0.95⁵⁰`).
c) Expliquer en une phrase pourquoi le chemin du cell state limite le vanishing.

---

## PARTIE C — RL & robustesse (6 points)

**C1 (2 pts)** Q-Learning sur 2 états + terminal. `α = 0.5`, `γ = 0.5`. Init : `Q(S1,a)=0, Q(S1,b)=1, Q(S2,a)=2, Q(S2,b)=0`, `Q(T,·)=0`.
Transitions : `(S1,a) → r=1, S′=S2` ; `(S1,b) → r=0.5, S′=T` ; `(S2,a) → r=3, S′=T` ; `(S2,b) → r=0, S′=S1`.
Deux expériences arrivent dans l'ordre :
1. `(S1, a, r=1, S2)` : calculer la cible puis la nouvelle `Q(S1,a)`.
2. `(S2, b, r=0, S1)` (avec la Q-table **mise à jour**) : calculer la cible puis la nouvelle `Q(S2,b)`.
c) Donner la politique gloutonne finale (par état, l'action de plus grande Q).

**C2 (1.5 pt)** Trois politiques de règles de pare-feu testées comme bandits (récompense détection). À l'instant `t` avec `ln t = 4`, `c = 1` :
A : `n=4, x̄=0.6` — B : `n=16, x̄=0.9` — C : `n=2, x̄=0.8`.
a) Calculer les trois UCB.
b) Quelle règle est tirée ? Lequel des deux bras est « optimiste » ?
c) Que devient l'exploration quand `nₐ` grandit ?

**C3 (1 pt)** Évaluation d'une politique RL de réponse à incidents contre la baseline à règles :
retours par épisode — baseline `[5, 7, 6]`, RL `[8, 9, 4, 6]`.
a) Calculer la moyenne de chaque politique.
b) La RL est-elle meilleure ? Citer une précaution statistique indispensable avant de conclure.

**C4 (1 pt)** Formaliser en MDP la réponse à un **brute-force SSH** : donner pour chaque colonne un élément concret :
a) 3 états ; b) 2 actions par état (au moins) ; c) une récompense positive et une récompense négative plausibles ; d) pourquoi la modélisation est-elle **MDP** (propriété de Markov).

**C5 (0.5 pt)** Q-Learning vs SARSA : lequel est on-policy, lequel off-policy ? Une phrase sur la différence dans la cible.

---
*Fin du devoir — vérifiez vos réponses avant de rendre.*
