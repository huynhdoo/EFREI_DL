# DEVOIR 1 — Deep Learning & RL appliqués à la cybersécurité

**Durée : 2 h — Calculatrice non autorisée — Sans document**
**Barème : 20 points**

*Tables utiles (fournies) : σ(z) = 1/(1+e⁻ᶻ) ; σ(0)=0.5, σ(1)≈0.731, σ(−1)≈0.269 ; tanh(2)≈0.964, tanh(1.243)≈0.846 ; ln 0.9 ≈ −0.1054, ln 0.1 ≈ −2.3026 ; taille de sortie conv : (n−F+2p)/s+1 ; Q(s,a) ← Q(s,a) + α[r + γ·max Q(s′,·) − Q(s,a)].*

---

## PARTIE A — Fondements du deep learning (7 points)

**A1 (1.5 pt)** Un perceptron à seuil calcule `o = 1` si `w₁x₁ + w₂x₂ + b > 0`, sinon `0`, avec `w = (1, 1)` et `b = −1.5`.
a) Calculer la sortie pour les quatre entrées de la table XOR : `(0,0), (1,0), (0,1), (1,1)` (donner le pré-activation et la sortie).
b) Ce perceptron réalise-t-il le XOR ? Quelle fonction booléenne calcule-t-il à la place ?
c) Expliquer en une phrase pourquoi **aucun** perceptron unique ne peut réaliser le XOR.

**A2 (1.5 pt)** Soit `σ(z) = 1/(1+e⁻ᶻ)` (rappel : `σ′(z) = σ(z)(1−σ(z))`).
a) Calculer `σ(0)` puis `σ′(0)`.
b) Poser `z = ln 3` (donc `e⁻ᶻ = 1/3`). Calculer `σ(z)` puis `σ′(z)` **en fractions exactes**.
c) Soit `y(x) = σ(3x)`. Calculer `dy/dx` en `x = 0`.

**A3 (1.5 pt)** On ajuste un neurone linéaire `ŷ = w·x` sur les deux points `(x=1, y=3)` et `(x=2, y=5)` avec la perte `L(w) = ½ Σᵢ (ŷᵢ − yᵢ)²`.
a) Calculer `L(1)`.
b) Écrire puis calculer `dL/dw` en `w = 1` (indice : `dL/dw = Σᵢ (ŷᵢ − yᵢ)·xᵢ`).
c) Dans quel sens faut-il modifier `w` ? Vérifier que le minimiseur exact est `w* = 2.6`.

**A4 (1.5 pt)** Suite de l'entraînement : un seul échantillon `x = (1, 2)`, `y = 4`, `ŷ = w·x`, `L = ½(ŷ − y)²`, départ `w₀ = (0, 0)`, pas `η = 0.1`. Rappel : `∇L = (ŷ − y)·x`, `w ← w − η∇L`.
a) Étape 1 : calculer `ŷ₀`, le gradient, puis `w₁`.
b) Étape 2 : calculer `ŷ₁`, le gradient, puis `w₂`.
c) Calculer `L(w₂)`.

**A5 (1 pt)** Un classifieur de trafic malveillant utilise la perte BCE : `L = −[y·ln p + (1−y)·ln(1−p)]`.
a) Échantillon réellement malveillant (`y=1`) avec `p = 0.9` : calculer `L`.
b) Échantillon réellement bénin (`y=0`) avec `p = 0.9` : calculer `L`.
c) Pourquoi la pénalité (b) est-elle beaucoup plus grande ? Donner l'interprétation pour un détecteur.

---

## PARTIE B — Architectures (7 points)

**B1 (1.5 pt)** Utiliser la formule `(n − F + 2p)/s + 1` pour une couche de convolution 2D :
a) Entrée `13×13`, filtre `F=5`, padding `p=1`, stride `s=2`, 8 filtres : donner la taille de sortie (H×W×C).
b) Entrée `28×28`, `F=3`, `p=0`, `s=1` : taille de sortie ?
c) Quelle valeur de `p` (avec `F=5`, `s=1`) garde la taille d'entrée `7×7` ? Vérifier par le calcul.

**B2 (1.5 pt)** Sur la matrice
```
M = [ 1  3  2  0 ]
    [ 4  2  1  5 ]
    [ 0  6  3  2 ]
    [ 7  1  4  2 ]
```
a) Appliquer un max-pooling `2×2`, stride 2 : donner la matrice 2×2 de sortie.
b) Appliquer un avg-pooling `2×2`, stride 2 sur la même matrice.
c) Citer un risque de l'avg-pooling par rapport au max-pooling pour un détecteur de motifs rares (pics d'activation).

**B3 (2 pts)** Une cellule LSTM (1 neurone caché) observe une fenêtre de logs SSH. À l'instant `t` : `x_t = 1`, `h_{t−1} = 0.5`, `c_{t−1} = 2`. Les pré-activations viennent de (tableau : poids sur `x`, poids sur `h`, biais) :
- `f` : `U_f=1, W_f=0, b_f=−2`
- `i` : `U_i=0, W_i=1, b_i=0.5`
- `c̃` : `U_c=2, W_c=0, b_c=0`
- `o` : `U_o=1, W_o=−1, b_o=−0.5`

Équations : `f_t=σ(U_f x_t + W_f h_{t−1} + b_f)`, idem `i_t`, `c̃_t = tanh(·)`, `c_t = f_t c_{t−1} + i_t c̃_t`, `o_t = σ(·)`, `h_t = o_t·tanh(c_t)`.
a) Calculer les quatre pré-activations.
b) En déduire `f_t`, `i_t`, `c̃_t`, `o_t` (table fournie en en-tête).
c) Calculer `c_t` puis `h_t` (arrondir à 3 décimales).

**B4 (1 pt)** Pipeline sur des images de bytes de malware `16×16` : `conv 3×3 (p=1, s=1)` → `max-pool 2×2 (s=2)` → `conv 3×3 (p=0, s=1)`.
a) Donner la taille après chaque opération.
b) À quoi sert le padding `p=1` de la première couche ? À quoi sert le stride `s=2` du pooling ?

**B5 (1 pt)** Pour alimenter un RNN avec des séquences de logs, on fenêtre une session de `n = 7` événements avec une fenêtre de `w = 3` événements.
a) Combien de fenêtres avec stride 1 ? Lister les indices de départ.
b) Combien de fenêtres avec stride 2 ?
c) Pourquoi fenêtrer au lieu de donner toute la session ?

---

## PARTIE C — RL & robustesse (6 points)

**C1 (1.5 pt)** Agent SOC, chaîne de 3 états avec récompenses sur les transitions :
`A (scan détecté) —r=1→ B (confinement) —r=2→ C (forensique) —r=5→ T (incident clos, terminal)`. Rappel : `V(s) = r + γ·V(s′)`, `V(T) = 0`, `γ = 0.9`, politique déterministe fixée.
a) Calculer `V(C)`.
b) Calculer `V(B)`.
c) Calculer `V(A)`.

**C2 (1.5 pt)** Mise à jour Q-Learning : `Q(s,a) = 1`, `α = 0.5`, `γ = 0.9`, l'expérience est `(s, a, r=3, s′)` avec `max Q(s′, ·) = 4`.
a) Calculer la cible `r + γ·max Q(s′,·)`.
b) Calculer l'erreur TD puis la nouvelle valeur `Q(s,a)`.
c) Si `α = 1`, que vaut `Q(s,a)` ? Que signifie `α` ?

**C3 (1 pt)** Politique ε-greedy avec `ε = 0.2` sur 3 actions de valeurs `Q = (5, 3, 1)` (la meilleure est l'action 1).
a) Donner la probabilité de chaque action.
b) Sur 300 pas, combien de pas d'exploration en moyenne ? combien pour chacune des deux actions non gloutonnes ?

**C4 (1 pt)** Retour actualisé, `γ = 0.5`.
a) Trajectoire `r₀=2, r₁=0, r₂=10` : calculer `G₀ = r₀ + γr₁ + γ²r₂`.
b) Trajectoire infinie de récompense constante `r = 1`, `γ = 0.8` : donner `G = r/(1−γ)` et l'interpréter.

**C5 (1 pt)** Un détecteur a `TPR = 0.90` et `FPR = 0.05` sur données propres. Un attaquant perturbe ses entrées et fait échouer la détection sur **40 %** des malveillants auparavant détectés.
a) Calculer le TPR en présence de l'attaque.
b) Le FPR change-t-il ici ? Que faut-il mesurer en plus pour juger la robustesse réelle du détecteur ?

---
*Fin du devoir — vérifiez vos réponses avant de rendre.*
