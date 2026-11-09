# CORRIGÉ — DEVOIR 1 · Deep Learning & RL appliqués à la cybersécurité

*Valeurs numériques vérifiées numériquement (numpy).*

## PARTIE A (7 points)

**A1 (1.5)**
- a) `o(0,0)` : pré-act `−1.5` → `0` ; `o(1,0)` : pré-act `1−1.5 = −0.5` → `0` ; `o(0,1)` : pré-act `−0.5` → `0` ; `o(1,1)` : pré-act `2−1.5 = 0.5` → `1`.
- b) Non : le XOR exige `o(1,0)=o(0,1)=1`, ici `0`. La table obtenue `(0,0,0,1)` est la fonction **ET**.
- c) Un perceptron ne sépare que des classes **linéairement séparables** ; or `{(1,0),(0,1)}` et `{(0,0),(1,1)}` ne sont pas séparables par un hyperplan (aucun droite ne sépare les deux paires).

**A2 (1.5)**
- a) `σ(0) = 0.5` ; `σ′(0) = 0.5·(1−0.5) = 0.25`.
- b) `σ(ln 3) = 1/(1 + 1/3) = **3/4 = 0.75**` ; `σ′(ln 3) = (3/4)(1/4) = **3/16 = 0.1875**`.
- c) Chaîne : `dy/dx = σ(3x)(1−σ(3x))·3` ; en `x=0` : `0.25·3 = **0.75**`.

**A3 (1.5)**
- a) `ŷ = (1·1, 1·2) = (1, 2)` ; erreurs `1−3 = −2`, `2−5 = −3` ⇒ `L(1) = ½(4+9) = **6.5**`.
- b) `dL/dw = (−2)·1 + (−3)·2 = **−8**`.
- c) Gradient négatif ⇒ on **augmente** `w`. optimum : `w* = Σxᵢyᵢ/Σxᵢ² = (3+10)/(1+4) = 13/5 = **2.6**` ✓.

**A4 (1.5)**
- a) Étape 1 : `ŷ₀ = 0`, `err = −4`, `∇L = −4·(1,2) = (−4,−8)` ; `w₁ = (0,0) − 0.1·(−4,−8) = **(0.4, 0.8)**`.
- b) Étape 2 : `ŷ₁ = 0.4·1 + 0.8·2 = 2.0`, `err = −2`, `∇L = (−2,−4)` ; `w₂ = (0.4,0.8) − 0.1·(−2,−4) = **(0.6, 1.2)**`.
- c) `ŷ₂ = 0.6 + 2.4 = 3.0` ; `L(w₂) = ½(3−4)² = **0.5**`.

**A5 (1)**
- a) `L = −ln 0.9 ≈ **0.105**` (petite pénalité : bonne prédiction confiante).
- b) `L = −ln(1−0.9) = −ln 0.1 ≈ **2.303**`.
- c) La BCE punit logairement la **confiance erronée** : le détecteur est très pénalisé quand il déclare « sûr à 90 % » un trafic bénin — d'où l'importance de calibrer les scores et d'auditer les faux positifs confiants.

## PARTIE B (7 points)

**B1 (1.5)**
- a) `(13−5+2)/2 + 1 = 10/2 + 1 = **6**` ⇒ sortie `6×6×8`.
- b) `(28−3+0)/1 + 1 = **26**` ⇒ `26×26`.
- c) `p = 2` : `(7−5+4)/1 + 1 = 6+1 = 7` ✓ (padding « same » : `p = (F−1)/2` quand `s=1`).

**B2 (1.5)**
- a) Max-pooling 2×2, stride 2 :
`[[max(1,3,4,2), max(2,0,1,5)], [max(0,6,7,1), max(3,2,4,2)]] = **[[4, 5], [7, 4]]**`.
- b) Avg-pooling : `[(1+3+4+2)/4, (2+0+1+5)/4 ; (0+6+7+1)/4, (3+2+4+2)/4] = **[[2.5, 2.0], [3.5, 2.75]]**`.
- c) L'avg-pooling **dilue un pic isolé** (un activateur rare noyé dans la moyenne) : le max conserve le pic — préférer le max pour des motifs rares (signatures d'attaque ponctuelles).

**B3 (2)**
- a) Pré-activations : `f : 1·1 + 0·0.5 − 2 = −1` ; `i : 0 + 1·0.5 + 0.5 = 1` ; `c̃ : 2·1 + 0 + 0 = 2` ; `o : 1 − 0.5 − 0.5 = 0`.
- b) `f_t = σ(−1) ≈ 0.269` ; `i_t = σ(1) ≈ 0.731` ; `c̃_t = tanh(2) ≈ 0.964` ; `o_t = σ(0) = 0.5`.
- c) `c_t = 0.269·2 + 0.731·0.964 = 0.538 + 0.7047 ≈ **1.243**` ;
`h_t = 0.5·tanh(1.243) ≈ 0.5·0.846 = **0.423**`.

**B4 (1)**
- a) `16×16` → conv(p=1) : `(16−3+2)/1+1 = 16` → pool(s=2) : `8×8` → conv(p=0) : `(8−3)/1+1 = **6**` ⇒ `6×6`.
- b) Le padding `p=1` compense la perte de taille du filtre 3×3 et préserve les bords (les premiers/derniers bytes d'un fichier comptent souvent) ; le stride 2 du pooling **divise la résolution par 2** (invariance locale, coûts réduits).

**B5 (1)**
- a) `(7−3)/1 + 1 = **5**` fenêtres, départs `0,1,2,3,4`.
- b) `(7−3)/2 + 1 = **3**` fenêtres (départs `0,2,4`).
- c) Fenêtrer rend la séquence **fixe et Stationnaire** (taille d'entrée constante du RNN, augmentation implicite du nombre d'exemples), au prix d'une perte de contexte global.

## PARTIE C (6 points)

**C1 (1.5)** `V(T) = 0`.
- a) `V(C) = 5 + 0.9·0 = **5**`.
- b) `V(B) = 2 + 0.9·5 = 2 + 4.5 = **6.5**`.
- c) `V(A) = 1 + 0.9·6.5 = 1 + 5.85 = **6.85**`.

**C2 (1.5)**
- a) Cible `= 3 + 0.9·4 = **6.6**`.
- b) Erreur TD `= 6.6 − 1 = 5.6` ; `Q ← 1 + 0.5·5.6 = **3.8**`.
- c) Avec `α=1`, `Q = 6.6` : `α` est le **taux d'apprentissage** de la Q-table (1 = oublie l'historique, 0 = n'apprend pas).

**C3 (1)**
- a) `P(a₁) = (1−ε) + ε/3 = 0.8 + 0.0667 = **0.8667**` ; `P(a₂) = P(a₃) = ε/3 ≈ **0.0667**`.
- b) Exploration : `ε·300 = **60 pas**`, soit `60/2 = **30 pas**` par action non gloutonne.

**C4 (1)**
- a) `G₀ = 2 + 0.5·0 + 0.25·10 = **4.5**`.
- b) `G = 1/(1−0.8) = **5**` : une récompense constante vaut géométriquement `r/(1−γ)` — justifie de privilégier les récompenses **durables** (ne pas « éteindre » un incident pour encaisser r puis le subir à nouveau).

**C5 (1)**
- a) `TPR_adv = 0.90 × (1 − 0.40) = **0.54**`.
- b) Ici les perturbations visent les malveillants ⇒ le FPR est supposé inchangé (0.05). Il faut mesurer les **deux taux sous attaque** (TPR_adv, FPR_adv), idéalement une courbe de robustesse en fonction de ε, et tester l'attaque adaptative ; un TPR de 0.54 est inutilisable en production SOC.
