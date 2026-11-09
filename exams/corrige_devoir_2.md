# CORRIGÉ — DEVOIR 2 · Deep Learning & RL appliqués à la cybersécurité

*Valeurs numériques vérifiées numériquement (numpy).*

## PARTIE A (7 points)

**A1 (2)**
- a) `z₁ = (1·2+0·1, −1·2+3·1) = (2, 1)` ; `a₁ = ReLU(z₁) = (2, 1)` (les deux sont > 0) ; `a₂ = 1·2 + 2·1 = **4**`.
- b) `δ₂ = a₂ − y = 4 − 2 = **2**` ; `∂L/∂w₂ = δ₂·a₁ = **(4, 2)**` ; `∂L/∂b₂ = δ₂ = **2**`.
- c) `ReLU′(z₁) = (1, 1)` ; `δ₁ = (w₂ δ₂) ⊙ (1,1) = (1·2, 2·2) = **(2, 4)**` ;
`∂L/∂W₁ = δ₁ xᵀ = [[2·2, 2·1], [4·2, 4·1]] = **[[4, 2], [8, 4]]**` ; `∂L/∂b₁ = **(2, 4)**`.
Contrôle de cohérence : `‖∂L/∂w₂‖ ≠ 0` et la chaîne `W₁` reçoit bien un gradient non nul (pas de vanishing ici, `z₁ > 0` partout).

**A2 (1.5)**
- a) `ReLU(−3)=0`, `ReLU(0)=0`, `ReLU(2)=2` ; dérivée : `0` en `−1`, `1` en `0.5`.
- b) `f(−2) = 0.1·(−2) = **−0.2**`.
- c) **Neurone mort** (dying ReLU) : la dérivée est 0 sur tout le batch ⇒ aucun gradient. Remèdes : initialisation de **He**, **Leaky-ReLU**, baisser le pas d'apprentissage.

**A3 (1.5)**
- a) Exponentielles : `(1, 2, 2)`, somme `5` ⇒ `p = (0.2, 0.4, 0.4)`.
- b) Si `y = phishing` : `L = −ln 0.2 ≈ **1.609**` ; si `y = malware` : `L = −ln 0.4 ≈ **0.916**`.
- c) L'argmax est **malware** (p=0.4, ex æquo avec bénin) : la vraie classe « phishing » ne reçoit que `p = 0.2` ⇒ prédiction erronée alors que la CE « sait » déjà différer les classes.

**A4 (1)**
- a) `0.25⁴ = 1/256 ≈ **0.0039**`.
- b) `0.25¹⁰ ≈ **9.5·10⁻⁷**` : le gradient du poids de la 1ʳᵉ couche est multiplié par ~10⁻⁶ ⇒ **vanishing gradients**.
- c) Ex. : ReLU/ReLU améliorées, initialisation Xavier/He adaptée à l'activation, normalisation (BatchNorm), connexions résiduelles.

**A5 (1)**
- a) Exemple 1 : `0.9 − 1 = **−0.1**` ; exemple 2 : `0.6 − 0 = **+0.6**`.
- b) Moyenne : `(−0.1 + 0.6)/2 = **0.25**`. Signe positif pour l'exemple 2 : le logit `z` doit **diminuer** (on descend `z ← z − η·0.6`) pour que `p` s'éloigne de 1 — cohérent avec un faux positif.

## PARTIE B (7 points)

**B1 (2)**
- a) Conv 1 : `8·(3·3·3 + 1) = 8·28 = **224**` ; sortie `16×16×8` (padding same).
- b) Conv 2 : `16·(3·3·8 + 1) = 16·73 = **1168**` (entrée 8 canaux) ; sortie `8×8×16` après pool.
- c) Après le 2ᵉ pool : `4×4×16` ⇒ flatten `= **256**` ; FC : `256·10 + 10 = **2570**`.
**Total = 224 + 1168 + 2570 = 3962 paramètres.** (on voit que le FC concentre ~65 % des paramètres d'où le risque d'overfitting et l'intérêt du global-average-pooling.)

**B2 (1.5)** (valeurs issues de la table)
- a) `h₁ = tanh(1) = 0.762` ; `h₂ = tanh(−1 + 0.5·0.762) = tanh(−0.619) = −0.549` ; `h₃ = tanh(1 + 0.5·(−0.549)) = tanh(0.7255) = 0.620` ; `ŷ₃ = 2·0.620 = **1.24**`.
- b) `δ₃ = ŷ₃ − y = 0.24` ; `∂L/∂w_hy = 0.24·0.620 = **0.1488**`.
- c) `1 − h₃² = 1 − 0.3844 = 0.6156` ; `∂L/∂h₂ = 0.24·2·0.6156 = **0.2955**` ; `∂L/∂w_hh = 0.2955·h₂ = 0.2955·(−0.549) ≈ **−0.162**`.

**B3 (1.5)**
- a) `z = (3 + 1)/√2 = 4/√2 = **2√2 ≈ 2.828**`.
- b) `x̂ = z·(1/√2, 1/√2) = (4/2, 4/2) = **(2, 2)**` ; erreur `x − x̂ = **(1, −1)**`.
- c) `SSE = 1 + 1 = **2**` ; `MSE = ½·2 = **1**`. L'AE linéaire projette sur la droite `(1,1)/√2` : c'est exactement la **1ʳᵉ composante principale (PCA)** des données centrées — la reconstruction est la projection orthogonale.

**B4 (1)** (alerte si `> 0.20`)
- a) Malveillants alertés : `0.25, 0.40` (0.18 < 0.20) ⇒ `TPR = 2/3 ≈ **0.667**`.
- b) Aucun bénin > 0.20 ⇒ `FPR = **0**`.
- c) Toutes les alertes sont vraies ⇒ `précision = 2/2 = **1**`. On sacrifie le rappel pour un SOC saturé de FPs.

**B5 (1)**
- a) `0.5⁵⁰ ≈ **8.9·10⁻¹⁶**` : gradient anéanti — le RNN n'apprend plus les dépendances longues.
- b) `0.95⁵⁰ ≈ **0.077**` : dégradé mais exploitable.
- c) La mise à jour `c_t = f_t⊙c_{t−1} + …` est **additive** : la dérivée `∂c_t/∂c_{t−1} = f_t` ne subit pas la saturation tanh répétée, le gradient circule si `f_t ≈ 1`.

## PARTIE C (6 points)

**C1 (2)**
- a) Cible `= 1 + 0.5·max(Q(S2,a), Q(S2,b)) = 1 + 0.5·2 = **2**` ; `Q(S1,a) ← 0 + 0.5·(2 − 0) = **1**`.
- b) Cible `= 0 + 0.5·max(Q(S1,a), Q(S1,b)) = 0 + 0.5·max(1, 1) = **0.5**` ; `Q(S2,b) ← 0 + 0.5·(0.5 − 0) = **0.25**`.
- c) Politique gloutonne : en `S1` : `a` ou `b` (égalité `1 = 1`, ex æquo) ; en `S2` : `a` (`2 > 0.25`).

**C2 (1.5)**
- a) `UCB(A) = 0.6 + √(4/4) = 0.6 + 1 = **1.6**` ; `UCB(B) = 0.9 + √(4/16) = 0.9 + 0.5 = **1.4**` ; `UCB(C) = 0.8 + √(4/2) = 0.8 + √2 ≈ **2.214**`.
- b) On tire **C** : c'est le bras peu exploré (`n=2`) — l'optimisme face à l'incertitude l'emporte sur la meilleure moyenne observée (B).
- c) Le terme `c√(ln t / nₐ)` décroît en `1/√nₐ` : l'exploration se **réduit automatiquement** quand un bras est beaucoup tiré (convergence vers l'exploitation).

**C3 (1)**
- a) Baseline : `(5+7+6)/3 = **6.0**` ; RL : `(8+9+4+6)/4 = **6.75**`.
- b) Supérieure sur l'échantillon, mais **4 épisodes ≠ preuve** : variance d'épisode élevée ; il faut comparer sur un grand nombre d'épisodes avec IC95 (ou test apparié), et un jeu d'évaluation **fixé et disjoint** de l'entraînement (sinon optimisation du bruit / overfitting de l'environnement).

**C4 (1)** Exemple cohérent attendu :
- a) États : `{normal, SSH suspect (tentatives en hausse), SSH compromis}`.
- b) Actions : `{surveiller, bannir l'IP / défier captcha, isoler le hôte + reset sessions}`.
- c) Récompense : `+10` si l'attaque est stoppée avant exfiltration ; `−5` pour un bannissement d'IP légitime (faux positif coûteux) ; éventuellement `−0.1`/pas pour laisser l'attaque courir.
- d) MDP car la probabilité de l'état suivant ne dépend **que de l'état courant et de l'action** (pas du passé complet) — à justifier : ici on agrège le taux de tentatives des dernières minutes dans l'état pour respecter cette propriété.

**C5 (0.5)** SARSA est **on-policy** (cible avec l'action réellement choisie par la politique ε-greedy) ; Q-Learning est **off-policy** (cible avec `max_a Q(s′,a)`, la politique gloutonne, quelle que soit l'action jouée).
