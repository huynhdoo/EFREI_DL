# CORRIGÉ — DEVOIR 4 · Deep Learning & RL appliqués à la cybersécurité

*Valeurs numériques vérifiées numériquement (numpy).*

## PARTIE A (7 points)

**A1 (1.5)**
- a) `ŷ = 1·1 + 1·2 = 3` ; perte `½(3−4)² = 0.5` ; régularisation `(0.1/2)(1+1) = 0.1` ; `J = **0.6**`.
- b) `∇J = (3−4)·(1,2) + 0.1·(1,1) = (−1,−2) + (0.1,0.1) = **(−0.9, −1.9)**`.
- c) Le terme L2 **pénalise les poids grands** (réduit la variance / le sur-apprentissage, lisse la frontière) ; `λ` trop grand ⇒ **sous-apprentissage** (poids écrasés vers 0, modèle trop simple, biais élevé).

**A2 (2)**
- a) `m₁ = 0.9·0 + 0.1·2 = **0.2**` ; `v₁ = 0.999·0 + 0.001·4 = **0.004**`.
- b) `m̂₁ = 0.2/(1−0.9) = **2**` ; `v̂₁ = 0.004/(1−0.999) = **4**` (les moments bruts sont biaisés vers 0 au démarrage).
- c) Mise à jour `η·m̂₁/√v̂₁ = 0.1·2/2 = **0.1**` ⇒ `w₁ = 1 − 0.1 = **0.9**`.
Sans correction de biais, le pas vaudrait `η·m₁/√v₁ = 0.1·0.2/0.0632 ≈ **0.316**`, soit **~3.2× trop grand** et dépendant arbitrairement de `β₂` : la correction garantit un pas moyen non biaisé dès `k = 1` (sinon les premiers pas sont systématiquement mal calibrés).

**A3 (2)**
- a) `μ_B = (1+3+5+7)/4 = **4**` ; `Var_B = (9+1+1+9)/4 = **5**` ; `σ_B = √5 ≈ **2.2361**`.
- b) `x̂ = (−3, −1, 1, 3)/2.2361 = **(−1.3416, −0.4472, 0.4472, 1.3416)**`.
- c) `y = 2x̂ + 1 = **(−1.683, 0.106, 1.894, 3.683)**` ; contrôle : `moy(y) = **1**` (= β), `écart-type(y) = **2**` (= γ).
Bénéfice : chaque couche reçoit des entrées recentrées/réscaled ⇒ gradients stables, pas d'apprentissage plus grands, moins sensible à l'init.

**A4 (0.5)** LayerNorm quand (1) les **mini-batches sont petits** ou les statistiques batch bruitées/instables (et au **test**, où BatchNorm doit utiliser les statistiques accumulées), (2) en **séquentiel/RNN** : la normalisation par exemple ne mélange pas les pas de temps ni les exemples (BatchNorm casse l'indépendance des séquences et la cohérence train/test).

**A5 (1)**
- a) `g₀ = 2(0−3) = **−6**` ; `v₁ = 0.9·0 + (−6) = **−6**` ; `w₁ = 0 − 0.1·(−6) = **0.6**`.
- b) `g₁ = 2(0.6−3) = **−4.8**` ; `v₂ = 0.9·(−6) + (−4.8) = −5.4 − 4.8 = **−10.2**` ; `w₂ = 0.6 − 0.1·(−10.2) = **1.62**`.
- c) `v` est une **EMA des gradients** : si le signe est constant, les contributions s'accumulent (jusqu'à `g/(1−μ) ≈ 10g` en régime permanent) ⇒ accélération dans les vallées et amortissement des oscillations transversales.

## PARTIE B (7 points)

**B1 (1.5)**
- a) conv1 : `r = 1 + 2·1 = **3**`, `j = 1` ; pool : `r = 3 + (2−1)·1 = **4**`, `j = 1·2 = 2` ; conv3 : `r = 4 + 2·2 = **8**`, `j = 2`.
- b) Champ récepteur total `= **8**` (un neurone de sortie agrège une fenêtre de 8×8 pixels d'entrée), stride cumulé `j = **2**`.
- c) Un motif d'attaque (en-tête d'exécutable, séquence d'octets) ne peut être vu que s'il tient **dans** le champ récepteur : des couches successives élargissent `r` — d'où l'empilement de convolutions plutôt qu'un filtre géant.

**B2 (2)**
- a) `∏(1−h_t²) = (1−0.36)(1−0)(1−0.64) = 0.64·1·0.36 = **0.2304**`.
- b) `w_hh = 0.5` : `0.5³·0.2304 = 0.125·0.2304 = **0.0288**` : la sensibilité à `h₀` est divisée par ~35 en 3 pas ⇒ régime **vanishing** (exponentiel en `T` si `|w_hh|·max(1−h²) < 1`).
- c) `w_hh = 1.2` : `1.2³·0.2304 = 1.728·0.2304 ≈ **0.398**` : le produit sur 3 pas a **augmenté** au lieu de décroître (facteur par pas `|w_hh|·√⟨1−h²⟩ ≳ 1` par endroits) ⇒ régime **exploding** quand `T` grandit — d'où le gradient clipping et la régularisation du Jacobien récurrent.

**B3 (1.5)**
- a) Paires (positif vs 3 négatifs `[0.2, 0.5, 0.8]`) :
`0.4` : bat `0.2` (1) ; `0.5` : bat `0.2` (1), égalité avec `0.5` (0.5) ; `0.9` : bat les 3 (3).
Somme `= 1 + 1.5 + 3 = **5.5**`.
- b) `AUC = 5.5/9 = **11/18 ≈ 0.611**`.
- c) `0.5` = **aléatoire** (aucune capacité de classement). `0.61` : signal très faible — inexploitable seul en SOC (beaucoup de FP pour un bon TPR) ; il faut ajouter des features, ré-entraîner, ou combiner (boosting/ensemble) et vérifier la dérive.

**B4 (1)**
- a) `(100 − 9)/1 + 1 = **92**` positions × 16 filtres ⇒ `92×16`.
- b) `16·(9·1 + 1) = 16·10 = **160**` paramètres (partage de poids : 10 paramètres utiles par filtre). Pool `2×2, s=2` ⇒ longueur `92/2 = **46**`.
- c) Chaque filtre 1D est un **détecteur de motif d'octets** local (séquence d'opcodes, magic bytes, chaîne encodée), réutilisé sur toute la longueur — comme un filtre-image détecte une bordure à toute position (invariance par translation).

**B5 (1)**
- a) Par porte : `128·64 + 128·128 + 128 = 8192 + 16384 + 128 = **24 704**` ; LSTM : `4·24 704 = **98 816**`.
- b) GRU : `3·24 704 = **74 112**`.
- c) Ratio `4/3 ≈ **1.33**` : GRU = 25 % de calcul/paramètres en moins ⇒ **GRU** sur capteur embarqué à latence stricte (à qualité de séquence comparable, vérifiée au préalable).

## PARTIE C (6 points)

**C1 (2)**
- a) `s(x) = 3 + 2·1 = **5 < 6**` ⇒ classé bénin ✓.
- b) `sign(∇ₓs) = (1, 1)` ; `x_adv = (3 + 0.6, 1 + 0.6) = **(3.6, 1.6)**` ; `s(x_adv) = 3.6 + 3.2 = **6.8 ≥ 6**` ⇒ l'échantillon **bascule** de « bénin » vers « malveillant » : la classification a été retournée par une perturbation de taille `ε` dans la direction du gradient. (Une FGSM d'**évasion** applique le même calcul avec `−ε·sign(∇)` pour faire passer un malveillant **sous** le seuil.)
- c) `s_adv(ε) = 5 + ε·(1+2) = 5 + 3ε ≥ 6 ⇔ **ε* = 1/3 ≈ 0.333**`. Normes de `(0.6, 0.6)` : `L∞ = **0.6**`, `L₂ = 0.6√2 ≈ **0.849**` — une perturbation minuscule en L∞ suffit : la robustesse se mesure en ε, pas seulement en accuracy propre.

**C2 (1)**
- a) Malveillants propres : `0.15·10 000 = **1500**` ; retournés : `0.08·1500 = **120**` ; restants : `**1380**`.
- b) `1380/10 000 = **13.8 %**`.
- c) La frontière se déplace vers la région des malveillants retournés (ils «tirent» la classe bénin) ⇒ rappel en baisse sur les zones touchées, faux négatifs silencieux. Parade : **validation d'étiquettes** (double-annotation, détection d'étiquettes incohérentes par nettoyage type cleanlab / influence functions), suivi de la dérive de distribution des labels.

**C3 (1.5)**
- a) Blocage : `E = 0.6·50 + 0.4·0 = **30 k€**` ; Ignorer : `E = 0.4·200 = **80 k€**` ; Escalade : `E = 5 + 0.4·20 = **13 k€**`.
- b) **Escalade analyste** (`13 < 30 < 80` k€). Si le coût de brèche passe à 1000 : `E(ignorer) = 0.4·1000 = **400**` (le pire s'aggrave), `E(blocage) = 30`, `E(escalade) = 13` : l'escalade reste la meilleure décision et « ignorer » devient indéfendable. Règle générale : quand le coût de la brèche augmente, le seuil de `p` déclenchant une action automatique **baisse** (on devient plus prudent côté risque, pas côté service).

**C4 (1)**
- a) Rappel propre : `180/200 = **90 %**` ; rappel avec trigger : `5/200 = **2.5 %**`.
- b) Jeu de test **adversarial dédié** incluant les triggers candidats (NEAT / scans de motifs), audit des échantillons mal classés « bénins » par un analyste, monitoring des scores près de la frontière ; à la chaîne : validation des données d'entraînement (provenance, dedup, sanitisation des en-têtes).

**C5 (0.5)** Ex. attendus : **human-in-the-loop** sur les actions irréversibles (blocage, isolation) avec droit de veto ; **journalisation complète et immuable** des décisions (features, score, version du modèle, action) pour l'audit ; **réévaluation périodique** (dérive, performance par segments, red-team) + procédure de rollback ; aussi acceptés : explicabilité des alertes, séparation des rôles, seuils de confiance minimum.
