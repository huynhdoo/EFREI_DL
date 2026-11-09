# CORRIGÉ — DEVOIR 3 · Deep Learning & RL appliqués à la cybersécurité

*Valeurs numériques vérifiées numériquement (numpy).*

## PARTIE A (7 points)

**A1 (1.5)**
- a) Xavier : `σ_W = √(2/200) = √0.01 = **0.1**`.
- b) He : `σ_W = √(2/100) = √0.02 ≈ **0.141**`. Pour ReLU, **He** : la ReLU annule la moitié des activations en moyenne, il faut doubler la variance (`2/fan_in`) pour conserver `Var(z) ≈ Var(x)` (Xavier sous-échelle alors les gradients).
- c) `Var(z) = 100·0.05²·1 = 100·0.0025 = **0.25**` ⇒ `σ(z) = **0.5**` : la pré-activation est deux fois trop petite ; en profondeur elle s'éteint ⇒ **vanishing gradients**, apprentissage quasi nul des premières couches.

**A2 (1.5)**
- a) `1.5¹⁰ = (1.5⁵)² ≈ 7.594² ≈ **57.7**` : le gradient explose (~58× sur 10 pas) — un RNN avec `|w_hh| > 1` diverge.
- b) `‖g‖ = √(9+16) = **5**` > 2 ⇒ facteur `2/5 = **0.4**` ; gradient clippé `= (1.2, 1.6)` (direction conservée).
- c) Clipping par valeur : `gᵢ ← clip(gᵢ, −c, c)` composante par composante (peut changer la direction) ; par norme : redimensionnement global (direction conservée).

**A3 (1.5)**
- a) `a⊙m/(1−p) = (2,0,0,6)/0.5 = **(4, 0, 0, 12)**`.
- b) Au test : **aucun dropout**, activations inchangées (le scaling a déjà été fait à l'entraînement).
- c) `E[mᵢ/(1−p)] = (1−p)/(1−p) = 1` : chaque activation est mise à l'échelle ×2 la moitié du temps ⇒ espérance préservée, plus besoin de correction au test.

**A4 (1)**
- a) Arrêt à l'époque **4** (`val = 0.42`, minimum) : après, la perte de validation remonte (0.45, 0.55) alors que le train continue de baisser.
- b) Le train continuerait de chuter ⇒ **sur-apprentissage** : le modèle mémorise le train (bruit, spécificités d'échantillon) au détriment de la généralisation.

**A5 (1.5)**
- a) `μ = (1+3+5)/3 = **3**` ; `σ² = (4+0+4)/3 = 8/3 ≈ **2.667**` ; `σ = √(8/3) ≈ **1.633**`.
- b) `x̂ = ((1−3), 0, (5−3))/1.633 = **(−1.225, 0, 1.225)**` (exact : ∓√(3/2) ≈ ∓1.2247).
- c) **BatchNorm** dépend de la taille du mini-batch (statistiques estimées sur le batch) ; LayerNorm normalise chaque exemple individuellement — d'où LayerNorm pour séquences/RNN et petits batches.

## PARTIE B (7 points)

**B1 (1.5)**
- a) `(32−5)/1+1 = 28` → `28×28×8` ; pool → `14×14×8` ; `(14−3)/1+1 = **12**` → `12×12×16` ; pool → `6×6×16`.
- b) Flatten `= 6·6·16 = **576**`.
- c) FC : `576·10 + 10 = **5770**` paramètres.

**B2 (2)**
- a) `z_t = σ(0) = **0.5**` ; `r_t = σ(−1) ≈ **0.269**`.
- b) Pré-act `h̃ = 1·2 + 1·(0.269·1) = 2.269` ⇒ `h̃_t = tanh(2.269) ≈ **0.979**` ;
`h_t = (1−0.5)·1 + 0.5·0.979 = 0.5 + 0.4895 ≈ **0.989**` (≈ 0.9895).
- c) Ex. : (1) pas de cell state séparé — le GRU n'a que `h` ; (2) deux portes seulement (`z, r`) au lieu de trois (`f, i, o`) : le couplage `forget = 1 − update` lie l'oubli et l'écriture ; ~25 % de paramètres en moins.

**B3 (1.5)**
- a) `t = 0.5` : alertés `≥ 0.5` → malveillants `{0.5, 0.7, 0.9}` (3 TP), bénins `{0.6, 0.8}` (2 FP).
`TPR = 3/4 = **0.75**` ; `FPR = 2/4 = **0.50**` ; `précision = 3/5 = **0.60**`.
- b) `t = 0.7` : TP `{0.7, 0.9}` (2), FP `{0.8}` (1). `TPR = 2/4 = **0.50**` ; `FPR = 1/4 = **0.25**` ; `précision = 2/3 ≈ **0.667**`.
- c) Chaque `t` donne un point `(FPR, TPR)` de la courbe ROC ; `t` décroissant ⇒ on parcourt la courbe de `(0,0)` à `(1,1)` ; l'aire sous cette courbe = AUC.

**B4 (1)**
- a) Position `5.25` sur la liste triée ⇒ `seuil = x₍₅₎ + 0.25·(x₍₆₎ − x₍₅₎) = 0.25 + 0.25·(0.30−0.25) = **0.2625**`.
- b) Bénins > 0.2625 : `{0.30, 0.50}` ⇒ `FPR = 2/8 = **0.25**`.
- c) `0.28, 0.45, 0.60 > 0.2625` ⇒ `TPR = 3/3 = **1.0**`. Compromis : seuil plus bas ⇒ rappel maximal mais 25 % de faux positifs — en SOC on remonterait le seuil (ou seuil par percentiles de coût), le coût d'un FP étant structurellement plus faible que celui d'un FN manqué.

**B5 (1)**
- a) `c₁·c₂ = 9` ; `‖c₁‖ = √(9+16) = 5` ; `‖c₂‖ = 5` ; `cos = 9/25 = **0.36**`.
- b) `0.36 < 0.7` ⇒ **familles différentes**.
- c) Le cosinus compare les **directions** (patron sémantique du code), insensible à la norme (amplitude/quantité de trafic) — deux variantes d'une même famille restent proches même si leurs normes diffèrent.

## PARTIE C (6 points)

**C1 (1.5)**
- a) `V*(B) = max(2 + 0.9·0, 5 + 0.9·0) = **5**` ⇒ action **honeypot** (collecter les preuves avant de couper).
- b) `V*(A) = 1 + 0.9·V*(B) = 1 + 4.5 = **5.5**` (politique durcir → honeypot).
- c) « attendre » en A vaut `0 + 0.9·V*(A) = 0.9·5.5 = **4.95 < 5.5**` ⇒ incohérent avec l'optimalité : la politique optimale est **durcir en A, honeypot en B**, `V*(A) = 5.5`, `V*(B) = 5`. (Auto-vérification : aucune action déviante n'améliore la valeur — condition de Bellman saturée.)

**C2 (1.5)**
- a) `G(π₁) = 0 + 0.9·0 + 0.81·10 = **8.1**` ; `G(π₂) = 4 + 0.9·4 + 0.81·0 = **7.6**`.
- b) `π₁` (récompense retardée mais forte) gagne avec `γ = 0.9`.
- c) Avec `γ = 0.2` : `G(π₁) = 0.04·10 = **0.4**` ; `G(π₂) = 4 + 0.8 = **4.8**` ⇒ `π₂` gagne. `γ` est l'**horizon de planification** : petit γ ⇒ myopie (préfère les gains immédiats), grand γ ⇒ patience (vise les gains différés).

**C3 (1.5)**
- a) `bloc IP : 8 − 10·0.4 − 0.5·6 = 8 − 4 − 3 = **1**` ;
`quarantaine : 7 − 1 − 1 = **5**` ;
`surveillance : 3 − 0.5 − 0 = **2.5**`.
- b) **Quarantaine hôte** (r = 5) : bon compromis détection/FP/coût métier.
- c) durcir la pénalité (`d → +∞` interdit : grande pénalité, e.g. `−100·d`, ou contrainte dure `d ≤ 0` avec filtre d'actions) — on modifie la **fonction de récompense/contrainte**, l'agent et l'algo restent les mêmes.

**C4 (1)**
- a) **Reward hacking** (spécification trompeuse de la récompense) : l'agent optimise le proxy (fermer vite), pas l'objectif réel (neutraliser).
- b) Ex. : récompense = résolution vérifiée (alerte fermée **et** aucun incident lié rouvert dans les 24 h) ; pénalité si réouverture ; éventuellement Reward Modeling/audit humain d'échantillons d'alertes fermées.

**C5 (0.5)** Ex. attendus : **échantillonnage coûteux/lent** (peu d'incidents réels, exploration risquée) ; **exploration unsafe** en prod (l'agent doit « essayer » des actions dommageables) ; aussi acceptés : transfert simulation→réel (sim2real), non-stationnarité de l'adversaire, instabilité/variance des politiques, reproductibilité & auditabilité.
