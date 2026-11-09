# DEVOIR 3 — Deep Learning & RL appliqués à la cybersécurité

**Durée : 2 h — Calculatrice non autorisée — Sans document**
**Barème : 20 points**

*Tables utiles (fournies) : σ(0)=0.5, σ(−1)≈0.269 ; tanh(2.269)≈0.979 ; √2 ≈ 1.414 ; 1.5⁵ ≈ 7.594 ; √(8/3) ≈ 1.633 ; rappel quantile linéaire ; Bellman optimel : V*(s) = maxₐ [r(s,a) + γ·V*(s′)].*

---

## PARTIE A — Fondements du deep learning (7 points)

**A1 (1.5 pt)** Une couche linéaire a `fan_in = fan_out = 100`, entrée normalisée `Var(x) = 1`.
a) Calculer l'écart-type de l'initialisation **Xavier** : `σ_W = √(2/(fan_in+fan_out))`.
b) Calculer l'écart-type de l'initialisation **He** : `σ_W = √(2/fan_in)` ; quelle init choisir pour une cachée **ReLU** et pourquoi ?
c) Un stagiaire initialise avec `σ_W = 0.05`. Calculer `Var(z) = fan_in·σ_W²·Var(x)` et l'écart-type de la pré-activation. Quel symptôme d'entraînement attendre ?

**A2 (1.5 pt)** RNN scalaire `h_t = tanh(w_hh·h_{t−1} + …)`, `w_hh = 1.5`.
a) En supposant `tanh`≈identité autour de 0, la norme du gradient est multipliée par `|w_hh|` par pas. Calculer le facteur sur 10 pas (indice : `1.5⁵ ≈ 7.594`).
b) Le gradient du pas courant mesuré vaut `g = (3, 4)`. Le clipping par norme fixe un plafond `‖g‖ ≤ 2` : calculer `‖g‖`, le facteur d'échelle et le gradient clippé.
c) Différence entre clipping **par norme** et clipping **par valeur** ?

**A3 (1.5 pt)** Dropout inversé avec probabilité de drop `p = 0.5` (on divise par `1−p` à l'entraînement). Vecteur d'activations `a = (2, 4, 0, 6)` et masque tiré `m = (1, 0, 1, 1)`.
a) Calculer le vecteur de sortie **à l'entraînement**.
b) Que fait-on **au test** (dropout inversé) ?
c) Expliquer en une phrase pourquoi la division par `1−p` garde l'espérance des activations inchangée.

**A4 (1 pt)** Historique d'entraînement d'un classifieur de malwares :

| époque | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| perte train | 0.90 | 0.65 | 0.45 | 0.30 | 0.22 | 0.15 |
| perte val | 0.92 | 0.68 | 0.50 | **0.42** | 0.45 | 0.55 |

a) À quelle époque arrêter (early stopping) ? Justifier.
b) Que devient la perte de train après l'arrêt — que révèle la divergence train/val ?

**A5 (1.5 pt)** LayerNorm sur le vecteur `x = (1, 3, 5)` (γ=1, β=0, sans ε) : `x̂ᵢ = (xᵢ − μ)/σ_pop`.
a) Calculer `μ` puis la variance population `σ²` et `σ`.
b) Donner le vecteur normalisé (3 décimales).
c) BatchNorm vs LayerNorm : lequel dépend de la taille du mini-batch ?

---

## PARTIE B — Architectures (7 points)

**B1 (1.5 pt)** CNN `32×32×3` : `conv 5×5, 8 filtres (p=0, s=1)` → `max-pool 2×2 (s=2)` → `conv 3×3, 16 filtres (p=0, s=1)` → `max-pool 2×2 (s=2)` → `flatten` → `FC → 10`.
a) Taille après chacune des 4 opérations.
b) Taille du flatten.
c) Nombre de paramètres de la FC (avec biais).

**B2 (2 pts)** Cellule **GRU** (1 neurone) : `h_{t−1} = 1`, `x_t = 2`. Poids :
- `z` : `U_z=0, W_z=0, b_z=0`
- `r` : `U_r=0, W_r=−1, b_r=0`
- `h̃` : `U_h=1, W_h=1, b_h=0`
Équations : `z_t=σ(·)`, `r_t=σ(·)`, `h̃_t = tanh(U_h x_t + W_h·(r_t⊙h_{t−1}))`, `h_t = (1−z_t)⊙h_{t−1} + z_t⊙h̃_t`.
a) Calculer les portes `z_t` et `r_t`.
b) Calculer `h̃_t` puis `h_t` (3 décimales).
c) Donner **deux différences** structurelles GRU vs LSTM.

**B3 (1.5 pt)** Scores de classifieur (≥ t ⇒ alerte). Bénins : `[0.2, 0.4, 0.6, 0.8]` ; malveillants : `[0.3, 0.5, 0.7, 0.9]`.
a) Pour `t = 0.5` : calculer TPR, FPR, précision.
b) Pour `t = 0.7` : recalculer les trois métriques.
c) Que dit-on de la courbe ROC si on fait varier `t` de 0 à 1 ?

**B4 (1 pt)** Détection d'anomalies par AE. Erreurs de reconstruction des flux bénins du jeu de validation :
`[0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.50]`.
On fixe le seuil au **75ᵉ percentile** (interpolation linéaire : position `= 0.75·(n−1) = 5.25` sur la liste triée).
a) Calculer le seuil exact.
b) FPR sur les bénins (on alerte si `erreur > seuil`).
c) Sur 3 malveillants détectés avec erreurs `[0.28, 0.45, 0.60]`, donner le TPR ; conclure sur le compromis.

**B5 (1 pt)** Embeddings AE de familles de malwares (espace latent 3D) : `c₁ = (3, 4, 0)`, `c₂ = (3, 0, 4)`. Seuil de similarité : deux codes sont de la même famille si `cos(c₁,c₂) ≥ 0.7`.
a) Calculer `c₁·c₂`, `‖c₁‖`, `‖c₂‖`, puis le cosinus.
b) Même famille ou non ?
c) Pourquoi utiliser le cosinus plutôt que la distance euclidienne pour comparer des codes ?

---

## PARTIE C — RL & robustesse (6 points)

**C1 (1.5 pt)** MDP « kill chain » déterministe, `γ = 0.9`, `V*(T) = 0` (T : incident clos).
- État `A` (reconnaissance détectée) : `durcir` → `r=1, B` ; `attendre` → `r=0, A` (auto-boucle).
- État `B` (intrusion en cours) : `bloquer` → `r=2, T` ; `honeypot` → `r=5, T`.
a) Calculer `V*(B) = max(2, 5)` et l'action optimale.
b) En déduire `V*(A)` sous la politique `durcir` puis `honeypot`.
c) Vérifier qu'« attendre » en A ne bat pas « durcir » (`0.9·V*(A)` vs `V*(A)`). Donner la politique optimale complète.

**C2 (1.5 pt)** Deux politiques évaluées sur une même tâche de confinement, `γ = 0.9`, horizon 3 :
- `π₁` : récompenses `(0, 0, 10)` ; `π₂` : récompenses `(4, 4, 0)`.
a) Calculer les retours `G(π₁)` et `G(π₂)`.
b) Laquelle gagne ?
c) Avec `γ = 0.2`, laquelle gagne ? Que contrôle `γ` ici ?

**C3 (1.5 pt)** Récompense multi-objectif d'une action de défense :
`r = r_détection − 10·fp − 0.5·d`, où `fp` = faux positifs (fraction), `d` = minutes d'indisponibilité.
Trois actions :
- `bloc IP` : `r_dét=8, fp=0.4, d=6` ;
- `quarantaine hôte` : `r_dét=7, fp=0.1, d=2` ;
- `surveillance seule` : `r_dét=3, fp=0.05, d=0`.
a) Calculer le `r` de chaque action.
b) Quelle action la récompense recommande-t-elle ?
c) Un DSI refuse toute indisponibilité (`d` doit être 0) : comment modifier la récompense sans réécrire l'agent ?

**C4 (1 pt)** L'agent est récompensé par « nombre d'alertes fermées par heure ». Il se met à fermer automatiquement les alertes sans investiguer.
a) Nommer ce phénomène.
b) Proposer une récompense corrigée (ou deux termes) qui le décourage.

**C5 (0.5 pt)** Citer **deux** limites majeures du RL en production SOC (hors récompense).

---
*Fin du devoir — vérifiez vos réponses avant de rendre.*
