# DEVOIR 4 — Deep Learning & RL appliqués à la cybersécurité

**Durée : 2 h — Calculatrice non autorisée — Sans document**
**Barème : 20 points**

*Tables utiles (fournies) : √5 ≈ 2.2361 ; √2 ≈ 1.4142 ; FGSM : `x_adv = x + ε·sign(∇ₓ)` ; Adam : `m ← β₁m + (1−β₁)g`, `v ← β₂v + (1−β₂)g²`, `m̂ = m/(1−β₁ᵏ)`, `v̂ = v/(1−β₂ᵏ)`, `w ← w − η·m̂/(√v̂+ε₀)` ; champ récepteur : `r ← r + (F−1)·j`, `j ← j·s`.*

---

## PARTIE A — Fondements du deep learning (7 points)

**A1 (1.5 pt)** Régression régularisée L2 : un neurone linéaire `ŷ = w₁x₁ + w₂x₂`, un échantillon `x = (1, 2)`, `y = 4`, pondération `λ = 0.1`.
`J(w) = ½(ŷ − y)² + (λ/2)‖w‖²`, gradient `∇J = (ŷ − y)·x + λ·w`.
a) Pour `w = (1, 1)` : calculer `ŷ`, le terme de perte et le terme de régularisation, puis `J`.
b) Calculer `∇J` en ce point.
c) À quoi sert le terme `λ‖w‖²/2` et que provoque un `λ` trop grand ?

**A2 (2 pts)** Premier pas d'**Adam** sur un scalaire : `w₀ = 1`, gradient `g₀ = 2`, `β₁ = 0.9`, `β₂ = 0.999`, `η = 0.1`, `ε₀` négligeable, `m₀ = v₀ = 0`.
a) Calculer `m₁` et `v₁`.
b) Calculer les biais-corrigés `m̂₁` et `v̂₁`.
c) Calculer la mise à jour puis `w₁`. Pourquoi la correction de biais est-elle indispensable au pas 1 ?

**A3 (2 pts)** **BatchNorm** sur un mini-batch de 4 activations `B = [1, 3, 5, 7]`, paramètres `γ = 2`, `β = 1` (pas d'ε).
a) Calculer la moyenne et la variance (population) du batch, puis `σ`.
b) Calculer les 4 valeurs normalisées `x̂ᵢ`.
c) Calculer les sorties `yᵢ = γx̂ᵢ + β`, puis vérifier la moyenne et l'écart-type des sorties. Quel bénéfice pour l'entraînement ?

**A4 (0.5 pt)** Dans quels cas préfère-t-on **LayerNorm** à BatchNorm (deux situations) ?

**A5 (1 pt)** SGD avec **momentum** sur `L(w) = (w − 3)²`, donc `g(w) = 2(w − 3)` ; `μ = 0.9`, `η = 0.1`, `w₀ = 0`, `v₀ = 0`.
Rappel : `v ← μv + g`, puis `w ← w − η·v`.
a) Étape 1 : calculer `g₀`, `v₁`, `w₁`.
b) Étape 2 : calculer `g₁`, `v₂`, `w₂`.
c) Pourquoi la vitesse `v` dépasse-t-elle le gradient instantané ?

---

## PARTIE B — Architectures (7 points)

**B1 (1.5 pt)** CNN : `conv 3×3 (s=1)` → `max-pool 2×2 (s=2)` → `conv 3×3 (s=1)`.
Avec `r = 1`, `j = 1` initialement : `r ← r + (F−1)·j` à chaque couche puis `j ← j·s` si stride `s`.
a) Champ récepteur après chaque couche.
b) Champ récepteur total et stride cumulé `j` final.
c) Pourquoi le champ récepteur compte-t-il pour détecter des motifs dans des images de bytes ?

**B2 (2 pts)** RNN scalaire `h_t = tanh(w_hh·h_{t−1} + w_xh·x_t)`. On donne, pour une entrée fixée : `h₁ = 0.6`, `h₂ = 0`, `h₃ = 0.8`.
La dérivée se factorise : `∂h₃/∂h₀ = w_hh³ · ∏ₜ (1 − h_t²)`.
a) Calculer `∏(1 − h_t²)` sur les 3 pas.
b) Cas `w_hh = 0.5` : calculer `∂h₃/∂h₀`. Que constate-t-on ?
c) Cas `w_hh = 1.2` : recalculer. Vers quel régime entre-t-on quand `T` grandit ?

**B3 (1.5 pt)** Scores d'un détecteur de phishing : Bénins `[0.2, 0.5, 0.8]`, Phishing `[0.4, 0.5, 0.9]`.
L'AUC par comptage de paires (Mann–Whitney) : chaque paire (positif, négatif) vaut **1** si `score⁺ > score⁻`, **0.5** si égalité, **0** sinon ; `AUC = somme/9`.
a) Détailler le compte des 9 paires.
b) Donner l'AUC (fraction exacte puis décimale).
c) Une AUC de 0.5 correspond à quoi ? Interpréter 0.61 pour un SOC.

**B4 (1 pt)** Convolutions **1D** sur une séquence de `n = 100` bytes : filtre `F = 9`, `s = 1`, `p = 0`, **16 filtres** (entrée mono-canal), puis `max-pool 2×2 (s=2)`.
a) Longueur de sortie de la convolution et profondeur.
b) Nombre de paramètres de la couche conv (biais inclus).
c) Que « détecte » un filtre 1D dans ce contexte (analogie avec les filtres CNN-image) ?

**B5 (1 pt)** Comparaison **LSTM vs GRU**, entrée `d = 64`, cachée `h = 128`.
Paramètres par porte : `W_xh (h×d) + W_hh (h×h) + b (h)`.
a) Paramètres d'une porte, puis total **LSTM** (4 portes).
b) Total **GRU** (3 portes).
c) Ratio LSTM/GRU ; quel modèle choisir si la latence d'inférence est stricte sur un capteur embarqué ?

---

## PARTIE C — RL & robustesse (6 points)

**C1 (2 pts)** Détecteur linéaire de malware sur features `x = (x₁, x₂)` : score `s(x) = x₁ + 2x₂`, classé **malveillant si `s ≥ 6`**. L'échantillon `x = (3, 1)` est actuellement classé bénin.
Attaque **FGSM** : `x_adv = x + ε·sign(∇ₓs)` (ici `∇ₓs = w = (1, 2)`).
a) Calculer `s(x)` et confirmer le classement initial.
b) Avec `ε = 0.6` : calculer `x_adv` puis `s(x_adv)` ; l'échantillon est-il détourné ?
c) Quelle est la **plus petite** valeur de `ε` (seuil exact) qui retourne la classification ? Normes `L∞` et `L₂` de la perturbation de la question b.

**C2 (1 pt)** Empoisonnement d'étiquettes : jeu d'entraînement de 10 000 échantillons dont **15 % malveillants**. L'attaquant re-étiquette « bénin » **8 %** des échantillons malveillants.
a) Nombre de malveillants dans l'entraînement propre, puis nombre retourné et restant.
b) Nouvelle proportion apparente de malveillants.
c) Effet attendu sur la frontière de décision et une parade côté données.

**C3 (1.5 pt)** Alerte avec probabilité de malveillance `p = 0.4`. Coûts (k€) :
- **Blocage automatique** : si l'hôte est bénin → 50 (rupture de service) ; si malveillant → 0.
- **Ignorer** : si malveillant → 200 (brèche) ; si bénin → 0.
- **Escalade analyste** : 5 fixe, + 20 de délai si l'hôte est malveillant.
a) Calculer le coût espéré des trois décisions.
b) Quelle décision prendre ? Comment le seuil optimal de `p` change-t-il si le coût de brèche passe à 1000 ?

**C4 (1 pt)** Backdoor : le détecteur (empoisonné) a un rappel de **90 %** sur les 200 malveillants propres du test, mais seulement **5 détections** sur les 200 malveillants portant le trigger `0xDEADBEEF` dans l'en-tête.
a) Calculer les deux rappels.
b) Quelle supervision/technique de test aurait révélé la backdoor avant mise en production ?

**C5 (0.5 pt)** Gouvernance d'un SOC piloté par IA : citer **trois** exigences de supervision humaine / traçabilité à imposer avant mise en production.

---
*Fin du devoir — vérifiez vos réponses avant de rendre.*
