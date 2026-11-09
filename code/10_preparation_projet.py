#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DLCyber — 10_préparation_projet (miroir exécutable du notebook 10)
===================================================================
Le module est évalué à 100 % par un projet DL/RL appliqué à la cybersécurité.
Ce script exécute les 15 drills « papier » (calculs manuels type soutenance)
avec corrigés pas-à-pas et asserts numériques — MÊME source de vérité que
`code/_build_nb_10.py` (le builder importe la liste CELLS de ce fichier).

Lancer :  python code/10_preparation_projet.py
"""
import os
import sys

# ======================================================================
# Contenu du notebook : liste (type, source). Source de vérité unique.
# ======================================================================
CELLS = [
("md", """# T10 · Préparation au projet — DL/RL appliqués à la cybersécurité

**Le module est évalué à 100 % par un projet** : une réalisation DL ou RL
appliquée à un problème de cybersécurité (détection d'intrusion, phishing,
malwares, exfiltration, réponse à incident…), rendue **en soutenance**.

**Objectifs de ce notebook :**
- Revenir sur **tout le module** (T01–T09) sous forme de **15 drills « papier »**
  — calculs manuels du type de ceux attendus à la soutenance.
- Verrouiller la **rubrique d'évaluation** du projet (ci-dessous).
- Vérifier chaque calcul à la main **puis** avec Python (cellules de contrôle).

**Rubrique du projet (100 % de la note) :**

| Critère | Pondération | Ce qui est attendu |
|---|---|---|
| **Analyse du problème cyber** | 25 % | menace visée, données, hypothèses, baseline à règles |
| **Choix & justification des architectures** | 30 % | pourquoi CE modèle (MLP/CNN/LSTM/AE/Q-Learning) ? alternatives écartées |
| **Interprétation critique des résultats** | 25 % | métriques adaptées au déséquilibre, courbes, erreurs analysées — pas seulement « accuracy = 0.98 » |
| **Limites & risques** | 20 % | robustesse (FGSM, drift), biais, coût opérationnel, gouvernance |

> **Conseil d'étude :** refaire chaque drill **à la main, sans Python**, puis
> exécuter la cellule de contrôle. Les questions de soutenance ressemblent à
> s'y méprendre aux drills ci-dessous.

**Prérequis :** T01–T09 (backprop, gradients, CNN, LSTM, auto-encodeurs, RL, robustesse)."""),

("md", """## 📌 Bloc 0 — Environnement & imports"""),

("code", """import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
np.set_printoptions(precision=4, suppress=True, linewidth=100)
print("numpy", np.__version__, "- pret pour les 15 drills.")"""),

# ---------------------------------------------------------------- D1
("md", """## Drill 1 · Rétropropagation à la main (1 couche cachée)

**Énoncé (papier).** Réseau : entrée $x=\\begin{pmatrix}1\\\\2\\end{pmatrix}$, une couche cachée de
2 neurones à **sigmoïde** ($W_1=I_2$, $b_1=0$), sortie **linéaire** unique
($w_2=(1,1)^{\\top}$, $b_2=0$). Cible $y=1$, perte $L=\\tfrac12(\\hat y-y)^2$.

Calculer à la main : la propagation avant ($a_1$, $\\hat y$, $L$), puis
$\\partial L/\\partial w_2$ et $\\partial L/\\partial W_1$ par remontée des gradients.
On utilisera $\\sigma'(z)=\\sigma(z)(1-\\sigma(z))$ et la sortie linéaire donne
$\\delta_{\\text{out}}=\\hat y-y$."""),

("code", """# Corrigé pas-à-pas — Drill 1 : retropropagation 1 couche cachee
x_d1 = np.array([1.0, 2.0]); y_d1 = 1.0
W1 = np.eye(2); b1 = np.zeros(2); w2 = np.ones(2); b2 = 0.0
sig = lambda z: 1.0 / (1.0 + np.exp(-z))
# 1) propagation avant
z1 = W1 @ x_d1 + b1
a1 = sig(z1)
yhat = w2 @ a1 + b2
L = 0.5 * (yhat - y_d1) ** 2
print(f"Drill 1 : a1 = {a1} (sigmo(1), sigmo(2)), yhat = {yhat:.6f}, L = {L:.6f}")
# 2) remontee des gradients
d_out = yhat - y_d1
dL_w2 = d_out * a1
delta1 = d_out * w2 * a1 * (1 - a1)
dL_W1 = np.outer(delta1, x_d1)
print(f"  dL/dw2   = {dL_w2}")
print(f"  delta_1  = {delta1}")
print(f"  dL/dW1   =\\n{dL_W1}")
# 3) controle independant : gradient numerique (differences centrales)
def loss_d1(W1, b1, w2, b2):
    return 0.5 * (w2 @ sig(W1 @ x_d1 + b1) + b2 - y_d1) ** 2
eps = 1e-6
num_W1 = np.zeros_like(W1); num_w2 = np.zeros_like(w2)
for i in range(2):
    for j in range(2):
        Wp = W1.copy(); Wp[i, j] += eps; Wm = W1.copy(); Wm[i, j] -= eps
        num_W1[i, j] = (loss_d1(Wp, b1, w2, b2) - loss_d1(Wm, b1, w2, b2)) / (2 * eps)
    wp = w2.copy(); wp[i] += eps; wm = w2.copy(); wm[i] -= eps
    num_w2[i] = (loss_d1(W1, b1, wp, b2) - loss_d1(W1, b1, wm, b2)) / (2 * eps)
print(f"  ecart max vs gradient numerique : "
      f"{max(np.abs(dL_W1 - num_W1).max(), np.abs(dL_w2 - num_w2).max()):.1e}")
assert np.allclose(dL_w2, num_w2, atol=1e-6)
assert np.allclose(dL_W1, num_W1, atol=1e-6)
assert np.isclose(yhat, 1.611856, atol=1e-6)
assert np.allclose(dL_w2, [0.447302, 0.538921], atol=1e-5)
assert np.isclose(np.abs(dL_W1).max(), 0.240596, atol=1e-5)
print("  ✅ gradients analytiques = gradients numeriques (controle independant)")"""),

# ---------------------------------------------------------------- D2
("md", """## Drill 2 · Dérivée de la sigmoïde en $z=-1.2$

**Énoncé (papier).** Soit $\\sigma(z)=\\dfrac{1}{1+e^{-z}}$.
1. Calculer $\\sigma(-1.2)$ puis $\\sigma'(-1.2)$ (on rappellera la formule
   $\\sigma'(z)=\\sigma(z)\\,(1-\\sigma(z))$).
2. Quelle est la **valeur maximale** possible de $\\sigma'$ et où est-elle atteinte ?
3. Pourquoi la sigmoïde sature-t-elle en réseau profond (rappel T02) ?"""),

("code", """# Corrigé pas-à-pas — Drill 2 : derivee de la sigmoide en -1.2
z = -1.2
s = 1.0 / (1.0 + np.exp(-z))
d = s * (1.0 - s)
print(f"Drill 2 : sigmo(-1.2) = 1/(1+e^1.2) = {s:.6f}")
print(f"  sigmo'(-1.2) = s(1-s) = {d:.6f}")
print(f"  maximum de sigmo' : en z=0, sigmo'(0) = 0.25")
assert np.isclose(s, 0.231475, atol=1e-6)
assert np.isclose(d, 0.177894, atol=1e-6)
assert not np.isclose(d, 0.25)                # loin du maximum
assert d < 0.25                                 # jamais au-dessus de 1/4
s2 = 1.0/(1.0+np.exp(1.2))                      # par l'autre forme
assert np.isclose(s, s2, atol=1e-12)            # cohérence des deux écritures
print("  ✅ sigmo(-1.2) = 0.231475, sigmo'(-1.2) = 0.177894 (max theorique 0.25 en 0)")"""),

# ---------------------------------------------------------------- D3
("md", """## Drill 3 · Choix de la variance d'initialisation (Xavier / He)

**Énoncé (papier).** Couche de 64 entrées et 64 sorties ($fan_{in}=fan_{out}=64$).
1. Donner la variance et l'écart-type **Xavier (Glorot)** :
   $\\mathrm{Var}=\\dfrac{2}{fan_{in}+fan_{out}}$, et la borne uniforme
   $\\pm\\sqrt{\\dfrac{6}{fan_{in}+fan_{out}}}$.
2. Même question avec **He** (ReLU) : $\\mathrm{Var}=\\dfrac{2}{fan_{in}}$.
3. Vérifier numériquement qu'un tirage uniforme borné reproduit bien la variance visée."""),

("code", """# Corrigé pas-à-pas — Drill 3 : Xavier vs He
fan_in, fan_out = 64, 64
var_x = 2.0 / (fan_in + fan_out)
std_x = np.sqrt(var_x)
bnd = np.sqrt(6.0 / (fan_in + fan_out))
print(f"Drill 3 : Xavier var = 2/128 = {var_x:.6f}, std = {std_x:.4f}, borne unif = +/-{bnd:.6f}")
he_std = np.sqrt(2.0 / fan_in)
print(f"  He (ReLU) : std = sqrt(2/64) = {he_std:.6f} (= sqrt(2) * Xavier ici car fan_in=fan_out)")
rng = np.random.default_rng(42)                 # rng isole pour cette cellule
W = rng.uniform(-bnd, bnd, (fan_out, fan_in))
emp_var = W.var()
print(f"  verif numerique : var empirique = {emp_var:.6f} (visee {var_x:.6f})")
assert np.isclose(var_x, 1.0/64.0)              # = 0.015625 exactement
assert np.isclose(std_x, 0.125)                 # = 1/8 exactement
assert np.isclose(bnd, np.sqrt(6.0/128.0))
assert abs(emp_var - var_x) / var_x < 0.05      # variance empirique a +/-5 %
assert np.abs(W).max() <= bnd
assert np.isclose(he_std, np.sqrt(2.0/64.0), atol=1e-12)
print("  ✅ Xavier std = 0.125 ; He std = 0.1768 ; variance empirique conforme")"""),

# ---------------------------------------------------------------- D4
("md", """## Drill 4 · Taille de sortie d'une convolution & compte de paramètres

**Énoncé (papier).** Entrée $28\\times28\\times 3$, couche convolutive : **16 filtres**
de taille $5\\times5$, stride 1, **padding valide** (aucun).
1. Taille spatiale de la sortie ? (formule : $\\lfloor (H-K)/s\\rfloor+1$)
2. Nombre de **paramètres apprenables** de la couche ?
3. Combien de paramètres aurait une couche **entièrement connectée**
   $2352\\to16$ sur la même entrée ? Conclure (T04 · P4)."""),

("code", """# Corrigé pas-à-pas — Drill 4 : conv size + params
H, Wd, C, K, F, s = 28, 28, 3, 5, 16, 1
out = (H - K) // s + 1
print(f"Drill 4 : sortie = ({H}-{K})//{s}+1 = {out} x {out} x {F}")
params = K * K * C * F + F          # poids + biais
print(f"  parametres conv = 5*5*3*16 + 16 = {params}")
fc = (H * Wd * C) * F + F           # entierement connectee 2352 -> 16
print(f"  parametres fully-connected = 2352*16 + 16 = {fc}")
print(f"  ratio = {fc/params:.1f}x plus de parametres pour le fully-connected")
assert out == 24
assert params == 1216               # 1200 poids + 16 biais
assert fc == 37648
assert fc / params > 30
print("  ✅ 24x24x16 en sortie ; 1 216 parametres vs 37 648 en fully-connected")"""),

# ---------------------------------------------------------------- D5
("md", """## Drill 5 · Max-pooling 2×2 sur une matrice 4×4

**Énoncé (papier).** Appliquer un **max-pooling** $2\\times2$ (stride 2) à la matrice

$$X=\\begin{pmatrix}1&2&3&4\\\\5&6&7&8\\\\9&10&11&12\\\\13&14&15&16\\end{pmatrix}$$

Donner la matrice de sortie $2\\times2$ et expliquer ce que le pooling rend
le réseau invariant (T04 · P2)."""),

("code", """# Corrigé pas-à-pas — Drill 5 : max-pooling 2x2 stride 2
X = np.arange(1, 17).reshape(4, 4)
# 4 fenetres 2x2 : (1..4)->6? non : (1,2,5,6)->6 ; (3,4,7,8)->8 ; (9,10,13,14)->14 ; (11..16)->16
out_p = X.reshape(2, 2, 2, 2).max(axis=(1, 3))
print(f"Drill 5 : entree =\\n{X}")
print(f"  sortie (max-pool 2x2, stride 2) =\\n{out_p}")
expected = np.array([[6, 8], [14, 16]])
assert np.array_equal(out_p, expected)
# controle fenetre par fenetre (papier)
assert out_p[0, 0] == X[0:2, 0:2].max() == 6
assert out_p[0, 1] == X[0:2, 2:4].max() == 8
assert out_p[1, 0] == X[2:4, 0:2].max() == 14
assert out_p[1, 1] == X[2:4, 2:4].max() == 16
assert out_p.shape == (2, 2) and X.size // out_p.size == 4   # division par 4 des activations
print("  ✅ sortie = [[6,8],[14,16]] : invariance translationnelle locale, x4 downsampling")"""),

# ---------------------------------------------------------------- D6
("md", """## Drill 6 · BPTT : chaîne de Jacobien sur 3 pas de temps

**Énoncé (papier).** RNN scalaire $h_t=\\tanh(w\\,h_{t-1}+x_t)$, $h_0=0$,
$w=0.5$, entrée $(x_1,x_2,x_3)=(0.1,0.2,0.3)$, perte $L=\\tfrac12(h_3-1)^2$.

1. Calculer $h_1,h_2,h_3$ à la main.
2. Écrire la chaîne $\\dfrac{\\partial h_3}{\\partial w}=\\sum$ en développant les dérivées
   $\\dfrac{\\partial h_t}{\\partial w}=(1-h_t^2)\\big(h_{t-1}+w\\dfrac{\\partial h_{t-1}}{\\partial w}\\big)$.
3. En déduire $\\partial L/\\partial w$ et expliquer le risque de **vanishing** (T05 · P2)."""),

("code", """# Corrigé pas-à-pas — Drill 6 : BPTT sur 3 pas
w = 0.5
xs = [0.1, 0.2, 0.3]
yv = 1.0
hs = []
h = 0.0
for xt in xs:
    h = np.tanh(w * h + xt)
    hs.append(h)
h1, h2, h3 = hs
print(f"Drill 6 : h1 = {h1:.6f}, h2 = {h2:.6f}, h3 = {h3:.6f}")
# chaine des derives partielles dh_t/dw
d1 = (1 - h1**2) * (0.0 + w * 0.0)          # h0 = 0
d2 = (1 - h2**2) * (h1 + w * d1)
d3 = (1 - h3**2) * (h2 + w * d2)
dL_dw = (h3 - yv) * d3
print(f"  dh1/dw = {d1:.6f}, dh2/dw = {d2:.6f}, dh3/dw = {d3:.6f}")
print(f"  dL/dw = (h3-1) * dh3/dw = {dL_dw:.6f}")
# controle : gradient numerique
def loss_d6(w):
    hh = 0.0
    for xt in xs:
        hh = np.tanh(w * hh + xt)
    return 0.5 * (hh - yv) ** 2
eps = 1e-6
num = (loss_d6(w + eps) - loss_d6(w - eps)) / (2 * eps)
print(f"  controle numerique : dL/dw = {num:.6f}")
assert np.isclose(h3, 0.398935, atol=1e-6)
assert np.isclose(dL_dw, num, atol=1e-6)
assert np.isclose(dL_dw, -0.147382, atol=1e-5)
# chaque facteur (1-h_t^2) < 1 : la chaine s'amortit -> vanishing sur sequences longues
factors = [1 - hi**2 for hi in hs]
assert all(0 < fa < 1 for fa in factors)
print(f"  facteurs (1-h_t^2) = {factors} (tous < 1 -> amortissement multiplicatif)")
print("  ✅ dL/dw = -0.147382 (BPTT) = gradient numerique")"""),

# ---------------------------------------------------------------- D7
("md", """## Drill 7 · LSTM : valeur de la porte d'oubli et état de cellule

**Énoncé (papier).** À l'instant $t$ :
$h_{t-1}=(0.5,-0.5)$, $x_t=(1,1)$, $W_f=(2,-2,1,-1)$, $b_f=0$ (porte d'oubli
$z_f = W_f\\,[h_{t-1},x_t]+b_f$). Par ailleurs $c_{t-1}=4$, porte d'entrée
$i_t=\\sigma(0)$, candidat $\\tilde c_t=\\tanh(1)$, porte de sortie $o_t=\\sigma(1)$.

1. Calculer $f_t=\\sigma(z_f)$ : **que devient la mémoire de la cellule ?**
2. Calculer $c_t = f_t\\,c_{t-1} + i_t\\,\\tilde c_t$ puis $h_t = o_t\\tanh(c_t)$.
3. Que vaut $f_t$ si l'on veut **tout oublier** ? **tout conserver** ?"""),

("code", """# Corrigé pas-à-pas — Drill 7 : porte d'oubli LSTM
h_prev = np.array([0.5, -0.5]); x_t = np.array([1.0, 1.0])
Wf = np.array([2.0, -2.0, 1.0, -1.0]); bf = 0.0
z_f = Wf @ np.concatenate([h_prev, x_t]) + bf
f = 1.0 / (1.0 + np.exp(-z_f))
print(f"Drill 7 : z_f = 2*0.5 - 2*(-0.5) + 1 - 1 = {z_f}")
print(f"  f = sigmo(2) = {f:.6f}  -> on conserve ~88 % de la memoire precedente")
c_prev = 4.0
i = 1.0 / (1.0 + np.exp(-0.0))
c_tilde = np.tanh(1.0)
c_t = f * c_prev + i * c_tilde
o = 1.0 / (1.0 + np.exp(-1.0))
h_t = o * np.tanh(c_t)
print(f"  i = sigmo(0) = {i:.4f}, c_tilde = tanh(1) = {c_tilde:.6f}")
print(f"  c_t = {f:.6f}*4 + {i:.4f}*{c_tilde:.6f} = {c_t:.6f}")
print(f"  h_t = sigmo(1)*tanh(c_t) = {o:.6f} * {np.tanh(c_t):.6f} = {h_t:.6f}")
assert np.isclose(z_f, 2.0)
assert np.isclose(f, 1/(1+np.exp(-2)), atol=1e-12)
assert np.isclose(f, 0.880797, atol=1e-6)
assert np.isclose(c_t, 3.903985, atol=1e-6)
assert np.isclose(h_t, 0.730464, atol=1e-6)
assert np.isclose(1.0/(1.0+np.exp(-0.0)), 0.5)   # f=0.5 : oublie la moitie
print("  f -> 0 : tout oublier ; f -> 1 : tout conserver ; f = 0.5 : moitie-moitie")
print("  ✅ f = 0.881 : l'etat de cellule survit presque intact (c_t ~ 3.90)")"""),

# ---------------------------------------------------------------- D8
("md", """## Drill 8 · Auto-encodeur linéaire = PCA (vérification numérique)

**Énoncé (papier).** Soit $X$ centrée, un auto-encodeur **linéaire** à un neurone
de goulot, poids $w$ de norme 1, décodage symétrique ($\\hat x = zw^{\\top}$,
$z = Xw$). Montrer que l'erreur de reconstruction minimale est atteinte pour
$w = v_1$ (premier vecteur propre de la covariance) et vaut

$$\\frac1n\\lVert X - zw^{\\top}\\rVert_F^2 = \\mathrm{tr}(\\Sigma) - \\lambda_{\\max}.$$

Vérifier numériquement **et** contre `sklearn.decomposition.PCA` (T06 · P2)."""),

("code", """# Corrigé pas-à-pas — Drill 8 : AE lineaire = PCA
rng = np.random.default_rng(7)                  # rng isole pour cette cellule
base = rng.normal(size=(200, 1))
X = base @ np.array([[2.0, 1.0, 0.5]]) + rng.normal(0, 0.05, (200, 3))
Xc = X - X.mean(axis=0)                          # centree
cov = Xc.T @ Xc / len(Xc)
lam, V = np.linalg.eigh(cov)
v1, lam_max = V[:, -1], lam[-1]
print(f"Drill 8 : valeurs propres de Sigma = {lam}")
# AE lineaire a 1 neurone : w = v1
w_ae = v1 / np.linalg.norm(v1)
z = Xc @ w_ae
rec = np.outer(z, w_ae)
ae_err = np.sum((Xc - rec) ** 2) / len(Xc)
resid = np.trace(cov) - lam_max
print(f"  erreur AE = {ae_err:.6f} ; tr(Sigma) - lambda_max = {resid:.6f}")
assert np.isclose(np.linalg.norm(w_ae), 1.0)
assert np.isclose(ae_err, resid, rtol=1e-9)
# controle croise sklearn
p = PCA(n_components=1).fit(Xc)
rec_s = p.inverse_transform(p.transform(Xc))
sk_err = np.sum((Xc - rec_s) ** 2) / len(Xc)
assert np.isclose(ae_err, sk_err, atol=1e-8)
# les poids AE appris = chargement PCA (au signe pres)
assert np.isclose(abs(w_ae @ p.components_[0]), 1.0, atol=1e-6)
print(f"  controle sklearn PCA : erreur = {sk_err:.6f} ; |<w_ae, pc1>| = {abs(w_ae @ p.components_[0]):.6f}")
print("  ✅ AE lineaire (1 goulot, decodage symetrique) == PCA : erreur = tr(Sigma) - lambda_max")"""),

# ---------------------------------------------------------------- D9
("md", """## Drill 9 · Score d'anomalie : seuil par quantile et taux de fausses alertes

**Énoncé (papier).** Les scores de reconstruction $s$ des flux **bénins** suivent
$\\mathcal N(2.0, 0.5^2)$ (échantillon observé). Les **attaques** suivent
$\\mathcal N(4.5, 0.7^2)$.

1. Fixer le seuil au **quantile 95 %** des bénins (entraînés sur 1000 flux).
2. Estimer le **taux de fausses alertes (FPR)** sur 1000 flux bénins *indépendants*,
   le comparer à la valeur théorique $1-\\Phi\\big(\\frac{\\tau-\\mu}{\\sigma}\\big)$.
3. En déduire le **TPR** sur les attaques. Discuter le coût des fausses alertes en SOC."""),

("code", """# Corrigé pas-à-pas — Drill 9 : seuil quantile + FPR
rng = np.random.default_rng(11)                 # rng isole pour cette cellule
benign_tr = rng.normal(2.0, 0.5, 1000)
benign_te = rng.normal(2.0, 0.5, 1000)
attacks = rng.normal(4.5, 0.7, 500)
thr = np.quantile(benign_tr, 0.95)
print(f"Drill 9 : seuil = quantile95(benins) = {thr:.6f}")
print(f"  valeur theorique mu + 1.6449*sigma = {2.0 + 0.5*1.6449:.4f}")
fpr = np.mean(benign_te > thr)
fpr_theory = 1 - stats.norm.cdf((thr - 2.0) / 0.5)
tpr = np.mean(attacks > thr)
print(f"  FPR empirique (1000 flux indep.) = {fpr:.3f} ; theorique = {fpr_theory:.4f}")
print(f"  TPR sur attaques = {tpr:.3f}")
assert np.isclose(thr, 2.814705, atol=1e-5)
assert np.isclose(fpr, 0.048, atol=1e-9)        # deterministe (graine fixee)
assert abs(fpr_theory - 0.05) <= 0.01            # ~5 % de fausses alertes visees
assert tpr >= 0.95
assert thr > 2.5 and thr < 3.1
print("  ✅ seuil 2.815 -> ~5 % de fausses alertes, ~98.6 % de vraies attaques detectees")
print("  => en SOC : 5 fausses alertes pour 1000 flux = cout humain ; voir T09 · P6")"""),

# ---------------------------------------------------------------- D10
("md", """## Drill 10 · Équation de Bellman sur une chaîne à 3 états

**Énoncé (papier).** Chaîne déterministe $s_1 \\xrightarrow{a} s_2 \\xrightarrow{a} s_3
\\xrightarrow{a} \\text{terminal}$, une seule action, récompense $+1$ en entrant dans
l'état terminal, $0$ sinon, facteur d'actualisation $\\gamma=0.9$.

1. Écrire l'équation de Bellman $V(s)=R(s)+\\gamma V(s')$ pour chaque état.
2. Résoudre **à la main** par substitution arrière.
3. Confirmer par itération sur les valeurs (T07 · P3)."""),

("code", """# Corrigé pas-à-pas — Drill 10 : Bellman chaine 3 etats, gamma=0.9
gamma = 0.9
# Substitution arriere (papier) : V(terminal)=0
V3 = 1.0 + gamma * 0.0            # recompense +1 en terminal
V2 = 0.0 + gamma * V3             # = 0.9
V1 = 0.0 + gamma * V2             # = 0.81
print(f"Drill 10 : V(s3) = 1 ; V(s2) = gamma*V(s3) = {V2} ; V(s1) = gamma^2 = {V1}")
# confirmation par iteration sur les valeurs
V = np.zeros(3)
for _ in range(200):
    Vn = np.array([gamma * V[1], gamma * V[2], 1.0 + gamma * 0.0])
    if np.allclose(V, Vn, atol=1e-14):
        V = Vn
        break
    V = Vn
print(f"  iteration sur les valeurs : V = {V}")
assert np.allclose(V, [0.81, 0.9, 1.0], atol=1e-12)
assert np.isclose(V[0], gamma ** 2) and np.isclose(V[1], gamma)
print("  ✅ V = (gamma^2, gamma, 1) = (0.81, 0.90, 1.00) — la valeur decroît avec la distance au but")"""),

# ---------------------------------------------------------------- D11
("md", """## Drill 11 · Une mise à jour Q-Learning à la main

**Énoncé (papier).** $Q(s,a)=1.0$, la transition $(s,a)$ donne $r=3$ vers $s'$,
avec $Q(s',\\cdot)=(0.5,\\,2.0)$. Taux d'apprentissage $\\alpha=0.5$, $\\gamma=0.9$.

1. Calculer l'**erreur TD** et la nouvelle valeur $Q(s,a)$ (Q-Learning, hors-politique :
   $\\max_{a'}Q(s',a')$).
2. Refaire la mise à jour côté **SARSA** (dans-politique) si l'action réellement
   choisie en $s'$ est la première ($Q=0.5$). Que constate-t-on ?"""),

("code", """# Corrigé pas-à-pas — Drill 11 : une mise a jour Q-Learning
Q_sa, r, alpha, gamma = 1.0, 3.0, 0.5, 0.9
Q_next = np.array([0.5, 2.0])
td = r + gamma * Q_next.max() - Q_sa
Q_new = Q_sa + alpha * td
print(f"Drill 11 : cible = r + gamma*max Q(s',a') = 3 + 0.9*2 = {r + gamma*Q_next.max()}")
print(f"  erreur TD = {td} ; Q <- 1 + 0.5*{td} = {Q_new}")
td_sarsa = r + gamma * Q_next[0] - Q_sa
Q_sarsa = Q_sa + alpha * td_sarsa
print(f"  SARSA (a' = 1ere action, Q=0.5) : TD = {td_sarsa}, Q <- {Q_sarsa}")
assert np.isclose(td, 3.8, atol=1e-12)
assert np.isclose(Q_new, 2.9, atol=1e-12)
assert np.isclose(td_sarsa, 2.45, atol=1e-12)
assert np.isclose(Q_sarsa, 2.225, atol=1e-12)
assert Q_sarsa < Q_new          # SARSA est plus conservateur (penalise l'exploration future)
print("  ✅ Q-Learning : Q=2.9 ; SARSA : Q=2.225 — Q-Learning est optimiste (max), SARSA prudent")"""),

# ---------------------------------------------------------------- D12
("md", """## Drill 12 · Valeur espérée d'une politique ε-greedy

**Énoncé (papier).** Deux bras de valeur moyenne $v_A=1.0$ (optimal) et $v_B=0.5$.
Politique ε-greedy avec $\\varepsilon=0.2$ : avec probabilité $1-\\varepsilon$ on joue le
meilleur bras estimé (A, suppose bien estimé), avec probabilité $\\varepsilon$ un bras
uniforme parmi les deux.

1. Calculer la récompense **espérée par pas** de cette politique.
2. La comparer au **toujours-greedy** (1.0) et au **toujours-aléatoire** (0.75).
3. Vérifier par simulation Monte-Carlo. À quoi sert l'exploration résiduelle ?"""),

("code", """# Corrigé pas-à-pas — Drill 12 : epsilon-greedy, valeur esperee
eps_g = 0.2
vals = np.array([1.0, 0.5])
E_closed = (1 - eps_g) * vals.max() + eps_g * vals.mean()
print(f"Drill 12 : E[r] = (1-eps)*1 + eps*(1+0.5)/2 = 0.8 + 0.2*0.75 = {E_closed:.4f}")
print(f"  toujours-greedy : 1.0 (mais bloque si mauvaise estimation) ; toujours-aleatoire : {vals.mean()}")
rng = np.random.default_rng(3)                  # rng isole pour cette cellule
N = 200_000
explore = rng.random(N) < eps_g
arms = np.where(explore, rng.integers(0, 2, N), 0)
E_emp = vals[arms].mean()
print(f"  simulation MC (N={N:,}) : E[r] = {E_emp:.5f}")
assert np.isclose(E_closed, 0.95, atol=1e-12)
assert np.isclose(E_emp, 0.95, atol=0.01)
assert vals.mean() < E_closed < vals.max()      # entre aléatoire et greedy
print("  ✅ E[r] = 0.95 : on sacrifie 5 % de récompense contre la garantie d'explorer")
print("  => sans exploration residuelle, un bras sous-estime n'est jamais reteste (regret lineaire)")"""),

# ---------------------------------------------------------------- D13
("md", """## Drill 13 · Sélection UCB : lequel de ces bras essayer ?

**Énoncé (papier).** 3 bras, moyennes observées $\\bar x=(0.9,0.5,0.8)$,
nombres de tirages $n=(100,5,50)$, total $t=155$, $c=2$ :

$$\\mathrm{UCB}_i = \\bar x_i + c\\sqrt{\\frac{\\ln t}{n_i}}.$$

1. Calculer les trois UCB (arrondir à $10^{-3}$) et désigner le bras sélectionné.
2. Pourquoi ce résultat illustre-t-il **l'optimisme face à l'incertitude** (T07 · P4) ?"""),

("code", """# Corrigé pas-à-pas — Drill 13 : selection UCB1
means = np.array([0.9, 0.5, 0.8])
counts = np.array([100, 5, 50])
c = 2.0
t = counts.sum()
ucb = means + c * np.sqrt(np.log(t) / counts)
print(f"Drill 13 : t = {t}, ln t = {np.log(t):.6f}")
for i in range(3):
    print(f"  bras {i+1} : {means[i]} + 2*sqrt({np.log(t):.4f}/{counts[i]}) = {ucb[i]:.6f}")
choice = int(ucb.argmax())
print(f"  -> UCB choisit le bras {choice+1} (moyenne la plus FAIBLE mais tres peu essayee)")
expected = np.array([1.349151, 2.508666, 1.435196])
assert np.allclose(ucb, expected, atol=1e-5)
assert choice == 1
assert ucb[1] > ucb[0] and ucb[1] > ucb[2]      # l'incertitude fait gagner le bras 2
print("  ✅ bras 2 (moyenne 0.5, n=5) l'emporte : incertitude = bonus d'exploration")
print("  => regret subi = prix a payer pour confirmer/refuter les moyennes observees")"""),

# ---------------------------------------------------------------- D14
("md", """## Drill 14 · Récompense multi-objectif en cyberdéfense adaptative

**Énoncé (papier).** Un agent de réponse à incident choisit entre :
- **isoler l'hôte** : bloque 100 % de l'attaque, coût = 30 min d'indisponibilité ;
- **limiter le débit** : bloque 80 % de l'attaque, coût = 5 min.

Récompense $R=\\alpha\\cdot\\text{blocage} - \\beta\\cdot\\text{temps (en heures)}$,
avec $\\alpha=1$.

1. Calculer $R$ des deux actions pour $\\beta=0.5$ puis $\\beta=0.3$.
2. Trouver la **valeur de bascule** $\\beta^*$ où l'agent change de préférence.
3. Que se passe-t-il si $\\beta$ est mal spécifié (T08 · P3 & P6) ?"""),

("code", """# Corrigé pas-à-pas — Drill 14 : recompense multi-objectif
def recomp(beta):
    iso = 1.0 - beta * (30 / 60)     # bloque 100 %, 0.5 h d'arret
    rl = 0.8 - beta * (5 / 60)       # bloque 80 %, 1/12 h d'arret
    return iso, rl
r05 = recomp(0.5)
r03 = recomp(0.3)
print(f"Drill 14 : beta=0.5 -> isoler = {r05[0]:.4f}, limiter = {r05[1]:.6f}  (limiter gagne)")
print(f"  beta=0.3 -> isoler = {r03[0]:.4f}, limiter = {r03[1]:.4f}  (isoler gagne)")
# bascule : 1 - beta/2 = 0.8 - beta/12  ->  0.2 = beta*(1/2-1/12) = beta*5/12
b_star = 0.2 / (0.5 - 1 / 12)
rb = recomp(b_star)
print(f"  valeur de bascule beta* = 0.2/(5/12) = {b_star:.4f} ; R(isoler)=R(limiter)={rb[0]:.4f}")
assert np.isclose(b_star, 0.48, atol=1e-12)
assert np.isclose(r05[1], 0.758333, atol=1e-5) and r05[1] > r05[0]
assert np.isclose(r03[0], 0.85) and r03[0] > r03[1]
assert np.isclose(rb[0], rb[1], atol=1e-12)
assert (recomp(0.6)[1] > recomp(0.6)[0]) and (recomp(0.1)[0] > recomp(0.1)[1])
print("  ✅ beta* = 0.48 : au-dela, l'agent privilegie la continuite de service ; en dessous, la securite")
print("  => recompense mal specifiee = agent qui laisse passer des attaques (ou coupe des services) ;")
print("     c'est LE risque n°1 du RL en cyber reel (T08 · P6)")"""),

# ---------------------------------------------------------------- D15
("md", """## Drill 15 · FGSM : perturbation ε à partir d'un vecteur de poids

**Énoncé (papier).** Classifieur **linéaire**, logit $m(x)=w^{\\top}x$ avec
$w=(2,-3,1)$ ; l'attaque maximise la perte, gradient $\\nabla_x J = w$.
FGSM : $x_{adv}=x+\\varepsilon\\,\\mathrm{sign}(\\nabla_x J)$ avec $\\varepsilon=0.5$.

1. Écrire la perturbation $\\delta$ et le gain de logit $\\Delta m = w^{\\top}\\delta$.
2. Montrer que $\\Delta m = \\varepsilon\\,\\lVert w\\rVert_1$ (borne maximale en $\\ell_\\infty\\le\\varepsilon$).
3. Une marge initiale de $+2$ tient-elle ? Donner le **$\\varepsilon$ critique** qui inverse la décision."""),

("code", """# Corrigé pas-à-pas — Drill 15 : FGSM avec w = (2,-3,1)
w = np.array([2.0, -3.0, 1.0])
eps_f = 0.5
delta = eps_f * np.sign(w)
dl = w @ delta
print(f"Drill 15 : sign(w) = {np.sign(w)}, delta = eps*sign(w) = {delta}")
print(f"  gain de logit : w.delta = 2*0.5 + (-3)*(-0.5) + 1*0.5 = {dl:.4f}")
print(f"  borne theorique eps*||w||_1 = {eps_f}*6 = {eps_f*np.abs(w).sum():.4f}")
assert np.allclose(delta, [0.5, -0.5, 0.5])
assert np.isclose(dl, eps_f * np.abs(w).sum())
assert np.isclose(dl, 3.0)
margin = 2.0
print(f"  marge initiale +{margin} : apres attaque marge = {margin - dl:.1f} -> DECISION INVERSEE")
assert margin - dl < 0
eps_star = margin / np.abs(w).sum()
print(f"  epsilon critique : eps* = marge/||w||_1 = 2/6 = {eps_star:.6f}")
assert np.isclose(eps_star, 1/3, atol=1e-12)
assert 4.0 - dl > 0                              # une marge de 4 tiendrait
print("  ✅ chaque dimension bouge de +/-eps dans le sens du gradient : ||w||_1 = sensibilité L∞")
print("  => defenses : clipping, reentrainement adversarial, distance au seuil (T09 · P4-P5)")"""),

# ---------------------------------------------------------------- final
("md", """## Conseils soutenance

- **Saviez-vous justifier votre architecture ?** « J'ai pris un LSTM parce que les logs
  sont séquentiels et que la porte d'oubli filtre le bruit entre événements » bat
  « j'ai pris un LSTM parce que ça marche bien ». Attendez-vous aux drills 6–7.
- **Ne présentez jamais une accuracy seule** sur des données déséquilibrées : apportez
  F1, AUC-ROC, matrice de confusion, et un coût métier des fausses alertes (drills 9, 12).
- **Anticipez la question adversariale** : « et si l'attaquant perturbe vos entrées ? »
  → FGSM, marge, coût de l'attaque (drill 15).
- **Nommez vos limites** : drift des attaques, récompense RL mal spécifiée (drill 14),
  robustesse non garantie. Une limite identifiée vaut plus qu'un résultat sur-vendu.
- **Rejouez les calculs** de base au tableau (drills 1–5, 10–11) : le jury teste la
  compréhension, pas la mémoire.

### Checklist projet (à cocher avant le rendu)
- [ ] Problème cyber formulé : menace, adversaire, données, baseline à règles.
- [ ] Architecture choisie **justifiée** contre au moins une alternative écartée.
- [ ] Métriques adaptées au déséquilibre (F1/AUC/confusion) + courbes d'apprentissage.
- [ ] Analyse d'erreurs : exemples de faux positifs/negatifs commentés.
- [ ] Au moins un test de robustesse (bruit, FGSM, dérive simulée).
- [ ] Section « limites & risques » honnête (biais, drift, adversaire adaptatif).
- [ ] Reproductibilité : graines fixées, environnement figé, code exécutable.
- [ ] Soutenance chronométrée : 10 min démo + 10 min questions « papier » (les drills)."""),

("code", """print("NB 10 — les 15 drills sont vérifiés. ✅")"""),
]


def main():
    """Execute every code cell in a shared namespace (mirror of the notebook)."""
    ns = {"__name__": "__drill__"}
    for kind, src in CELLS:
        if kind == "code":
            exec(compile(src, "<drill>", "exec"), ns)
    print("10_preparation_projet — exercices vérifiés. ✅")


if __name__ == "__main__":
    main()
