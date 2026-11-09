# Plan détaillé des notebooks T01–T09

> Module **Deep Learning & Reinforcement Learning appliqués à la cybersécurité**
> (EFREI, 35 h, évaluation = projet DL/RL cyber 100 %).
>
> Chaque notebook **T01–T09** est organisé en **6 points-sujets** (`## 1 ·` … `## 6 ·`),
> chacun contenant son **exercice corrigé Pn** (bloc de code, valeurs vérifiées et assertées).
> Le **point 7** (`## 7 ·`) n'est pas un nouveau sujet : il est réservé à la **synthèse
> globale** — tableau récapitulatif des 6 points + exemple intégrateur.
> Les points signalés `*(réf. X)*` renvoient à un chapitre d'un livre de référence
> (voir Bibliographie dans `PLAN_D_ETUDE.md`) — les livres ne sont pas inclus (droits d'auteur),
> seulement référencés.

**Convention de lecture**

| Symbole | Signification |
|---|---|
| `Pn` | Point n du notebook (1–6) : théorie + exemples exécutés (sorties visibles) |
| `P7` | Synthèse globale : récap des 6 points + exemple intégrateur (jamais un nouveau sujet) |
| `→ exo Pn` | Exercice corrigé du point n (à refaire sans regarder) |
| `*(réf. X)*` | Ajout aligné sur la référence X (Goodfellow, Géron, Sutton & Barto, Goodfellow adversarial) |
| `###` | Sous-point (développement du point parent) |

**Contrainte d'implémentation** : tous les modèles sont implémentés **en numpy pur**
(propagation avant + rétropropagation écrites à la main) — l'objectif du module est la
*compréhension des mécanismes*, pas l'usage d'un framework. `scipy`/`sklearn` sont
utilisés pour les métriques, les baselines et la vérification. Données **synthétiques
seedées** (aucun téléchargement) — reproductibles à la virgule près, y compris sur Colab.

## Sommaire

- **T01 — Fondements : perceptron, descente de gradient, backpropagation** (`notebooks/01_fondements_perceptron_backprop.ipynb`) — **P1** Le perceptron & séparabilité linéaire (XOR) · **P2** Fonctions d'activation & leurs dérivées · **P3** Fonctions de perte (MSE, entropie croisée) · **P4** Descente de gradient (GD, mini-batch, pas) · **P5** Rétropropagation d'un MLP (numpy + gradient check) · **P6** Application cyber : détecteur de scan de ports · **P7** Synthèse globale + exemple intégrateur (MLP 2 couches numpy sur données d'intrusion synthétiques)
- **T02 — Réseaux profonds : gradients & sur-apprentissage** (`notebooks/02_dnn_gradients_surapprentissage.ipynb`) — **P1** Anatomie d'un DNN (profondeur, paramètres, mémoire) · **P2** Vanishing gradients (mesure par couche) · **P3** Exploding gradients & clipping · **P4** Initialisations (zéro, naive, Xavier, He) · **P5** Overfitting en deep learning (courbes train/val, capacité) · **P6** Spécificités des données cyber (déséquilibre, bruit, drift) · **P7** Synthèse globale + exemple intégrateur (diagnostic gradient + profond vs peu profond)
- **T03 — Régularisation, normalisation, optimisation** (`notebooks/03_regularisation_normalisation_optimisation.ipynb`) — **P1** Régularisation L1/L2 · **P2** Dropout (inverted dropout, entraînement vs inférence) · **P3** Early stopping & validation · **P4** Normalisation (BatchNorm, LayerNorm) *(réf. Géron ch. 11)* · **P5** Optimiseurs : SGD+momentum, RMSProp, Adam *(réf. Géron ch. 11)* · **P6** Choix des hyperparamètres (grille mini) · **P7** Synthèse globale + exemple intégrateur (même détecteur cyber, 4 configs comparées)
- **T04 — Réseaux convolutifs (CNN) pour la cybersécurité** (`notebooks/04_cnn_cyber.ipynb`) — **P1** L'opération de convolution (stride, padding, taille de sortie) · **P2** Pooling & invariance translationnelle · **P3** Anatomie d'un CNN (blocs, champ récepteur) *(réf. Goodfellow ch. 9)* · **P4** Compte de paramètres CNN vs MLP · **P5** Filtres & cartographies de caractères (interprétabilité) · **P6** Cas cyber : classification malware « en images » (mini-CNN numpy entraîné + gradient check im2col) · **P7** Synthèse globale + exemple intégrateur (pipeline détection complet + matrice de confusion)
- **T05 — Réseaux récurrents : RNN, LSTM, GRU** (`notebooks/05_rnn_lstm_gru_logs.ipynb`) — **P1** La boucle temporelle d'un RNN & BPTT · **P2** Vanishing sur séquences longues (chaîne de Jacobien) *(réf. Géron ch. 17)* · **P3** LSTM : les portes & la cellule à ligne droite · **P4** GRU : variante 2 portes · **P5** Fenêtrage, vocabulaire & padding de logs · **P6** Cas cyber : détection d'exfiltration dans des logs séquentiels (RNN numpy entraîné) · **P7** Synthèse globale + exemple intégrateur (classifieur de sessions normale/exfiltration)
- **T06 — Auto-encodeurs & modèles de représentation** (`notebooks/06_autoencodeurs_anomalies.ipynb`) — **P1** L'auto-encodeur : encodage, goulot, reconstruction · **P2** AE linéaire = PCA (lien valeurs propres) · **P3** Denoising & régularisation de l'AE *(réf. Goodfellow ch. 14)* · **P4** Erreur de reconstruction = score d'anomalie (seuil, ROC) · **P5** Représentations pour la similarité (embeddings, k-NN) · **P6** Cas cyber : détection d'anomalies sur flux réseau (AUC) · **P7** Synthèse globale + exemple intégrateur (détecteur non supervisé complet)
- **T07 — Fondements du Reinforcement Learning** (`notebooks/07_rl_fondements_qlearning.ipynb`) — **P1** Agent, environnement, état, action, récompense *(réf. Sutton & Barto ch. 3)* · **P2** RL vs apprentissage supervisé · **P3** Politiques, valeur & équations de Bellman (calcul à la main) *(réf. Sutton & Barto ch. 4)* · **P4** Exploration / exploitation (ε-greedy, UCB, regret) · **P5** Q-Learning tabulaire (mise à jour, convergence) *(réf. Sutton & Barto ch. 6)* · **P6** Au-delà : SARSA & DQN (vue conceptuelle) · **P7** Synthèse globale + exemple intégrateur (agent de défense réseau sur grille)
- **T08 — RL pour la cyberdéfense adaptative** (`notebooks/08_rl_cyberdefense_adaptative.ipynb`) — **P1** Modéliser un incident en MDP (kill chain simplifiée) · **P2** Réponse automatique aux incidents (politique Q-Learning isoler/surveiller/bloquer) · **P3** Adaptation dynamique des défenses (récompense multi-objectif) · **P4** Orchestration intelligente (séquences de playbooks, coûts) · **P5** Évaluer une politique (retours, variance, baseline à règles) · **P6** Limites & risques du RL en cyber réel (récompense mal spécifiée, adversaire adaptatif, sim2real) · **P7** Synthèse globale + exemple intégrateur (comparaison politique apprises vs baseline)
- **T09 — Robustesse, attaques & gouvernance des modèles** (`notebooks/09_robustesse_adversarial_gouvernance.ipynb`) — **P1** Robustesse & stabilité (sensibilité au bruit, variance entre graines) · **P2** Biais & dérives des modèles IA (biais de catégories, dérive temporelle) · **P3** Vulnérabilités spécifiques du DL (opacité, empoisonnement de données — vue défensive) · **P4** Attaques adaptatives : FGSM & baisse d'exactitude *(réf. Goodfellow ch. 12)* · **P5** Défenses défensives (clipping, réentraînement adversarial, détection) · **P6** Gouvernance, validation & supervision humaine *(réf. EU AI Act — vue pratique)* · **P7** Synthèse globale + exemple intégrateur (scorecard robustesse d'un détecteur)

## Notebooks d'accompagnement

| Fichier | Contenu |
|---|---|
| `10_preparation_projet.ipynb` | 15 drills « papier » (calculs manuels type soutenance) avec corrigés pas-à-pas + vérification numérique ; guide du projet évalué (rubrique : analyse du problème, choix d'architectures, interprétation critique, limites & risques) |
| `11_graphiques.ipynb` | 8 figures pédagogiques inline + PNG dans `outputs/` : perceptron & frontière, vanishing gradients, descente de gradient (pas), sur-apprentissage, cartographies CNN, portes LSTM, score d'anomalie AE/ROC, convergence Q-Learning |

## Miroirs `code/`

Chaque notebook `NN_x.ipynb` a son script miroir auto-exécutable `code/NN_x.py`
(mêmes calculs, mêmes asserts, sortie texte + marqueur de succès), les builders de
notebooks sont dans `code/_build_nb_*.py`. Les logs d'exécution sont dans `outputs/NN_x.log`.

## Statut de vérification

*(rempli après exécution complète — les compteurs sont dérivés des artefacts, pas mémorisés)*
