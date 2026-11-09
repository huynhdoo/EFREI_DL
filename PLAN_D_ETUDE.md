# Plan d'étude — Deep Learning & RL appliqués à la cybersécurité (2025-2026)

> Base : `docs/Syllabus_DL_RL_Cybersecurite.pdf`. Objectif : couvrir le programme
> en 35 h (10 séquences de 3 h 30) de façon progressive, avec de la théorie,
> des exemples et des exercices corrigés en Python **numpy pur** (`code/`), pour
> comprendre les mécanismes avant de les industrialiser.
>
> **Évaluation du module : un projet DL/RL appliqué à la cybersécurité (100 %)** —
> analyse d'un problème cyber complexe, choix et justification des architectures,
> interprétation critique des résultats, analyse des limites & risques.
> Les notebooks d'entraînement et les devoirs blancs (`exams/`) préparent à la
> maîtrise conceptuelle exigée par le projet et sa soutenance.

## Prérequis vérifiés
Python 3.12 avec `numpy`, `scipy`, `pandas`, `sklearn`, `matplotlib`.
Validation des modules Machine Learning (M1) et Data Science & IA (M2) ;
très bonne maîtrise de la cybersécurité ; aisance en programmation ;
bases solides en mathématiques appliquées (souhaitées).
Avant de commencer, réviser : gradients & dérivées (rappel fait au T01),
algèbre matricielle de base (produit matriciel, normes), probabilités (lois usuelles).

## Jupyter Notebooks (`notebooks/`)
11 notebooks complets, avec explications détaillées et sorties exécutées
(outputs pré-calculées dans le `.ipynb`) — disponibles sur **Google Colab**
(`File → Open notebook from → Google Drive`, dossier `EFREI/DL/notebooks`)
ou JupyterLab local (`jupyter notebook notebooks/`). **Aucun framework DL requis** :
tous les modèles sont implémentés en numpy pur (objectif « comprendre les mécanismes »).

| Notebook | Partie | Contenu (6 points-sujets + synthèse) |
|---|---|---|
| `01_fondements_perceptron_backprop.ipynb` | 1 | perceptron & XOR, activations, pertes, GD, backprop numpy + gradient check, détecteur de scan |
| `02_dnn_gradients_surapprentissage.ipynb` | 1 | anatomie DNN, vanishing/exploding gradients + clipping, Xavier/He, overfitting, données cyber |
| `03_regularisation_normalisation_optimisation.ipynb` | 1 | L1/L2, dropout, early stopping, BatchNorm/LayerNorm, SGD/momentum/RMSProp/Adam, hyperparamètres |
| `04_cnn_cyber.ipynb` | 2 | convolution (stride/padding), pooling, anatomie CNN, paramètres, filtres, mini-CNN numpy sur « malware-images » |
| `05_rnn_lstm_gru_logs.ipynb` | 2 | RNN & BPTT, vanishing séquentiel, LSTM, GRU, fenêtrage de logs, détection d'exfiltration |
| `06_autoencodeurs_anomalies.ipynb` | 2 | AE & goulot, AE linéaire = PCA, denoising, erreur de reconstruction = score d'anomalie, embeddings, AUC |
| `07_rl_fondements_qlearning.ipynb` | 3 | agent/environnement/état/action/récompense, RL vs supervisé, Bellman à la main, ε-greedy/UCB, Q-Learning, SARSA/DQN (concept) |
| `08_rl_cyberdefense_adaptative.ipynb` | 3 | incident en MDP (kill chain), réponse à incidents, récompense multi-objectif, orchestration, évaluation vs baseline, limites du RL |
| `09_robustesse_adversarial_gouvernance.ipynb` | 4 | robustesse/stabilité, biais & dérives, empoisonnement, FGSM (vue défensive), défenses, gouvernance & supervision humaine |
| `10_preparation_projet.ipynb` | — | 15 drills papier + corrigés + rubrique du projet évalué |
| `11_graphiques.ipynb` | — | 8 figures pédagogiques (PNG dans `outputs/`) |

Détail point par point et exercices P1–P7 : voir [`PLAN_NOTEBOOKS.md`](PLAN_NOTEBOOKS.md).

## Organisation en 10 séquences de 3 h 30
Chaque séquence = 3 h 30. 10 séquences couvrent les 4 parties du syllabus (35 h).
Rythme interne type (à adapter selon le topic) :
- **00:00–00:15** (15 min) — raccroche + objectifs de la session
- **00:15–02:30** (135 min) — points 1–6 du notebook : théorie + exemples exécutés,
  **exercice Pn corrigé intégré dans chaque point n** (à refaire sans regarder)
- **02:30–03:00** (30 min) — point 7 (synthèse/récap du topic)
- **03:00–03:30** (30 min) — drill papier (type soutenance) + auto-évaluation

> Matériel par séquence : 1 notebook (`notebooks/`) + 1 script (`code/`) du même
> numéro, exécutés ensemble. Support de séance (déroulé + programme indicatif) :
> `slides/EFREI - DL.pptx`.

### S1 · T01 — Perceptron, descente de gradient, backpropagation (Partie 1)
**Objectifs :** refaire à la main une propagation avant/arrière ; entraîner un MLP numpy.
- 00:00–00:15 · Du ML classique au deep : pourquoi « apprendre les représentations »
- 00:15–01:30 · Points 1–3 : perceptron/XOR, activations, fonctions de perte
- 01:30–02:30 · Points 4–5 : descente de gradient, rétropropagation + gradient check
- 02:30–03:00 · Point 6 : détecteur de scan de ports
- 03:00–03:30 · Point 7 (synthèse) + drill papier (backprop 1 couche à la main)
> **Exercices P1–P7** : intégrés dans chaque point du notebook.

### S2 · T02 — Réseaux profonds : gradients & sur-apprentissage (Partie 1)
**Objectifs :** diagnostiquer vanishing/exploding gradients, choisir une initialisation.
- 00:00–00:15 · Rappels S1 (chaîne de dérivées)
- 01:15… voir notebook : points 1–6 (anatomie DNN, vanishing, exploding/clipping,
  Xavier/He, overfitting, déséquilibre cyber)
- 03:00–03:30 · Synthèse + drill papier (calcul de variance propagée)

### S3 · T03 — Régularisation, normalisation, optimisation (Partie 1)
**Objectifs :** stabiliser et régulariser un réseau ; choisir l'optimiseur.
- Points 1–6 : L1/L2, dropout, early stopping, BatchNorm, Adam, grille d'hyperparamètres
- 03:00–03:30 · Synthèse « quelle config pour quel problème » + drill papier (pas d'Adam)

### S4 · T04 — CNN pour la cybersécurité (Partie 2)
**Objectifs :** calculer une taille de sortie, un compte de paramètres ; lire une cartographie de caractères.
- Points 1–6 : convolution, pooling, anatomie CNN, paramètres, filtres, mini-CNN malware
- 03:00–03:30 · Synthèse + drill papier (taille de sortie, max-pool à la main)

### S5 · T05 — RNN, LSTM, GRU (Partie 2)
**Objectifs :** dérouler un BPTT sur 3 pas ; écrire les équations des portes LSTM.
- Points 1–6 : RNN/BPTT, vanishing séquentiel, LSTM, GRU, fenêtrage de logs, exfiltration
- 03:00–03:30 · Synthèse + drill papier (portes LSTM sur un pas)

### S6 · T06 — Auto-encodeurs & représentations (Partie 2)
**Objectifs :** expliquer pourquoi AE linéaire = PCA ; poser un seuil d'anomalie.
- Points 1–6 : AE, lien PCA, denoising, score d'anomalie/ROC, embeddings, AUC cyber
- 03:00–03:30 · Synthèse + drill papier (reconstruction d'un flux, seuillage)

### S7 · T07 — Fondements du RL (Partie 3)
**Objectifs :** formaliser un problème cyber en MDP ; dérouler Bellman et Q-Learning à la main.
- Points 1–6 : boucle RL, RL vs supervisé, Bellman à la main, ε-greedy/UCB, Q-Learning, SARSA/DQN
- 03:00–03:30 · Synthèse + drill papier (mise à jour Q à la main)

### S8 · T08 — RL pour la cyberdéfense adaptative (Partie 3)
**Objectifs :** évaluer une politique apprised vs une baseline à règles ; critiquer.
- Points 1–6 : MDP incident, réponse à incidents, récompense multi-objectif, orchestration, évaluation, limites
- 03:00–03:30 · Synthèse + revue critique (que ne PAS automatiser en premier)

### S9 · T09 — Robustesse, attaques & gouvernance (Partie 4)
**Objectifs :** mesurer la robustesse d'un détecteur ; attaquer (FGSM) et défendre ; gouverner.
- Points 1–6 : robustesse, biais/dérive, empoisonnement, FGSM, défenses, gouvernance
- 03:00–03:30 · Synthèse + drill papier (perturbation FGSM sur 1 exemple)

### S10 · Préparation projet (100 % de l'évaluation)
**Objectifs :** consolider, s'entraîner au papier, monter le projet.
- 00:00–00:30 · Relecture ciblée des 15 drills du notebook 10 (sans exécuter)
- 00:30–01:30 · **Simulation blanche #1 (60 min, papier)** : `exams/devoir_1.md`
- 01:30–02:00 · Correction + repérage des points faibles (`exams/corrige_devoir_1.md`)
- 02:00–03:00 · **Simulation blanche #2 (60 min, papier)** : `exams/devoir_2.md`
  + entraînement oral du projet (binômes)
- 03:00–03:30 · Simulation graphique (notebook 11) + planning de révision finale

> **Les 4 devoirs blancs** (`exams/devoir_{1..4}.md`, chacun 2 h · 20 points ·
> sans document · A 7 pt / B 7 pt / C 6 pt, corrigés vérifiés dans
> `exams/corrige_devoir_{1..4}.md`) sont à utiliser en hors-séquence pour
> l'entraînement papier : 1 devoir blanc par semaine minimum, à commencer au plus
> tard à la S7. Objectif : ≥ 14/20 sur au moins 2 devoirs avant le rendu du projet.

## Barème et stratégie d'évaluation
- **Projet DL/RL appliqué à la cybersécurité : 100 %** — analyse d'un problème cyber
  complexe, choix et justification des architectures, interprétation critique des
  résultats, analyse des limites et risques.
- → L'évaluation mesure la **maîtrise conceptuelle**, la **rigueur d'analyse** et la
  capacité à relier IA avancée et cyberdéfense.
- → Les devoirs blancs (`exams/`) sont des auto-évaluations quantitatives : ils
  vérifient que vous savez faire les calculs « à la main » qu'une soutenance exige
  de savoir expliquer (backprop, Bellman, tailles CNN, FGSM…).
- → Pour le projet : préparez des exemples à justifier oralement — pourquoi cette
  architecture plutôt qu'une autre, quels résultats sont crédibles, quels risques
  résiduels (biais, adversarial, dérive).

## Bibliographie (ordre de lecture conseillé — livres référencés, non inclus)
1. Goodfellow, Bengio, Courville — *Deep Learning* (mitpress, freely available at
   deeplearningbook.org ; ch. 6, 8, 9, 12, 14 → T01–T04, T06, T09).
2. Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*
   (O'Reilly ; ch. 10, 11, 16, 17 → T02–T03, T05).
3. Sutton & Barto — *Reinforcement Learning: An Introduction* (MIT Press, freely
   available at incompleteideas.net ; ch. 3–6, 16 → T07–T08).
4. Goodfellow, Shafiee, Papernot et al. — *Explaining and Harnessing Adversarial
   Examples* (arXiv:1412.6572 → FGSM, T09).
5. MITRE ATLAS & OWASP Machine Learning Security Top 10 (référentiels cyber → T08–T09).

## Conventions des fichiers python
- Code avec docstrings et commentaires en **anglais** ; prose pédagogique en français.
- Chaque fichier est **auto-exécutable** (`python3 code/0X_....py`) et affiche les
  résultats + des assertions qui vérifient les calculs (marqueur ✅ final).
- Tous les modèles sont en **numpy pur** ; `sklearn`/`scipy` pour les métriques et
  la vérification croisée uniquement. Données synthétiques seedées (Colab-ready).
- Les exercices sont numérotés ; le corrigé est inline avec le bloc
  `# ---- Exercice Pn : ... ----`.

## Suivi de progression
- [ ] S1 · T01 Perceptron, GD, backprop (Partie 1)
- [ ] S2 · T02 Réseaux profonds : gradients & sur-apprentissage (Partie 1)
- [ ] S3 · T03 Régularisation, normalisation, optimisation (Partie 1)
- [ ] S4 · T04 CNN pour la cybersécurité (Partie 2)
- [ ] S5 · T05 RNN, LSTM, GRU (Partie 2)
- [ ] S6 · T06 Auto-encodeurs & représentations (Partie 2)
- [ ] S7 · T07 Fondements du RL (Partie 3)
- [ ] S8 · T08 RL pour la cyberdéfense adaptative (Partie 3)
- [ ] S9 · T09 Robustesse, attaques & gouvernance (Partie 4)
- [ ] S10 · Préparation projet (2 simulations blanches + oral du projet)

## Devoirs blancs (`exams/`)
4 devoirs complets (2 h · 20 points · sans document · A fondements DL 7 pt /
B architectures 7 pt / C RL & robustesse 6 pt), chacun avec son corrigé détaillé.
Toutes les réponses corrigées ont été **vérifiées numériquement**
(numpy/scipy) avant mise en ligne.

| Devoir | Part A (fondements DL) | Part B (architectures) | Part C (RL & robustesse) |
|---|---|---|---|
| `devoir_1.md` | perceptron/XOR, dérivée sigmoid, MSE+gradient, GD 2 pas | conv size/padding, max-pool, portes LSTM à la main | Bellman 3 états, mise à jour Q, ε-greedy |
| `devoir_2.md` | backprop 1 couche cachée (chaîne complète), ReLU, softmax/CE | compte de paramètres CNN, BPTT 3 pas, AE linéaire | Q-Learning 2 itérations, UCB, baseline vs politique |
| `devoir_3.md` | Xavier/He variance, exploding + clipping, dropout scaling | pooling/stride chaîne, GRU vs LSTM (portes), seuil AE/ROC | MDP kill chain, retour d'une politique, récompense multi-objectif |
| `devoir_4.md` | L2 dans le loss, Adam premier pas, BatchNorm sur mini-batch | champ récepteur CNN, vanishing RNN (chaîne Jacobien), AUC | FGSM ε calculé, empoisonnement étiquettes, décision supervisée humaine |

Corrigés : `corrige_devoir_1.md` … `corrige_devoir_4.md`.
