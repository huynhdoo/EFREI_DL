# EFREI_DL — Notebooks

Notebooks du module **Deep Learning & Reinforcement Learning appliqués à la
cybersécurité** (EFREI, 35 h, évaluation = projet 100 %).
Contenu en français ; sorties pré-exécutées dans les `.ipynb` (aucune installation
requise pour la lecture). **Tous les modèles sont implémentés en numpy pur** —
objectif : comprendre les mécanismes, pas utiliser un framework.

## Notebooks

Chaque notebook T01–T09 est organisé en **6 points-sujets** numérotés, chacun avec son
**exercice corrigé Pn** intégré ; le **point 7** est réservé à la **synthèse globale**
(tableau récapitulatif des 6 points + exemple intégrateur réutilisant tous les outils).
Voir le détail dans [`PLAN_NOTEBOOKS.md`](PLAN_NOTEBOOKS.md).

| # | Notebook | Contenu |
|---|---|---|
| 01 | `01_fondements_perceptron_backprop.ipynb` | Perceptron/XOR, activations, pertes, GD, backprop numpy + gradient check, détecteur de scan |
| 02 | `02_dnn_gradients_surapprentissage.ipynb` | Anatomie DNN, vanishing/exploding + clipping, Xavier/He, overfitting, données cyber |
| 03 | `03_regularisation_normalisation_optimisation.ipynb` | L1/L2, dropout, early stopping, BatchNorm/LayerNorm, momentum/RMSProp/Adam, hyperparamètres |
| 04 | `04_cnn_cyber.ipynb` | Convolution, pooling, anatomie CNN, paramètres, filtres, mini-CNN « malware-images » |
| 05 | `05_rnn_lstm_gru_logs.ipynb` | RNN/BPTT, vanishing séquentiel, LSTM, GRU, fenêtrage de logs, exfiltration |
| 06 | `06_autoencodeurs_anomalies.ipynb` | AE & goulot, AE linéaire = PCA, denoising, score d'anomalie/ROC, embeddings, AUC |
| 07 | `07_rl_fondements_qlearning.ipynb` | MDP, RL vs supervisé, Bellman à la main, ε-greedy/UCB, Q-Learning, SARSA/DQN (concept) |
| 08 | `08_rl_cyberdefense_adaptative.ipynb` | Incident en MDP (kill chain), réponse à incidents, multi-objectif, orchestration, limites du RL |
| 09 | `09_robustesse_adversarial_gouvernance.ipynb` | Robustesse, biais & dérives, empoisonnement, FGSM (vue défensive), défenses, gouvernance |
| 10 | `10_preparation_projet.ipynb` | 15 drills « papier » du projet avec corrigés pas-à-pas + rubrique d'évaluation |
| 11 | `11_graphiques.ipynb` | 8 figures pédagogiques (perceptron, vanishing, GD, overfitting, CNN, LSTM, ROC, Q-Learning) |

## Documents de référence (`docs/`)

| Fichier | Contenu |
|---|---|
| `Syllabus_DL_RL_Cybersecurite.pdf` | Syllabus officiel du module (35 h, 4 parties) |

> *Note : les livres de référence (Goodfellow *Deep Learning*, Géron, Sutton & Barto)
> ne sont pas inclus (droit d'auteur) ; ils sont référencés dans les notebooks par
> leurs chapitres (`*(réf. ...)*`). Référentiels cyber cités : MITRE ATLAS,
> OWASP ML Security Top 10.*

## Fiches, cas d'étude & supports (`courses/`, `exams/`, `slides/`)

- `slides/EFREI - DL.pptx` — support de séance : déroulé des 10 séances,
  programme indicatif par topic (T01–T09 + préparation) et rubrique du projet
  évalué (miroir de `EFREI/MATHS/slides/EFREI - MATHS.pptx`).
- `courses/T0N_synthese_*.docx` — fiches de synthèse P1–P6 de chaque topic
  (à quoi ça sert / calcul manuel / code Python).
- `courses/T0N_projet_*.docx` — cas d'étude cyber complet, calculs détaillés étape par étape.
- `exams/devoir_{1..4}.md` + `exams/corrige_devoir_{1..4}.md` — 4 devoirs blancs
  d'auto-évaluation (2 h · 20 pts) avec corrigés vérifiés numériquement.

## Exécuter

- **Local** : `jupyter notebook notebooks/`
- **Google Colab** : `File → Open notebook from → Google Drive` (dossier `EFREI/DL/notebooks`)
- **Scripts miroirs** : `python3 code/01_fondements_perceptron_backprop.py` (auto-vérifiés)

## Dépendances
Python 3.10+ · `numpy` · `scipy` · `pandas` · `sklearn` · `matplotlib`
(pas de PyTorch/TensorFlow : tout est implémenté en numpy pur)

---
*Généré le 09/11/2026 — Module DL & RL appliqués à la cybersécurité 2025-2026.*
