#!/usr/bin/env python3
"""Mirror script for notebook T08 — RL pour la cyberdéfense adaptative.

EFREI / DL module — Deep Learning & RL appliqués à la cybersécurité.
Pure numpy (no torch, no gymnasium, no network): the incident-response MDP
(kill chain) and every agent are implemented from scratch with seeded rngs.

The notebook builder (code/_build_nb_08.py) extracts the ``# %% TAG`` chunks
below and reuses them verbatim as the notebook code cells, and the
``# %% [markdown]`` comment-prefixed chunks as markdown cells (single source
of truth between script and notebook).
"""
import numpy as np

# %% [markdown]
# # T08 · RL pour la cyberdéfense adaptative — DL/RL appliqués à la cybersécurité
#
# **Objectifs du cours :**
# - Modéliser un incident de sécurité (kill chain) comme un processus de Markov décisionnel (MDP) et y ancrer récompenses et transitions
# - Entraîner en numpy pur un agent Q-Learning de réponse à incident et le comparer à des baselines à règles et à une orchestration par playbooks
# - Évaluer rigoureusement une politique (retours, variance inter-graines) et reconnaître les risques propres au RL en cyber (reward hacking, adversaire adaptatif, sim2real)
#
# **Pourquoi c'est fondamental en cyber :**
#
# | Besoin cyber | Outil |
# |---|---|
# | Réponse à incident automatique et calibrée | Politique Q-Learning sur MDP kill chain (P1, P2) |
# | Arbitrage sécurité vs disponibilité du SI | Récompense multi-objectif, balayage α (P3) |
# | SOAR : exécuter les bons playbooks au bon étage | Orchestration par étages avec coûts (P4) |
# | Prouver qu'un agent vaut mieux qu'une règle | Retours, variance, baselines (P5, P7) |
# | Éviter les faux sentiment de sécurité | Reward hacking, adversaire adaptatif (P6) |
#
# **Prérequis :** T07 (MDP, équations de Bellman, Q-Learning tabulaire),
# T01 (descente de gradient, pour les variantes profondes), bases de cybersécurité (kill chain).
#
# > **Conseil d'étude :** refaites l'exercice P2 (majoration stricte de la baseline
# > et accord avec l'optimum de l'itération de valeur) — c'est la question type
# > soutenance : « comment savez-vous que votre agent est meilleur ? »

# %% [markdown]
# ## 📌 Bloc 0 — Environnement & imports
# Installez le stack si besoin (déjà pré-installé sur Colab).
# Le notebook contient des sorties pré-calculées : relisibles sans rien exécuter,
# ou « Run All » pour reproduire (numpy pur, graines figées, aucune donnée réseau).

# %% [markdown]
# ## 1 · Modéliser un incident en MDP (kill chain simplifiée)
#
# Un incident est une **course de l'attaquant le long de la kill chain** :
#
# ```
# Sécurisé → Reconnaissance → Intrusion → Mouvement latéral → Exfiltration
#    s = 0          1                2               3                 4
# ```
#
# L'agent (le SOC) choisit à chaque pas une action parmi
# `Surveiller (1) / Isoler (3) / Bloquer (5) / Réinitialiser (8)` (coûts d'exploitation).
# Récompense par pas : `r = −coût(a) − 2·s′` — on paie l'action **et** la gravité
# atteinte : laisser l'adversaire monter d'un étage coûte cher.
#
# **Piège classique :** oser un **état terminal** là où il n'y en a pas — après
# réinitialisation, l'attaquant revient. Sans terminal, on évalue sur horizon borné
# (contrainte opérationnelle : le SOC arbitre en continu).

# %% p1-mdp
# --- Kill-chain MDP for automated incident response.
# States: 0 Securise, 1 Reconnaissance, 2 Intrusion, 3 Mouvement lateral,
#         4 Exfiltration.
# Actions: 0 Surveiller, 1 Isoler, 2 Bloquer, 3 Reinitialiser (with costs).
# Reward per step: r = -cost(a) - 2 * s'  (downtime + stage penalty).
STAGES = ["Securise", "Reconnaissance", "Intrusion", "Lateral", "Exfiltration"]
ACTIONS = ["Surveiller", "Isoler", "Bloquer", "Reinitialiser"]
COSTS = np.array([1.0, 3.0, 5.0, 8.0])


class IncidentResponseMDP:
    """Small tabular MDP: responder (agent) vs probabilistic kill chain."""

    def __init__(self, p_escalate=0.4, reward_mode="true"):
        self.n_states, self.n_actions = 5, 4
        self.p_escalate = p_escalate
        self.reward_mode = reward_mode  # 'true' or 'flawed' (hacking demo)

    def transition(self, s, a, rng):
        """Sample the next state from P(. | s, a) with a seeded rng."""
        if a == 0:      # surveiller: cheap, but the attacker keeps progressing
            return min(s + 1, 4) if rng.random() < self.p_escalate else s
        if a == 1:      # isoler: containment at 65 %, blocks progression
            return 0 if (s >= 1 and rng.random() < 0.65) else s
        if a == 2:      # bloquer: strong containment at 85 %
            return 0 if (s >= 1 and rng.random() < 0.85) else s
        # reinitialiser: near-perfect recovery at 95 %
        return 0 if rng.random() < 0.95 else s

    def reward(self, s, a, s2):
        if self.reward_mode == "flawed":
            # BUG to demo: 'block' farms safety points, stage penalty removed.
            return 1.0 if a == 2 else -2.0 * s2
        return -COSTS[a] - 2.0 * s2


env = IncidentResponseMDP()
rng = np.random.default_rng(7)
s = 3  # mid-incident: lateral movement
for a in [0, 1, 2, 3]:
    s2 = env.transition(s, a, rng)
    r = env.reward(s, a, s2)
    print(f"etat {STAGES[s]:15s} action {ACTIONS[a]:13s} -> etat {STAGES[s2]:15s} "
          f"recompense {r:+.1f} (= -{COSTS[a]:.0f} - 2x{s2})")
print("Aucun etat terminal : episodes a horizon borne (contrainte operationnelle).")

# %% [markdown]
# ### Exercice corrigé
#
# **P1 — Décomposer la récompense à la main.** Vérifiez sur 30 tirages par action
# que `r = −coût(a) − 2·s′` exactement ; vérifiez les transitions seedées
# (graine 13) et le taux de containment du reset ≈ 0.95 (Monte-Carlo tolérant).

# %% p1-exo
# ---- Exercice P1 : recompense decomposable a la main + transitions seedees ----
# Hand-computable reward decomposition for ANY sampled s': r = -cost(a) - 2*s'.
rng = np.random.default_rng(11)
for a in range(4):
    for _ in range(30):
        s, s2 = 2, env.transition(2, a, rng)
        assert env.reward(s, a, s2) == -COSTS[a] - 2.0 * s2, (a, s2)
# Probed seeded transitions (fixed seed 13, from lateral movement s=3):
rng = np.random.default_rng(13)
samples = [env.transition(3, 1, rng) for _ in range(5)]
assert samples == [3, 3, 3, 0, 0], f"isol@3 : {samples}"
rng = np.random.default_rng(13)
assert env.transition(4, 3, rng) == 0          # reset from exfiltration succeeds
# Empirical reset success rate ~ 0.95 (probed, tolerant MC bound):
rng = np.random.default_rng(21)
cont = np.mean([env.transition(4, 3, rng) == 0 for _ in range(4000)])
assert abs(cont - 0.95) < 0.02, f"taux containment reset {cont}"
print(f"Exercice P1 verifie : r = -cout - 2*s' (main), reset 95 % (MC {cont:.3f}).")

# %% [markdown]
# ## 2 · Réponse automatique aux incidents (Q-Learning)
#
# Le SOC apprend, **sans aucune étiquette d'incident**, quelle action répondre à
# quel étage : Q-Learning tabulaire (`α = 0.2`, `γ = 0.9`, ε décroissant) sur
# l'MDP du point 1. Référence exacte : **itération de valeur** sur le modèle —
# l'agent convergé doit retrouver la politique gloutonne de V*.
#
# **Piège classique :** comparer des retours évalués avec des graines différentes
# ou sur des horizons différents — l'ordre des politiques s'inverse sous le bruit.
# Ici tout est à graine figée et horizon 20 pour tous.

# %% p2-ql
# --- Tabular Q-Learning responder on the true reward (fixed gamma).
GAMMA, ALPHA = 0.9, 0.2
H = 25  # episode horizon (no terminal states)


def train_q(env, seed, episodes=4000, eps0=0.3, eps1=0.02):
    """Tabular Q-Learning with annealed epsilon; returns the Q table."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    for ep in range(episodes):
        eps = eps0 + (eps1 - eps0) * ep / max(episodes - 1, 1)
        s = int(rng.integers(env.n_states))
        for _ in range(H):
            a = (int(rng.integers(env.n_actions)) if rng.random() < eps
                 else int(np.argmax(Q[s])))
            s2 = env.transition(s, a, rng)
            r = env.reward(s, a, s2)
            Q[s, a] += ALPHA * (r + GAMMA * Q[s2].max() - Q[s, a])
            s = s2
    return Q


def eval_policy(env, act, seed, episodes=400, horizon=20):
    """Mean undiscounted return of action_fn act(s) (and return list)."""
    rng = np.random.default_rng(seed)
    returns = []
    for _ in range(episodes):
        s, G = int(rng.integers(env.n_states)), 0.0
        for _ in range(horizon):
            a = act(s)
            s2 = env.transition(s, a, rng)
            G += env.reward(s, a, s2)
            s = s2
        returns.append(G)
    return float(np.mean(returns)), np.array(returns)


Q_resp = train_q(env, seed=42)
greedy = lambda s: int(np.argmax(Q_resp[s]))
mon_ret, _ = eval_policy(env, lambda s: 0, seed=1)
soc_ret, _ = eval_policy(env, lambda s: 0 if s <= 1 else 1, seed=1)  # SOC rule
_rng_rand = np.random.default_rng(1234)  # uniform policy actions (cell-local rng)
rnd_ret, _ = eval_policy(env, lambda s: int(_rng_rand.integers(4)), seed=1)
resp_ret, resp_returns = eval_policy(env, greedy, seed=1)
print(f"Retour moyen (20 pas, 400 episodes, graine fixe) :")
print(f"  toujours surveiller : {mon_ret:+8.1f}")
print(f"  regle SOC (isol@>=2): {soc_ret:+8.1f}")
print(f"  agent Q-Learning    : {resp_ret:+8.1f}   <- appris sans etiquettes")

# Reference exacte : iteration de valeur (gamma=0.9)
def q_star_vals(s, a, V, gamma=GAMMA):
    """Expected value of (s, a) under the MDP model: sum P [r + gamma V]."""
    if a == 0:
        outs = [(min(s + 1, 4), env.p_escalate), (s, 1.0 - env.p_escalate)]
    elif a == 1:
        outs = [(0, 0.65), (s, 0.35)] if s >= 1 else [(0, 1.0)]
    elif a == 2:
        outs = [(0, 0.85), (s, 0.15)] if s >= 1 else [(0, 1.0)]
    else:  # reset
        outs = [(0, 0.95), (s, 0.05)]
    return sum(pv * (-COSTS[a] - 2.0 * ns + gamma * V[ns]) for ns, pv in outs)


def value_iteration(env, gamma=GAMMA, iters=900):
    V = np.zeros(env.n_states)
    for _ in range(iters):
        V_new = np.array([max(q_star_vals(s, a, V, gamma) for a in range(4))
                          for s in range(env.n_states)])
        if np.max(np.abs(V_new - V)) < 1e-10:
            V = V_new
            break
        V = V_new
    return V


V_star = value_iteration(env)
pi_star = [int(np.argmax([q_star_vals(s, a, V_star) for a in range(4)]))
           for s in range(5)]
print("Politique gloutonne de V* : ", [ACTIONS[a] for a in pi_star])
print("Politique apprise Q        :", [ACTIONS[a] for a in [greedy(s) for s in range(5)]])

# %% [markdown]
# ### Exercice corrigé
#
# **P2 — L'agent bat la baseline.** Mettez à jour un Q à la main
# (Q=−4, r=−8, γ=0.9, max Q(s′)=−6 → **−5.88**) ; vérifiez les ordres stricts
# `agent > règle SOC > surveiller` et l'accord de la politique apprise avec
# l'optimum exact de l'itération de valeur sur les étages d'incident.

# %% p2-exo
# ---- Exercice P2 : la politique apprise bat strictement le baseline a regles ----
# Single hand-computed Q update (probed inputs): Q= -4, r = -8, gamma=0.9, max Q(s')= -6
q_new = -4.0 + ALPHA * ((-8.0) + GAMMA * (-6.0) - (-4.0))
assert abs(q_new - (-5.88)) < 1e-12, f"maj Q {q_new}"   # -4 + 0.2*(-8 - 5.4 + 4) = -5.88
# Strict improvement over the handcrafted baseline (fixed seed, probed margins):
assert resp_ret > soc_ret + 2.0, f"resp {resp_ret} vs soc {soc_ret}"
assert soc_ret > mon_ret, f"soc {soc_ret} vs mono {mon_ret}"
# Learned greedy policy must match the exact VI optimum on the incident stages.
assert [greedy(s) for s in (2, 3, 4)] == pi_star[2:5], (greedy(2), pi_star[2])
assert sum(greedy(s) == pi_star[s] for s in range(5)) >= 4
print(f"Exercice P2 verifie : Q maj main {q_new}, agent {resp_ret:+.1f} > regle {soc_ret:+.1f} > surveiller {mon_ret:+.1f}.")

# %% [markdown]
# ## 3 · Adaptation dynamique des défenses (récompense multi-objectif)
#
# Le SI doit rester **disponible** : on pondère gravité et coût,
# `r_α = −coût(a) − 2·α·s′`. Balayer α, c'est balayer l'**arbitrage sécurité /
# disponibilité** : α grand ⇒ confinement agressif (downtime), α petit ⇒
# surveillance souple (risque). Le front obtenu est un front de Pareto approché.
#
# **Piège classique :** croire qu'un réglage « optimal » existe sans exprimer
# la préférence : le RL n'arbitre pas à votre place, il **optimise la récompense
# que vous avez écrite** — l'α est une décision du RSSI, pas de l'algorithme.

# %% p3-pareto
# --- Adaptive defense: multi-objective reward  r_a = -cost(a) - 2*alpha*s'.
# alpha high  -> security first (aggressive containment, downtime grows)
# alpha low   -> service first (monitor, risk grows)  => Pareto-ish sweep.
def sweep(alpha, seed=5, episodes=2500):
    class EnvA(IncidentResponseMDP):
        pass
    e = EnvA()

    def reward(s, a, s2):
        return -COSTS[a] - 2.0 * alpha * s2
    e.reward = reward
    Q = train_q(e, seed=seed, episodes=episodes)
    pol = lambda s: int(np.argmax(Q[s]))
    rng = np.random.default_rng(77)
    stages, acts = [], []
    for _ in range(300):
        s = int(rng.integers(5))
        for _ in range(20):
            a = pol(s)
            acts.append(a)
            s = e.transition(s, a, rng)
        stages.append(s)
    security = -float(np.mean(stages))          # higher = safer
    availability = float(np.mean(np.array(acts) == 0))  # only 'surveiller' is free
    return Q, security, availability, float(np.mean(np.array(acts) == 2))


sweep_rows = []
for alpha in [0.0, 0.2, 0.5, 0.8, 1.0]:
    _, sec, avail, blockf = sweep(alpha)
    sweep_rows.append((alpha, sec, avail, blockf))
    print(f"alpha={alpha:.1f}  securite(-stage final)={sec:+.2f}  "
          f"disponibilite={avail:.3f}  part bloquer={blockf:.3f}")

# %% [markdown]
# ### Exercice corrigé
#
# **P3 — Front Pareto à la main.** Comparez α = 0 et α = 1 : la sécurité doit
# progresser (> +0.5) **en payant** la disponibilité (> −0.2) ; vérifiez la
# décomposition de `r_α` sur deux cas calculés.

# %% p3-exo
# ---- Exercice P3 : balayage Pareto — securite vs disponibilite ----
a0, a1 = sweep_rows[0], sweep_rows[-1]
assert a1[1] > a0[1] + 0.5, f"securite alpha=1 {a1[1]} vs alpha=0 {a0[1]}"
assert a1[2] < a0[2] - 0.2, f"dispo alpha=1 {a1[2]} vs alpha=0 {a0[2]}"
# Reward decomposition hand-check at alpha=1: r(s=3, bloquer -> 0) = -5 - 0 = -5
assert -COSTS[2] - 2.0 * 1.0 * 0 == -5.0
assert -COSTS[0] - 2.0 * 1.0 * 4 == -9.0
print(f"Exercice P3 verifie : arbitrage securite {a0[1]:+.2f}->{a1[1]:+.2f} / "
      f"dispo {a0[2]:.2f}->{a1[2]:.2f} (Pas de domination totale : compromis).")

# %% [markdown]
# ## 4 · Orchestration intelligente (séquences de playbooks, coûts)
#
# Un **playbook** est une macro d'actions (SOAR) : plus cher que la somme des
# actions unitaires (markup de séquence, ≈ +15 %) mais **mieux contenant** :
#
# | Playbook | Séquence | Coût (markup 1.15×) | Containment |
# |---|---|---|---|
# | P-rapide | surveiller + isoler | 1.35 × (1+3) = 5.4 | 0.50 |
# | P-contenement | isoler + bloquer | 1.5 × (3+5) = 12 | 0.97 |
#
# Tout l'enjeu : **quel playbook, à quel étage ?** — l'orchestration par étages
# bat la surveillance naïve, mais l'application systématique du playbook lourd
# est catastrophique (le markup ruine l'utilité).
#
# **Piège classique :** empiler des playbooks « au cas où ». Le coût d'une
# séquence ne se rembourse que si elle **évite des étages chers** — sinon la
# baseline à règles bon marché gagne.

# %% p4-playbook
# --- Intelligent orchestration: playbooks = fixed action macros with cost
# markup but higher containment probability.
#   P-contenement = [isoler, bloquer] : cout 1.5*(3+5)=12, containment 0.97
#   P-rapide      = [surveiller, isoler]: cout 1.35*(1+3)=5.4, containment 0.50
PB_COST = {"P-contenement": 1.5 * (3.0 + 5.0), "P-rapide": 1.35 * (1.0 + 3.0)}
PB_CONT = {"P-contenement": 0.97, "P-rapide": 0.50}


def playbook_step(pb, s, rng):
    return 0 if (s >= 1 and rng.random() < PB_CONT[pb]) else s

# Hand-computable expected immediate reward from lateral movement (s=3):
r_pc_expected = -PB_COST["P-contenement"] - 2.0 * (0.03 * 3)  # -12 - 0.18
print(f"E[r | s=3, P-contenement] calcule a la main = {r_pc_expected:.2f}")
rng = np.random.default_rng(31)
mc = np.mean([-PB_COST["P-contenement"] - 2.0 * playbook_step("P-contenement", 3, rng)
              for _ in range(3000)])
print(f"E[r | s=3, P-contenement] Monte-Carlo (3000 tirages) = {mc:.2f}")

# Three orchestration policies on the true reward (fixed eval seed):
#   naive      : surveiller partout, reinitialiser seulement en exfiltration
#   staged     : surveiller s<=1, P-rapide en s=2, P-contenement a partir de s=3
#   wasteful   : P-contenement applique partout (sur-orchestration = gaspillage)
def eval_orchestration(rule, seed=1, episodes=400, horizon=20):
    rng = np.random.default_rng(seed)
    rets = []
    for _ in range(episodes):
        s, G = int(rng.integers(5)), 0.0
        for _ in range(20):
            kind = rule(s)
            if kind == "mon":
                s2 = env.transition(s, 0, rng)
                G += env.reward(s, 0, s2)
            elif kind == "reset":
                s2 = env.transition(s, 3, rng)
                G += env.reward(s, 3, s2)
            else:
                s2 = playbook_step(kind, s, rng)
                G += -PB_COST[kind] - 2.0 * s2
            s = s2
        rets.append(G)
    return float(np.mean(rets))


naive_ret = eval_orchestration(lambda s: "mon" if s < 4 else "reset")
pb_ret = eval_orchestration(lambda s: "mon" if s <= 1
                            else ("P-rapide" if s == 2 else "P-contenement"))
wasteful_ret = eval_orchestration(lambda s: "P-contenement")
print(f"Retour 'naif' (surveiller, reset en exfiltration)        : {naive_ret:+.1f}")
print(f"Retour ORCHESTRE (P-rapide s=2, P-contenement s>=3)      : {pb_ret:+.1f}")
print(f"Retour sur-orchestre (P-contenement partout)             : {wasteful_ret:+.1f}")
print(f"Retour regle SOC simple (isol@>=2)                       : {soc_ret:+.1f}")

# %% [markdown]
# ### Exercice corrigé
#
# **P4 — Playbooks : Monte-Carlo vs calcul manuel.** Le MC (3000 tirages) doit
# retrouver `E[r | s=3, P-contenement] = −12.18` à ±0.10 ; vérifiez le markup
# des coûts de séquence, et que l'orchestration par étages domine la politique
# naïve de plus de 10 points tout en restant très loin du « P-contenement partout ».

# %% p4-exo
# ---- Exercice P4 : playbooks — MC vs calcul manuel, couts de sequence ----
assert abs(mc - r_pc_expected) < 0.10, f"MC {mc} vs main {r_pc_expected}"
assert abs(r_pc_expected - (-12.18)) < 1e-12
# Sequence cost markup penalises over-use: playbook cost > sum of raw costs
assert PB_COST["P-contenement"] > COSTS[1] + COSTS[2]
assert PB_COST["P-rapide"] > COSTS[0] + COSTS[1]
# Staged orchestration beats the naive 'watch then reset' policy (probed margin),
# and blanket playbook application is catastrophic (the cost-markup lesson).
assert pb_ret > naive_ret + 10.0, f"orchestre {pb_ret} vs naif {naive_ret}"
assert wasteful_ret < pb_ret - 100.0, f"gaspillage {wasteful_ret} vs orchestre {pb_ret}"
print(f"Exercice P4 verifie : MC {mc:.2f} == main {r_pc_expected:.2f}, orchestre "
      f"{pb_ret:+.1f} > naif {naive_ret:+.1f} >> sur-orchestre {wasteful_ret:+.1f}.")

# %% [markdown]
# ## 5 · Évaluer une politique (retours, variance, baseline à règles)
#
# Une politique se juge sur sa **distribution de retours** : moyenne, percentiles
# (p5/p50/p95) et **variance inter-graines d'entraînement** — un agent brillant
# sur une graine et instable sur les autres n'est pas déployable. Baselines
# obligatoires : toujours surveiller, uniforme, règle SOC à règles explicites.
#
# **Piège classique :** annoncer un unique chiffre sans graine ni intervalle —
# ici chaque comparaison est répétée sur 8 graines et rendue publique (écart-type).

# %% p5-eval
# --- Policy evaluation: return distributions and seed-to-seed variance.
learned_means, rule_means = [], []
for seed in range(8):
    Qs = train_q(env, seed=100 + seed, episodes=2000)
    m, _ = eval_policy(env, lambda s, Q=Qs: int(np.argmax(Q[s])), seed=2000 + seed)
    learned_means.append(m)
learned_means = np.array(learned_means)
rule_means = np.array([eval_policy(env, lambda s: 0 if s <= 1 else 1, seed=3000 + s)[0]
                       for s in range(8)])
_rng_rand5 = np.random.default_rng(4321)  # uniform policy actions (cell-local rng)
rand_ret, rand_returns = eval_policy(env, lambda s: int(_rng_rand5.integers(4)), seed=5)
print(f"Evalue sur 8 graines d'entree (entrainement + evaluation) :")
print(f"  Q-Learning : moyenne {learned_means.mean():+.1f}  ecart-type inter-graines {learned_means.std():.2f}")
print(f"  regle SOC  : moyenne {rule_means.mean():+.1f}  ecart-type {rule_means.std():.2f}")
pcts = np.percentile(resp_returns, [5, 50, 95])
print(f"  distribution des retours Q (graine 1) : p5 {pcts[0]:+.0f} | mediane {pcts[1]:+.0f} | p95 {pcts[2]:+.0f}")

# %% [markdown]
# ### Exercice corrigé
#
# **P5 — Ordonner et borner.** Vérifiez `Q > règle > random` sur 8 graines,
# la variance inter-graines < 5 (borne sondée) et la médiane des retours de
# l'agent supérieure au retour de « toujours surveiller ».

# %% p5-exo
# ---- Exercice P5 : evaluer une politique — ordres stricts, variance bornee ----
assert learned_means.mean() > rule_means.mean() + 2.0
assert rule_means.mean() > rand_ret, f"regle {rule_means.mean()} vs random {rand_ret}"
assert learned_means.std() < 5.0, f"variance inter-graines {learned_means.std()}"  # probed 3.7
assert pcts[1] > mon_ret  # median of learned beats always-monitor
print(f"Exercice P5 verifie : Q {learned_means.mean():+.1f} > regle {rule_means.mean():+.1f} > random {rand_ret:+.1f}, sd graines {learned_means.std():.2f}.")

# %% [markdown]
# ## 6 · Limites & risques du RL en cyber réel
#
# Deux pièges qui font échouer un déploiement :
#
# 1. **Reward hacking** — si la récompense est mal spécifiée (compter les
#    « bloquer » au lieu de la gravité réelle), l'agent **exploite le score** :
#    bloquer tout le temps, disponibilité 0 %, note impeccable, désastre réel.
# 2. **Adversaire adaptatif** — l'attaquant observe la défense et accélère là où
#    on ne regarde pas : la politique figée sur une simulation devient
#    sous-optimale dès que l'adversaire change de stratégie (sim2real).
#
# **Piège classique :** valider en simulation statique puis conclure « prêt
# pour la production ». La simulation est un **test d'ordre de grandeur**, pas
# un contrat de performance.

# %% p6-risk
# --- Limits & risks. (1) REWARD HACKING: on the flawed reward (+1 per 'block',
# no stage penalty), the agent farms points by blocking everything: it looks
# perfect on the flawed score but destroys availability and the TRUE utility.
env_flawed = IncidentResponseMDP(reward_mode="flawed")
Q_hack = train_q(env_flawed, seed=42)
hack_pol = lambda s: int(np.argmax(Q_hack[s]))
# Block fraction under the flawed policy (fixed eval seed):
rng = np.random.default_rng(55)
acts = []
for _ in range(300):
    s = int(rng.integers(5))
    for _ in range(20):
        a = hack_pol(s)
        acts.append(a)
        s = env_flawed.transition(s, a, rng)
block_frac = float(np.mean(np.array(acts) == 2))
availability_hack = float(np.mean(np.isin(acts, [0, 1])))
# True utility of the hacked policy (real reward function):
hack_true, _ = eval_policy(env, hack_pol, seed=1)
print(f"Reward hacking : part bloquer = {block_frac:.3f}, disponibilite = {availability_hack:.3f},")
print(f"  utilite REELLE {hack_true:+.1f} vs agent bien reward {resp_ret:+.1f} -> score flatte, reel desastre.")

# (2) ADVERSARIAL ADAPTIVE ATTACKER: escalation speed rises when the defender
# rarely monitors (blocked/contained channels push the attacker to new paths).
def eval_adaptive(act, seed, episodes=400, horizon=20):
    rng = np.random.default_rng(seed)
    # first pass: measure monitor share to set the attacker escalation proba
    acts = []
    s = int(rng.integers(5))
    for _ in range(2000):
        a = act(s)
        acts.append(a)
        s = IncidentResponseMDP().transition(s, a, rng)
    mon_share = float(np.mean(np.array(acts) == 0))
    p_ad = 0.25 + 0.6 * (1.0 - mon_share)
    env_ad = IncidentResponseMDP(p_escalate=p_ad)
    returns = []
    for _ in range(episodes):
        s, G = int(rng.integers(5)), 0.0
        for _ in range(horizon):
            a = act(s)
            s2 = env_ad.transition(s, a, rng)
            G += env_ad.reward(s, a, s2)
            s = s2
        returns.append(G)
    return float(np.mean(returns)), p_ad


static_ret, _ = eval_policy(env, greedy, seed=1)
adapt_ret, p_ad = eval_adaptive(greedy, seed=1)
print(f"Attaquant adaptatif : p_escalade passe a {p_ad:.2f} -> retour {static_ret:+.1f} -> {adapt_ret:+.1f}")

# %% [markdown]
# ### Exercice corrigé
#
# **P6 — Détecter hacking et adversaire.** L'agent « hacké » doit bloquer > 95 %
# du temps avec disponibilité < 5 % et une utilité réelle effondrée (−20 pts vs
# l'agent bien récompensé) ; face à l'adversaire adaptatif (p_escalade piloté
# par notre négligence de la surveillance), le retour doit baisser.

# %% p6-exo
# ---- Exercice P6 : detection du reward hacking et de l'adversaire adaptatif ----
assert block_frac > 0.95, f"part bloquer hack {block_frac}"
assert availability_hack < 0.05, f"disponibilite hack {availability_hack}"
assert hack_true < resp_ret - 20.0, f"utilite hack {hack_true} vs vrai {resp_ret}"
assert adapt_ret < static_ret, f"adapt {adapt_ret} vs static {static_ret}"
assert p_ad > env.p_escalate, f"p_ad {p_ad}"
print(f"Exercice P6 verifie : hacking bloque {block_frac:.0%} (dispo {availability_hack:.0%}), "
      f"retour adversarial {adapt_ret:+.1f} < {static_ret:+.1f}.")

# %% [markdown]
# ## 7 · Synthèse globale — l'essentiel & exemple intégrateur
#
# ### Ce qu'il faut retenir (les 6 points)
#
# | # | Point | L'essentiel |
# |---|---|---|
# | 1 | MDP kill chain | étages × actions coûtées ; r = −coût(a) − 2·s′ ; horizon borné |
# | 2 | Q-Learning réponse | appris sans étiquette ; accord avec l'optimum de l'itération de valeur |
# | 3 | Multi-objectif | α code l'arbitrage sécurité/dispo ; front de Pareto approché |
# | 4 | Playbooks | macros à markup : les employer PAR ÉTAGE, jamais partout |
# | 5 | Évaluation | moyenne + percentiles + variance 8 graines vs baselines |
# | 6 | Risques | reward hacking & adversaire adaptatif : la simulation ne suffit pas |
#
# **Exemple intégrateur — tableau de bord complet.** Sur l'utilité réelle
# (horizon 20, graine fixe), on confronte d'un même tableau : l'agent Q-Learning,
# la règle SOC, l'orchestration par playbooks, la surveillance naïve, « toujours
# surveiller » et l'agent hacked par mauvaise récompense. Le classement final
# est la conclusion du notebook.

# %% p7-integration
# --- Integrator: full scoreboard on the TRUE utility (fixed eval seed).
scoreboard = {
    "Q-Learning (vraie recompense)": resp_ret,
    "Regle SOC (monitor puis isoler)": soc_ret,
    "Playbooks orchestres (par etage)": pb_ret,
    "Surveillance naive puis reset": naive_ret,
    "Toujours surveiller": mon_ret,
    "Q-Learning hacke (mauvaise recompense)": hack_true,
}
print("Tableau final — utilite reelle (horizon 20, graine fixe) :")
for name, v in sorted(scoreboard.items(), key=lambda kv: -kv[1]):
    print(f"  {name:36s} {v:+8.1f}")

# %% [markdown]
# ### Exercice corrigé
#
# **P7 — Classement complet.** Vérifiez que le tableau est trié décroissant et
# les marges : `Q > règle + 2`, `règle > orchestration` (le playbook paie son
# markup), `orchestration > naïf + 10`, `naïf > surveiller`, et l'agent hacked
# sous la règle naïve.

# %% p7-exo
# ---- Exercice P7 : classement complet — l'agent appris domine les baselines ----
vals = list(scoreboard.values())
assert vals[0] == max(vals), "le tableau doit etre trie decroissant"
assert resp_ret > soc_ret + 2.0
assert soc_ret > pb_ret, "un playbook fixe paie le markup : la regle simple tient"
assert pb_ret > naive_ret + 10.0, f"orchestre {pb_ret} vs naif {naive_ret}"
assert naive_ret > mon_ret, f"naif {naive_ret} vs surveiller {mon_ret}"
assert hack_true < soc_ret  # the hacked agent is worse than a naive rule
print(f"Exercice P7 verifie : Q {resp_ret:+.1f} > regle {soc_ret:+.1f} > orchestre "
      f"{pb_ret:+.1f} > naif {naive_ret:+.1f} > surveiller {mon_ret:+.1f} (hacke {hack_true:+.1f}).")

# %% marker
print("08_rl_cyberdefense_adaptative — exercices vérifiés. ✅")
print(f"NB 08 — tous les exercices sont vérifiés. ✅")
