"""Diagnòstic de sobreestimació del crític (apartat 4: discussió basada en mesures).

Compara la predicció Q1(s_t, μ(s_t)) amb el retorn descomptat observat
G_t = Σ_k γ^k r_{t+k} als episodis d'avaluació. Només fa forwards de les xarxes:
no afegeix cap pas d'entorn.
"""
import numpy as np
import torch

from experiments.evaluation import EpisodeTrace


def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """G_t per a cada t d'un episodi. :param rewards: [T]. :return: [T]."""
    returns = np.empty(len(rewards), dtype=np.float64)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def valid_mask(trace: EpisodeTrace, min_horizon: int) -> np.ndarray:
    """Estats on G_t és fiable. :return: [T] booleà.

    En un episodi truncat, G_t s'ha tallat: a prop del final no inclou la cua que
    el crític sí que estima, i el biaix mesurat seria fals. Per això s'exclouen els
    estats amb menys de min_horizon passos per davant (error residual ≤ γ^h·|V|).
    """
    length = len(trace.rewards)
    if not trace.truncated:
        return np.ones(length, dtype=bool)
    return (length - np.arange(length)) >= min_horizon


def overestimation_bias(agent, traces: list, gamma: float, min_horizon: int) -> dict:
    """Biaix de sobreestimació de Q1 sobre els episodis d'avaluació.

    :return: dict amb bias_all = mitjana de (Q1 − G_t) sobre tots els estats vàlids,
    bias_s0 = el mateix només amb s_0, mean_q, mean_g i n_bias_states.
    Valors None si cap estat és vàlid.
    """
    q_values, returns, q0, g0 = [], [], [], []
    for trace in traces:
        mask = valid_mask(trace, min_horizon)
        if not mask.any():
            continue
        q = _q1_of_policy(agent, trace.states)
        g = discounted_returns(trace.rewards, gamma)
        q_values.append(q[mask])
        returns.append(g[mask])
        if mask[0]:
            q0.append(q[0])
            g0.append(g[0])
    if not q_values:
        return dict.fromkeys(("bias_all", "bias_s0", "mean_q", "mean_g", "n_bias_states"))
    q_all, g_all = np.concatenate(q_values), np.concatenate(returns)
    return {"bias_all": float(np.mean(q_all - g_all)),
            "bias_s0": float(np.mean(np.subtract(q0, g0))) if q0 else None,
            "mean_q": float(q_all.mean()), "mean_g": float(g_all.mean()),
            "n_bias_states": int(len(q_all))}


def _q1_of_policy(agent, states: np.ndarray) -> np.ndarray:
    """Q1(s, μ(s)) per a un lot d'estats. :param states: [T,obs]. :return: [T]."""
    device = next(agent.Critic.parameters()).device
    with torch.no_grad():
        s = torch.as_tensor(states, dtype=torch.float32, device=device)
        return agent.Critic(s, agent.Actor(s)).squeeze(1).cpu().numpy().astype(np.float64)
