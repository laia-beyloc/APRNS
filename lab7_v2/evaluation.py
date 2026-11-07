"""Episodis d'avaluació sense soroll d'exploració (apartats 2 i 3.1.1)."""
import json
from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeTrace:
    """Trajectòria d'un episodi d'avaluació.

    states [T,obs]: estats s_0..s_{T-1} on s'ha actuat; rewards [T]: r_0..r_{T-1};
    truncated: True si l'episodi s'ha tallat per límit de temps (G_t queda incomplet).
    """

    states: np.ndarray
    rewards: np.ndarray
    truncated: bool

    @property
    def episode_return(self) -> float:
        return float(self.rewards.sum())


def run_evaluation_episodes(agent, env, seeds: list) -> list:
    """Executa un episodi per llavor amb la política determinista μ(s).

    No consumeix cap generador global: choose_action() no és aleatori i l'entorn
    es torna a sembrar amb reset(seed=...). Així avaluar no altera l'entrenament.
    """
    return [_run_episode(agent, env, seed) for seed in seeds]


def _run_episode(agent, env, seed: int) -> EpisodeTrace:
    obs, _ = env.reset(seed=seed)
    states, rewards = [], []
    while True:
        states.append(obs)
        obs, reward, terminated, truncated, _ = env.step(agent.choose_action(obs))
        rewards.append(reward)
        if terminated or truncated:
            return EpisodeTrace(np.asarray(states, dtype=np.float32),
                                np.asarray(rewards, dtype=np.float64), bool(truncated and not terminated))


def summarize_returns(traces: list) -> dict:
    """Estadístics dels retorns d'avaluació per a una fila del CSV."""
    returns = [trace.episode_return for trace in traces]
    return {
        "eval_mean_return": float(np.mean(returns)),
        "eval_std_return": float(np.std(returns)),
        "eval_mean_length": float(np.mean([len(trace.rewards) for trace in traces])),
        "eval_returns": json.dumps([round(r, 3) for r in returns]),
    }
