"""Bucle d'entrenament comú a DDPG i TD3 amb pressupost fix en transicions (apartats 2 i 4).

Només fa servir la interfície comuna dels agents: choose_action(obs), update(batch),
replay_buffer i batch_size. No hi ha parada per recompensa ni selecció de llavors.
"""
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from experiments.config import ExperimentConfig
from experiments.diagnostics import overestimation_bias
from experiments.evaluation import run_evaluation_episodes, summarize_returns
from experiments.run_records import RunRecords


@dataclass
class MetricWindow:
    """Mitjana de les mètriques d'update() entre dues avaluacions (valors None s'ignoren)."""

    names: tuple = ()

    def __post_init__(self):
        # Claus fixes des del principi: la capçalera del CSV no depèn de si ja s'ha après.
        self.reset()

    def reset(self) -> None:
        self.sums = dict.fromkeys(self.names, 0.0)
        self.counts = dict.fromkeys(self.names, 0)

    def add(self, metrics: dict) -> None:
        for key, value in metrics.items():
            if value is not None:
                self.sums[key] += value
                self.counts[key] += 1

    def pop_means(self) -> dict:
        """Retorna les mitjanes de la finestra i la reinicia."""
        means = {k: (self.sums[k] / self.counts[k] if self.counts[k] else None) for k in self.names}
        self.reset()
        return means


@dataclass
class LoopState:
    """Estat del bucle que cal desar al checkpoint per poder reprendre exactament."""

    window: MetricWindow
    step: int = 0
    episode: int = 0
    evaluation_rows: list = field(default_factory=list)
    episode_rows: list = field(default_factory=list)


def train(agent, env, eval_env, noise, cfg: ExperimentConfig, records: RunRecords, seed: int,
          state: Optional[LoopState] = None,
          on_episode_end: Optional[Callable[[LoopState], None]] = None) -> LoopState:
    """Entrena `agent` fins a cfg.total_steps transicions d'entrenament.

    :param noise: soroll d'exploració amb sample() (i reset() opcional, com l'OU).
    :param seed: llavor del primer reset de l'entorn d'entrenament (només si state és nou).
    :param state: estat d'un checkpoint per reprendre; None per començar de zero.
    :param on_episode_end: crida a cada frontera d'episodi (p. ex. per desar el checkpoint).
    :return: estat final del bucle.
    """
    resuming = state is not None
    state = state or LoopState(window=MetricWindow(agent.metric_names))
    if resuming:
        _replay_history(records, state)
    # En reprendre, el RNG de l'entorn ja ve restaurat del checkpoint: no es torna a sembrar.
    obs, _ = env.reset(seed=None if resuming else seed)
    episode_return, episode_length = 0.0, 0
    start_time = time.time()

    while state.step < cfg.total_steps:
        action = np.clip(agent.choose_action(obs) + noise.sample(), -1.0, 1.0)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.replay_buffer.put(obs, action, reward, next_obs, terminated, truncated)
        obs = next_obs
        episode_return += float(reward)
        episode_length += 1
        state.step += 1

        if len(agent.replay_buffer) > agent.batch_size:
            state.window.add(agent.update(agent.replay_buffer.get(agent.batch_size)))

        if state.step % cfg.eval_every == 0:
            row = _evaluation_row(agent, eval_env, cfg, state, time.time() - start_time)
            state.evaluation_rows.append(row)
            records.log_evaluation(row)
            print(f"{records.run_dir} {state.step}/{cfg.total_steps} retorn={row['eval_mean_return']:.1f}", flush=True)

        if terminated or truncated:
            state.episode += 1
            row = {"episode": state.episode, "transitions": state.step, "return": episode_return,
                   "length": episode_length, "terminated": bool(terminated), "truncated": bool(truncated)}
            state.episode_rows.append(row)
            records.log_episode(row)
            if hasattr(noise, "reset"):  # l'OU té memòria; el gaussià no
                noise.reset()
            if on_episode_end is not None:
                on_episode_end(state)
            obs, _ = env.reset()
            episode_return, episode_length = 0.0, 0
    return state


def _evaluation_row(agent, eval_env, cfg: ExperimentConfig, state: LoopState, elapsed: float) -> dict:
    """Una fila de evaluations.csv: retorns d'avaluació, mètriques de la finestra i comptadors."""
    traces = run_evaluation_episodes(agent, eval_env, cfg.eval_seeds())
    return {"transitions": state.step, "train_episodes": state.episode,
            "critic_updates": agent.critic_updates, "actor_updates": agent.actor_updates,
            **summarize_returns(traces),
            **overestimation_bias(agent, traces, agent.gamma, cfg.bias_min_horizon),
            **state.window.pop_means(), "wall_time_s": round(elapsed, 1)}


def _replay_history(records: RunRecords, state: LoopState) -> None:
    """En una execució represa, la carpeta nova comença amb el tram compartit fins al checkpoint."""
    for row in state.evaluation_rows:
        records.log_evaluation(row)
    for row in state.episode_rows:
        records.log_episode(row)
