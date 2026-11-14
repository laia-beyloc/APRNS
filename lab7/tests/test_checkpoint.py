"""Tests del checkpoint: reprendre ha de reproduir exactament l'execució original."""
import csv

import numpy as np
import pytest

from experiments.checkpoint import CheckpointAtStep, PeriodicCheckpoint, load_checkpoint
from experiments.config import VARIANTS, ExperimentConfig
from Noise import NormalActionNoise
from experiments.run_records import RunRecords
from experiments.seeding import seed_everything
from TD3 import TD3
from experiments.training import LoopState, MetricWindow, train
from conftest import FakeEnv

CFG = ExperimentConfig(total_steps=400, eval_every=100, eval_episodes=2)


def _agent_env(variant="full"):
    seed_everything(0)
    env = FakeEnv()
    return TD3(env, replay_size=1000, batch_size=16, **VARIANTS[variant]), env


def _rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return [{k: v for k, v in r.items() if k != "wall_time_s"} for r in csv.DictReader(stream)]


def test_resume_reproduces_the_original_run(tmp_path):
    agent, env = _agent_env()
    hook = CheckpointAtStep(tmp_path / "ckpt.pt", agent, env, at_step=150)
    train(agent, env, FakeEnv(), NormalActionNoise(np.zeros(2), 0.1), CFG,
          RunRecords(tmp_path / "original", {}), seed=0, on_episode_end=hook)
    assert hook.saved_at is not None and hook.saved_at >= 150

    # Agent nou (pesos diferents a propòsit) que es restaura des del checkpoint.
    seed_everything(99)
    env2 = FakeEnv()
    resumed = TD3(env2, replay_size=1000, batch_size=16, **VARIANTS["full"])
    state = load_checkpoint(tmp_path / "ckpt.pt", resumed, env2)
    assert state.step == hook.saved_at
    train(resumed, env2, FakeEnv(), NormalActionNoise(np.zeros(2), 0.1), CFG,
          RunRecords(tmp_path / "resumed", {}), seed=0, state=state)

    for name in ("evaluations.csv", "train_episodes.csv"):
        assert _rows(tmp_path / "original" / name) == _rows(tmp_path / "resumed" / name)
    assert (resumed.critic_updates, resumed.actor_updates) == (agent.critic_updates, agent.actor_updates)


def test_branches_from_one_checkpoint_are_independent(tmp_path):
    agent, env = _agent_env()
    hook = CheckpointAtStep(tmp_path / "ckpt.pt", agent, env, at_step=150)
    train(agent, env, FakeEnv(), NormalActionNoise(np.zeros(2), 0.1), CFG,
          RunRecords(tmp_path / "full", {}), seed=0, on_episode_end=hook)

    a, env_a = _agent_env("no_delay")
    b, env_b = _agent_env("no_minimum")
    load_checkpoint(tmp_path / "ckpt.pt", a, env_a)
    load_checkpoint(tmp_path / "ckpt.pt", b, env_b)
    a.replay_buffer.arrays[2][0] = 1e6     # modificar una branca no pot afectar l'altra
    assert b.replay_buffer.arrays[2][0] != 1e6
    assert a.policy_delay == 1 and b.use_minimum is False   # els interruptors són els de la variant nova


class Crash(Exception):
    pass


def test_cut_and_continue_gives_identical_measurements(tmp_path):
    # Referència sense talls.
    agent, env = _agent_env()
    train(agent, env, FakeEnv(), NormalActionNoise(np.zeros(2), 0.1), CFG,
          RunRecords(tmp_path / "ref", {}), seed=0)

    # Mateixa execució, tallada després del checkpoint periòdic (simula un tall de la VM).
    agent, env = _agent_env()
    latest = tmp_path / "cut" / "checkpoint_latest.pt"
    periodic = PeriodicCheckpoint(latest, agent, env, every=150)

    def hook(state):
        periodic(state)
        if state.step >= 320:
            raise Crash
    with pytest.raises(Crash):
        train(agent, env, FakeEnv(), NormalActionNoise(np.zeros(2), 0.1), CFG,
              RunRecords(tmp_path / "cut", {"seed": 0}), seed=0, on_episode_end=hook)

    # Continuació en un procés "nou": agent amb altres pesos, restaurat des del checkpoint.
    seed_everything(99)
    env2 = FakeEnv()
    resumed = TD3(env2, replay_size=1000, batch_size=16, **VARIANTS["full"])
    records = RunRecords.reopen(tmp_path / "cut")
    state = load_checkpoint(latest, resumed, env2)
    assert 150 <= state.step < 320 and records.config == {"seed": 0}
    train(resumed, env2, FakeEnv(), NormalActionNoise(np.zeros(2), 0.1), CFG, records, seed=0, state=state)

    for name in ("evaluations.csv", "train_episodes.csv"):
        assert _rows(tmp_path / "ref" / name) == _rows(tmp_path / "cut" / name)


def test_periodic_checkpoint_schedule(tmp_path):
    agent, env = _agent_env()
    saved = []
    periodic = PeriodicCheckpoint(tmp_path / "latest.pt", agent, env, every=100, start_step=130)
    for step in (150, 199, 205, 230, 310):
        before = periodic.next_step
        periodic(LoopState(window=MetricWindow(agent.metric_names), step=step))
        saved.append(periodic.next_step != before)
    assert saved == [False, False, True, False, True]
