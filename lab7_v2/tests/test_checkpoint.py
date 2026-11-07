"""Tests del checkpoint: reprendre ha de reproduir exactament l'execució original."""
import csv

import numpy as np

from checkpoint import CheckpointAtStep, load_checkpoint
from config import VARIANTS, ExperimentConfig
from Noise import NormalActionNoise
from run_records import RunRecords
from seeding import seed_everything
from TD3 import TD3
from training import train
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
