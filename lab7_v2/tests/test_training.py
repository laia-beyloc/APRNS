"""Tests del bucle comú, l'avaluació i el registre d'execucions."""
import csv
import random

import numpy as np
import pytest
import torch

from experiments.config import VARIANTS, ExperimentConfig
from DDPG import DDPG
from experiments.evaluation import run_evaluation_episodes
from Noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from TD3 import TD3
from experiments.run_records import RunRecords
from experiments.seeding import seed_everything
from experiments.training import train
from conftest import FakeEnv

CFG = ExperimentConfig(total_steps=300, eval_every=100, eval_episodes=2)


def _run(run_dir, seed=0):
    seed_everything(seed)
    agent = DDPG(FakeEnv(), replay_size=1000, batch_size=16)
    records = RunRecords(run_dir, {"seed": seed})
    state = train(agent, FakeEnv(), FakeEnv(), OrnsteinUhlenbeckActionNoise(mu=np.zeros(2)), CFG, records, seed)
    return agent, state


def _read(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_fixed_budget_and_evaluation_schedule(tmp_path):
    agent, state = _run(tmp_path / "run")
    assert len(agent.replay_buffer) == CFG.total_steps == state.step
    rows = _read(tmp_path / "run" / "evaluations.csv")
    assert [int(r["transitions"]) for r in rows] == [100, 200, 300]
    assert {"critic_loss", "actor_loss", "eval_mean_return", "critic_updates"} <= set(rows[0])
    assert len(_read(tmp_path / "run" / "train_episodes.csv")) == state.episode


def test_same_seed_gives_identical_measurements(tmp_path):
    _run(tmp_path / "a")
    _run(tmp_path / "b")
    strip = lambda rows: [{k: v for k, v in r.items() if k != "wall_time_s"} for r in rows]
    assert strip(_read(tmp_path / "a" / "evaluations.csv")) == strip(_read(tmp_path / "b" / "evaluations.csv"))


def test_evaluation_does_not_consume_global_rng(fake_env):
    seed_everything(0)
    agent = DDPG(fake_env, replay_size=100)
    before = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state().clone())
    traces = run_evaluation_episodes(agent, FakeEnv(), [1, 2])
    after = (random.getstate(), np.random.get_state()[1], torch.get_rng_state())
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and torch.equal(before[2], after[2])
    assert len(traces) == 2 and traces[0].states.shape[1] == 8


def test_common_loop_runs_td3_without_special_cases(tmp_path):
    seed_everything(0)
    agent = TD3(FakeEnv(), replay_size=1000, batch_size=16, **VARIANTS["full"])
    records = RunRecords(tmp_path / "td3", {})
    train(agent, FakeEnv(), FakeEnv(), NormalActionNoise(np.zeros(2), 0.1), CFG, records, 0)
    rows = _read(tmp_path / "td3" / "evaluations.csv")
    assert {"critic2_loss", "critic_disagreement", "min_reduction"} <= set(rows[0])
    assert int(rows[-1]["actor_updates"]) == int(rows[-1]["critic_updates"]) // 2


def test_run_folder_is_never_overwritten(tmp_path):
    RunRecords(tmp_path / "run", {})
    with pytest.raises(FileExistsError):
        RunRecords(tmp_path / "run", {})
