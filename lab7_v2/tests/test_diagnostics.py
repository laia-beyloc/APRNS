"""Tests del diagnòstic de sobreestimació."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.diagnostics import discounted_returns, overestimation_bias, valid_mask
from experiments.evaluation import EpisodeTrace


class ConstantCritic(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, s, a):
        return self.value.expand(s.shape[0], 1)


class ZeroActor(nn.Module):
    def forward(self, s):
        return torch.zeros(s.shape[0], 2)


def _trace(rewards, truncated):
    return EpisodeTrace(np.zeros((len(rewards), 8), np.float32), np.asarray(rewards, float), truncated)


def test_discounted_returns():
    np.testing.assert_allclose(discounted_returns(np.array([1.0, 2.0, 3.0]), 0.5), [2.75, 3.5, 3.0])


def test_truncated_episode_excludes_last_states():
    mask = valid_mask(_trace(np.ones(10), truncated=True), min_horizon=4)
    np.testing.assert_array_equal(mask, [True] * 7 + [False] * 3)   # T − t ≥ 4
    assert valid_mask(_trace(np.ones(10), truncated=False), 4).all()


def test_bias_is_q_minus_observed_return():
    agent = SimpleNamespace(Actor=ZeroActor(), Critic=ConstantCritic(5.0))
    gamma = 0.5
    terminated = _trace([1.0, 2.0, 3.0], truncated=False)          # G = [2.75, 3.5, 3.0]
    truncated = _trace([1.0] * 6, truncated=True)                   # vàlids t = 0..2
    out = overestimation_bias(agent, [terminated, truncated], gamma, min_horizon=4)
    g_trunc = discounted_returns(np.ones(6), gamma)[:3]
    g_all = np.concatenate([[2.75, 3.5, 3.0], g_trunc])
    assert out["n_bias_states"] == 6
    assert out["bias_all"] == pytest.approx(5.0 - g_all.mean())
    assert out["bias_s0"] == pytest.approx(5.0 - np.mean([2.75, g_trunc[0]]))


def test_no_valid_state_gives_none():
    agent = SimpleNamespace(Actor=ZeroActor(), Critic=ConstantCritic(0.0))
    out = overestimation_bias(agent, [_trace([1.0, 1.0], truncated=True)], 0.99, min_horizon=300)
    assert out["bias_all"] is None and out["n_bias_states"] is None
