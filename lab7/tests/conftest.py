"""Entorn fals i utilitats comunes dels tests (no necessiten Box2D)."""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(1)


class FakeEnv(gym.Env):
    """Entorn mínim amb les dimensions de LunarLander continu (obs 8, acció 2 a [-1,1]).

    Fa servir self.np_random perquè la llavor de reset() determini la trajectòria.
    """

    observation_space = gym.spaces.Box(-np.inf, np.inf, (8,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    def __init__(self, max_steps: int = 40, p_terminate: float = 0.05):
        self.max_steps, self.p_terminate = max_steps, p_terminate

    def _obs(self):
        return self.np_random.normal(size=8).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        reward = float(-np.sum(np.square(action)) + 0.1 * self.np_random.normal())
        terminated = bool(self.np_random.uniform() < self.p_terminate)
        truncated = self.t >= self.max_steps and not terminated
        return self._obs(), reward, terminated, truncated, {}


@pytest.fixture
def fake_env():
    return FakeEnv()


def random_batch(batch_size: int = 16, obs_dim: int = 8, act_dim: int = 2, seed: int = 0):
    """Minibatch sintètic amb el format de ReplayBuffer.get()."""
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(batch_size, obs_dim)).astype(np.float32),
            rng.uniform(-1, 1, size=(batch_size, act_dim)).astype(np.float32),
            rng.normal(size=batch_size).astype(np.float32),
            rng.normal(size=(batch_size, obs_dim)).astype(np.float32),
            np.zeros(batch_size, dtype=np.float32),
            np.zeros(batch_size, dtype=np.float32))


def params_equal(net_a, net_b) -> bool:
    return all(torch.equal(a, b) for a, b in zip(net_a.parameters(), net_b.parameters()))


def snapshot(net) -> list:
    return [p.detach().clone() for p in net.parameters()]
