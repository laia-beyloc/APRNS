"""Tests de les equacions de DDPG (figura 1 de l'enunciat)."""
import numpy as np
import pytest
import torch

import DDPG as ddpg_module
from DDPG import DDPG
from conftest import FakeEnv, params_equal, random_batch, snapshot


@pytest.fixture
def agent(fake_env):
    torch.manual_seed(0)
    return DDPG(fake_env, replay_size=1000, batch_size=16, gamma=0.99)


def test_targets_start_as_exact_copies(agent):
    assert params_equal(agent.Critic, agent.Critic_target)
    assert params_equal(agent.Actor, agent.Actor_target)


def test_critic_output_layer_uses_paper_init(agent):
    # initialize_weights() posa la darrera capa a U(-3e-3, 3e-3).
    assert agent.Critic.l4.weight.abs().max().item() <= 0.003


def test_choose_action_is_deterministic_numpy(agent):
    obs = np.ones(8, dtype=np.float32)
    a1, a2 = agent.choose_action(obs), agent.choose_action(obs)
    assert isinstance(a1, np.ndarray) and a1.shape == (2,)
    np.testing.assert_array_equal(a1, a2)
    assert np.all(np.abs(a1) <= 1.0)


def _expected_critic_loss(agent, batch):
    s, a, r, s2, term, _ = (torch.as_tensor(x, dtype=torch.float32) for x in batch)
    with torch.no_grad():
        y = r.unsqueeze(1) + agent.gamma * (1 - term.unsqueeze(1)) * agent.Critic_target(s2, agent.Actor_target(s2))
        return ((agent.Critic(s, a) - y) ** 2).mean().item()


def test_critic_target_cuts_bootstrap_only_on_termination(agent):
    s, a, r, s2, term, trunc = random_batch(4)
    term[:2] = 1.0           # terminacions reals: y = r
    trunc[2:] = 1.0          # truncacions per temps: y = r + γQ'
    batch = (s, a, r, s2, term, trunc)
    loss = agent.compute_critic_loss(batch).item()
    assert loss == pytest.approx(_expected_critic_loss(agent, batch), rel=1e-5)
    # La marca de truncació no ha de canviar res.
    batch_no_trunc = (s, a, r, s2, term, np.zeros_like(trunc))
    assert loss == pytest.approx(agent.compute_critic_loss(batch_no_trunc).item(), rel=1e-6)


def test_terminal_target_ignores_next_state(agent):
    s, a, r, s2, term, trunc = random_batch(4)
    term[:] = 1.0
    loss_a = agent.compute_critic_loss((s, a, r, s2, term, trunc)).item()
    loss_b = agent.compute_critic_loss((s, a, r, s2 + 100.0, term, trunc)).item()
    assert loss_a == pytest.approx(loss_b, rel=1e-6)


def test_actor_loss_is_minus_mean_q(agent):
    batch = random_batch(8)
    s = torch.as_tensor(batch[0])
    with torch.no_grad():
        expected = -agent.Critic(s, agent.Actor(s)).mean().item()
    assert agent.compute_actor_loss(batch).item() == pytest.approx(expected, rel=1e-6)


def test_update_trains_both_and_soft_updates_targets(agent):
    old_critic_target, old_actor_target = snapshot(agent.Critic_target), snapshot(agent.Actor_target)
    old_actor = snapshot(agent.Actor)
    metrics = agent.update(random_batch(16))
    assert set(metrics) == set(agent.metric_names)
    assert not params_equal(agent.Actor, agent.Actor_target)
    assert any(not torch.equal(p, q) for p, q in zip(old_actor, agent.Actor.parameters()))
    for online, target, old in ((agent.Critic, agent.Critic_target, old_critic_target),
                                (agent.Actor, agent.Actor_target, old_actor_target)):
        for p, t, o in zip(online.parameters(), target.parameters(), old):
            torch.testing.assert_close(t, agent.tau * p + (1 - agent.tau) * o)
    assert (agent.critic_updates, agent.actor_updates) == (1, 1)


def test_learn_uses_update_and_resets_ou_each_episode(monkeypatch):
    resets = []

    class SpyOU(ddpg_module.OrnsteinUhlenbeckActionNoise):
        def reset(self):
            resets.append(1)
            super().reset()

    monkeypatch.setattr(ddpg_module, "OrnsteinUhlenbeckActionNoise", SpyOU)
    monkeypatch.setattr(ddpg_module, "episode_reward_plot", lambda *args, **kwargs: None)
    torch.manual_seed(0)
    agent = DDPG(FakeEnv(max_steps=10, p_terminate=0.0), replay_size=1000, batch_size=8)
    agent.learn(60)
    assert agent.critic_updates == 60 - 8      # s'aprèn quan len(buffer) > batch_size
    assert len(resets) == 1 + 6                # reset del constructor + un per episodi
