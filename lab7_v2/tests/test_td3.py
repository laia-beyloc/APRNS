"""Tests de TD3 (apartat 3, figura 5) i dels interruptors d'ablació (apartat 4)."""
import numpy as np
import pytest
import torch

import TD3 as td3_module
from experiments.config import VARIANTS
from TD3 import TD3
from conftest import FakeEnv, params_equal, random_batch, snapshot


def make_agent(seed=0, **switches):
    torch.manual_seed(seed)
    return TD3(FakeEnv(), replay_size=1000, batch_size=16, **switches)


def _tensors(batch):
    return [torch.as_tensor(x, dtype=torch.float32) for x in batch]


# ---------- inicialització ----------

def test_twin_critics_are_independent_and_targets_are_copies():
    agent = make_agent()
    assert not params_equal(agent.Critic, agent.Critic2)
    for online, target in ((agent.Critic, agent.Critic_target), (agent.Critic2, agent.Critic2_target),
                           (agent.Actor, agent.Actor_target)):
        assert params_equal(online, target)


# ---------- 3.1.1 suavitzat de l'acció objectiu ----------

def test_target_noise_is_clipped_and_action_bounded():
    agent = make_agent(target_noise=10.0)   # σ enorme: el retall c ha de dominar
    s2 = torch.randn(256, 8)
    with torch.no_grad():
        mu = agent.Actor_target(s2)
        smoothed = agent.target_action(s2)
    assert smoothed.abs().max() <= 1.0
    assert (smoothed - mu).abs().max() <= agent.noise_clip + 1e-6


def test_no_smoothing_gives_exact_target_action_but_consumes_rng():
    s2 = torch.randn(32, 8)
    states_after = []
    for sigma in (0.0, 0.2):
        agent = make_agent(target_noise=sigma)
        torch.manual_seed(123)
        with torch.no_grad():
            smoothed = agent.target_action(s2)
            if sigma == 0.0:
                assert torch.equal(smoothed, agent.Actor_target(s2).clamp(-1, 1))
        states_after.append(torch.get_rng_state())
    assert torch.equal(states_after[0], states_after[1])


# ---------- 3.1.2 dos crítics i mínim al target ----------

def _expected_target(agent, batch, use_minimum):
    s, a, r, s2, term, _ = _tensors(batch)
    with torch.no_grad():
        a2 = agent.target_action(s2)
        q1, q2 = agent.Critic_target(s2, a2), agent.Critic2_target(s2, a2)
        next_q = torch.minimum(q1, q2) if use_minimum else q1
        return r.unsqueeze(1) + agent.gamma * (1 - term.unsqueeze(1)) * next_q


@pytest.mark.parametrize("use_minimum", [True, False])
def test_both_critics_regress_to_the_same_target(use_minimum):
    agent = make_agent(use_minimum=use_minimum)
    s, a, r, s2, term, trunc = random_batch(8)
    term[:3] = 1.0
    trunc[3:5] = 1.0
    batch = (s, a, r, s2, term, trunc)
    torch.manual_seed(7)
    loss, metrics = agent.compute_critic_loss(batch)
    torch.manual_seed(7)
    y = _expected_target(agent, batch, use_minimum)
    st, at = torch.as_tensor(s), torch.as_tensor(a)
    with torch.no_grad():
        l1 = ((agent.Critic(st, at) - y) ** 2).mean().item()
        l2 = ((agent.Critic2(st, at) - y) ** 2).mean().item()
    assert metrics["critic_loss"] == pytest.approx(l1, rel=1e-5)
    assert metrics["critic2_loss"] == pytest.approx(l2, rel=1e-5)
    assert loss.item() == pytest.approx(l1 + l2, rel=1e-5)
    assert metrics["min_reduction"] >= 0.0


def test_truncation_flag_does_not_change_target():
    agent = make_agent()
    s, a, r, s2, term, trunc = random_batch(8)
    torch.manual_seed(1)
    loss_a, _ = agent.compute_critic_loss((s, a, r, s2, term, np.ones_like(trunc)))
    torch.manual_seed(1)
    loss_b, _ = agent.compute_critic_loss((s, a, r, s2, term, trunc))
    assert loss_a.item() == pytest.approx(loss_b.item(), rel=1e-6)


def test_actor_maximizes_q1_only():
    agent = make_agent()
    batch = random_batch(8)
    s = torch.as_tensor(batch[0])
    with torch.no_grad():
        expected = -agent.Critic(s, agent.Actor(s)).mean().item()
    assert agent.compute_actor_loss(batch).item() == pytest.approx(expected, rel=1e-6)
    with torch.no_grad():   # canviar Q2 no pot afectar l'objectiu de l'actor
        for p in agent.Critic2.parameters():
            p.add_(1.0)
    assert agent.compute_actor_loss(batch).item() == pytest.approx(expected, rel=1e-6)


def test_no_minimum_still_trains_both_critics():
    agent = make_agent(**VARIANTS["no_minimum"])
    before1, before2 = snapshot(agent.Critic), snapshot(agent.Critic2)
    agent.update(random_batch(16))
    assert any(not torch.equal(p, q) for p, q in zip(before1, agent.Critic.parameters()))
    assert any(not torch.equal(p, q) for p, q in zip(before2, agent.Critic2.parameters()))


# ---------- 3.1.3 actualitzacions retardades ----------

def _targets(agent):
    return [snapshot(n) for n in (agent.Critic_target, agent.Critic2_target, agent.Actor_target)]


def _changed(before, nets):
    return [any(not torch.equal(p, q) for p, q in zip(b, n.parameters())) for b, n in zip(before, nets)]


def test_delay_counts_critic_updates_and_moves_actor_with_all_targets():
    agent = make_agent(policy_delay=2)
    nets = (agent.Critic_target, agent.Critic2_target, agent.Actor_target)
    actor_before, targets_before = snapshot(agent.Actor), _targets(agent)

    metrics = agent.update(random_batch(16, seed=1))          # iteració 1: només crítics
    assert metrics["actor_loss"] is None
    assert params_equal(agent.Actor, agent.Actor_target)
    assert _changed(targets_before, nets) == [False, False, False]
    assert not _changed([actor_before], [agent.Actor])[0]

    metrics = agent.update(random_batch(16, seed=2))          # iteració 2: actor + tots els targets
    assert metrics["actor_loss"] is not None
    assert _changed(targets_before, nets) == [True, True, True]
    assert _changed([actor_before], [agent.Actor])[0]

    for seed in range(3, 7):
        agent.update(random_batch(16, seed=seed))
    assert (agent.critic_updates, agent.actor_updates) == (6, 3)


def test_no_delay_updates_actor_every_iteration():
    agent = make_agent(**VARIANTS["no_delay"])
    for seed in range(5):
        assert agent.update(random_batch(16, seed=seed))["actor_loss"] is not None
    assert (agent.critic_updates, agent.actor_updates) == (5, 5)
    assert agent.tau == 0.001


# ---------- apartat 4: cada interruptor canvia només el seu mecanisme ----------

SWITCHES = ("use_minimum", "target_noise", "policy_delay")


@pytest.mark.parametrize("variant", ["no_minimum", "no_smoothing", "no_delay"])
def test_variant_changes_only_its_switch(variant):
    full, ablated = make_agent(**VARIANTS["full"]), make_agent(**VARIANTS[variant])
    differing = [k for k in SWITCHES if getattr(full, k) != getattr(ablated, k)]
    assert len(differing) == 1
    # Mateixos pesos inicials, mateixa exploració/τ/batch i mateixos optimitzadors.
    for name in ("Actor", "Critic", "Critic2", "Actor_target", "Critic_target", "Critic2_target"):
        assert params_equal(getattr(full, name), getattr(ablated, name))
    assert (full.tau, full.batch_size, full.gamma, full.noise_clip) == \
           (ablated.tau, ablated.batch_size, ablated.gamma, ablated.noise_clip)
    for opt in ("optim_dqn", "optim_critic2", "optim_actor"):
        assert getattr(full, opt).defaults == getattr(ablated, opt).defaults


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_variants_consume_the_same_random_sequence(variant):
    # Amb el mateix batch i la mateixa llavor, totes les variants deixen el RNG
    # de torch en el mateix estat: les diferències venen del mecanisme, no del sorteig.
    reference = make_agent(**VARIANTS["full"])
    agent = make_agent(**VARIANTS[variant])
    states = []
    for a in (reference, agent):
        torch.manual_seed(5)
        a.update(random_batch(16))
        states.append(torch.get_rng_state())
    assert torch.equal(states[0], states[1])


# ---------- integració amb l'esquelet ----------

def test_learn_uses_gaussian_exploration_and_delay(monkeypatch):
    monkeypatch.setattr(td3_module, "episode_reward_plot", lambda *args, **kwargs: None)
    torch.manual_seed(0)
    agent = TD3(FakeEnv(max_steps=10, p_terminate=0.0), replay_size=1000, batch_size=8)
    agent.learn(60)
    assert agent.critic_updates == 52 and agent.actor_updates == 26
