"""Checkpoint complet d'una execució i represa (apartat 4, opció d'estalvi de còmput).

Es desa a una frontera d'episodi: xarxes en línia i objectiu, optimitzadors, memòria
de repetició, comptadors d'actualitzacions, estat del bucle i tots els generadors
aleatoris (python, numpy, torch, cuda i entorn). Així una branca continua exactament
com ho hauria fet l'execució original.
"""
from pathlib import Path

import torch
from torch import nn, optim

from experiments.seeding import capture_rng_state, restore_rng_state
from experiments.training import LoopState


def _stateful_parts(agent) -> dict:
    """Xarxes i optimitzadors de l'agent, trobats pel tipus (serveix per a DDPG i TD3)."""
    return {name: obj for name, obj in vars(agent).items() if isinstance(obj, (nn.Module, optim.Optimizer))}


def save_checkpoint(path: Path, agent, env, state: LoopState) -> None:
    """Desa a `path` tot el necessari per reprendre l'execució des de `state`."""
    torch.save({
        "parts": {name: obj.state_dict() for name, obj in _stateful_parts(agent).items()},
        "replay_buffer": agent.replay_buffer.state_dict(),
        "counters": {"critic_updates": agent.critic_updates, "actor_updates": agent.actor_updates},
        "loop_state": state,
        "rng": capture_rng_state(),
        "env_rng": env.unwrapped.np_random.bit_generator.state,
    }, path)


def load_checkpoint(path: Path, agent, env) -> LoopState:
    """Restaura agent, entorn i generadors des de `path`; retorna l'estat del bucle.

    Cada càrrega crea una còpia independent: diverses variants poden partir del
    mateix fitxer sense compartir res.
    """
    payload = torch.load(path, weights_only=False)  # fitxers propis, no de tercers
    parts = _stateful_parts(agent)
    for name, saved in payload["parts"].items():
        parts[name].load_state_dict(saved)
    agent.replay_buffer.load_state_dict(payload["replay_buffer"])
    agent.critic_updates = payload["counters"]["critic_updates"]
    agent.actor_updates = payload["counters"]["actor_updates"]
    env.unwrapped.np_random.bit_generator.state = payload["env_rng"]
    restore_rng_state(payload["rng"])
    return payload["loop_state"]


class CheckpointAtStep:
    """Callback per a train(): desa un sol checkpoint a la primera frontera d'episodi ≥ at_step."""

    def __init__(self, path: Path, agent, env, at_step: int):
        self.path, self.agent, self.env, self.at_step = Path(path), agent, env, at_step
        self.saved_at = None

    def __call__(self, state: LoopState) -> None:
        if self.saved_at is None and state.step >= self.at_step:
            save_checkpoint(self.path, self.agent, self.env, state)
            self.saved_at = state.step
