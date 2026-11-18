"""Comprovació de comportament: què fa la política final quan l'episodi s'esgota (truncat).

Carrega els final_model.pt de les execucions que en tenen (prova de control) i en juga
episodis deterministes amb llavors de l'avaluació final. Per a cada episodi truncat
resumeix els últims 200 passos: alçada i posició del mòdul, contacte de les dues potes,
velocitat vertical i fracció de passos amb el motor principal encès. Una alçada > 0 sense
contacte, velocitat vertical ~0 i motor encès indica que el mòdul es manté suspès (planeja).

Ús (des de lab7/):  python -m analysis.hovering --results results --out analysis/supplementary
Necessita els final_model.pt (no es versionen: es generen en acabar cada execució).
"""
import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gymnasium as gym
import numpy as np
import torch

from experiments.config import ENV_ID, ExperimentConfig, td3_settings
from experiments.run_experiment import FINAL_MODEL_NAME
from TD3 import TD3

TAIL = 200          # passos finals que es resumeixen
EPISODES = 20       # episodis per execució (primeres llavors de l'avaluació final)


def load_agent(env, variant: str, model_path: Path) -> TD3:
    """Agent TD3 amb les xarxes finals desades."""
    switches, kwargs, _ = td3_settings(variant)
    agent = TD3(env, **kwargs, **switches)
    for name, state in torch.load(model_path, weights_only=False).items():
        getattr(agent, name).load_state_dict(state)
    return agent


def episode_summary(agent, env, seed: int) -> dict:
    """Juga un episodi determinista i en resumeix el final. Observació: [x, y, vx, vy, θ, ω, pota1, pota2]."""
    obs, _ = env.reset(seed=seed)
    states, actions, total = [], [], 0.0
    while True:
        action = agent.choose_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        states.append(obs)
        actions.append(action)
        total += reward
        if terminated or truncated:
            break
    tail_s, tail_a = np.array(states[-TAIL:]), np.array(actions[-TAIL:])
    return {"eval_seed": seed, "return": total, "length": len(states), "truncated": bool(truncated and not terminated),
            "tail_y": tail_s[:, 1].mean(), "tail_abs_x": np.abs(tail_s[:, 0]).mean(),
            "tail_both_legs": (tail_s[:, 6] * tail_s[:, 7]).mean(), "tail_abs_vy": np.abs(tail_s[:, 3]).mean(),
            "tail_main_engine_on": (tail_a[:, 0] > 0).mean()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("analysis/supplementary"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    env = gym.make(ENV_ID, continuous=True)
    seeds = ExperimentConfig().final_eval_seeds()[:EPISODES]
    rows = []
    for model_path in sorted(args.results.glob(f"*/seed_*/{FINAL_MODEL_NAME}")):
        variant, seed = model_path.parent.parent.name, int(model_path.parent.name[5:])
        agent = load_agent(env, variant, model_path)
        rows += [{"variant": variant, "seed": seed, **episode_summary(agent, env, s)} for s in seeds]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "hovering.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    truncated = [r for r in rows if r["truncated"]]
    print(f"{len(truncated)} de {len(rows)} episodis truncats. Mitjanes als últims {TAIL} passos dels truncats:")
    for key in ("tail_y", "tail_abs_x", "tail_both_legs", "tail_abs_vy", "tail_main_engine_on", "return"):
        print(f"  {key}: {np.mean([r[key] for r in truncated]):.3f}")


if __name__ == "__main__":
    main()
