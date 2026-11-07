"""Executa UNA execució (algorisme, variant, llavor) en un procés i una carpeta.

Exemples:
    python run_experiment.py --algo ddpg --seed 0
    python run_experiment.py --algo td3 --variant no_delay --seed 3
    python run_experiment.py --algo td3 --variant no_minimum --seed 0 \\
        --resume results/full/seed_0/checkpoint.pt --results results_branched

Resultat: <results>/<variant>/seed_<n>/ amb config.json, evaluations.csv,
train_episodes.csv i, a TD3 complet, checkpoint.pt (ignorat per git).
"""
import argparse
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

from config import AGENT_KWARGS, ENV_ID, TARGET_NOISE_CLIP, TD3_EXPLORATION_SIGMA, VARIANTS, ExperimentConfig

# Només les execucions de TD3 complet desen checkpoint: és la branca de la qual
# poden partir les altres variants (apartat 4).
CHECKPOINT_VARIANTS = ("full",)
CHECKPOINT_NAME = "checkpoint.pt"


def parse_args(argv=None) -> argparse.Namespace:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Una execució de DDPG o d'una variant de TD3.")
    parser.add_argument("--algo", choices=("ddpg", "td3"), required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="full", help="només per a td3")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=defaults.total_steps)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--eval-episodes", type=int, default=defaults.eval_episodes)
    parser.add_argument("--device", choices=("cpu", "auto"), default=defaults.device)
    parser.add_argument("--threads", type=int, default=defaults.num_threads)
    parser.add_argument("--results", type=Path, default=defaults.results_dir)
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint de TD3 complet de la mateixa llavor")
    return parser.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    if args.device == "cpu":
        # S'ha de fer abans d'importar torch: les xarxes de l'esquelet es mouen soles
        # a CUDA si la veuen, i així no cal tocar el codi donat.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return _run(args)


def _run(args: argparse.Namespace) -> Path:
    # Imports aquí perquè torch es carregui després de fixar CUDA_VISIBLE_DEVICES.
    import gymnasium as gym
    import torch

    from checkpoint import CheckpointAtStep, load_checkpoint
    from run_records import RunRecords
    from seeding import seed_everything
    from training import train

    cfg = ExperimentConfig(total_steps=args.steps, eval_every=args.eval_every, eval_episodes=args.eval_episodes,
                           results_dir=args.results, device=args.device, num_threads=args.threads)
    torch.set_num_threads(cfg.num_threads)
    seed_everything(args.seed)
    env = gym.make(ENV_ID, continuous=True)
    eval_env = gym.make(ENV_ID, continuous=True)
    agent, noise = build_agent_and_noise(args.algo, args.variant, env)

    run_name = "ddpg" if args.algo == "ddpg" else args.variant
    run_dir = cfg.results_dir / run_name / f"seed_{args.seed}"
    records = RunRecords(run_dir, run_config(args, cfg, agent, noise))

    state, hook = None, None
    if args.resume is not None:
        state = load_checkpoint(args.resume, agent, env)
    elif args.algo == "td3" and args.variant in CHECKPOINT_VARIANTS:
        hook = CheckpointAtStep(run_dir / CHECKPOINT_NAME, agent, env, int(cfg.total_steps * cfg.checkpoint_fraction))
    records.write_config(branch_step=state.step if state else 0)

    start = time.time()
    final = train(agent, env, eval_env, noise, cfg, records, args.seed, state=state, on_episode_end=hook)
    records.write_config(completed=True, final_step=final.step, train_episodes=final.episode,
                         critic_updates=agent.critic_updates, actor_updates=agent.actor_updates,
                         checkpoint_step=hook.saved_at if hook else None, wall_time_s=round(time.time() - start, 1))
    env.close()
    eval_env.close()
    return run_dir


def build_agent_and_noise(algo: str, variant: str, env):
    """Agent i soroll d'exploració: OU de l'esquelet per a DDPG, gaussià per a TD3 (3.1.1)."""
    import numpy as np
    from Noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

    act_dim = env.action_space.shape[0]
    if algo == "ddpg":
        from DDPG import DDPG
        # Mateixa crida que fa learn() de l'esquelet (paràmetres per defecte de Noise.py).
        return DDPG(env, **AGENT_KWARGS), OrnsteinUhlenbeckActionNoise(mu=np.zeros(act_dim))
    from TD3 import TD3
    return TD3(env, **AGENT_KWARGS, **VARIANTS[variant]), NormalActionNoise(np.zeros(act_dim), TD3_EXPLORATION_SIGMA)


def run_config(args: argparse.Namespace, cfg: ExperimentConfig, agent, noise) -> dict:
    """Configuració completa registrada a config.json (llavors, hiperparàmetres, versions)."""
    import gymnasium as gym
    import numpy as np
    import torch

    return {
        "algorithm": args.algo, "variant": args.variant if args.algo == "td3" else None,
        "switches": VARIANTS[args.variant] if args.algo == "td3" else None,
        "seed": args.seed, "env_id": ENV_ID, "experiment": asdict(cfg), "eval_seeds": cfg.eval_seeds(),
        "agent_kwargs": AGENT_KWARGS, "tau": agent.tau,
        "optimizers": {name: {k: v for k, v in opt.defaults.items() if k in ("lr", "weight_decay")}
                       for name, opt in vars(agent).items() if isinstance(opt, torch.optim.Optimizer)},
        "exploration": {"type": type(noise).__name__,
                        **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(noise).items()
                           if k in ("sigma", "_sigma", "theta", "dt")}},
        "target_noise_clip": TARGET_NOISE_CLIP if args.algo == "td3" else None,
        "resumed_from": str(args.resume) if args.resume else None,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "gymnasium": gym.__version__, "numpy": np.__version__},
        "device": str(next(agent.Actor.parameters()).device), "threads": torch.get_num_threads(),
        "command": " ".join(sys.argv), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


if __name__ == "__main__":
    print(main())
