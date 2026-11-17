"""Executa UNA execució (algorisme, variant, llavor) en un procés i una carpeta.

Exemples:
    python -m experiments.run_experiment --algo ddpg --seed 0
    python -m experiments.run_experiment --algo td3 --variant no_delay --seed 3
    python -m experiments.run_experiment --algo td3 --variant no_minimum --seed 0 \\
        --resume results/full/seed_0/checkpoint.pt --results results_branched

    python -m experiments.run_experiment --algo td3 --variant full --seed 2 --continue

Resultat: <results>/<variant>/seed_<n>/ amb config.json, evaluations.csv,
train_episodes.csv i, a TD3 complet, checkpoint.pt (branca de l'apartat 4). En acabar
es desen final_model.pt (xarxes finals) i final_evaluation.csv (avaluació final amb
llavors noves).
Durant l'execució es manté checkpoint_latest.pt (represa després d'un tall; s'esborra
en acabar). Els .pt els ignora git.
"""
import argparse
import json
import os
import shutil
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

from experiments.config import (AGENT_KWARGS, ENV_ID, SANITY_CONFIGS, TARGET_NOISE_CLIP, TD3_EXPLORATION_SIGMA, VARIANTS,
                                ExperimentConfig, td3_settings)

# Només les execucions de TD3 complet desen checkpoint: és la branca de la qual
# poden partir les altres variants (apartat 4).
CHECKPOINT_VARIANTS = ("full",)
CHECKPOINT_NAME = "checkpoint.pt"
LATEST_CHECKPOINT_NAME = "checkpoint_latest.pt"
FINAL_MODEL_NAME = "final_model.pt"
FINAL_EVALUATION_CSV = "final_evaluation.csv"


def parse_args(argv=None) -> argparse.Namespace:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Una execució de DDPG o d'una variant de TD3.")
    parser.add_argument("--algo", choices=("ddpg", "td3"), required=True)
    parser.add_argument("--variant", choices=(*VARIANTS, *SANITY_CONFIGS), default="full",
                        help="només per a td3 (les de SANITY_CONFIGS són la prova de control fora de protocol)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=defaults.total_steps)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--eval-episodes", type=int, default=defaults.eval_episodes)
    parser.add_argument("--device", choices=("cpu", "auto"), default=defaults.device)
    parser.add_argument("--threads", type=int, default=defaults.num_threads)
    parser.add_argument("--results", type=Path, default=defaults.results_dir)
    parser.add_argument("--final-eval-episodes", type=int, default=defaults.final_eval_episodes,
                        help="episodis de l'avaluació final de la política (0 la desactiva)")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint de TD3 complet de la mateixa llavor")
    parser.add_argument("--continue", dest="continue_run", action="store_true",
                        help="si la carpeta ja existeix i no està completada, la continua des del seu checkpoint")
    return parser.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    if args.device == "cpu":
        # S'ha de fer abans d'importar torch: les xarxes de l'esquelet es mouen soles
        # a CUDA si la veuen, i així no cal tocar el codi donat.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return _run(args)


def run_directory(args: argparse.Namespace) -> Path:
    """<results>/<variant|ddpg>/seed_<n>/."""
    return args.results / ("ddpg" if args.algo == "ddpg" else args.variant) / f"seed_{args.seed}"


def _run(args: argparse.Namespace) -> Path:
    if args.continue_run and is_completed(run_directory(args)):
        return run_directory(args)  # res a fer
    # Imports aquí perquè torch es carregui després de fixar CUDA_VISIBLE_DEVICES.
    import gymnasium as gym
    import torch

    from experiments.checkpoint import CheckpointAtStep, PeriodicCheckpoint, load_checkpoint
    from experiments.run_records import RunRecords
    from experiments.seeding import seed_everything
    from experiments.training import train

    random_start = td3_settings(args.variant)[2] if args.algo == "td3" else 0
    cfg = ExperimentConfig(total_steps=args.steps, eval_every=args.eval_every, eval_episodes=args.eval_episodes,
                           results_dir=args.results, device=args.device, num_threads=args.threads,
                           random_start_steps=random_start, final_eval_episodes=args.final_eval_episodes)
    torch.set_num_threads(cfg.num_threads)
    seed_everything(args.seed)
    env = gym.make(ENV_ID, continuous=True)
    eval_env = gym.make(ENV_ID, continuous=True)
    agent, noise = build_agent_and_noise(args.algo, args.variant, env)

    run_dir = run_directory(args)
    latest = run_dir / LATEST_CHECKPOINT_NAME
    records, state = None, None
    if args.continue_run and run_dir.exists():
        records, state = _reopen_interrupted(run_dir, latest, agent, env, RunRecords, load_checkpoint)
    if records is None:
        records = RunRecords(run_dir, run_config(args, cfg, agent, noise))
        if args.resume is not None:
            state = load_checkpoint(args.resume, agent, env)
        records.write_config(branch_step=state.step if state else 0)

    hooks = [PeriodicCheckpoint(latest, agent, env, cfg.checkpoint_every, state.step if state else 0)]
    if (args.algo == "td3" and args.variant in CHECKPOINT_VARIANTS and args.resume is None
            and not (run_dir / CHECKPOINT_NAME).exists()):
        hooks.append(CheckpointAtStep(run_dir / CHECKPOINT_NAME, agent, env,
                                      int(cfg.total_steps * cfg.checkpoint_fraction),
                                      on_saved=lambda step: records.write_config(checkpoint_step=step)))

    start = time.time()
    final = train(agent, env, eval_env, noise, cfg, records, args.seed, state=state,
                  on_episode_end=lambda loop_state: [hook(loop_state) for hook in hooks])
    if cfg.final_eval_episodes > 0:
        records.write_config(final_evaluation=final_evaluation(agent, eval_env, cfg, run_dir))
    records.write_config(completed=True, final_step=final.step, train_episodes=final.episode,
                         critic_updates=agent.critic_updates, actor_updates=agent.actor_updates,
                         wall_time_s=round(time.time() - start, 1))  # del darrer tram si s'ha continuat
    latest.unlink(missing_ok=True)  # ja no cal: l'execució és completa
    env.close()
    eval_env.close()
    return run_dir


def final_evaluation(agent, eval_env, cfg: ExperimentConfig, run_dir: Path) -> dict:
    """Desa les xarxes finals i avalua la política final en cfg.final_eval_episodes episodis.

    Les llavors són noves (disjuntes de les de l'avaluació periòdica), així que mesura la
    política en condicions inicials no vistes durant el seguiment. Escriu un CSV per
    episodi i retorna el resum que es desa a config.json.
    """
    import csv

    import numpy as np

    from experiments.checkpoint import save_final_model
    from experiments.diagnostics import overestimation_bias
    from experiments.evaluation import run_evaluation_episodes

    save_final_model(run_dir / FINAL_MODEL_NAME, agent)
    seeds = cfg.final_eval_seeds()
    traces = run_evaluation_episodes(agent, eval_env, seeds)
    returns = np.array([t.episode_return for t in traces])
    lengths = np.array([len(t.rewards) for t in traces])
    truncated = np.array([t.truncated for t in traces])
    with (run_dir / FINAL_EVALUATION_CSV).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["eval_seed", "return", "length", "truncated"])
        writer.writerows(zip(seeds, returns.tolist(), lengths.tolist(), truncated.tolist()))
    return {"episodes": len(traces), "mean_return": float(returns.mean()), "std_return": float(returns.std()),
            "success_rate": float(np.mean(returns >= 200)), "truncated_rate": float(truncated.mean()),
            **overestimation_bias(agent, traces, agent.gamma, cfg.bias_min_horizon)}


def _reopen_interrupted(run_dir: Path, latest: Path, agent, env, records_cls, load_checkpoint):
    """Retorna (records, state) per continuar una carpeta interrompuda, o (None, None) per tornar a començar.

    Sense checkpoint_latest.pt, el tall va ser abans del primer checkpoint periòdic: es
    descarta la carpeta (menys de checkpoint_every passos, que es regeneren idènticament
    perquè l'execució és determinista per llavor).
    """
    if latest.exists():
        records = records_cls.reopen(run_dir)
        state = load_checkpoint(latest, agent, env)
        records.write_config(continued_from=records.config.get("continued_from", []) + [state.step])
        return records, state
    shutil.rmtree(run_dir)
    return None, None


def is_completed(run_dir: Path) -> bool:
    """True si la carpeta conté una execució acabada (config.json amb completed: true)."""
    try:
        return json.loads((run_dir / "config.json").read_text(encoding="utf-8")).get("completed") is True
    except (OSError, ValueError):
        return False


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
    switches, agent_kwargs, _ = td3_settings(variant)
    return TD3(env, **agent_kwargs, **switches), NormalActionNoise(np.zeros(act_dim), TD3_EXPLORATION_SIGMA)


def run_config(args: argparse.Namespace, cfg: ExperimentConfig, agent, noise) -> dict:
    """Configuració completa registrada a config.json (llavors, hiperparàmetres, versions)."""
    import gymnasium as gym
    import numpy as np
    import torch

    return {
        "algorithm": args.algo, "variant": args.variant if args.algo == "td3" else None,
        "switches": td3_settings(args.variant)[0] if args.algo == "td3" else None,
        "sanity_check": SANITY_CONFIGS.get(args.variant) if args.algo == "td3" else None,
        "seed": args.seed, "env_id": ENV_ID, "experiment": asdict(cfg), "eval_seeds": cfg.eval_seeds(),
        "agent_kwargs": {"replay_size": agent.replay_buffer.capacity, "batch_size": agent.batch_size,
                         "gamma": agent.gamma}, "tau": agent.tau,
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
