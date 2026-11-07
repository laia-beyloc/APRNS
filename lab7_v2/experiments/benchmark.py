"""Benchmark de cost: passos/s d'entrenament, cost d'una avaluació i estimació de l'estudi.

Exemples:
    python -m experiments.benchmark                       # 1 procés
    python -m experiments.benchmark --parallel 1 8 14     # també amb K processos simultanis (contenció)

Cada mesura és un procés "worker" (1 fil, CPU) que entrena DDPG i TD3 complet
uns quants passos sense avaluar i després mesura l'avaluació de l'agent resultant.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ALGOS = (("ddpg", "full"), ("td3", "full"))
WORST_CASE_EVAL_LENGTH = 1000  # episodi truncat pel límit de temps de LunarLander
# Arrel del projecte (lab7/): el worker es llança com a mòdul del paquet des d'aquí.
ROOT = Path(__file__).resolve().parents[1]
SINGLE_THREAD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def measure_one(algo: str, variant: str, steps: int, eval_episodes: int) -> dict:
    """Mesura (dins d'un worker) passos/s d'entrenament i cost d'avaluació d'un agent."""
    import gymnasium as gym
    import torch

    from experiments.config import ENV_ID, ExperimentConfig
    from experiments.diagnostics import overestimation_bias
    from experiments.evaluation import run_evaluation_episodes
    from experiments.run_experiment import build_agent_and_noise
    from experiments.run_records import RunRecords
    from experiments.seeding import seed_everything
    from experiments.training import train

    torch.set_num_threads(1)
    seed_everything(0)
    env, eval_env = gym.make(ENV_ID, continuous=True), gym.make(ENV_ID, continuous=True)
    agent, noise = build_agent_and_noise(algo, variant, env)
    cfg = ExperimentConfig(total_steps=steps, eval_every=steps + 1)  # cap avaluació durant la mesura
    with tempfile.TemporaryDirectory() as tmp:
        records = RunRecords(Path(tmp) / "run", {})
        start = time.perf_counter()
        train(agent, env, eval_env, noise, cfg, records, seed=0)
        train_s = time.perf_counter() - start

    seeds = [cfg.eval_seed_base + i for i in range(eval_episodes)]
    start = time.perf_counter()
    traces = run_evaluation_episodes(agent, eval_env, seeds)
    overestimation_bias(agent, traces, agent.gamma, cfg.bias_min_horizon)
    eval_s = time.perf_counter() - start
    eval_steps = sum(len(t.rewards) for t in traces)
    return {"algo": algo if algo == "ddpg" else f"td3_{variant}", "train_sps": steps / train_s,
            "eval_s_per_episode": eval_s / eval_episodes, "eval_mean_length": eval_steps / eval_episodes,
            "eval_s_per_step": eval_s / eval_steps}


def worker_main(steps: int, eval_episodes: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    results = [measure_one(a, v, steps, eval_episodes) for a, v in ALGOS]
    print("RESULT " + json.dumps(results), flush=True)


def run_parallel(k: int, steps: int, eval_episodes: int) -> list:
    """Llança k workers a la vegada i retorna la mitjana per algorisme."""
    cmd = [sys.executable, "-m", "experiments.benchmark", "--worker", "--steps", str(steps), "--eval-episodes", str(eval_episodes)]
    env = {**os.environ, "PYTHONPATH": "", **SINGLE_THREAD_ENV}
    procs = [subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=env, cwd=ROOT) for _ in range(k)]
    per_worker = [json.loads(_result_line(p.communicate()[0])) for p in procs]
    return [_mean_dicts([w[i] for w in per_worker]) for i in range(len(ALGOS))]


def _result_line(output: str) -> str:
    return next(line[len("RESULT "):] for line in output.splitlines() if line.startswith("RESULT "))


def _mean_dicts(rows: list) -> dict:
    return {key: (rows[0][key] if key == "algo" else sum(r[key] for r in rows) / len(rows)) for key in rows[0]}


def job_hours(train_sps: float, eval_s_per_step: float, eval_length: float,
              steps: int, eval_every: int, eval_episodes: int) -> float:
    """Hores d'una execució: entrenament + (steps/eval_every) avaluacions."""
    evals = steps // eval_every
    return (steps / train_sps + evals * eval_episodes * eval_length * eval_s_per_step) / 3600


def study_hours(job_h: float, n_jobs: int, workers: int) -> float:
    """Temps de paret: les execucions s'agrupen en tandes de `workers`."""
    return math.ceil(n_jobs / workers) * job_h


def print_estimates(measured: dict, steps: int, eval_every: int, n_variants: int = 4) -> None:
    """Taula de l'estudi: 1 DDPG + n_variants×seeds TD3, per eval_episodes, N processos i longitud d'avaluació."""
    td3_key = "td3_full"
    for seeds in (5, 10):
        n_td3 = n_variants * seeds
        print(f"\n== Estudi amb {seeds} llavors ({1 + n_td3} execucions, T={steps}, avaluació cada {eval_every}) ==")
        print("hores de paret | eval_ep  longitud_eval  " + "  ".join(f"N={n:<3}" for n in sorted(measured)))
        for eval_episodes in (1, 3, 5):
            for length_label, length in (("mesurada", None), ("pitjor", WORST_CASE_EVAL_LENGTH)):
                cells = []
                for n in sorted(measured):
                    rows = {r["algo"]: r for r in measured[n]}
                    hours = []
                    for algo, count in (("ddpg", 1), (td3_key, n_td3)):
                        r = rows[algo]
                        length_used = length or r["eval_mean_length"]
                        hours.append((job_hours(r["train_sps"], r["eval_s_per_step"], length_used,
                                                steps, eval_every, eval_episodes), count))
                    # Cost total en hores·procés, repartit entre n processos (aprox. de tandes).
                    total = sum(h * c for h, c in hours)
                    longest = max(h for h, _ in hours)
                    cells.append(f"{max(total / n, longest):6.1f}")
                print(f"               | {eval_episodes:^7}  {length_label:<13}  " + "  ".join(f"{c:<5}" for c in cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=5000, help="passos d'entrenament mesurats")
    parser.add_argument("--eval-episodes", type=int, default=10, help="episodis d'avaluació mesurats")
    parser.add_argument("--parallel", type=int, nargs="+", default=[1], help="nombre de processos simultanis a provar")
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--worker", action="store_true", help="ús intern")
    args = parser.parse_args()
    if args.worker:
        worker_main(args.steps, args.eval_episodes)
        return

    measured = {}
    for k in args.parallel:
        measured[k] = run_parallel(k, args.steps, args.eval_episodes)
        for r in measured[k]:
            print(f"K={k:<3} {r['algo']:<9} {r['train_sps']:7.1f} passos/s  "
                  f"avaluació {r['eval_s_per_episode']:.2f} s/episodi, longitud mitjana {r['eval_mean_length']:.0f}")
    print_estimates(measured, args.total_steps, args.eval_every)


if __name__ == "__main__":
    main()
