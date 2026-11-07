"""Llançador local: executa les execucions de l'estudi en paral·lel, un procés per execució.

Exemples:
    python -m experiments.launch --dry-run                 # només llista les execucions i el seu estat
    python -m experiments.launch --workers 12              # DDPG llavor 0 + 4 variants × llavors de config.py
    python -m experiments.launch --workers 12 --seeds 0 1 2 3 4 5 6 7 8 9

Salta les execucions ja completades (config.json amb completed: true). Una carpeta
incompleta no s'esborra ni es reprèn: només s'avisa (run_experiment.py no sobreescriu).
El stdout/stderr de cada execució va a <results>/<nom>/seed_<n>.log.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from experiments.config import VARIANTS, ExperimentConfig

# Arrel del projecte (lab7/): els mòduls es criden com a paquet des d'aquí.
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Job:
    algo: str      # "ddpg" o "td3"
    variant: str   # només rellevant per a td3
    seed: int

    @property
    def name(self) -> str:
        """Nom de la carpeta de l'execució (com a run_experiment.py)."""
        return "ddpg" if self.algo == "ddpg" else self.variant

    def run_dir(self, results: Path) -> Path:
        return results / self.name / f"seed_{self.seed}"

    def log_path(self, results: Path) -> Path:
        return results / self.name / f"seed_{self.seed}.log"


def default_jobs(seeds: tuple, names: Optional[list] = None) -> list:
    """DDPG (llavor 0) i cada variant de TD3 per a cada llavor; `names` filtra ("ddpg" o noms de variant)."""
    jobs = [Job("ddpg", "full", 0)]
    jobs += [Job("td3", variant, seed) for variant in VARIANTS for seed in seeds]
    return jobs if names is None else [job for job in jobs if job.name in names]


def job_status(job: Job, results: Path) -> str:
    """'done' (completed: true), 'incomplete' (carpeta sense completar) o 'pending'."""
    run_dir = job.run_dir(results)
    if not run_dir.exists():
        return "pending"
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "incomplete"
    return "done" if config.get("completed") is True else "incomplete"


def build_command(job: Job, results: Path, extra: list) -> list:
    cmd = [sys.executable, "-m", "experiments.run_experiment", "--algo", job.algo, "--seed", str(job.seed),
           "--threads", "1", "--device", "cpu", "--results", str(results)]
    if job.algo == "td3":
        cmd += ["--variant", job.variant]
    return cmd + extra


def run_job(job: Job, results: Path, extra: list) -> int:
    """Executa un treball com a subprocés i en desa la sortida al log."""
    log = job.log_path(results)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    with log.open("a", encoding="utf-8") as stream:
        code = subprocess.run(build_command(job, results, extra), cwd=ROOT, env=env,
                              stdout=stream, stderr=subprocess.STDOUT).returncode
    print(f"[{'ok' if code == 0 else 'ERROR'}] {job.name} seed {job.seed} (codi {code}, log {log})", flush=True)
    return code


def extra_args(args: argparse.Namespace) -> list:
    """Opcions de run_experiment.py que només es passen si l'usuari les ha indicat."""
    pairs = (("--steps", args.steps), ("--eval-every", args.eval_every), ("--eval-episodes", args.eval_episodes))
    return [token for flag, value in pairs if value is not None for token in (flag, str(value))]


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8, help="màxim d'execucions simultànies")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ExperimentConfig().seeds))
    parser.add_argument("--variants", nargs="+", choices=["ddpg", *VARIANTS], default=None,
                        help="només aquestes execucions (p. ex. ddpg full no_delay); per defecte totes")
    parser.add_argument("--results", type=Path, default=ExperimentConfig().results_dir)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="només llista les execucions")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    jobs = default_jobs(tuple(args.seeds), args.variants)
    status = {job: job_status(job, args.results) for job in jobs}
    todo = [job for job in jobs if status[job] == "pending"]
    for job in jobs:
        print(f"{status[job]:<10} {job.name:<13} seed {job.seed}")
        if status[job] == "incomplete":
            print(f"AVÍS: {job.run_dir(args.results)} existeix però no està completada; no s'executa ni s'esborra.")
    print(f"{len(todo)} per executar, {sum(s == 'done' for s in status.values())} completades, "
          f"{sum(s == 'incomplete' for s in status.values())} incompletes.")
    if args.dry_run or not todo:
        return 0
    extra = extra_args(args)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(lambda job: run_job(job, args.results, extra), todo))
    return 1 if any(codes) else 0


if __name__ == "__main__":
    sys.exit(main())
