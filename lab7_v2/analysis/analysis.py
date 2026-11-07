"""Anàlisi dels resultats: genera figures (PNG) i taules (CSV) de l'informe.

Només numpy/matplotlib. No genera text de discussió: això ho escriu l'usuari.
Només s'usen execucions amb config.json["completed"] == True.

Ús (des de lab7/):  python -m analysis.analysis --results results --out analysis
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.config import VARIANTS, ExperimentConfig

BOOTSTRAP_SEED = 12345
DIAGNOSTICS = ("bias_all", "bias_s0", "critic_disagreement", "min_reduction")
METRICS = ("last_n_mean", "auc")


# ---------------------------------------------------------------- càrrega

def read_csv_columns(path: Path) -> dict:
    """Llegeix un CSV com a dict columna -> array float (buit/no numèric -> NaN)."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    columns = {}
    for name in rows[0].keys() if rows else []:
        values = []
        for row in rows:
            try:
                values.append(float(row[name]))
            except (TypeError, ValueError):
                values.append(np.nan)
        columns[name] = np.array(values)
    return columns


def load_runs(results_dir: Path) -> dict:
    """Carrega les execucions completades: {variant: {seed: columnes}}."""
    runs = {}
    for config_path in sorted(Path(results_dir).glob("*/seed_*/config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        csv_path = config_path.parent / "evaluations.csv"
        if not config.get("completed") or not csv_path.exists():
            continue
        columns = read_csv_columns(csv_path)
        if "transitions" not in columns:
            continue
        variant = config.get("variant", config_path.parent.parent.name)
        runs.setdefault(variant, {})[int(config["seed"])] = columns
    return runs


# ---------------------------------------------------------------- estadística

def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator,
                 alpha: float = 0.05) -> tuple:
    """IC bootstrap percentil de la mitjana. :param values: [n_seeds] o [n_seeds, T].

    Remostreja entre llavors (files). Retorna (low, high) amb la forma de values[0].
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = np.nanmean(values[idx], axis=1)  # [n_boot, ...]
    low, high = np.nanpercentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return low, high


def stack_metric(seed_runs: dict, column: str) -> tuple:
    """Apila una columna de totes les llavors sobre les transicions comunes.

    :return: (seeds, transitions [T], matriu [n_seeds, T])
    """
    seeds = sorted(seed_runs)
    common = seed_runs[seeds[0]]["transitions"]
    for seed in seeds[1:]:
        common = np.intersect1d(common, seed_runs[seed]["transitions"])
    rows = []
    for seed in seeds:
        run = seed_runs[seed]
        keep = np.isin(run["transitions"], common)
        values = run.get(column)
        rows.append(values[keep] if values is not None else np.full(len(common), np.nan))
    return seeds, common, np.vstack(rows)


def run_metrics(transitions: np.ndarray, returns: np.ndarray, last_n: int,
                total_steps: float) -> dict:
    """Mètriques pre-registrades d'una execució: mitjana de les últimes N i AUC/T."""
    x, y = np.asarray(transitions, float), np.asarray(returns, float)
    auc = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))
    return {"last_n_mean": float(np.mean(y[-last_n:])), "auc": auc / total_steps}


# ---------------------------------------------------------------- taules

def write_table(path: Path, rows: list) -> None:
    """Escriu una llista de dicts com a CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def per_run_table(runs: dict, variants: list, last_n: int, total_steps: float) -> list:
    """Una fila per (variant, llavor) amb les dues mètriques."""
    rows = []
    for variant in variants:
        for seed, run in sorted(runs[variant].items()):
            metrics = run_metrics(run["transitions"], run["eval_mean_return"], last_n, total_steps)
            rows.append({"variant": variant, "seed": seed, **metrics})
    return rows


def summary_table(per_run: list, n_boot: int, rng: np.random.Generator) -> list:
    """Mitjana i IC 95 % bootstrap per variant i mètrica."""
    rows = []
    for variant in dict.fromkeys(r["variant"] for r in per_run):
        for metric in METRICS:
            values = np.array([r[metric] for r in per_run if r["variant"] == variant])
            low, high = bootstrap_ci(values, n_boot, rng)
            rows.append({"variant": variant, "metric": metric, "n_seeds": len(values),
                         "mean": values.mean(), "ci_low": low, "ci_high": high})
    return rows


def paired_table(per_run: list, n_boot: int, rng: np.random.Generator,
                 baseline: str = "full") -> list:
    """Diferències per llavor (variant − full) amb IC bootstrap de la mitjana."""
    by_key = {(r["variant"], r["seed"]): r for r in per_run}
    rows = []
    for variant in dict.fromkeys(r["variant"] for r in per_run):
        if variant == baseline:
            continue
        seeds = sorted(s for (v, s) in by_key if v == variant and (baseline, s) in by_key)
        if not seeds:
            continue
        for metric in METRICS:
            diff = np.array([by_key[(variant, s)][metric] - by_key[(baseline, s)][metric]
                             for s in seeds])
            low, high = bootstrap_ci(diff, n_boot, rng)
            rows.append({"variant": variant, "metric": metric, "n_seeds": len(seeds),
                         "mean_diff": diff.mean(), "ci_low": low, "ci_high": high})
    return rows


def curve_table(runs: dict, variants: list, column: str, n_boot: int,
                rng: np.random.Generator) -> list:
    """Corba mitjana amb IC bootstrap entre llavors, per variant."""
    rows = []
    for variant in variants:
        _, x, matrix = stack_metric(runs[variant], column)
        if np.all(np.isnan(matrix)):
            continue
        low, high = bootstrap_ci(matrix, n_boot, rng)
        mean = np.nanmean(matrix, axis=0)
        rows += [{"variant": variant, "transitions": int(t), "mean": m, "ci_low": lo, "ci_high": hi}
                 for t, m, lo, hi in zip(x, mean, low, high)]
    return rows


# ---------------------------------------------------------------- figures

def plot_curves(rows: list, ylabel: str, title: str, path: Path) -> None:
    """Dibuixa corbes mitjana + banda per variant des d'una curve_table."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for variant in dict.fromkeys(r["variant"] for r in rows):
        sub = [r for r in rows if r["variant"] == variant]
        x = [r["transitions"] for r in sub]
        ax.plot(x, [r["mean"] for r in sub], label=variant)
        ax.fill_between(x, [r["ci_low"] for r in sub], [r["ci_high"] for r in sub], alpha=0.2)
    ax.set(xlabel="Transicions", ylabel=ylabel, title=title)
    ax.legend()
    save(fig, path)


def plot_exploratory(runs: dict, path: Path, seed: int = 0) -> list:
    """DDPG vs TD3 full amb una sola llavor: sense bandes. Retorna les files del CSV."""
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ("ddpg", "full"):
        run = runs.get(name, {}).get(seed)
        if run is None:
            continue
        ax.plot(run["transitions"], run["eval_mean_return"], label=f"{name} (llavor {seed})")
        rows += [{"algorithm": name, "seed": seed, "transitions": int(t), "eval_mean_return": r}
                 for t, r in zip(run["transitions"], run["eval_mean_return"])]
    ax.set(xlabel="Transicions", ylabel="Retorn d'avaluació",
           title="Comparació exploratòria (una llavor)")
    ax.legend()
    save(fig, path)
    return rows


def plot_estimates(per_run: list, summary: list, paired: list, path: Path) -> None:
    """Mètriques per variant (punts per llavor + IC) i diferències aparellades."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for col, metric in enumerate(METRICS):
        ax = axes[0, col]
        for i, row in enumerate(r for r in summary if r["metric"] == metric):
            pts = [r[metric] for r in per_run if r["variant"] == row["variant"]]
            ax.scatter([i] * len(pts), pts, color="gray", s=12, alpha=0.6)
            ax.errorbar(i, row["mean"], yerr=[[row["mean"] - row["ci_low"]],
                        [row["ci_high"] - row["mean"]]], fmt="o", color="C0", capsize=4)
            ax.set_xticks(range(len([r for r in summary if r["metric"] == metric])))
            ax.set_xticklabels([r["variant"] for r in summary if r["metric"] == metric])
        ax.set(title=metric)
        ax = axes[1, col]
        sub = [r for r in paired if r["metric"] == metric]
        for i, row in enumerate(sub):
            ax.errorbar(i, row["mean_diff"], yerr=[[row["mean_diff"] - row["ci_low"]],
                        [row["ci_high"] - row["mean_diff"]]], fmt="o", capsize=4)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([r["variant"] for r in sub])
        ax.set(title=f"{metric}: variant − full")
    save(fig, path)


def save(fig, path: Path) -> None:
    """Desa la figura i la tanca."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- principal

def run_analysis(results_dir: Path, out_dir: Path, n_boot: int = 10_000,
                 seed: int = BOOTSTRAP_SEED) -> None:
    """Genera totes les figures i taules a out_dir/figures i out_dir/tables."""
    cfg = ExperimentConfig()
    rng = np.random.default_rng(seed)
    figures, tables = Path(out_dir) / "figures", Path(out_dir) / "tables"
    runs = load_runs(results_dir)
    variants = [v for v in VARIANTS if v in runs]

    explo = plot_exploratory(runs, figures / "exploratory_ddpg_vs_td3.png")
    if explo:
        write_table(tables / "exploratory_ddpg_vs_td3.csv", explo)
    if not variants:
        return

    curves = curve_table(runs, variants, "eval_mean_return", n_boot, rng)
    write_table(tables / "ablation_curves.csv", curves)
    plot_curves(curves, "Retorn d'avaluació mitjà", "Ablacions TD3 (IC 95 % bootstrap entre llavors)",
                figures / "ablation_curves.png")

    per_run = per_run_table(runs, variants, cfg.summary_last_n, cfg.total_steps)
    summary = summary_table(per_run, n_boot, rng)
    paired = paired_table(per_run, n_boot, rng)
    write_table(tables / "metrics_per_run.csv", per_run)
    write_table(tables / "metrics_by_variant.csv", summary)
    if paired:
        write_table(tables / "paired_differences_vs_full.csv", paired)
    plot_estimates(per_run, summary, paired, figures / "metrics_and_paired_differences.png")

    for column in DIAGNOSTICS:
        diag = curve_table(runs, variants, column, n_boot, rng)
        if diag:
            write_table(tables / f"diagnostic_{column}.csv", diag)
            plot_curves(diag, column, f"Diagnòstic: {column}", figures / f"diagnostic_{column}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("analysis"))
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    run_analysis(args.results, args.out, args.n_boot, args.seed)


if __name__ == "__main__":
    main()
