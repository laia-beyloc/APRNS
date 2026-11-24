"""Figures de l'informe a partir dels resultats reals (results/, analysis/tables/ i analysis/prof/tables/).

Ús (des de lab7/):  python informe/figuras/figures.py
Sortida: informe/img/fig_ddpg_td3.png i informe/img/fig_ablacio.png
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # lab7/
RESULTS, TABLES, OUT = ROOT / "results", ROOT / "analysis" / "tables", ROOT / "informe" / "img"
PROF_TABLES = ROOT / "analysis" / "prof" / "tables"   # configuració B (prefix prof_)
VARIANTS = {"full": "TD3 complet", "no_minimum": "sense mínim", "no_smoothing": "sense suavitzat",
            "no_delay": "sense retard"}
COLORS = {"full": "C0", "no_minimum": "C1", "no_smoothing": "C2", "no_delay": "C3", "ddpg": "C7", "prof_full": "C4"}
SMOOTH = 10   # finestra mòbil (avaluacions) de la fracció d'episodis resolts

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7, "lines.linewidth": 1.0})


def read(path: Path) -> list:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def column(rows: list, name: str) -> np.ndarray:
    return np.array([float(r[name]) if r[name] not in ("", "nan") else np.nan for r in rows])


def thousands(ax) -> None:
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
    ax.set_xlabel("Transicions (milers)")


def fig_ddpg_td3() -> None:
    """Retorn d'avaluació i biaix Q − G de DDPG i TD3 complet (configuracions A i B), llavor 0."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    for name, label in (("ddpg", "DDPG"), ("full", "TD3 (A)"), ("prof_full", "TD3 (B)")):
        rows = read(RESULTS / name / "seed_0" / "evaluations.csv")
        x = column(rows, "transitions")
        axes[0].plot(x, column(rows, "eval_mean_return"), color=COLORS[name], label=label)
        axes[1].plot(x, column(rows, "bias_all"), color=COLORS[name], label=label)
    axes[0].set(ylabel="Retorn d'avaluació", ylim=(-600, 320), title="(a) Retorn (mitjana de 5 episodis)")
    axes[0].axhline(200, color="k", ls="--", lw=0.6)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set(ylabel=r"$Q_1(s,\mu(s)) - G$", ylim=(-60, 200), title="(b) Biaix d'estimació del crític")
    for ax in axes:
        thousands(ax)
    axes[0].legend(loc="lower right")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig_ddpg_td3.png", dpi=300)
    plt.close(fig)


def fig_ablacio() -> None:
    """Fila superior configuració A, inferior B: (a) retorn mitjà amb IC 95 %, (b) fracció
    d'episodis resolts, (c) biaix mitjà; 10 llavors per variant."""
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.25), sharex=True)
    for row, (prefix, tables, name) in enumerate((("", TABLES, "A"), ("prof_", PROF_TABLES, "B"))):
        curves = read(tables / "ablation_curves.csv")
        bias = read(tables / "diagnostic_bias_all.csv")
        ax = axes[row]
        for variant, label in VARIANTS.items():
            run = prefix + variant
            sub = [r for r in curves if r["variant"] == run]
            x = column(sub, "transitions")
            ax[0].plot(x, column(sub, "mean"), color=COLORS[variant], label=label, lw=0.8)
            ax[0].fill_between(x, column(sub, "ci_low"), column(sub, "ci_high"), color=COLORS[variant], alpha=0.15, lw=0)
            solved = []
            for seed_dir in sorted((RESULTS / run).glob("seed_*")):
                rows = read(seed_dir / "evaluations.csv")
                solved.append([np.mean(np.array(json.loads(r["eval_returns"])) >= 200) for r in rows])
            kernel = np.ones(SMOOTH) / SMOOTH
            ax[1].plot(x[SMOOTH - 1:], np.convolve(np.mean(solved, 0), kernel, mode="valid") * 100, color=COLORS[variant])
            sub = [r for r in bias if r["variant"] == run]
            ax[2].plot(column(sub, "transitions"), column(sub, "mean"), color=COLORS[variant])
        ax[0].axhline(200, color="k", ls="--", lw=0.6)
        ax[0].set(ylabel="Retorn d'avaluació", ylim=(-500, 300), title=f"({name}) Retorn mitjà i IC 95 %")
        ax[1].set(ylabel="Episodis resolts (%)", ylim=(0, 100), title=f"({name}) Episodis amb retorn ≥ 200")
        ax[2].axhline(0, color="k", lw=0.6)
        ax[2].set(ylabel=r"$Q_1(s,\mu(s)) - G$", ylim=(-40, 100), title=f"({name}) Biaix d'estimació mitjà")
    for ax in axes[1]:
        thousands(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.945))
    fig.savefig(OUT / "fig_ablacio.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_ddpg_td3()
    fig_ablacio()
