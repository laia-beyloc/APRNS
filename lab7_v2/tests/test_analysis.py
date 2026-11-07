"""Tests d'analysis.py amb CSV sintètics (sense entrenar)."""
import csv
import json

import numpy as np

import analysis
from config import ExperimentConfig


def make_run(root, name, seed, offset, completed=True, n=30):
    run_dir = root / name / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"completed": completed, "seed": seed, "variant": name}), encoding="utf-8")
    with (run_dir / "evaluations.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["transitions", "eval_mean_return", "bias_all", "bias_s0",
                    "critic_disagreement", "min_reduction"])
        for i in range(n):
            w.writerow([(i + 1) * 1000, offset + i + seed, 1.0, "", 0.5, 0.1])


def test_only_completed_runs_loaded(tmp_path):
    make_run(tmp_path, "full", 0, 0)
    make_run(tmp_path, "full", 1, 0, completed=False)
    runs = analysis.load_runs(tmp_path)
    assert list(runs["full"]) == [0]
    assert np.isnan(runs["full"][0]["bias_s0"]).all()


def test_run_metrics_constant_return():
    x = np.arange(1, 11) * 1000.0
    m = analysis.run_metrics(x, np.full(10, 5.0), last_n=3, total_steps=x[-1] - x[0])
    assert m["last_n_mean"] == 5.0 and np.isclose(m["auc"], 5.0)


def test_bootstrap_deterministic_and_contains_mean():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    a = analysis.bootstrap_ci(values, 500, np.random.default_rng(1))
    b = analysis.bootstrap_ci(values, 500, np.random.default_rng(1))
    assert a == b and a[0] <= values.mean() <= a[1]


def test_paired_difference_exact_when_constant_shift(tmp_path):
    for s in range(3):
        make_run(tmp_path, "full", s, 0)
        make_run(tmp_path, "no_delay", s, 10)
    runs = analysis.load_runs(tmp_path)
    per_run = analysis.per_run_table(runs, ["full", "no_delay"], 10, 30_000)
    rows = analysis.paired_table(per_run, 200, np.random.default_rng(0))
    last = next(r for r in rows if r["metric"] == "last_n_mean")
    assert np.isclose(last["mean_diff"], 10) and np.isclose(last["ci_low"], 10)


def test_run_analysis_writes_outputs(tmp_path):
    res = tmp_path / "results"
    for s in range(3):
        for name, off in (("full", 0), ("no_minimum", 5), ("no_smoothing", 2), ("no_delay", 1)):
            make_run(res, name, s, off)
    make_run(res, "ddpg", 0, -3)
    out = tmp_path / "out"
    analysis.run_analysis(res, out, n_boot=50)
    for f in ("figures/exploratory_ddpg_vs_td3.png", "figures/ablation_curves.png",
              "tables/metrics_by_variant.csv", "tables/paired_differences_vs_full.csv",
              "tables/diagnostic_bias_all.csv", "figures/diagnostic_min_reduction.png"):
        assert (out / f).exists(), f
    assert ExperimentConfig().summary_last_n == 10
