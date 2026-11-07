"""Prova de fum a LunarLander-v3 real (uns 2.000 passos). Necessita Box2D."""
import csv
import json

import pytest

pytest.importorskip("Box2D")

from experiments.run_experiment import CHECKPOINT_NAME, main

STEPS = ["--steps", "2000", "--eval-every", "1000", "--eval-episodes", "1"]


def _evaluations(run_dir):
    with open(run_dir / "evaluations.csv", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_smoke_ddpg_td3_and_branch(tmp_path):
    results = ["--results", str(tmp_path)]
    ddpg_dir = main(["--algo", "ddpg", "--seed", "0", *STEPS, *results])
    full_dir = main(["--algo", "td3", "--variant", "full", "--seed", "0", *STEPS, *results])

    for run_dir in (ddpg_dir, full_dir):
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        assert config["completed"] and config["final_step"] == 2000
        rows = _evaluations(run_dir)
        assert [r["transitions"] for r in rows] == ["1000", "2000"]
        assert rows[-1]["bias_s0"] != ""   # el diagnòstic de sobreestimació s'ha calculat
    assert (full_dir / CHECKPOINT_NAME).exists()

    branch_dir = main(["--algo", "td3", "--variant", "no_minimum", "--seed", "0", *STEPS,
                       "--results", str(tmp_path / "branched"), "--resume", str(full_dir / CHECKPOINT_NAME)])
    config = json.loads((branch_dir / "config.json").read_text(encoding="utf-8"))
    assert config["branch_step"] > 0 and config["switches"]["use_minimum"] is False
    assert len(_evaluations(branch_dir)) == 2
