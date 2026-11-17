"""Prova de fum a LunarLander-v3 real (uns 2.000 passos). Necessita Box2D."""
import csv
import json

import pytest

pytest.importorskip("Box2D")

from experiments.run_experiment import CHECKPOINT_NAME, FINAL_EVALUATION_CSV, FINAL_MODEL_NAME, main

STEPS = ["--steps", "2000", "--eval-every", "1000", "--eval-episodes", "1", "--final-eval-episodes", "2"]


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
    for run_dir in (ddpg_dir, full_dir):   # política final desada i avaluada amb llavors noves
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        assert (run_dir / FINAL_MODEL_NAME).exists() and config["final_evaluation"]["episodes"] == 2
        assert len((run_dir / FINAL_EVALUATION_CSV).read_text(encoding="utf-8").splitlines()) == 3

    branch_dir = main(["--algo", "td3", "--variant", "no_minimum", "--seed", "0", *STEPS,
                       "--results", str(tmp_path / "branched"), "--resume", str(full_dir / CHECKPOINT_NAME)])
    config = json.loads((branch_dir / "config.json").read_text(encoding="utf-8"))
    assert config["branch_step"] > 0 and config["switches"]["use_minimum"] is False
    assert len(_evaluations(branch_dir)) == 2


def test_continue_skips_completed_and_restarts_partial(tmp_path):
    args = ["--algo", "ddpg", "--seed", "0", *STEPS, "--results", str(tmp_path), "--continue"]
    run_dir = main(args)
    config_path = run_dir / "config.json"
    before = config_path.read_text(encoding="utf-8")
    assert not (run_dir / "checkpoint_latest.pt").exists()   # s'esborra en acabar
    main(args)                                                # completada: no es toca
    assert config_path.read_text(encoding="utf-8") == before

    # Carpeta a mitges sense checkpoint (tall abans del primer checkpoint periòdic).
    config_path.write_text(json.dumps({"completed": False}), encoding="utf-8")
    (run_dir / "evaluations.csv").write_text("brossa\n", encoding="utf-8")
    main(args)
    assert json.loads(config_path.read_text(encoding="utf-8"))["completed"] is True
    assert [r["transitions"] for r in _evaluations(run_dir)] == ["1000", "2000"]


def test_smoke_sanity_config(tmp_path):
    run_dir = main(["--algo", "td3", "--variant", "td3_paper", "--seed", "0", *STEPS, "--results", str(tmp_path)])
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["completed"] and config["tau"] == 0.005 and config["agent_kwargs"]["batch_size"] == 100
    assert config["switches"] == {"use_minimum": True, "target_noise": 0.2, "policy_delay": 2}
    assert config["experiment"]["random_start_steps"] == 10_000
    assert config["critic_updates"] == 0          # 2.000 passos < 10.000 passos aleatoris inicials
    assert not (run_dir / CHECKPOINT_NAME).exists()  # la branca de l'apartat 4 només és de "full"
