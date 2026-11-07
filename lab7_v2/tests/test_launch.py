"""Tests del llançador: llista de treballs, salt de completades i dry-run (sense entrenar)."""
import json

from experiments import launch
from experiments.config import VARIANTS, ExperimentConfig


def make_run(results, job, completed):
    run_dir = job.run_dir(results)
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"completed": completed}), encoding="utf-8")


def test_default_jobs_list():
    seeds = ExperimentConfig().seeds
    jobs = launch.default_jobs(seeds)
    assert len(jobs) == 1 + len(VARIANTS) * len(seeds)
    assert jobs[0] == launch.Job("ddpg", "full", 0)
    assert {j.name for j in jobs} == {"ddpg", *VARIANTS}


def test_status_and_command(tmp_path):
    done, partial, new = launch.Job("td3", "full", 0), launch.Job("td3", "full", 1), launch.Job("td3", "full", 2)
    make_run(tmp_path, done, True)
    make_run(tmp_path, partial, False)
    assert [launch.job_status(j, tmp_path) for j in (done, partial, new)] == ["done", "incomplete", "pending"]
    cmd = launch.build_command(new, tmp_path, ["--steps", "10"])
    assert cmd[-2:] == ["--steps", "10"]
    assert cmd[cmd.index("--variant") + 1] == "full" and cmd[cmd.index("--threads") + 1] == "1"


def test_dry_run_does_not_launch(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launch, "run_job", lambda *a: (_ for _ in ()).throw(AssertionError("no s'ha de llançar")))
    assert launch.main(["--dry-run", "--results", str(tmp_path)]) == 0
    assert "21 per executar" in capsys.readouterr().out


def test_skips_completed_and_warns_incomplete(tmp_path, monkeypatch, capsys):
    make_run(tmp_path, launch.Job("ddpg", "full", 0), True)
    make_run(tmp_path, launch.Job("td3", "full", 0), False)
    launched = []
    monkeypatch.setattr(launch, "run_job", lambda job, results, extra: launched.append(job) or 0)
    assert launch.main(["--seeds", "0", "1", "--results", str(tmp_path), "--workers", "2"]) == 0
    names = {(j.name, j.seed) for j in launched}
    assert ("ddpg", 0) not in names and ("full", 0) not in names
    assert ("full", 1) in names and len(launched) == 1 + 4 * 2 - 2
    assert "AVÍS" in capsys.readouterr().out
    assert (tmp_path / "full" / "seed_0" / "config.json").exists()  # no s'ha esborrat res
