import json
from src.rigor.tracking import ExperimentTracker


def test_tracker_writes_fallback_when_disabled(tmp_path):
    tracker = ExperimentTracker(enabled=False, output_dir=str(tmp_path))
    tracker.log({"metric": 1.5}, step=3)
    tracker.finish()

    fallback = tmp_path / "tracker_fallback.jsonl"
    assert fallback.exists()
    rows = fallback.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0]) == {"step": 3, "metric": 1.5}


def test_tracker_enabled_uses_wandb_and_falls_back_on_log_error(tmp_path, monkeypatch):
    class FakeRun:
        def __init__(self):
            self.finished = False

        def finish(self):
            raise RuntimeError("finish failure")

    class FakeWandb:
        def __init__(self):
            self.run = FakeRun()

        def init(self, **kwargs):
            return self.run

        def log(self, payload, step=None):
            raise RuntimeError("log failure")

    fake_wandb = FakeWandb()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    tracker = ExperimentTracker(enabled=True, output_dir=str(tmp_path), project="p")
    tracker.log({"loss": 0.2}, step=5)
    tracker.finish()  # should swallow finish exception

    rows = (tmp_path / "tracker_fallback.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(rows[0]) == {"step": 5, "loss": 0.2}
