"""Experiment tracking abstraction with optional Weights & Biases integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json
import os


class ExperimentTracker:
    """Light wrapper around W&B that gracefully no-ops when disabled."""

    def __init__(
        self,
        enabled: bool = False,
        project: str = "microglia-pruning",
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        group: Optional[str] = None,
        tags: Optional[list[str]] = None,
        output_dir: str = "wandb_offline",
    ) -> None:
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._fallback_path = self.output_dir / "tracker_fallback.jsonl"
        self._wandb = None
        self._run = None
        self._local_buffer: list[Dict[str, Any]] = []
        if enabled:
            import wandb  # lazy import so core workflows don't require wandb

            self._wandb = wandb
            self._run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=config or {},
                group=group,
                tags=tags or [],
            )

    def _write_fallback(self, payload: Dict[str, Any], step: Optional[int] = None) -> None:
        record = {"step": step, **payload}
        self._local_buffer.append(record)
        with self._fallback_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log(self, payload: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled and self._wandb is not None:
            try:
                self._wandb.log(payload, step=step)
            except Exception:
                self._write_fallback(payload, step=step)
            return
        self._write_fallback(payload, step=step)

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            try:
                self._run.finish()
            except Exception:
                pass
