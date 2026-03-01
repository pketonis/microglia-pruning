"""Experiment tracking abstraction with optional Weights & Biases integration."""

from __future__ import annotations

from typing import Any, Dict, Optional


class ExperimentTracker:
    """Light wrapper around W&B that gracefully no-ops when disabled."""

    def __init__(
        self,
        enabled: bool = False,
        project: str = "microglia-pruning",
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.enabled = enabled
        self._wandb = None
        self._run = None
        if enabled:
            import wandb  # lazy import so core workflows don't require wandb

            self._wandb = wandb
            self._run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=config or {},
            )

    def log(self, payload: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.log(payload, step=step)

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            self._run.finish()
