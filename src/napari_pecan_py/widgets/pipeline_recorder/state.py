"""Shared in-memory pipeline state for the Pipeline Recorder widget."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class PipelineStep:
    kind: str
    description: str
    params: dict[str, Any]
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "description": self.description,
            "params": self.params,
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PipelineStep":
        return cls(
            kind=str(raw.get("kind", "unknown")),
            description=str(raw.get("description", "")),
            params=dict(raw.get("params", {}) or {}),
            enabled=bool(raw.get("enabled", True)),
        )


class PipelineStore:
    """Simple observer-backed store used by all widgets."""

    def __init__(self) -> None:
        self._steps: list[PipelineStep] = []
        self._listeners: list[Callable[[], None]] = []

    @property
    def steps(self) -> list[PipelineStep]:
        return list(self._steps)

    def set_steps(self, steps: list[PipelineStep]) -> None:
        self._steps = list(steps)
        self._emit()

    def add_step(self, step: PipelineStep) -> None:
        self._steps.append(step)
        self._emit()

    def upsert_step(
        self,
        *,
        match,
        new_step: PipelineStep,
    ) -> None:
        """Replace first matching step, else append."""
        for i, st in enumerate(self._steps):
            try:
                ok = bool(match(st))
            except Exception:
                ok = False
            if ok:
                # Preserve enabled state when replacing.
                new_step.enabled = bool(st.enabled)
                self._steps[i] = new_step
                self._emit()
                return
        self._steps.append(new_step)
        self._emit()

    def clear(self) -> None:
        self._steps.clear()
        self._emit()

    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        self._listeners.append(callback)

        def _unsubscribe() -> None:
            if callback in self._listeners:
                self._listeners.remove(callback)

        return _unsubscribe

    def _emit(self) -> None:
        for cb in list(self._listeners):
            try:
                cb()
            except Exception:
                continue


PIPELINE_STORE = PipelineStore()


def record_pipeline_step(kind: str, description: str, params: dict[str, Any]) -> None:
    PIPELINE_STORE.add_step(
        PipelineStep(kind=kind, description=description, params=dict(params or {}), enabled=True)
    )


def upsert_pipeline_step(
    *,
    kind: str,
    description: str,
    params: dict[str, Any],
    match,
) -> None:
    PIPELINE_STORE.upsert_step(
        match=match,
        new_step=PipelineStep(
            kind=kind,
            description=description,
            params=dict(params or {}),
            enabled=True,
        ),
    )
