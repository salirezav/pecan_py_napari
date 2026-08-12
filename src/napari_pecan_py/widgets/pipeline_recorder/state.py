"""Shared in-memory pipeline state for the Pipeline Recorder widget."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable

ROOT_PLACEHOLDER = "$root"
_LAYER_PARAM_KEYS = (
    "source_layer",
    "output_layer",
    "output_mask_layer",
    "mask_layer",
    "a_layer",
    "b_layer",
    "edge_layer",
    "limit_mask_layer",
    "ellipse_layer",
    "shapes_layer",
    "output_shapes_layer",
)


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
        self.root_layer: str | None = None

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
        self.root_layer = None
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

_pipeline_applying = False
_last_pipeline_record_mono = 0.0
_PIPELINE_RECORD_SUPPRESS_SEC = 0.5


def set_pipeline_applying(active: bool) -> None:
    global _pipeline_applying
    _pipeline_applying = bool(active)


def is_pipeline_applying() -> bool:
    return _pipeline_applying


def mark_pipeline_recorded() -> None:
    global _last_pipeline_record_mono
    _last_pipeline_record_mono = time.monotonic()


def is_recent_pipeline_record() -> bool:
    return (time.monotonic() - _last_pipeline_record_mono) < _PIPELINE_RECORD_SUPPRESS_SEC


def _infer_root_from_params(params: dict[str, Any]) -> str | None:
    for key in _LAYER_PARAM_KEYS:
        value = params.get(key)
        if not value or str(value) == ROOT_PLACEHOLDER:
            continue
        return str(value).split(" - ")[0]
    return None


def infer_recorded_root(steps: list[dict[str, Any]], explicit: str | None = None) -> str | None:
    """Infer the recorded video stem used as a placeholder during replay."""
    if explicit and str(explicit) != ROOT_PLACEHOLDER:
        return str(explicit)
    names: list[str] = []
    for step in steps or []:
        params = step.get("params") or {}
        for key in _LAYER_PARAM_KEYS:
            value = params.get(key)
            if value and str(value) != ROOT_PLACEHOLDER:
                names.append(str(value))
    if not names:
        return None
    roots: dict[str, int] = {}
    for name in names:
        root = name.split(" - ")[0]
        score = sum(
            1
            for other in names
            if other == root or other.startswith(f"{root} - ") or root in other
        )
        roots[root] = max(roots.get(root, 0), score)
    return max(roots, key=roots.get)


def record_pipeline_step(kind: str, description: str, params: dict[str, Any]) -> None:
    mark_pipeline_recorded()
    if PIPELINE_STORE.root_layer is None:
        root = _infer_root_from_params(params or {})
        if root:
            PIPELINE_STORE.root_layer = root
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
    mark_pipeline_recorded()
    if PIPELINE_STORE.root_layer is None:
        root = _infer_root_from_params(params or {})
        if root:
            PIPELINE_STORE.root_layer = root
    PIPELINE_STORE.upsert_step(
        match=match,
        new_step=PipelineStep(
            kind=kind,
            description=description,
            params=dict(params or {}),
            enabled=True,
        ),
    )


def rename_layer_references(old_name: str, new_name: str) -> int:
    """Rewrite exact layer-name references across recorded pipeline steps.

    Updates every known layer-name param key and any description text that
    contains the old name (avoiding accidental rewrites of ``name [N]`` when
    renaming the unindexed base name). Returns the number of steps changed.
    """
    old = str(old_name or "")
    new = str(new_name or "")
    if not old or not new or old == new:
        return 0

    # Do not turn "video - adjusted [2]" into "video - glare [2]" when renaming
    # the base name "video - adjusted".
    desc_pattern = re.compile(re.escape(old) + r"(?! \[\d+\])")

    changed = 0
    updated: list[PipelineStep] = []
    for st in PIPELINE_STORE.steps:
        params = dict(st.params or {})
        step_changed = False
        for key in _LAYER_PARAM_KEYS:
            if str(params.get(key, "")) == old:
                params[key] = new
                step_changed = True
        desc = str(st.description or "")
        new_desc = desc_pattern.sub(new, desc)
        if new_desc != desc:
            desc = new_desc
            step_changed = True
        if step_changed:
            changed += 1
            updated.append(
                PipelineStep(
                    kind=st.kind,
                    description=desc,
                    params=params,
                    enabled=bool(st.enabled),
                )
            )
        else:
            updated.append(st)

    if changed:
        mark_pipeline_recorded()
        PIPELINE_STORE.set_steps(updated)
    return changed
