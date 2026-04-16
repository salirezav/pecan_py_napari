"""Pipeline Recorder widget for capture/save/load/apply workflows."""

from __future__ import annotations

import json
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .logic import apply_pipeline_step
from .state import PIPELINE_STORE, PipelineStep


def _yaml_available() -> bool:
    try:
        import yaml  # noqa: F401

        return True
    except Exception:
        return False


class PipelineRecorderWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._building = False
        self._unsubscribe = PIPELINE_STORE.subscribe(self._rebuild_list)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(QLabel("Recorded adjustments (execution order)"))
        self._list = QListWidget()
        self._list.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._list)

        row = QHBoxLayout()
        self._btn_apply_selected = QPushButton("Apply selected step(s)")
        self._btn_apply_selected.clicked.connect(self._apply_selected)
        row.addWidget(self._btn_apply_selected)
        self._btn_apply_all = QPushButton("Apply pipeline on video")
        self._btn_apply_all.clicked.connect(self._apply_all_checked)
        row.addWidget(self._btn_apply_all)
        layout.addLayout(row)

        row2 = QHBoxLayout()
        self._btn_save = QPushButton("Save selected as YAML/JSON")
        self._btn_save.clicked.connect(self._save_selected)
        row2.addWidget(self._btn_save)
        self._btn_load = QPushButton("Load pipeline")
        self._btn_load.clicked.connect(self._load_pipeline)
        row2.addWidget(self._btn_load)
        self._btn_clear = QPushButton("Clear")
        self._btn_clear.clicked.connect(lambda: PIPELINE_STORE.clear())
        row2.addWidget(self._btn_clear)
        layout.addLayout(row2)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #888;")
        layout.addWidget(self._status)
        layout.addStretch(1)
        self._rebuild_list()

    def closeEvent(self, event):  # noqa: N802
        try:
            self._unsubscribe()
        except Exception:
            pass
        super().closeEvent(event)

    def _rebuild_list(self) -> None:
        self._building = True
        try:
            self._list.clear()
            for i, step in enumerate(PIPELINE_STORE.steps):
                label = f"{i+1}. {step.description}"
                item = QListWidgetItem(label)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                item.setCheckState(Qt.CheckState.Checked if step.enabled else Qt.CheckState.Unchecked)
                self._list.addItem(item)
        finally:
            self._building = False

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._building:
            return
        idx = self._list.row(item)
        steps = PIPELINE_STORE.steps
        if not (0 <= idx < len(steps)):
            return
        steps[idx].enabled = item.checkState() == Qt.CheckState.Checked
        PIPELINE_STORE.set_steps(steps)

    def _selected_steps(self, checked_only: bool = True) -> list[dict]:
        out: list[dict] = []
        for s in PIPELINE_STORE.steps:
            if checked_only and not s.enabled:
                continue
            out.append(s.to_dict())
        return out

    def _save_selected(self) -> None:
        steps = self._selected_steps(checked_only=True)
        if not steps:
            self._status.setText("No checked steps to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save pipeline",
            "pipeline.yml",
            "YAML (*.yml *.yaml);;JSON (*.json)",
        )
        if not path:
            return
        payload = {"version": 1, "steps": steps}
        p = Path(path)
        try:
            if p.suffix.lower() == ".json":
                p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            else:
                if not _yaml_available():
                    raise RuntimeError("PyYAML is not installed. Save as .json or install pyyaml.")
                import yaml

                p.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        except Exception as exc:
            self._status.setText(f"Could not save pipeline: {exc}")
            return
        self._status.setText(f"Saved {len(steps)} step(s) to {p.name}.")

    def _load_pipeline(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load pipeline",
            "",
            "YAML/JSON (*.yml *.yaml *.json)",
        )
        if not path:
            return
        p = Path(path)
        try:
            txt = p.read_text(encoding="utf-8")
            if p.suffix.lower() == ".json":
                raw = json.loads(txt)
            else:
                if not _yaml_available():
                    raise RuntimeError("PyYAML is not installed. Load a .json file or install pyyaml.")
                import yaml

                raw = yaml.safe_load(txt)
            steps_raw = list((raw or {}).get("steps", []))
            steps = [PipelineStep.from_dict(x) for x in steps_raw if isinstance(x, dict)]
        except Exception as exc:
            self._status.setText(f"Could not load pipeline: {exc}")
            return
        PIPELINE_STORE.set_steps(steps)
        self._status.setText(f"Loaded {len(steps)} step(s) from {p.name}.")

    def _apply_selected(self) -> None:
        idx = self._list.currentRow()
        steps = PIPELINE_STORE.steps
        if not (0 <= idx < len(steps)):
            self._status.setText("Select a step first.")
            return
        try:
            msg = apply_pipeline_step(self._viewer, steps[idx].to_dict())
        except Exception as exc:
            self._status.setText(f"Apply failed: {exc}")
            return
        self._status.setText(msg)

    def _apply_all_checked(self) -> None:
        steps = self._selected_steps(checked_only=True)
        if not steps:
            self._status.setText("No checked steps to apply.")
            return
        applied = 0
        messages = []
        for st in steps:
            try:
                messages.append(apply_pipeline_step(self._viewer, st))
                applied += 1
            except Exception as exc:
                messages.append(f"Skipped {st.get('description', st.get('kind'))}: {exc}")
        self._status.setText(f"Applied {applied}/{len(steps)} step(s). {messages[-1] if messages else ''}")
