"""Pipeline Recorder widget for capture/save/load/apply workflows."""

from __future__ import annotations

import json
from pathlib import Path

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .logic import apply_pipeline_step_with_context, create_apply_context
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
        self._is_processing = False
        self._processing_steps: list[dict] = []
        self._processing_descriptions: list[str] = []
        self._processing_indices: list[int] = []
        self._processing_i = 0
        self._apply_ctx = None
        self._disabled_widgets: list[QWidget] = []
        self._cancel_requested = False

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
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.clicked.connect(self._request_stop)
        self._btn_stop.setEnabled(False)
        row.addWidget(self._btn_stop)
        layout.addLayout(row)

        row2 = QHBoxLayout()
        self._btn_save = QPushButton("Save selected as YAML/JSON")
        self._btn_save.clicked.connect(self._save_selected)
        row2.addWidget(self._btn_save)
        self._btn_load = QPushButton("Load pipeline")
        self._btn_load.clicked.connect(self._load_pipeline)
        row2.addWidget(self._btn_load)
        self._btn_edit = QPushButton("Edit selected step")
        self._btn_edit.clicked.connect(self._edit_selected_step)
        row2.addWidget(self._btn_edit)
        self._btn_clear = QPushButton("Clear")
        self._btn_clear.clicked.connect(lambda: PIPELINE_STORE.clear())
        row2.addWidget(self._btn_clear)
        layout.addLayout(row2)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #888;")
        layout.addWidget(self._status)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.hide()
        layout.addWidget(self._progress)
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
                label = self._format_step_label(i, step.description, "idle")
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

    def _format_step_label(self, idx: int, description: str, state: str) -> str:
        if state == "running":
            tag = "[RUN]"
        elif state == "done":
            tag = "[OK]"
        elif state == "error":
            tag = "[ERR]"
        else:
            tag = "[ ]"
        return f"{idx+1}. {tag} {description}"

    def _set_item_state(self, idx: int, state: str) -> None:
        if not (0 <= idx < self._list.count()):
            return
        item = self._list.item(idx)
        if item is None:
            return
        raw = PIPELINE_STORE.steps
        if 0 <= idx < len(raw):
            desc = raw[idx].description
        else:
            desc = item.text()
        item.setText(self._format_step_label(idx, desc, state))

    def _selected_steps(self, checked_only: bool = True) -> list[dict]:
        out: list[dict] = []
        for s in PIPELINE_STORE.steps:
            if checked_only and not s.enabled:
                continue
            out.append(s.to_dict())
        return out

    def _save_selected(self) -> None:
        if self._is_processing:
            self._status.setText("Pipeline is running. Wait for completion.")
            return
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
        if self._is_processing:
            self._status.setText("Pipeline is running. Wait for completion.")
            return
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

    def _edit_selected_step(self) -> None:
        if self._is_processing:
            self._status.setText("Pipeline is running. Wait for completion.")
            return
        idx = self._list.currentRow()
        steps = PIPELINE_STORE.steps
        if not (0 <= idx < len(steps)):
            self._status.setText("Select a step first.")
            return
        step = steps[idx]
        edited = self._edit_step_dialog(step)
        if edited is None:
            return
        steps[idx] = edited
        PIPELINE_STORE.set_steps(steps)
        self._list.setCurrentRow(idx)
        self._status.setText(f"Updated step {idx+1}: {edited.description}")

    def _edit_step_dialog(self, step: PipelineStep) -> PipelineStep | None:
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit step ({step.kind})")
        dlg.setMinimumWidth(700)
        lay = QVBoxLayout(dlg)

        lay.addWidget(QLabel("Description"))
        desc = QLineEdit(step.description)
        lay.addWidget(desc)

        lay.addWidget(QLabel("Params (JSON object)"))
        params_text = QPlainTextEdit()
        params_text.setPlainText(json.dumps(step.params, indent=2, sort_keys=True))
        params_text.setMinimumHeight(260)
        lay.addWidget(params_text)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() != QDialog.Accepted:
            return None

        new_desc = desc.text().strip() or step.description
        raw = params_text.toPlainText().strip()
        if not raw:
            new_params = {}
        else:
            try:
                parsed = json.loads(raw)
            except Exception as exc:
                QMessageBox.warning(self, "Invalid JSON", f"Could not parse params JSON:\n{exc}")
                return None
            if not isinstance(parsed, dict):
                QMessageBox.warning(self, "Invalid JSON", "Params must be a JSON object (dictionary).")
                return None
            new_params = parsed

        return PipelineStep(
            kind=step.kind,
            description=new_desc,
            params=new_params,
            enabled=step.enabled,
        )

    def _apply_selected(self) -> None:
        if self._is_processing:
            self._status.setText("Pipeline is already running.")
            return
        idx = self._list.currentRow()
        steps = PIPELINE_STORE.steps
        if not (0 <= idx < len(steps)):
            self._status.setText("Select a step first.")
            return
        self._start_processing([steps[idx].to_dict()], [steps[idx].description], [idx])

    def _apply_all_checked(self) -> None:
        if self._is_processing:
            self._status.setText("Pipeline is already running.")
            return
        steps = self._selected_steps(checked_only=True)
        if not steps:
            self._status.setText("No checked steps to apply.")
            return
        all_steps = PIPELINE_STORE.steps
        indices = [i for i, s in enumerate(all_steps) if s.enabled]
        descriptions = [all_steps[i].description for i in indices]
        for st in steps:
            if not isinstance(st, dict):
                self._status.setText("Apply failed: Invalid step format in pipeline.")
                return
        self._start_processing(steps, descriptions, indices)

    def _start_processing(self, steps: list[dict], descriptions: list[str], indices: list[int]) -> None:
        if not steps:
            return
        self._is_processing = True
        self._cancel_requested = False
        self._processing_steps = list(steps)
        self._processing_descriptions = list(descriptions)
        self._processing_indices = list(indices)
        self._processing_i = 0
        self._apply_ctx = create_apply_context(self._viewer)
        self._set_controls_enabled(False)
        self._progress.setValue(0)
        self._progress.show()
        for idx in self._processing_indices:
            self._set_item_state(idx, "idle")
        QTimer.singleShot(0, self._process_next_step)

    def _request_stop(self) -> None:
        if not self._is_processing:
            return
        self._cancel_requested = True
        self._status.setText("Stopping after current operation checkpoint...")

    def _is_cancel_requested(self) -> bool:
        return bool(self._cancel_requested)

    def _process_next_step(self) -> None:
        if not self._is_processing:
            return
        i = self._processing_i
        total = len(self._processing_steps)
        if i >= total:
            self._status.setText(f"Applied {total}/{total} step(s).")
            self._finish_processing()
            return
        step = self._processing_steps[i]
        desc = self._processing_descriptions[i] if i < len(self._processing_descriptions) else step.get("description", step.get("kind", "step"))
        idx = self._processing_indices[i] if i < len(self._processing_indices) else -1
        if idx >= 0:
            self._set_item_state(idx, "running")
        self._status.setText(f"Applying {desc} ({i+1}/{total})...")
        self._set_step_progress(i, total, 0, 1, "")
        QApplication.processEvents()
        try:
            msg = apply_pipeline_step_with_context(
                self._apply_ctx,
                step,
                progress_callback=self._on_step_progress,
                cancel_callback=self._is_cancel_requested,
            )
        except InterruptedError:
            if idx >= 0:
                self._set_item_state(idx, "idle")
            self._status.setText(f"Stopped at step {i+1}/{total}.")
            self._finish_processing()
            return
        except Exception as exc:
            if idx >= 0:
                self._set_item_state(idx, "error")
            self._status.setText(f"Apply failed at step {i+1}/{total}: {exc}")
            self._finish_processing()
            return
        if idx >= 0:
            self._set_item_state(idx, "done")
        self._status.setText(msg)
        self._set_step_progress(i, total, 1, 1, "")
        self._processing_i += 1
        QTimer.singleShot(0, self._process_next_step)

    def _on_step_progress(self, current: int, total: int, phase: str = "") -> None:
        if not self._is_processing:
            return
        if self._cancel_requested:
            raise InterruptedError("Pipeline apply cancelled by user.")
        step_idx = self._processing_i
        step_total = len(self._processing_steps)
        self._set_step_progress(step_idx, step_total, current, total, phase)
        QApplication.processEvents()

    def _set_step_progress(self, step_idx: int, step_total: int, current: int, total: int, phase: str) -> None:
        step_total_safe = max(1, int(step_total))
        total_safe = max(1, int(total))
        step_idx_clamped = max(0, min(int(step_idx), step_total_safe))
        current_clamped = max(0, min(int(current), total_safe))

        done = (step_idx_clamped + (current_clamped / total_safe)) / step_total_safe
        pct = max(0, min(100, int(round(done * 100.0))))
        self._progress.setValue(pct)
        self._progress.setFormat(f"{pct}%")

        if step_idx_clamped < step_total_safe:
            desc = self._processing_descriptions[step_idx_clamped]
            phase_txt = f" - {phase}" if phase else ""
            self._status.setText(
                f"Applying {desc} ({step_idx_clamped + 1}/{step_total_safe}){phase_txt}: "
                f"{current_clamped}/{total_safe}"
            )

    def _set_controls_enabled(self, enabled: bool) -> None:
        self._btn_apply_selected.setEnabled(enabled)
        self._btn_apply_all.setEnabled(enabled)
        self._btn_stop.setEnabled(not enabled)
        self._btn_save.setEnabled(enabled)
        self._btn_load.setEnabled(enabled)
        self._btn_edit.setEnabled(enabled)
        self._btn_clear.setEnabled(enabled)
        if enabled:
            for w in self._disabled_widgets:
                try:
                    w.setEnabled(True)
                except Exception:
                    pass
            self._disabled_widgets.clear()
            return
        win = getattr(self._viewer, "window", None)
        dock_map = getattr(win, "_dock_widgets", {}) if win is not None else {}
        for dock_obj in dock_map.values():
            dock = dock_obj[0] if isinstance(dock_obj, tuple) else dock_obj
            dwidget = None
            if hasattr(dock, "widget"):
                try:
                    dwidget = dock.widget()
                except Exception:
                    dwidget = None
            if dwidget is None or dwidget is self:
                continue
            try:
                if dwidget.isEnabled():
                    dwidget.setEnabled(False)
                    self._disabled_widgets.append(dwidget)
            except Exception:
                continue

    def _finish_processing(self) -> None:
        self._is_processing = False
        self._cancel_requested = False
        self._processing_steps = []
        self._processing_descriptions = []
        self._processing_indices = []
        self._processing_i = 0
        self._apply_ctx = None
        self._progress.hide()
        self._set_controls_enabled(True)
