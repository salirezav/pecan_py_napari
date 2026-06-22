"""Batch Pipeline widget for applying saved pipelines to multiple videos."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from qtpy.QtCore import QSize, QTimer, Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from napari_pecan_py._reader import VIDEO_EXTENSIONS

from ..pipeline_recorder.logic import apply_pipeline_step_with_context, create_apply_context
from ..pipeline_recorder.state import set_pipeline_applying
from .logic import create_headless_apply_context, load_pipeline_file, load_video_into_viewer


class _VideoStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class _SpinnerLabel(QLabel):
    _FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._frame = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.setFixedWidth(16)
        self.hide()

    def start(self) -> None:
        self._frame = 0
        self.setText(self._FRAMES[0])
        self.show()
        self._timer.start(80)

    def stop(self) -> None:
        self._timer.stop()
        self.hide()
        self.setText("")

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(self._FRAMES)
        self.setText(self._FRAMES[self._frame])


class _VideoListRow(QWidget):
    def __init__(
        self,
        video_path: str,
        *,
        on_remove,
        can_remove: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.video_path = str(video_path)
        self._on_remove = on_remove
        self._status = _VideoStatus.PENDING

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        self._spinner = _SpinnerLabel(self)
        layout.addWidget(self._spinner)

        self._check = QLabel("✓", self)
        self._check.setStyleSheet("color: #2ecc71; font-weight: bold;")
        self._check.setFixedWidth(16)
        self._check.hide()
        layout.addWidget(self._check)

        self._error = QLabel("✕", self)
        self._error.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self._error.setFixedWidth(16)
        self._error.hide()
        layout.addWidget(self._error)

        self._name = QLabel(Path(self.video_path).name, self)
        self._name.setToolTip(self.video_path)
        layout.addWidget(self._name, 1)

        self._remove_btn = QPushButton(self)
        self._remove_btn.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
        self._remove_btn.setToolTip("Remove video from list")
        self._remove_btn.setFixedSize(24, 24)
        self._remove_btn.clicked.connect(self._emit_remove)
        self._remove_btn.setEnabled(can_remove)
        layout.addWidget(self._remove_btn)

    def set_status(self, status: _VideoStatus) -> None:
        self._status = status
        self._spinner.stop()
        self._check.hide()
        self._error.hide()
        if status == _VideoStatus.RUNNING:
            self._spinner.start()
        elif status == _VideoStatus.DONE:
            self._check.show()
        elif status == _VideoStatus.ERROR:
            self._error.show()

    def set_removable(self, removable: bool) -> None:
        self._remove_btn.setEnabled(removable)

    def _emit_remove(self) -> None:
        self._on_remove(self.video_path)


class BatchPipelineWidget(QWidget):
    def __init__(self, napari_viewer) -> None:
        super().__init__()
        self._viewer = napari_viewer
        self._pipeline_path: str | None = None
        self._pipeline_steps: list[dict] = []
        self._pipeline_root_layer: str | None = None
        self._video_paths: list[str] = []
        self._row_widgets: list[_VideoListRow] = []

        self._is_processing = False
        self._cancel_requested = False
        self._video_idx = 0
        self._step_idx = 0
        self._apply_ctx = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(QLabel("Pipeline file"))
        row_pipe = QHBoxLayout()
        self._pipeline_label = QLabel("(none loaded)")
        self._pipeline_label.setStyleSheet("color: #888;")
        self._pipeline_label.setWordWrap(True)
        row_pipe.addWidget(self._pipeline_label, 1)
        self._btn_load_pipeline = QPushButton("Load pipeline")
        self._btn_load_pipeline.clicked.connect(self._load_pipeline)
        row_pipe.addWidget(self._btn_load_pipeline)
        layout.addLayout(row_pipe)

        layout.addWidget(QLabel("Videos to process"))
        self._video_list = QListWidget()
        self._video_list.setSelectionMode(QListWidget.NoSelection)
        layout.addWidget(self._video_list)

        row_videos = QHBoxLayout()
        self._btn_add_videos = QPushButton("Browse videos…")
        self._btn_add_videos.clicked.connect(self._browse_videos)
        row_videos.addWidget(self._btn_add_videos)
        self._btn_clear_videos = QPushButton("Clear list")
        self._btn_clear_videos.clicked.connect(self._clear_videos)
        row_videos.addWidget(self._btn_clear_videos)
        layout.addLayout(row_videos)

        self._chk_headless = QCheckBox("Process without adding layers to the viewer")
        self._chk_headless.setChecked(True)
        self._chk_headless.setToolTip(
            "When checked, videos and pipeline outputs stay off-screen. "
            "Side effects such as saved mask files still run."
        )
        layout.addWidget(self._chk_headless)

        row_run = QHBoxLayout()
        self._btn_run = QPushButton("Run batch")
        self._btn_run.clicked.connect(self._start_batch)
        row_run.addWidget(self._btn_run)
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.clicked.connect(self._request_stop)
        self._btn_stop.setEnabled(False)
        row_run.addWidget(self._btn_stop)
        layout.addLayout(row_run)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.hide()
        layout.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #888;")
        layout.addWidget(self._status)
        layout.addStretch(1)

    def _video_filter(self) -> str:
        patterns = sorted({f"*{ext}" for ext in VIDEO_EXTENSIONS}, key=str.lower)
        return f"Video files ({' '.join(patterns)});;All files (*.*)"

    def _load_pipeline(self) -> None:
        if self._is_processing:
            self._status.setText("Batch is running. Wait for completion or stop first.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load pipeline",
            "",
            "YAML/JSON (*.yml *.yaml *.json)",
        )
        if not path:
            return
        try:
            steps, name, root_layer = load_pipeline_file(path)
        except Exception as exc:
            self._status.setText(f"Could not load pipeline: {exc}")
            return
        self._pipeline_path = path
        self._pipeline_steps = steps
        self._pipeline_root_layer = root_layer
        self._pipeline_label.setText(f"{name} ({len(steps)} step(s))")
        self._pipeline_label.setStyleSheet("")
        self._status.setText(f"Loaded {len(steps)} step(s) from {name}.")

    def _browse_videos(self) -> None:
        if self._is_processing:
            self._status.setText("Batch is running. Wait for completion or stop first.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select videos",
            "",
            self._video_filter(),
        )
        if not paths:
            return
        added = 0
        existing = set(self._video_paths)
        for raw in paths:
            path = str(Path(raw).resolve())
            if path in existing:
                continue
            self._append_video(path)
            existing.add(path)
            added += 1
        if added:
            self._status.setText(f"Added {added} video(s). Total: {len(self._video_paths)}.")
        else:
            self._status.setText("No new videos added (duplicates skipped).")

    def _append_video(self, path: str) -> None:
        row = _VideoListRow(path, on_remove=self._remove_video, can_remove=not self._is_processing)
        item = QListWidgetItem()
        item.setSizeHint(QSize(0, max(28, row.sizeHint().height())))
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        self._video_list.addItem(item)
        self._video_list.setItemWidget(item, row)
        self._video_paths.append(path)
        self._row_widgets.append(row)

    def _remove_video(self, path: str) -> None:
        if self._is_processing:
            return
        try:
            idx = self._video_paths.index(path)
        except ValueError:
            return
        self._video_paths.pop(idx)
        self._row_widgets.pop(idx)
        self._video_list.takeItem(idx)
        self._status.setText(f"Removed {Path(path).name}. {len(self._video_paths)} video(s) remaining.")

    def _clear_videos(self) -> None:
        if self._is_processing:
            self._status.setText("Batch is running. Wait for completion or stop first.")
            return
        self._video_paths.clear()
        self._row_widgets.clear()
        self._video_list.clear()
        self._status.setText("Video list cleared.")

    def _set_controls_enabled(self, enabled: bool) -> None:
        self._btn_load_pipeline.setEnabled(enabled)
        self._btn_add_videos.setEnabled(enabled)
        self._btn_clear_videos.setEnabled(enabled)
        self._btn_run.setEnabled(enabled)
        self._btn_stop.setEnabled(not enabled)
        self._chk_headless.setEnabled(enabled)
        for row in self._row_widgets:
            row.set_removable(enabled)

    def _start_batch(self) -> None:
        if self._is_processing:
            self._status.setText("Batch is already running.")
            return
        if not self._pipeline_steps:
            self._status.setText("Load a pipeline first.")
            return
        if not self._video_paths:
            self._status.setText("Add at least one video.")
            return

        self._is_processing = True
        self._cancel_requested = False
        set_pipeline_applying(True)
        self._video_idx = 0
        self._step_idx = 0
        self._apply_ctx = None
        for row in self._row_widgets:
            row.set_status(_VideoStatus.PENDING)
        self._set_controls_enabled(False)
        self._progress.setValue(0)
        self._progress.show()
        self._status.setText("Starting batch…")
        QTimer.singleShot(0, self._process_next)

    def _request_stop(self) -> None:
        if not self._is_processing:
            return
        self._cancel_requested = True
        self._status.setText("Stopping after current checkpoint…")

    def _is_cancel_requested(self) -> bool:
        return bool(self._cancel_requested)

    def _current_row(self) -> _VideoListRow | None:
        if 0 <= self._video_idx < len(self._row_widgets):
            return self._row_widgets[self._video_idx]
        return None

    def _set_batch_progress(
        self,
        *,
        video_idx: int,
        video_total: int,
        step_idx: int,
        step_total: int,
        frame_current: int = 0,
        frame_total: int = 1,
        phase: str = "",
    ) -> None:
        video_total_safe = max(1, int(video_total))
        step_total_safe = max(1, int(step_total))
        frame_total_safe = max(1, int(frame_total))
        video_idx_clamped = max(0, min(int(video_idx), video_total_safe))
        step_idx_clamped = max(0, min(int(step_idx), step_total_safe))
        frame_clamped = max(0, min(int(frame_current), frame_total_safe))

        video_fraction = (step_idx_clamped + (frame_clamped / frame_total_safe)) / step_total_safe
        overall = (video_idx_clamped + video_fraction) / video_total_safe
        pct = max(0, min(100, int(round(overall * 100.0))))
        self._progress.setValue(pct)
        self._progress.setFormat(f"{pct}%")

        video_name = Path(self._video_paths[video_idx_clamped]).name
        phase_txt = f" - {phase}" if phase else ""
        self._status.setText(
            f"Processing {video_name} ({video_idx_clamped + 1}/{video_total_safe})"
            f", step {step_idx_clamped + 1}/{step_total_safe}{phase_txt}"
        )

    def _on_step_progress(self, current: int, total: int, phase: str = "") -> None:
        if not self._is_processing:
            return
        if self._cancel_requested:
            raise InterruptedError("Batch pipeline cancelled by user.")
        self._set_batch_progress(
            video_idx=self._video_idx,
            video_total=len(self._video_paths),
            step_idx=self._step_idx,
            step_total=len(self._pipeline_steps),
            frame_current=current,
            frame_total=total,
            phase=phase,
        )
        QApplication.processEvents()

    def _process_next(self) -> None:
        if not self._is_processing:
            return

        video_total = len(self._video_paths)
        step_total = len(self._pipeline_steps)

        if self._video_idx >= video_total:
            self._status.setText(f"Batch complete: processed {video_total} video(s).")
            self._finish_batch()
            return

        row = self._current_row()
        video_path = self._video_paths[self._video_idx]
        video_name = Path(video_path).name

        if self._step_idx == 0:
            if row is not None:
                row.set_status(_VideoStatus.RUNNING)
            self._set_batch_progress(
                video_idx=self._video_idx,
                video_total=video_total,
                step_idx=0,
                step_total=step_total,
            )
            headless = self._chk_headless.isChecked()
            verb = "Preparing" if headless else "Loading"
            self._status.setText(
                f"{verb} {video_name} ({self._video_idx + 1}/{video_total})…"
            )
            QApplication.processEvents()
            try:
                if headless:
                    self._apply_ctx = create_headless_apply_context(
                        video_path,
                        steps=self._pipeline_steps,
                        recorded_root=self._pipeline_root_layer,
                    )
                else:
                    load_video_into_viewer(self._viewer, video_path)
                    self._apply_ctx = create_apply_context(
                        self._viewer,
                        steps=self._pipeline_steps,
                        recorded_root=self._pipeline_root_layer,
                    )
            except Exception as exc:
                if row is not None:
                    row.set_status(_VideoStatus.ERROR)
                self._status.setText(
                    f"Failed to load {video_name} ({self._video_idx + 1}/{video_total}): {exc}"
                )
                self._finish_batch()
                return

        if self._step_idx >= step_total:
            if row is not None:
                row.set_status(_VideoStatus.DONE)
            self._video_idx += 1
            self._step_idx = 0
            self._apply_ctx = None
            QTimer.singleShot(0, self._process_next)
            return

        step = self._pipeline_steps[self._step_idx]
        desc = str(step.get("description") or step.get("kind") or "step")
        self._set_batch_progress(
            video_idx=self._video_idx,
            video_total=video_total,
            step_idx=self._step_idx,
            step_total=step_total,
        )
        self._status.setText(
            f"{video_name} ({self._video_idx + 1}/{video_total}): applying {desc}…"
        )
        QApplication.processEvents()

        try:
            msg = apply_pipeline_step_with_context(
                self._apply_ctx,
                step,
                progress_callback=self._on_step_progress,
                cancel_callback=self._is_cancel_requested,
            )
        except InterruptedError:
            if row is not None:
                row.set_status(_VideoStatus.PENDING)
            self._status.setText(
                f"Stopped while processing {video_name} ({self._video_idx + 1}/{video_total})."
            )
            self._finish_batch()
            return
        except Exception as exc:
            if row is not None:
                row.set_status(_VideoStatus.ERROR)
            self._status.setText(
                f"Failed on {video_name} ({self._video_idx + 1}/{video_total}), "
                f"step {self._step_idx + 1}/{step_total}: {exc}"
            )
            self._finish_batch()
            return

        self._status.setText(msg)
        self._step_idx += 1
        QTimer.singleShot(0, self._process_next)

    def _finish_batch(self) -> None:
        set_pipeline_applying(False)
        self._is_processing = False
        self._cancel_requested = False
        self._video_idx = 0
        self._step_idx = 0
        self._apply_ctx = None
        self._progress.hide()
        self._set_controls_enabled(True)
