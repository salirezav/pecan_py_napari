"""Tests for segmentation training-run recorder and naming."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from napari_pecan_py.widgets.yolo_seg.training_run import (
    TrainingRunRecorder,
    default_run_name,
    resolve_run_name,
    sanitize_run_name,
    serialize_label_ids_by_class,
    video_entries_for_manifest,
)


def test_default_run_name_format():
    name = default_run_name(when=datetime(2026, 8, 12, 15, 30, 42))
    assert name == "2026-08-12_153042"


def test_sanitize_and_resolve_run_name():
    assert sanitize_run_name("  My Run: v1?.pt ") == "My_Run_v1"
    auto = resolve_run_name("")
    assert len(auto) >= 15
    assert "_" in auto
    assert resolve_run_name("  pecan-kernel ") == "pecan-kernel"
    assert resolve_run_name("experiment A") == "experiment_A"


def test_serialize_helpers(tmp_path: Path):
    video = tmp_path / "a.mp4"
    mask = tmp_path / "a - [Pecan].tiff"
    video.write_bytes(b"x")
    mask.write_bytes(b"y")
    entries = video_entries_for_manifest([(str(video), {"Pecan": str(mask)})])
    assert entries[0]["path"].endswith("a.mp4")
    assert "Pecan" in entries[0]["masks"]
    assert serialize_label_ids_by_class({"Pecan": {3}, "Crack": None}) == {
        "Pecan": [3],
        "Crack": None,
    }


def test_training_run_recorder_writes_artifacts(tmp_path: Path):
    recorder = TrainingRunRecorder(tmp_path, "demo-run")
    assert recorder.run_dir == tmp_path / "demo-run"
    recorder.start(
        {
            "backend": "yolo",
            "dataset": {"train_frames": 10, "val_frames": 2, "test_frames": 0},
            "hyperparameters": {"epochs": 2, "batch_size": 4},
        }
    )
    recorder.log_line("hello")
    recorder.record_epoch({"epoch": 1, "train_loss": 1.5, "val_loss": 1.2})
    recorder.record_epoch({"epoch": 2, "train_loss": 1.1, "val_loss": 0.9, "val_iou_mean": 0.7})

    ckpt = tmp_path / "src.pt"
    ckpt.write_bytes(b"weights")
    dest = recorder.save_checkpoint_copy(ckpt)
    recorder.finish(
        status="completed",
        summary={"best_epoch": 2, "best_val_loss": 0.9},
        checkpoint_path=dest,
    )

    assert dest == recorder.checkpoint_path
    assert dest.is_file()
    assert recorder.manifest_path.is_file()
    assert recorder.metrics_path.is_file()
    assert recorder.log_path.is_file()

    manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_name"] == "demo-run"
    assert manifest["status"] == "completed"
    assert manifest["dataset"]["train_frames"] == 10
    assert manifest["summary"]["best_epoch"] == 2
    assert "final_epoch_metrics" in manifest
    assert "best_epoch_metrics" in manifest

    with recorder.metrics_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[-1]["val_iou_mean"] == "0.7"
    assert "hello" in recorder.log_path.read_text(encoding="utf-8")


def test_training_run_recorder_avoids_name_collision(tmp_path: Path):
    first = TrainingRunRecorder(tmp_path, "same")
    first.start({"backend": "unet"})
    second = TrainingRunRecorder(tmp_path, "same")
    assert second.run_name == "same_2"
    assert second.run_dir == tmp_path / "same_2"
