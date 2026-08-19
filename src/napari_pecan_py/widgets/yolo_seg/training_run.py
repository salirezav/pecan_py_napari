"""Local training-run artifacts: manifest, epoch metrics, and checkpoint naming."""

from __future__ import annotations

import csv
import json
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

MANIFEST_FILENAME = "run_manifest.json"
METRICS_FILENAME = "metrics_epoch.csv"
LOG_FILENAME = "console.log.txt"

_INVALID_NAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_WHITESPACE_RE = re.compile(r"\s+")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")


def default_run_name(*, when: datetime | None = None) -> str:
    """Return a filesystem-safe timestamp name (local time)."""
    stamp = when or datetime.now().astimezone()
    return stamp.strftime("%Y-%m-%d_%H%M%S")


def sanitize_run_name(name: str) -> str:
    """Sanitize a user-provided run / checkpoint stem."""
    cleaned = str(name).strip()
    if cleaned.lower().endswith(".pt"):
        cleaned = cleaned[:-3]
    cleaned = _INVALID_NAME_RE.sub("_", cleaned)
    cleaned = _WHITESPACE_RE.sub("_", cleaned)
    cleaned = _MULTI_UNDERSCORE_RE.sub("_", cleaned)
    cleaned = cleaned.strip(" ._")
    return cleaned or default_run_name()


def resolve_run_name(user_name: str | None = None) -> str:
    """Use the user's name when provided; otherwise auto-name with date/time."""
    text = (user_name or "").strip()
    if not text:
        return default_run_name()
    return sanitize_run_name(text)


def serialize_label_ids_by_class(
    label_ids_by_class: Mapping[str, set[int] | None] | None,
) -> Dict[str, list[int] | None]:
    if not label_ids_by_class:
        return {}
    out: Dict[str, list[int] | None] = {}
    for name, ids in label_ids_by_class.items():
        out[str(name)] = None if ids is None else sorted(int(v) for v in ids)
    return out


def video_entries_for_manifest(
    video_entries: Sequence[tuple[str, Mapping[str, str]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for video_path, masks in video_entries:
        rows.append(
            {
                "path": str(Path(video_path).resolve()),
                "masks": {
                    str(cls): str(Path(path).resolve()) for cls, path in masks.items()
                },
            }
        )
    return rows


def _try_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            commit = (result.stdout or "").strip()
            return commit or None
    except Exception:
        return None
    return None


def _package_version() -> str | None:
    try:
        from napari_pecan_py._version import __version__

        return str(__version__)
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


class TrainingRunRecorder:
    """Writes run folder artifacts next to the exported checkpoint."""

    def __init__(self, output_dir: str | Path, run_name: str | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        requested = resolve_run_name(run_name)
        self.run_dir = self._allocate_run_dir(self.output_dir, requested)
        self.run_name = self.run_dir.name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.run_dir / f"{self.run_name}.pt"
        self.manifest_path = self.run_dir / MANIFEST_FILENAME
        self.metrics_path = self.run_dir / METRICS_FILENAME
        self.log_path = self.run_dir / LOG_FILENAME
        self._manifest: Dict[str, Any] = {}
        self._metric_fields: list[str] = []
        self._metric_rows: list[dict[str, Any]] = []
        self._started_at = datetime.now(timezone.utc)

    @staticmethod
    def _allocate_run_dir(output_dir: Path, run_name: str) -> Path:
        candidate = output_dir / run_name
        if not candidate.exists():
            return candidate
        index = 2
        while True:
            alt = output_dir / f"{run_name}_{index}"
            if not alt.exists():
                return alt
            index += 1

    def start(self, payload: Mapping[str, Any] | None = None) -> Path:
        """Write the initial manifest and return the run directory."""
        self._started_at = datetime.now(timezone.utc)
        self._manifest = {
            "schema_version": 1,
            "run_name": self.run_name,
            "run_dir": str(self.run_dir.resolve()),
            "checkpoint_path": str(self.checkpoint_path.resolve()),
            "started_at_utc": self._started_at.isoformat(),
            "status": "running",
            "host": {
                "hostname": platform.node(),
                "system": platform.system(),
                "release": platform.release(),
                "python": platform.python_version(),
                "machine": platform.machine(),
            },
            "software": {
                "napari_pecan_py": _package_version(),
                "git_commit": _try_git_commit(),
            },
        }
        if payload:
            self._manifest.update(_json_safe(dict(payload)))
        self._write_manifest()
        return self.run_dir

    def update(self, **fields: Any) -> None:
        self._manifest.update(_json_safe(fields))
        self._write_manifest()

    def log_line(self, message: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(message.rstrip("\n") + "\n")

    def record_epoch(self, metrics: Mapping[str, Any]) -> None:
        row = {str(k): _json_safe(v) for k, v in metrics.items()}
        if "epoch" not in row:
            raise ValueError("record_epoch requires an 'epoch' field")
        for key in row:
            if key not in self._metric_fields:
                if key == "epoch":
                    self._metric_fields.insert(0, "epoch")
                else:
                    self._metric_fields.append(key)
        self._metric_rows.append(row)
        self._write_metrics_csv()

    def import_csv_metrics(self, source_csv: str | Path) -> None:
        """Copy an external epoch metrics table (e.g. Ultralytics results.csv)."""
        source = Path(source_csv)
        if not source.is_file():
            return
        shutil.copy2(source, self.metrics_path)
        with source.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            self._metric_fields = list(reader.fieldnames or [])
            self._metric_rows = [dict(row) for row in reader]

    def copy_sidecar(self, source: str | Path, dest_name: str) -> Path | None:
        source_path = Path(source)
        if not source_path.is_file():
            return None
        dest = self.run_dir / dest_name
        shutil.copy2(source_path, dest)
        return dest

    def save_checkpoint_copy(self, source: str | Path) -> Path:
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {source_path}")
        shutil.copy2(source_path, self.checkpoint_path)
        return self.checkpoint_path

    def finish(
        self,
        *,
        status: str = "completed",
        summary: Mapping[str, Any] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> Path:
        ended = datetime.now(timezone.utc)
        ckpt = Path(checkpoint_path) if checkpoint_path else self.checkpoint_path
        self._manifest["status"] = status
        self._manifest["ended_at_utc"] = ended.isoformat()
        self._manifest["duration_seconds"] = round(
            (ended - self._started_at).total_seconds(), 3
        )
        self._manifest["checkpoint_path"] = str(ckpt.resolve()) if ckpt else None
        if summary:
            self._manifest["summary"] = _json_safe(dict(summary))
        if self._metric_rows:
            self._manifest["epoch_count_recorded"] = len(self._metric_rows)
            self._manifest["final_epoch_metrics"] = dict(self._metric_rows[-1])
            best = self._pick_best_epoch_row()
            if best is not None:
                self._manifest["best_epoch_metrics"] = best
        self._write_manifest()
        return self.manifest_path

    def _pick_best_epoch_row(self) -> dict[str, Any] | None:
        if not self._metric_rows:
            return None
        preference = (
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
            "val_iou_mean",
            "val_loss",
            "train_loss",
        )
        minimize = {"val_loss", "train_loss"}
        for key in preference:
            scored: list[tuple[float, dict[str, Any]]] = []
            for row in self._metric_rows:
                raw = row.get(key)
                if raw in (None, "", "n/a"):
                    continue
                try:
                    scored.append((float(raw), row))
                except (TypeError, ValueError):
                    continue
            if not scored:
                continue
            scored.sort(key=lambda item: item[0], reverse=key not in minimize)
            return dict(scored[0][1])
        return dict(self._metric_rows[-1])

    def _write_manifest(self) -> None:
        with self.manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(self._manifest, fh, indent=2, sort_keys=False)
            fh.write("\n")

    def _write_metrics_csv(self) -> None:
        with self.metrics_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._metric_fields, extrasaction="ignore")
            writer.writeheader()
            for row in self._metric_rows:
                writer.writerow(row)


def mean_metric(values: Iterable[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None and v == v]
    if not nums:
        return None
    return sum(nums) / len(nums)
