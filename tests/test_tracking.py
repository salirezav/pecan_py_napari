"""Tests for instance Labels tracking."""

from __future__ import annotations

import numpy as np

from napari_pecan_py.widgets.tracking.logic import (
    TrackingConfig,
    extract_detections,
    format_tracking_summary,
    track_label_volume,
)


def _blob(h: int, w: int, y0: int, x0: int, size: int, value: int) -> np.ndarray:
    frame = np.zeros((h, w), dtype=np.uint16)
    frame[y0 : y0 + size, x0 : x0 + size] = value
    return frame


def test_extract_detections_min_area():
    frame = np.zeros((20, 20), dtype=np.uint16)
    frame[2:4, 2:4] = 1  # area 4
    frame[10:18, 10:18] = 2  # area 64
    dets = extract_detections(frame, min_area=10)
    assert [d.local_id for d in dets] == [2]


def test_track_keeps_id_while_moving_right():
    # Same object, new per-frame local IDs, moving +x each frame.
    frames = []
    local_ids = [3, 7, 2, 9, 1]
    for t, lid in enumerate(local_ids):
        frames.append(_blob(40, 120, y0=10, x0=10 + 15 * t, size=8, value=lid))
    vol = np.stack(frames, axis=0)

    result = track_label_volume(
        vol,
        TrackingConfig(max_match_distance=40.0, max_age=3, min_area=10.0),
    )
    assert result.n_tracks == 1
    tracked_ids = []
    for t in range(vol.shape[0]):
        vals = [int(v) for v in np.unique(result.labels[t]) if int(v) > 0]
        assert vals == [1]
        tracked_ids.append(vals[0])
    assert len(set(tracked_ids)) == 1


def test_track_assigns_new_id_for_second_object():
    frames = []
    # Frame 0: one object on the left.
    frames.append(_blob(40, 120, 10, 5, 8, 1))
    # Frame 1: first object moved right; second appears on the left with local id 5.
    f1 = _blob(40, 120, 10, 20, 8, 2)
    f1[10:18, 5:13] = 5
    frames.append(f1)
    # Frame 2: both continue rightward with fresh local ids.
    f2 = _blob(40, 120, 10, 35, 8, 9)
    f2[10:18, 20:28] = 4
    frames.append(f2)
    vol = np.stack(frames, axis=0)

    result = track_label_volume(
        vol,
        TrackingConfig(
            max_match_distance=40.0,
            max_age=3,
            min_area=10.0,
            allow_birth_anywhere=True,
        ),
    )
    assert result.n_tracks == 2

    ids0 = {int(v) for v in np.unique(result.labels[0]) if int(v) > 0}
    ids1 = {int(v) for v in np.unique(result.labels[1]) if int(v) > 0}
    ids2 = {int(v) for v in np.unique(result.labels[2]) if int(v) > 0}
    assert ids0 == {1}
    assert ids1 == {1, 2}
    assert ids2 == {1, 2}

    # Object that started on the left should keep track 1 on the right side later.
    assert int(result.labels[2, 12, 38]) == 1
    assert int(result.labels[2, 12, 22]) == 2


def test_track_single_frame_passthrough_shape():
    frame = _blob(16, 16, 4, 4, 6, 3)
    result = track_label_volume(frame, TrackingConfig(min_area=5.0))
    assert result.labels.ndim == 2
    assert result.n_tracks == 1
    assert int(result.labels[5, 5]) == 1
    assert "1 track" in format_tracking_summary(result)
