"""Tests for SAM2 prompt gathering (no GPU / sam2 required)."""

import numpy as np

from napari_pecan_py.widgets.sam2_seg.logic import (
    conditioning_masks_from_labels,
    frame_rgb_uint8,
    gather_prompts,
    labels_2d_at_frame,
    merge_class_into_labels,
    n_frames,
    propagation_progress,
    sam2_decord_available,
    summarize_prompts,
    video_path_for_sam2,
)


class _FakeLazyVideo:
    """Mimics LazyVideoArray: 4D shape but only supports per-frame __getitem__."""

    shape = (4, 12, 10, 3)

    def __getitem__(self, index: int) -> np.ndarray:
        return np.full((12, 10, 3), int(index), dtype=np.uint8)


def test_frame_rgb_uint8_lazy_video_indexing():
    data = _FakeLazyVideo()
    assert n_frames(data) == 4
    frame = frame_rgb_uint8(data, 2)
    assert frame.shape == (12, 10, 3)
    assert frame[0, 0, 0] == 2


def test_summarize_prompts_with_numpy_labels():
    prompts = {
        "point_coords": np.array([[1.0, 2.0]], dtype=np.float32),
        "point_labels": np.array([1], dtype=np.int32),
    }
    text = summarize_prompts(prompts)
    assert "1 point(s)" in text


def test_labels_2d_at_frame_from_3d_stack():
    vol = np.arange(2 * 3 * 4, dtype=np.uint32).reshape(2, 3, 4)
    sl = labels_2d_at_frame(vol, 1)
    assert sl.shape == (3, 4)
    assert int(sl[0, 0]) == int(vol[1, 0, 0])


def test_merge_video_stack_masks():
    vol = np.zeros((3, 8, 6), dtype=np.uint32)
    masks = np.zeros((3, 8, 6), dtype=bool)
    masks[1, 2:5, 2:4] = True
    masks[2, 1:3, 1:3] = True
    out = merge_class_into_labels(vol, masks, 4)
    assert int(out[1].sum()) == 4 * 3 * 2
    assert int(out[2].sum()) == 4 * 2 * 2
    assert int(out[0].sum()) == 0


def test_merge_into_video_labels_single_frame():
    vol = np.zeros((5, 10, 8), dtype=np.uint32)
    mask = np.zeros((10, 8), dtype=bool)
    mask[2:5, 3:6] = True
    out = merge_class_into_labels(vol, mask, 4, frame_index=2)
    assert int(out[2].sum()) == 4 * 3 * 3
    assert int(out[0].sum()) == 0
    assert int(out[1].sum()) == 0


def test_propagation_progress_counts_unique_frames():
    filled: set[int] = set()
    assert propagation_progress(filled, 5, 177) == (1, 177)
    assert propagation_progress(filled, 5, 177) == (1, 177)
    assert propagation_progress(filled, 10, 177) == (2, 177)
    assert propagation_progress(filled, 0, 177) == (3, 177)


def test_sam2_decord_available_is_bool():
    assert isinstance(sam2_decord_available(), bool)


def test_video_path_for_sam2_lazy_reader():
    class _Lazy:
        path = r"C:\data\clip.MP4"

    assert video_path_for_sam2(_Lazy()) == r"C:\data\clip.MP4"
    assert video_path_for_sam2(np.zeros((3, 4, 5, 3))) is None


def test_conditioning_masks_from_labels():
    vol = np.zeros((4, 6, 8), dtype=np.uint32)
    vol[1, 1:3, 2:4] = 4
    vol[3, 0:2, 0:2] = 4
    seeds = conditioning_masks_from_labels(vol, 4)
    assert [t for t, _ in seeds] == [1, 3]


def test_gather_points_and_mask():
    h, w = 32, 40
    points = np.array([[0, 10, 12, 1], [0, 20, 22, 0]], dtype=float)
    brush = np.zeros((h, w), dtype=np.uint32)
    brush[5:15, 8:18] = 1
    prompts = gather_prompts(
        frame_index=0,
        shape_hw=(h, w),
        points_layer_data=points,
        brush_labels_2d=brush,
        shapes_data=None,
        pipeline_mask_2d=None,
        pipeline_label_id=None,
    )
    assert prompts["point_coords"] is not None
    assert len(prompts["point_coords"]) == 2
    assert prompts["mask_input"] is not None
    assert np.any(prompts["prompt_mask"])
