"""Time index resolution for lazy video (no napari GUI)."""

import numpy as np

from napari_pecan_py.widgets.pecan_ellipse.logic import resolve_time_index_for_volume


class _FakeLazyVideo:
    shape = (177, 228, 436)

    def __getitem__(self, key):
        raise NotImplementedError("must not materialize full video")


class _FakeDims:
    def __init__(self, nsteps, current_step):
        self.nsteps = nsteps
        self.current_step = current_step


class _FakeViewer:
    def __init__(self, nsteps, current_step):
        self.dims = _FakeDims(nsteps, current_step)


def test_resolve_time_index_lazy_video_matches_slider():
    viewer = _FakeViewer(nsteps=(177, 228, 436), current_step=(102, 100, 200))
    t = resolve_time_index_for_volume(_FakeLazyVideo(), viewer)
    assert t == 102
