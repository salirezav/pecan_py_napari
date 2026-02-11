"""Utility functions for video analysis (e.g. edge-based frame selection)."""

import numpy as np
import cv2


def get_min_max_edge_frames(edge_video):
    """Return the frame indices with the least and most edge pixels.

    Counts lit (non-zero) pixels per frame and returns the index of the
    frame with the minimum count and the index with the maximum count.
    Useful for edge-detection videos (e.g. from PecanVideo.edge_detection()).

    Parameters
    ----------
    edge_video : BaseVideo or PecanVideo
        Video whose frames are edge maps (0/255 or BGR). Must have a
        .frames attribute (ndarray of shape (N, H, W) or (N, H, W, 3)).

    Returns
    -------
    tuple of (int, int)
        (min_idx, max_idx): index of frame with fewest lit pixels,
        then index of frame with most lit pixels.
    """
    frames = edge_video.frames
    counts = []
    for frame in frames:
        if frame.ndim == 3:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            g = frame
        counts.append(np.count_nonzero(g))
    counts = np.array(counts)
    min_idx = int(np.argmin(counts))
    max_idx = int(np.argmax(counts))
    return (min_idx, max_idx)


def add_min_max_edge_column(
    df,
    path_column="absolute_path",
    new_column="min_max_edge_frames",
    path_to_edge_video=None,
    save_path=None,
    frame_step=1,
):
    """Add a column of (min_edge_frame, max_edge_frame) per video path.

    For each path in the dataframe, loads (or builds) the corresponding
    edge video, computes the frame index with the least edges and the
    frame index with the most edges, and stores them as a string
    "(min_idx, max_idx)" in the new column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a column of video file paths.
    path_column : str
        Name of the column containing absolute paths. Default "absolute_path".
    new_column : str
        Name of the new column to add. Default "min_max_edge_frames".
    path_to_edge_video : callable or None
        If provided, must be a function that takes a single path (str) and
        returns a BaseVideo or PecanVideo (edge video). If None, the default
        is: load the video with PecanVideo(path) and call .edge_detection().
        To use pecan-only edges, pass a callable that loads the video,
        produces the pecan-only video, then returns its .edge_detection().
    save_path : str or None
        If provided, the updated dataframe is saved to this path (e.g. CSV)
        after adding the column. Default None.

    Returns
    -------
    pandas.DataFrame
        Copy of the dataframe with the new column added. If save_path was
        set, the file is written before returning.
    """
    import pandas as pd
    from tqdm import tqdm

    if path_to_edge_video is None:
        from pecan_py import PecanVideo

        def _default_edge_video(path):
            return PecanVideo(path, frame_step=frame_step).edge_detection()

        path_to_edge_video = _default_edge_video

    results = []
    for path in tqdm(df[path_column], desc="Min/max edge frames"):
        try:
            edge_video = path_to_edge_video(path)
            min_idx, max_idx = get_min_max_edge_frames(edge_video)
            results.append(f"({min_idx}, {max_idx})")
        except Exception:
            results.append(None)

    out = df.copy()
    out[new_column] = results

    if save_path:
        out.to_csv(save_path, index=False)

    return out
