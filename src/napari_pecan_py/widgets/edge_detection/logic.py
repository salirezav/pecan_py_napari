"""Edge-detection processing helpers."""

from __future__ import annotations

import cv2
import numpy as np


EDGE_METHODS: dict[str, str] = {
    "canny": "Canny",
    "sobel": "Sobel Magnitude",
    "scharr": "Scharr Magnitude",
    "laplacian": "Laplacian",
    "prewitt": "Prewitt",
    "roberts": "Roberts",
    "log": "Laplacian of Gaussian (LoG)",
    "dog": "Difference of Gaussians (DoG)",
    "morph_gradient": "Morphological Gradient",
    "structured_forest": "Structured Forest (ximgproc)",
}


def _odd(v: int, minimum: int = 1) -> int:
    v = int(v)
    if v < minimum:
        v = minimum
    if v % 2 == 0:
        v += 1
    return v


def _to_u8_gray(frame: np.ndarray) -> np.ndarray:
    f = np.asarray(frame)
    if f.ndim == 2:
        return _normalize_to_u8(f)
    if f.ndim == 3:
        if f.shape[-1] >= 3:
            rgb = _normalize_to_u8(f[..., :3])
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return _normalize_to_u8(f[..., 0])
    raise ValueError(f"Unsupported frame shape: {f.shape}")


def _normalize_to_u8(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return a
    if np.issubdtype(a.dtype, np.floating):
        mn = float(np.nanmin(a))
        mx = float(np.nanmax(a))
        if mn >= 0.0 and mx <= 1.0:
            return np.clip(a * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.clip(a, 0, 255).astype(np.uint8)


def _gradient_magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    mag = cv2.magnitude(np.asarray(gx, dtype=np.float32), np.asarray(gy, dtype=np.float32))
    return np.clip(mag, 0.0, 255.0).astype(np.uint8)


def _threshold_u8(img: np.ndarray, thr: int) -> np.ndarray:
    t = int(np.clip(int(thr), 0, 255))
    if t <= 0:
        return np.asarray(img, dtype=np.uint8)
    _, out = cv2.threshold(np.asarray(img, dtype=np.uint8), t, 255, cv2.THRESH_BINARY)
    return out


def _structured_forest_edges(gray_u8: np.ndarray, params: dict, state: dict | None) -> np.ndarray:
    if not hasattr(cv2, "ximgproc"):
        raise RuntimeError("OpenCV ximgproc module is not available.")
    model_path = str(params.get("model_path", "")).strip()
    if not model_path:
        raise ValueError("Structured Forest requires a model path (.yml.gz).")
    if state is None:
        state = {}
    key = f"structured_forest::{model_path}"
    detector = state.get(key)
    if detector is None:
        detector = cv2.ximgproc.createStructuredEdgeDetection(model_path)
        state[key] = detector
    rgb = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    edges_f = detector.detectEdges(rgb)
    edges_u8 = np.clip(edges_f * 255.0, 0.0, 255.0).astype(np.uint8)
    if bool(params.get("use_nms", True)):
        orientation = detector.computeOrientation(edges_f)
        radius = int(max(1, int(params.get("nms_radius", 2))))
        mult = float(params.get("nms_mult", 1.0))
        edges_f = detector.edgesNms(edges_f, orientation, r=radius, s=mult)
        edges_u8 = np.clip(edges_f * 255.0, 0.0, 255.0).astype(np.uint8)
    return _threshold_u8(edges_u8, int(params.get("threshold", 30)))


def apply_edge_method(frame: np.ndarray, method: str, params: dict, state: dict | None = None) -> np.ndarray:
    gray = _to_u8_gray(frame)
    m = str(method)

    if m == "canny":
        blur = _odd(int(params.get("blur_ksize", 3)), 1)
        src = cv2.GaussianBlur(gray, (blur, blur), float(params.get("blur_sigma", 0.0))) if blur > 1 else gray
        return cv2.Canny(
            src,
            threshold1=float(params.get("threshold1", 50)),
            threshold2=float(params.get("threshold2", 150)),
            apertureSize=int(params.get("aperture_size", 3)),
            L2gradient=bool(params.get("l2_gradient", False)),
        )

    if m == "sobel":
        ksize = _odd(int(params.get("ksize", 3)), 1)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize, scale=float(params.get("scale", 1.0)), delta=float(params.get("delta", 0.0)))
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize, scale=float(params.get("scale", 1.0)), delta=float(params.get("delta", 0.0)))
        return _threshold_u8(_gradient_magnitude(gx, gy), int(params.get("threshold", 40)))

    if m == "scharr":
        gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0, scale=float(params.get("scale", 1.0)), delta=float(params.get("delta", 0.0)))
        gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1, scale=float(params.get("scale", 1.0)), delta=float(params.get("delta", 0.0)))
        return _threshold_u8(_gradient_magnitude(gx, gy), int(params.get("threshold", 40)))

    if m == "laplacian":
        ksize = _odd(int(params.get("ksize", 3)), 1)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize, scale=float(params.get("scale", 1.0)), delta=float(params.get("delta", 0.0)))
        return _threshold_u8(np.clip(np.abs(lap), 0.0, 255.0).astype(np.uint8), int(params.get("threshold", 25)))

    if m == "prewitt":
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        gx = cv2.filter2D(gray, cv2.CV_32F, kx)
        gy = cv2.filter2D(gray, cv2.CV_32F, ky)
        return _threshold_u8(_gradient_magnitude(gx, gy), int(params.get("threshold", 40)))

    if m == "roberts":
        kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        gx = cv2.filter2D(gray, cv2.CV_32F, kx)
        gy = cv2.filter2D(gray, cv2.CV_32F, ky)
        return _threshold_u8(_gradient_magnitude(gx, gy), int(params.get("threshold", 35)))

    if m == "log":
        blur = _odd(int(params.get("blur_ksize", 5)), 1)
        sigma = float(params.get("sigma", 1.2))
        sm = cv2.GaussianBlur(gray, (blur, blur), sigmaX=sigma, sigmaY=sigma)
        lap = cv2.Laplacian(sm, cv2.CV_32F, ksize=_odd(int(params.get("lap_ksize", 3)), 1))
        return _threshold_u8(np.clip(np.abs(lap), 0.0, 255.0).astype(np.uint8), int(params.get("threshold", 20)))

    if m == "dog":
        k1 = _odd(int(params.get("ksize1", 3)), 1)
        k2 = _odd(int(params.get("ksize2", 7)), 1)
        s1 = float(params.get("sigma1", 1.0))
        s2 = float(params.get("sigma2", 2.0))
        g1 = cv2.GaussianBlur(gray, (k1, k1), sigmaX=s1, sigmaY=s1)
        g2 = cv2.GaussianBlur(gray, (k2, k2), sigmaX=s2, sigmaY=s2)
        diff = cv2.absdiff(g1, g2)
        return _threshold_u8(diff, int(params.get("threshold", 20)))

    if m == "morph_gradient":
        k = _odd(int(params.get("kernel_size", 3)), 1)
        iterations = max(1, int(params.get("iterations", 1)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
        return _threshold_u8(out, int(params.get("threshold", 20)))

    if m == "structured_forest":
        return _structured_forest_edges(gray, params, state)

    raise ValueError(f"Unknown edge method: {m}")


def apply_edges_to_volume(data, method: str, params: dict, state: dict | None = None, progress_callback=None) -> np.ndarray:
    shape = getattr(data, "shape", None)
    if shape is not None:
        try:
            shp = tuple(int(x) for x in shape)
        except Exception:
            shp = None
    else:
        shp = None

    if shp is None:
        arr = np.asarray(data)
        shp = tuple(int(x) for x in arr.shape)
        data_ref = arr
    else:
        data_ref = data

    if len(shp) == 2:
        if progress_callback is not None:
            progress_callback(0, 1)
        out = apply_edge_method(np.asarray(data_ref), method, params, state=state)
        if progress_callback is not None:
            progress_callback(1, 1)
        return out

    if len(shp) == 3:
        if shp[-1] in (3, 4):
            if progress_callback is not None:
                progress_callback(0, 1)
            out = apply_edge_method(np.asarray(data_ref), method, params, state=state)
            if progress_callback is not None:
                progress_callback(1, 1)
            return out
        total = int(shp[0])
        out_frames = []
        if progress_callback is not None:
            progress_callback(0, total)
        for i in range(total):
            out_frames.append(apply_edge_method(np.asarray(data_ref[i]), method, params, state=state))
            if progress_callback is not None:
                progress_callback(i + 1, total)
        return np.stack(out_frames, axis=0)

    if len(shp) == 4:
        total = int(shp[0])
        out_frames = []
        if progress_callback is not None:
            progress_callback(0, total)
        for i in range(total):
            out_frames.append(apply_edge_method(np.asarray(data_ref[i]), method, params, state=state))
            if progress_callback is not None:
                progress_callback(i + 1, total)
        return np.stack(out_frames, axis=0)

    raise ValueError(f"Unsupported data shape for edge detection: {shp}")
