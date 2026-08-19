"""Live FLIR + YOLO-seg overlay (OpenCV windows).

Run with the Spinnaker env:
  .venv-spinnaker\Scripts\python.exe scripts/live_yolo_flir.py
Or double-click the Desktop shortcut / .bat.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage
from ultralytics import YOLO


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255)
        if float(np.max(img)) <= 1.0:
            img = img * 255.0
        img = img.astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    return img


def to_yolo_predict_source(frame: np.ndarray):
    """Wrap an RGB frame for ``model.predict`` (PIL = RGB, unlike numpy = BGR)."""
    return PILImage.fromarray(_to_uint8_rgb(frame))


def inference_imgsz(height: int, width: int, model=None) -> int:
    """Pick an inference size aligned with YOLO stride (multiple of 32)."""
    if model is not None:
        for attr in ("overrides", "args"):
            args = getattr(model, attr, None)
            if isinstance(args, dict) and args.get("imgsz"):
                try:
                    saved = int(args["imgsz"])
                    if saved > 0:
                        return saved
                except (TypeError, ValueError):
                    pass
    size = max(int(height), int(width))
    return max(32, int(((size + 31) // 32) * 32))


def resolve_yolo_device(device: str) -> str:
    if str(device).strip().lower() != "auto":
        return str(device)
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "0"
    except Exception:
        pass
    return "cpu"


# --- model ---
RUNS_ROOT = Path(r"C:\Users\salir\Desktop\pecan_py_napari\yolo_seg_runs")
RUN_NAME = "2026-08-19_131846"
WEIGHTS_PATH = RUNS_ROOT / RUN_NAME / f"{RUN_NAME}.pt"
CONF = 0.25
DEVICE = "auto"  # auto | cpu | 0 (first CUDA GPU)

# --- overlay ---
MASK_ALPHA = 0.5
# BGR colors for OpenCV. Kernel is drawn on top of Pecan where they overlap.
CLASS_COLORS_BGR = {
    "pecan": (0, 140, 255),   # orange
    "kernel": (40, 220, 40),  # green
}

# --- camera ---
CAMERA_BACKEND = "spinnaker"  # "spinnaker" | "opencv"
CAMERA_INDEX = 0
SPINNAKER_CAMERA_INDEX = 0
SPINNAKER_ROOT = Path(r"C:\Program Files\Teledyne\Spinnaker")

# --- display / timing ---
WINDOW_NAME = "Pecan YOLO live"
CONTROLS_WINDOW_NAME = "Camera controls"
TARGET_FPS = 8  # initial display FPS; live slider can change this
# Full sensor grab + HQ demosaic/CCM, then software downscale (do NOT crop the sensor).
CAMERA_BINNING = 1          # 1 = full 5 MP. 2 = on-camera 2x2 (softer, faster).
DISPLAY_MAX_WIDTH = 1920
# Live "HQ Grab" switch in Camera controls (1 = quality, 0 = fast).
HQ_FRAME_MAX_WIDTH = 1920
HQ_INFER_IMGSZ = 1280
FAST_FRAME_MAX_WIDTH = 960
FAST_INFER_IMGSZ = 640
FAST_LIVE = False           # start in quality mode; toggle with HQ Grab
FRAME_MAX_WIDTH = HQ_FRAME_MAX_WIDTH
INFER_IMGSZ = HQ_INFER_IMGSZ

# --- Spinnaker color / white balance (matches SpinView "Color Correction" preset) ---
ENABLE_COLOR_CORRECTION = True
ENABLE_AUTO_WHITE_BALANCE = True  # BalanceWhiteAuto = Continuous (on-camera ISP)
CCM_APPLICATION = "GENERIC"
CCM_SENSOR = "IMX250"  # BFS-U3-51S5C sensor: IMX250
CCM_COLOR_TEMP = "GENERAL"
CCM_COLOR_SPACE = "SRGB"  # SpinView "Color Space: ON"
CCM_TYPE = "ADVANCED"

print(f"Python {sys.version.split()[0]}")
print(f"Weights path: {WEIGHTS_PATH}")
print(f"Weights exist: {WEIGHTS_PATH.exists()}")


def require_pyspin() -> None:
    """Verify Teledyne Spinnaker PySpin is available in this kernel."""
    wheel_dir = SPINNAKER_ROOT / "PySpin"
    wheels = sorted(wheel_dir.glob("spinnaker_python-*.whl"))
    wheel_tags = [w.name.split("-")[4] for w in wheels if len(w.name.split("-")) > 4]
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"

    try:
        import PySpin

        if not hasattr(PySpin, "System"):
            raise ImportError("wrong PySpin package (PyPI pyspin vs Spinnaker SDK)")
        ver = PySpin.System.GetInstance().GetLibraryVersion()
        print(f"PySpin OK — Spinnaker {ver.major}.{ver.minor}.{ver.type}.{ver.build} (Python {py_ver})")
    except ImportError as exc:
        raise RuntimeError(
            f"PySpin not available (Python {py_ver}).\n"
            f"Spinnaker wheels on disk: {wheel_tags or ['none']}\n\n"
            "Switch kernel to 'Pecan PySpin (Python 3.10)' or run:\n"
            "  powershell -File scripts/setup_spinnaker_env.ps1\n\n"
            "Or set CAMERA_BACKEND = 'opencv' for a generic webcam."
        ) from exc


if CAMERA_BACKEND == "spinnaker":
    require_pyspin()
else:
    print("Skipping PySpin check (opencv backend).")


_CCM_APPLICATION = {
    "GENERIC": "SPINNAKER_CCM_APPLICATION_GENERIC",
    "MICROSCOPY": "SPINNAKER_CCM_APPLICATION_MICROSCOPY",
}
_CCM_SENSOR = {
    "IMX250": "SPINNAKER_CCM_SENSOR_IMX250",
}
_CCM_COLOR_TEMP = {
    "GENERAL": "SPINNAKER_CCM_COLOR_TEMP_GENERAL",
    "SUNNY_5000K": "SPINNAKER_CCM_COLOR_TEMP_SUNNY_5000K",
    "CLOUDY_6500K": "SPINNAKER_CCM_COLOR_TEMP_CLOUDY_6500K",
    "TUNGSTEN_2800K": "SPINNAKER_CCM_COLOR_TEMP_TUNGSTEN_2800K",
    "DAYLIGHT_5034K": "SPINNAKER_CCM_COLOR_TEMP_DAYLIGHT_5034K",
}
_CCM_COLOR_SPACE = {
    "SRGB": "SPINNAKER_CCM_COLOR_SPACE_SRGB",
    "OFF": "SPINNAKER_CCM_COLOR_SPACE_OFF",
}
_CCM_TYPE = {
    "ADVANCED": "SPINNAKER_CCM_TYPE_ADVANCED",
    "LINEAR": "SPINNAKER_CCM_TYPE_LINEAR",
}


def _pyspin_enum_value(pyspin_module, name: str):
    return getattr(pyspin_module, name)


def _set_spinnaker_enum(nodemap, pyspin_module, node_name: str, entry_name: str) -> bool:
    node = pyspin_module.CEnumerationPtr(nodemap.GetNode(node_name))
    if not (pyspin_module.IsAvailable(node) and pyspin_module.IsWritable(node)):
        return False
    entry = node.GetEntryByName(entry_name)
    if not pyspin_module.IsReadable(entry):
        return False
    node.SetIntValue(entry.GetValue())
    return True


def _set_spinnaker_bool(nodemap, pyspin_module, node_name: str, value: bool) -> bool:
    node = pyspin_module.CBooleanPtr(nodemap.GetNode(node_name))
    if not (pyspin_module.IsAvailable(node) and pyspin_module.IsWritable(node)):
        return False
    node.SetValue(value)
    return True


def _set_spinnaker_int(nodemap, pyspin_module, node_name: str, value: int) -> bool:
    node = pyspin_module.CIntegerPtr(nodemap.GetNode(node_name))
    if not (pyspin_module.IsAvailable(node) and pyspin_module.IsWritable(node)):
        return False
    lo, hi = int(node.GetMin()), int(node.GetMax())
    inc = int(node.GetInc()) if hasattr(node, "GetInc") else 1
    inc = max(inc, 1)
    snapped = lo + ((int(value) - lo) // inc) * inc
    snapped = min(hi, max(lo, snapped))
    node.SetValue(snapped)
    return True


def _build_ccm_settings(pyspin_module):
    settings = pyspin_module.CCMSettings()
    settings.Application = _pyspin_enum_value(
        pyspin_module, _CCM_APPLICATION[CCM_APPLICATION]
    )
    settings.Sensor = _pyspin_enum_value(pyspin_module, _CCM_SENSOR[CCM_SENSOR])
    settings.ColorTemperature = _pyspin_enum_value(
        pyspin_module, _CCM_COLOR_TEMP[CCM_COLOR_TEMP]
    )
    settings.ColorSpace = _pyspin_enum_value(
        pyspin_module, _CCM_COLOR_SPACE[CCM_COLOR_SPACE]
    )
    settings.Type = _pyspin_enum_value(pyspin_module, _CCM_TYPE[CCM_TYPE])
    return settings


def configure_spinnaker_color(nodemap, pyspin_module) -> None:
    """Match Spin: auto white balance + on-host preset color correction."""
    if ENABLE_AUTO_WHITE_BALANCE:
        _set_spinnaker_bool(nodemap, pyspin_module, "IspEnable", True)
        _set_spinnaker_enum(nodemap, pyspin_module, "BalanceWhiteAuto", "Continuous")
        _set_spinnaker_enum(
            nodemap, pyspin_module, "RgbTransformLightSource", "General"
        )
    if ENABLE_COLOR_CORRECTION:
        # On-host CCM replaces on-camera ColorTransformationEnable.
        _set_spinnaker_bool(nodemap, pyspin_module, "ColorTransformationEnable", False)


def _read_spinnaker_pixel_format(nodemap, pyspin_module) -> str:
    """Read PixelFormat only — do not change it (ISP auto WB needs the camera default)."""
    node = pyspin_module.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
    if pyspin_module.IsAvailable(node) and pyspin_module.IsReadable(node):
        try:
            return str(node.GetCurrentEntry().GetSymbolic())
        except Exception:
            pass
    return "unknown"


def configure_spinnaker_capture_size(nodemap, pyspin_module) -> str:
    """Use the full sensor (optional binning only). Do not center-crop Width."""
    notes: list[str] = []
    binning = int(globals().get("CAMERA_BINNING", 1))
    if binning > 1:
        binned = _set_spinnaker_int(nodemap, pyspin_module, "BinningHorizontal", binning)
        binned = _set_spinnaker_int(nodemap, pyspin_module, "BinningVertical", binning) and binned
        if binned:
            notes.append(f"binning {binning}x{binning}")
        else:
            notes.append("full-res grab")
    else:
        _set_spinnaker_int(nodemap, pyspin_module, "BinningHorizontal", 1)
        _set_spinnaker_int(nodemap, pyspin_module, "BinningVertical", 1)
        notes.append("full-res grab")

    for name in ("OffsetX", "OffsetY"):
        _set_spinnaker_int(nodemap, pyspin_module, name, 0)

    w_node = pyspin_module.CIntegerPtr(nodemap.GetNode("Width"))
    h_node = pyspin_module.CIntegerPtr(nodemap.GetNode("Height"))
    if pyspin_module.IsAvailable(w_node) and pyspin_module.IsWritable(w_node):
        w_node.SetValue(int(w_node.GetMax()))
        notes.append(f"Width={int(w_node.GetValue())}")
    if pyspin_module.IsAvailable(h_node) and pyspin_module.IsWritable(h_node):
        h_node.SetValue(int(h_node.GetMax()))
        notes.append(f"Height={int(h_node.GetValue())}")
    return ", ".join(notes) or "default size"


def _downscale_rgb(rgb: np.ndarray, max_width: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    max_width = max(1, int(max_width))
    if w <= max_width:
        return rgb
    nh = max(1, int(round(h * (max_width / float(w)))))
    return cv2.resize(rgb, (max_width, nh), interpolation=cv2.INTER_AREA)


class LiveCamera:
    """Live camera via Spinnaker (FLIR) or OpenCV DirectShow."""

    def __init__(self, backend: str, *, index: int = 0, spinnaker_index: int = 0):
        self.backend = backend
        self.index = index
        self.spinnaker_index = spinnaker_index
        self._cap = None
        self._system = None
        self._cam_list = None
        self._cam = None
        self._processor = None
        self._ccm_settings = None

    def open(self) -> None:
        if self.backend == "opencv":
            self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                raise RuntimeError(f"OpenCV could not open camera index {self.index}")
            return

        if self.backend == "spinnaker":
            import PySpin

            self._system = PySpin.System.GetInstance()
            self._cam_list = self._system.GetCameras()
            n = self._cam_list.GetSize()
            if n == 0:
                self._release_spinnaker()
                raise RuntimeError("No Spinnaker cameras detected.")
            if self.spinnaker_index >= n:
                self._release_spinnaker()
                raise IndexError(f"Spinnaker camera index {self.spinnaker_index} >= {n}")

            self._cam = self._cam_list[self.spinnaker_index]
            self._cam.Init()
            nodemap = self._cam.GetNodeMap()

            stream_nodemap = self._cam.GetTLStreamNodeMap()
            node_mode = PySpin.CEnumerationPtr(stream_nodemap.GetNode("StreamBufferHandlingMode"))
            if PySpin.IsReadable(node_mode) and PySpin.IsWritable(node_mode):
                newest = node_mode.GetEntryByName("NewestOnly")
                if PySpin.IsReadable(newest):
                    node_mode.SetIntValue(newest.GetValue())

            acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
            if PySpin.IsReadable(acq_mode) and PySpin.IsWritable(acq_mode):
                continuous = acq_mode.GetEntryByName("Continuous")
                if PySpin.IsReadable(continuous):
                    acq_mode.SetIntValue(continuous.GetValue())

            configure_spinnaker_color(nodemap, PySpin)
            size_note = configure_spinnaker_capture_size(nodemap, PySpin)
            pixel_format = _read_spinnaker_pixel_format(nodemap, PySpin)
            print(f"Spinnaker capture: PixelFormat={pixel_format}, {size_note}")

            self._processor = PySpin.ImageProcessor()
            self.set_quality_mode(True)
            self.set_color_correction(bool(globals().get("ENABLE_COLOR_CORRECTION", True)))
            self._cam.BeginAcquisition()
            return

        raise ValueError(f"Unknown backend: {self.backend}")

    def set_quality_mode(self, high: bool) -> None:
        """HQ_LINEAR demosaic. Color correction is owned by ``set_color_correction``."""
        del high
        if self.backend != "spinnaker" or self._processor is None:
            return
        import PySpin

        try:
            self._processor.SetColorProcessing(
                PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR
            )
        except Exception:
            pass

    def set_color_correction(self, enabled: bool) -> None:
        """Turn on-host SpinView CCM on or off while acquiring."""
        globals()["ENABLE_COLOR_CORRECTION"] = bool(enabled)
        if self.backend != "spinnaker":
            return
        import PySpin

        if enabled:
            if self._ccm_settings is None:
                try:
                    self._ccm_settings = _build_ccm_settings(PySpin)
                except Exception:
                    self._ccm_settings = None
        else:
            self._ccm_settings = None

    def read_rgb(self) -> np.ndarray | None:
        t0 = time.perf_counter()
        max_width = int(globals().get("FRAME_MAX_WIDTH", 1280))
        if self.backend == "opencv":
            ok, frame_bgr = self._cap.read()
            if not ok:
                return None
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.last_native_hw = (int(rgb.shape[1]), int(rgb.shape[0]))
            out = _downscale_rgb(rgb, max_width)
            self.last_grab_ms = (time.perf_counter() - t0) * 1000.0
            return out

        if self.backend == "spinnaker":
            import PySpin

            image_result = self._cam.GetNextImage(1000)
            try:
                if image_result.IsIncomplete():
                    return None
                converted = self._processor.Convert(image_result, PySpin.PixelFormat_RGB8)
                if self._ccm_settings is not None:
                    converted = PySpin.ImageUtilityCCM_CreateColorCorrected(
                        converted, self._ccm_settings
                    )
                rgb = np.asarray(converted.GetNDArray(), dtype=np.uint8).copy()
                self.last_native_hw = (int(rgb.shape[1]), int(rgb.shape[0]))
                out = _downscale_rgb(rgb, max_width)
                self.last_grab_ms = (time.perf_counter() - t0) * 1000.0
                return out
            finally:
                image_result.Release()

        return None

    def get_enum(self, name: str) -> str | None:
        if self.backend != "spinnaker" or self._cam is None:
            return None
        import PySpin

        node = PySpin.CEnumerationPtr(self._cam.GetNodeMap().GetNode(name))
        if not (PySpin.IsAvailable(node) and PySpin.IsReadable(node)):
            return None
        try:
            return str(node.GetCurrentEntry().GetSymbolic())
        except Exception:
            return None

    def set_enum(self, name: str, entry: str) -> bool:
        if self.backend != "spinnaker" or self._cam is None:
            return False
        import PySpin

        return _set_spinnaker_enum(self._cam.GetNodeMap(), PySpin, name, entry)

    def get_float(self, name: str) -> float | None:
        if self.backend != "spinnaker" or self._cam is None:
            return None
        import PySpin

        node = PySpin.CFloatPtr(self._cam.GetNodeMap().GetNode(name))
        if not (PySpin.IsAvailable(node) and PySpin.IsReadable(node)):
            return None
        try:
            return float(node.GetValue())
        except Exception:
            return None

    def set_float(self, name: str, value: float) -> bool:
        if self.backend != "spinnaker" or self._cam is None:
            return False
        import PySpin

        node = PySpin.CFloatPtr(self._cam.GetNodeMap().GetNode(name))
        if not (PySpin.IsAvailable(node) and PySpin.IsWritable(node)):
            return False
        try:
            lo, hi = float(node.GetMin()), float(node.GetMax())
            node.SetValue(min(hi, max(lo, float(value))))
            return True
        except Exception:
            return False

    def float_range(self, name: str) -> tuple[float, float] | None:
        if self.backend != "spinnaker" or self._cam is None:
            return None
        import PySpin

        node = PySpin.CFloatPtr(self._cam.GetNodeMap().GetNode(name))
        if not (PySpin.IsAvailable(node) and PySpin.IsReadable(node)):
            return None
        try:
            return float(node.GetMin()), float(node.GetMax())
        except Exception:
            return None

    def get_balance_ratio(self, selector: str) -> float | None:
        if not self.set_enum("BalanceRatioSelector", selector):
            return None
        return self.get_float("BalanceRatio")

    def set_balance_ratio(self, selector: str, value: float) -> bool:
        if not self.set_enum("BalanceRatioSelector", selector):
            return False
        return self.set_float("BalanceRatio", value)

    def is_auto_on(self, enum_name: str) -> bool:
        current = self.get_enum(enum_name)
        return current is not None and current.lower() == "continuous"

    def _release_spinnaker(self) -> None:
        if self._cam is not None:
            try:
                self._cam.EndAcquisition()
            except Exception:
                pass
            try:
                self._cam.DeInit()
            except Exception:
                pass
            del self._cam
            self._cam = None
        if self._cam_list is not None:
            self._cam_list.Clear()
            del self._cam_list
            self._cam_list = None
        if self._system is not None:
            self._system.ReleaseInstance()
            del self._system
            self._system = None
        self._processor = None
        self._ccm_settings = None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self.backend == "spinnaker":
            self._release_spinnaker()


def list_spinnaker_cameras() -> list[str]:
    """List cameras without Init/ReleaseInstance (safe to call before opening)."""
    try:
        import PySpin
    except ImportError:
        return []

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    names: list[str] = []
    try:
        for i in range(cam_list.GetSize()):
            cam = cam_list[i]
            try:
                nodemap = cam.GetTLDeviceNodeMap()
                model_node = PySpin.CStringPtr(nodemap.GetNode("DeviceModelName"))
                serial_node = PySpin.CStringPtr(nodemap.GetNode("DeviceSerialNumber"))
                model = model_node.GetValue() if PySpin.IsReadable(model_node) else "?"
                serial = serial_node.GetValue() if PySpin.IsReadable(serial_node) else "?"
                names.append(f"[{i}] {model} (SN {serial})")
            except Exception as exc:
                names.append(f"[{i}] <error: {exc}>")
            finally:
                del cam
    finally:
        cam_list.Clear()
        del cam_list
    return names


def list_opencv_cameras(max_index: int = 4) -> list[str]:
    found: list[str] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            found.append(f"[{i}] {w}x{h}")
        cap.release()
    return found


print("Spinnaker cameras:", list_spinnaker_cameras() or ["(none)"])


if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(
        f"Weights not ready yet:\n  {WEIGHTS_PATH}\n\n"
        "Training is still running. Re-run this cell after the .pt file appears, "
        "then run the live-view cell."
    )

resolved_device = resolve_yolo_device(DEVICE)
model = YOLO(str(WEIGHTS_PATH))
try:
    model.fuse()
except Exception:
    pass
class_names = {int(k): str(v) for k, v in model.names.items()}

print(f"Classes: {class_names}")
print(f"Device: {resolved_device}")
try:
    import torch

    if str(resolved_device).lower() != "cpu" and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except Exception:
    pass


def _class_key(name: str) -> str:
    return str(name).strip().lower()


def _draw_priority(name: str) -> int:
    """Pecan first, kernel on top, everything else in between."""
    key = _class_key(name)
    if key == "pecan":
        return 0
    if key == "kernel":
        return 2
    return 1


def _normalize_roi(roi, w: int, h: int):
    if roi is None:
        return None
    x1, y1, x2, y2 = (int(v) for v in roi)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return x1, y1, x2, y2


def overlay_masks(
    frame_bgr: np.ndarray,
    result,
    names: dict[int, str],
    colors_bgr: dict[str, tuple[int, int, int]],
    alpha: float,
    roi=None,
) -> np.ndarray:
    """Tint instance masks at ``alpha``. If ``roi`` is set, only paint inside it."""
    out = frame_bgr.copy()
    if result is None or result.masks is None or result.boxes is None:
        return out

    masks = result.masks.data.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    h, w = out.shape[:2]
    roi = _normalize_roi(roi, w, h)
    overlay = out.copy()

    order = sorted(
        range(len(masks)),
        key=lambda i: _draw_priority(names.get(int(classes[i]), "")),
    )
    for i in order:
        name = names.get(int(classes[i]), str(int(classes[i])))
        color = colors_bgr.get(_class_key(name), (0, 255, 255))
        mask = masks[i]
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        keep = mask > 0.5
        if roi is not None:
            x1, y1, x2, y2 = roi
            inside = np.zeros_like(keep)
            inside[y1:y2, x1:x2] = keep[y1:y2, x1:x2]
            if not np.any(inside):
                continue
            keep = inside
        overlay[keep] = color

    cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, dst=out)
    return out


def draw_legend(frame_bgr: np.ndarray, names: dict[int, str]) -> np.ndarray:
    """Paint a small class-color key plus ROI hint."""
    y = 28
    seen: set[str] = set()
    for name in names.values():
        key = _class_key(name)
        if key in seen:
            continue
        seen.add(key)
        color = CLASS_COLORS_BGR.get(key, (0, 255, 255))
        cv2.rectangle(frame_bgr, (12, y - 16), (32, y + 2), color, thickness=-1)
        cv2.putText(
            frame_bgr,
            name,
            (40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        y += 26
    cv2.putText(
        frame_bgr,
        "drag ROI  |  c clear  |  q quit",
        (12, y + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return frame_bgr


def draw_fps_hud(
    frame_bgr: np.ndarray,
    fps: float,
    fps_max: float,
    device_text: str,
    extra_lines: list[str] | None = None,
) -> np.ndarray:
    """Draw measured pipeline FPS and a timing breakdown in the top-right corner."""
    _h, w = frame_bgr.shape[:2]
    lines = [f"{fps:.1f} fps", f"max {fps_max:.1f}", device_text]
    if extra_lines:
        lines.extend(extra_lines)
    y = 28
    for i, line in enumerate(lines):
        scale = 0.7 if i < 3 else 0.5
        thick = 2 if i < 3 else 1
        (tw, _th), _base = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x = max(12, w - tw - 14)
        cv2.putText(
            frame_bgr,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thick + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 255, 255),
            thick,
            cv2.LINE_AA,
        )
        y += 26 if i < 3 else 20
    return frame_bgr


def _mouse_to_image(x: int, y: int, view: dict) -> tuple[int, int]:
    """Map mouse coords on the shown (scaled) frame back to camera pixels."""
    scale = float(view.get("scale") or 1.0)
    if scale <= 0:
        scale = 1.0
    ox = float(view.get("ox") or 0.0)
    oy = float(view.get("oy") or 0.0)
    ix = int(round((x - ox) / scale))
    iy = int(round((y - oy) / scale))
    w = max(1, int(view.get("w") or 1))
    h = max(1, int(view.get("h") or 1))
    return int(np.clip(ix, 0, w - 1)), int(np.clip(iy, 0, h - 1))


def _letterbox_to_size(frame_bgr: np.ndarray, out_w: int, out_h: int):
    """Fit ``frame_bgr`` inside ``out_w x out_h`` with black bars; keep aspect ratio."""
    h, w = frame_bgr.shape[:2]
    out_w = max(1, int(out_w))
    out_h = max(1, int(out_h))
    scale = min(out_w / float(w), out_h / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = (
        frame_bgr
        if nw == w and nh == h
        else cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    )
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    ox = (out_w - nw) // 2
    oy = (out_h - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas, float(scale), float(ox), float(oy)


def _video_window_size(win_name: str, view: dict, frame_w: int, frame_h: int) -> tuple[int, int]:
    """Window client size for letterboxing, ignoring title-bar jitter."""
    default_w = min(int(globals().get("DISPLAY_MAX_WIDTH", 960)), max(1, int(frame_w)))
    default_h = max(1, int(round(default_w * frame_h / float(frame_w))))
    fallback_w = int(view.get("disp_w") or default_w)
    fallback_h = int(view.get("disp_h") or default_h)
    try:
        _x, _y, ww, wh = cv2.getWindowImageRect(win_name)
        ww, wh = int(ww), int(wh)
    except Exception:
        return fallback_w, fallback_h
    if ww < 64 or wh < 64:
        return fallback_w, fallback_h
    last_w = int(view.get("disp_w") or ww)
    last_h = int(view.get("disp_h") or wh)
    if abs(ww - last_w) <= 16 and 8 <= (wh - last_h) <= 56:
        return last_w, last_h
    if abs(ww - last_w) < 8 and abs(wh - last_h) < 8:
        return last_w, last_h
    return ww, wh


def _log_to_pos(value: float, lo: float, hi: float, steps: int) -> int:
    value = min(hi, max(lo, float(value)))
    t = (np.log(value) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return int(np.clip(round(t * steps), 0, steps))


def _pos_to_log(pos: int, lo: float, hi: float, steps: int) -> float:
    t = min(steps, max(0, int(pos))) / float(steps)
    return float(np.exp(np.log(lo) + t * (np.log(hi) - np.log(lo))))


def apply_quality_preset(high: bool) -> None:
    """Update live working size / YOLO size. Sensor binning stays as opened."""
    g = globals()
    g["FAST_LIVE"] = not high
    if high:
        g["FRAME_MAX_WIDTH"] = int(g.get("HQ_FRAME_MAX_WIDTH", 1920))
        g["INFER_IMGSZ"] = int(g.get("HQ_INFER_IMGSZ", 1280))
    else:
        g["FRAME_MAX_WIDTH"] = int(g.get("FAST_FRAME_MAX_WIDTH", 960))
        g["INFER_IMGSZ"] = int(g.get("FAST_INFER_IMGSZ", 640))


class CameraControlPanel:
    """Trackbars for analog camera settings; checkboxes for on/off switches."""

    WINDOW_WIDTH = 520
    WINDOW_HEIGHT = 420
    PANEL_W = 440
    PANEL_H = 360

    EXP_STEPS = 1000
    GAIN_STEPS = 270
    WB_STEPS = 400
    BOX = 18
    CHECKS = (
        ("hq_grab", "HQ Grab"),
        ("color_corr", "Color Corr"),
        ("exp_auto", "Exp Auto"),
        ("gain_auto", "Gain Auto"),
        ("wb_auto", "WB Auto"),
    )

    def __init__(self, camera: LiveCamera, default_fps: int):
        self.camera = camera
        self.win = CONTROLS_WINDOW_NAME
        self._last_auto_refresh = 0.0
        self._exp_range = (100.0, 20000.0)
        self._gain_range = (0.0, 27.0)
        self._wb_range = (0.25, 4.0)
        self._hitboxes: list[tuple[str, int, int, int, int]] = []
        self.checks = {
            "hq_grab": not bool(globals().get("FAST_LIVE", False)),
            "color_corr": bool(globals().get("ENABLE_COLOR_CORRECTION", True)),
            "exp_auto": True,
            "gain_auto": True,
            "wb_auto": bool(globals().get("ENABLE_AUTO_WHITE_BALANCE", True)),
        }
        self._quality_high = bool(self.checks["hq_grab"])
        self._color_corr = bool(self.checks["color_corr"])
        probed = camera.float_range("ExposureTime")
        if probed is not None:
            lo, hi = probed
            self._exp_range = (max(lo, 20.0), min(hi, 100000.0))
        probed = camera.float_range("Gain")
        if probed is not None:
            self._gain_range = probed
        probed = camera.float_range("BalanceRatio")
        if probed is not None:
            self._wb_range = probed

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        dummy = np.zeros((self.PANEL_H, self.PANEL_W, 3), dtype=np.uint8)
        cv2.imshow(self.win, dummy)

        fps0 = int(np.clip(default_fps, 1, 30))
        cv2.createTrackbar("FPS", self.win, fps0, 30, lambda _v: None)
        cv2.createTrackbar("Exposure", self.win, 0, self.EXP_STEPS, lambda _v: None)
        cv2.createTrackbar("Gain", self.win, 0, self.GAIN_STEPS, lambda _v: None)
        cv2.createTrackbar("WB Red", self.win, 0, self.WB_STEPS, lambda _v: None)
        cv2.createTrackbar("WB Blue", self.win, 0, self.WB_STEPS, lambda _v: None)
        cv2.resizeWindow(self.win, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        cv2.setMouseCallback(self.win, self._on_mouse)
        self._pull_autos_from_camera()
        self._pull_values_from_camera()
        self.render()

    def target_fps(self) -> int:
        return max(1, cv2.getTrackbarPos("FPS", self.win))

    def _lin_to_pos(self, value: float, lo: float, hi: float, steps: int) -> int:
        if hi <= lo:
            return 0
        t = (float(value) - lo) / (hi - lo)
        return int(np.clip(round(t * steps), 0, steps))

    def _pos_to_lin(self, pos: int, lo: float, hi: float, steps: int) -> float:
        t = min(steps, max(0, int(pos))) / float(steps)
        return float(lo + t * (hi - lo))

    def _safe_set_pos(self, name: str, pos: int) -> None:
        try:
            cv2.setTrackbarPos(name, self.win, int(pos))
        except Exception:
            pass

    def _pull_autos_from_camera(self) -> None:
        if self.camera.backend != "spinnaker":
            return
        self.checks["exp_auto"] = self.camera.is_auto_on("ExposureAuto")
        self.checks["gain_auto"] = self.camera.is_auto_on("GainAuto")
        self.checks["wb_auto"] = self.camera.is_auto_on("BalanceWhiteAuto")

    def _pull_values_from_camera(self) -> None:
        if self.camera.backend != "spinnaker":
            return
        exp = self.camera.get_float("ExposureTime")
        if exp is not None:
            self._safe_set_pos(
                "Exposure",
                _log_to_pos(exp, *self._exp_range, self.EXP_STEPS),
            )
        gain = self.camera.get_float("Gain")
        if gain is not None:
            self._safe_set_pos(
                "Gain",
                self._lin_to_pos(gain, *self._gain_range, self.GAIN_STEPS),
            )
        red = self.camera.get_balance_ratio("Red")
        if red is not None:
            self._safe_set_pos(
                "WB Red",
                self._lin_to_pos(red, *self._wb_range, self.WB_STEPS),
            )
        blue = self.camera.get_balance_ratio("Blue")
        if blue is not None:
            self._safe_set_pos(
                "WB Blue",
                self._lin_to_pos(blue, *self._wb_range, self.WB_STEPS),
            )

    def _mouse_to_panel(self, x: int, y: int) -> tuple[int, int]:
        try:
            _wx, _wy, ww, wh = cv2.getWindowImageRect(self.win)
            ww, wh = int(ww), int(wh)
        except Exception:
            return int(x), int(y)
        if ww < 8 or wh < 8:
            return int(x), int(y)
        cx = int(round(x * self.PANEL_W / float(ww)))
        cy = int(round(y * self.PANEL_H / float(wh)))
        return cx, cy

    def _on_mouse(self, event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONUP:
            return
        cx, cy = self._mouse_to_panel(x, y)
        for key, x1, y1, x2, y2 in self._hitboxes:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                self.checks[key] = not bool(self.checks[key])
                self._last_render = 0.0
                self.render()
                return

    def sync_quality(self) -> bool:
        """Apply HQ Grab checkbox. Returns True if the mode actually changed."""
        high = bool(self.checks["hq_grab"])
        if high == self._quality_high:
            return False
        self._quality_high = high
        apply_quality_preset(high)
        self.camera.set_quality_mode(high)
        print(
            "Quality: HQ (1920 / YOLO 1280)"
            if high
            else "Quality: FAST (960 / YOLO 640)"
        )
        return True

    def sync_color_corr(self) -> None:
        on = bool(self.checks["color_corr"])
        if on == self._color_corr:
            return
        self._color_corr = on
        self.camera.set_color_correction(on)
        print("Color correction: ON" if on else "Color correction: OFF")

    def apply(self) -> bool:
        changed = self.sync_quality()
        self.sync_color_corr()
        if self.camera.backend != "spinnaker":
            return changed
        exp_auto = bool(self.checks["exp_auto"])
        gain_auto = bool(self.checks["gain_auto"])
        wb_auto = bool(self.checks["wb_auto"])

        self.camera.set_enum("ExposureAuto", "Continuous" if exp_auto else "Off")
        if not exp_auto:
            us = _pos_to_log(
                cv2.getTrackbarPos("Exposure", self.win),
                *self._exp_range,
                self.EXP_STEPS,
            )
            self.camera.set_float("ExposureTime", us)

        self.camera.set_enum("GainAuto", "Continuous" if gain_auto else "Off")
        if not gain_auto:
            gain = self._pos_to_lin(
                cv2.getTrackbarPos("Gain", self.win),
                *self._gain_range,
                self.GAIN_STEPS,
            )
            self.camera.set_float("Gain", gain)

        if wb_auto:
            self.camera.set_enum("BalanceWhiteAuto", "Continuous")
        else:
            self.camera.set_enum("BalanceWhiteAuto", "Off")
            red = self._pos_to_lin(
                cv2.getTrackbarPos("WB Red", self.win),
                *self._wb_range,
                self.WB_STEPS,
            )
            blue = self._pos_to_lin(
                cv2.getTrackbarPos("WB Blue", self.win),
                *self._wb_range,
                self.WB_STEPS,
            )
            self.camera.set_balance_ratio("Red", red)
            self.camera.set_balance_ratio("Blue", blue)

        now = time.perf_counter()
        if now - self._last_auto_refresh >= 0.5:
            self._last_auto_refresh = now
            if exp_auto or gain_auto or wb_auto:
                self._pull_values_from_camera()
        return changed

    def current_readout(self) -> dict[str, str]:
        fps = self.target_fps()
        if self.camera.backend != "spinnaker":
            return {
                "FPS": str(fps),
                "camera": "opencv (no FLIR sliders)",
            }
        exp = self.camera.get_float("ExposureTime")
        gain = self.camera.get_float("Gain")
        red = self.camera.get_balance_ratio("Red")
        blue = self.camera.get_balance_ratio("Blue")
        return {
            "FPS": str(fps),
            "Exposure": (
                f"{'AUTO' if self.checks['exp_auto'] else 'MAN'}  "
                f"{exp:.0f} us" if exp is not None else "n/a"
            ),
            "Gain": (
                f"{'AUTO' if self.checks['gain_auto'] else 'MAN'}  "
                f"{gain:.2f} dB" if gain is not None else "n/a"
            ),
            "WB": (
                f"{'AUTO' if self.checks['wb_auto'] else 'MAN'}  "
                f"R {red:.2f}  B {blue:.2f}" if red is not None and blue is not None else "n/a"
            ),
        }

    def _draw_checkbox(self, canvas: np.ndarray, x: int, y: int, key: str, label: str) -> None:
        box = self.BOX
        on = bool(self.checks[key])
        cv2.rectangle(canvas, (x, y), (x + box, y + box), (210, 210, 210), 1)
        if on:
            cv2.rectangle(
                canvas,
                (x + 3, y + 3),
                (x + box - 3, y + box - 3),
                (70, 210, 90),
                thickness=-1,
            )
        cv2.putText(
            canvas,
            label,
            (x + box + 8, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        self._hitboxes.append((key, 8, y - 6, self.PANEL_W - 8, y + box + 8))

    def render(self) -> None:
        now = time.perf_counter()
        if now - getattr(self, "_last_render", 0.0) < 0.25:
            return
        self._last_render = now
        canvas = np.full((self.PANEL_H, self.PANEL_W, 3), 30, dtype=np.uint8)
        self._hitboxes = []
        row_y = 16
        row_h = 32
        for i, (key, label) in enumerate(self.CHECKS):
            self._draw_checkbox(canvas, 16, row_y + i * row_h, key, label)
        y = row_y + row_h * len(self.CHECKS) + 20
        for label, value in self.current_readout().items():
            cv2.putText(
                canvas,
                f"{label}: {value}",
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            y += 24
        cv2.putText(
            canvas,
            "Click a box to toggle. Uncheck Auto to use the sliders.",
            (16, self.PANEL_H - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(self.win, canvas)


def run_live_view() -> None:
    """Blocking OpenCV loop. Press q or Esc in the window to stop."""
    camera = LiveCamera(
        CAMERA_BACKEND,
        index=CAMERA_INDEX,
        spinnaker_index=SPINNAKER_CAMERA_INDEX,
    )
    camera.open()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    controls = CameraControlPanel(camera, TARGET_FPS)
    view = {"w": 1, "h": 1, "scale": 1.0, "ox": 0.0, "oy": 0.0}
    roi_state = {"pt1": None, "pt2": None, "dragging": False, "box": None}

    def on_mouse(event, x, y, _flags, _param) -> None:
        ix, iy = _mouse_to_image(x, y, view)
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_state["dragging"] = True
            roi_state["pt1"] = (ix, iy)
            roi_state["pt2"] = (ix, iy)
        elif event == cv2.EVENT_MOUSEMOVE and roi_state["dragging"]:
            roi_state["pt2"] = (ix, iy)
        elif event == cv2.EVENT_LBUTTONUP and roi_state["dragging"]:
            roi_state["dragging"] = False
            roi_state["pt2"] = (ix, iy)
            box = _normalize_roi(
                (*roi_state["pt1"], *roi_state["pt2"]),
                view["w"],
                view["h"],
            )
            roi_state["box"] = box
            if box is None:
                roi_state["pt1"] = roi_state["pt2"] = None

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    t0 = time.perf_counter()
    frames = 0

    print(
        f"Live view started — sliders in '{CONTROLS_WINDOW_NAME}', "
        f"drag ROI on '{WINDOW_NAME}', press q/Esc to stop."
    )

    try:
        while True:
            loop_start = time.perf_counter()
            if controls.apply():
                roi_state["box"] = None
                roi_state["pt1"] = roi_state["pt2"] = None
                roi_state["dragging"] = False
                view["logged"] = False
            target_fps = controls.target_fps()
            period = 1.0 / float(target_fps)

            rgb = camera.read_rgb()
            grab_ms = float(getattr(camera, "last_grab_ms", 0.0))
            if rgb is None:
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue

            frames += 1
            h, w = rgb.shape[:2]
            view["w"], view["h"] = w, h
            if not view.get("logged"):
                native = getattr(camera, "last_native_hw", None)
                native_s = f"{native[0]}x{native[1]}" if native else "?"
                print(
                    f"Camera frame: {native_s}  working: {w}x{h}  "
                    f"(YOLO imgsz={int(globals().get('INFER_IMGSZ', 640))}, device={resolved_device})"
                )
                view["logged"] = True

            t_infer = time.perf_counter()
            display_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            infer_imgsz = int(globals().get("INFER_IMGSZ", 640))
            results = model.predict(
                display_bgr,
                imgsz=infer_imgsz,
                conf=CONF,
                device=resolved_device,
                retina_masks=not bool(globals().get("FAST_LIVE", False)),
                quantize=16 if str(resolved_device).lower() not in {"cpu"} else None,
                verbose=False,
            )
            infer_ms = (time.perf_counter() - t_infer) * 1000.0

            t_mask = time.perf_counter()
            live_roi = roi_state["box"]
            if roi_state["dragging"] and roi_state["pt1"] and roi_state["pt2"]:
                live_roi = _normalize_roi(
                    (*roi_state["pt1"], *roi_state["pt2"]),
                    w,
                    h,
                )
            if results:
                display_bgr = overlay_masks(
                    display_bgr,
                    results[0],
                    class_names,
                    CLASS_COLORS_BGR,
                    MASK_ALPHA,
                    roi=live_roi,
                )
            if live_roi is not None:
                x1, y1, x2, y2 = live_roi
                cv2.rectangle(display_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
            draw_legend(display_bgr, class_names)
            mask_ms = (time.perf_counter() - t_mask) * 1000.0
            if not view.get("window_ready"):
                init_w = min(int(globals().get("DISPLAY_MAX_WIDTH", 960)), w)
                init_h = max(1, int(round(init_w * h / float(w))))
                cv2.resizeWindow(WINDOW_NAME, init_w, init_h)
                view["disp_w"], view["disp_h"] = init_w, init_h
                view["window_ready"] = True
            disp_w, disp_h = _video_window_size(WINDOW_NAME, view, w, h)
            shown, scale, ox, oy = _letterbox_to_size(display_bgr, disp_w, disp_h)
            view["scale"] = scale
            view["ox"] = ox
            view["oy"] = oy
            view["disp_w"], view["disp_h"] = shown.shape[1], shown.shape[0]
            work_s = max(1e-6, time.perf_counter() - loop_start)
            pipe_fps = 1.0 / work_s
            view["fps"] = 0.8 * float(view.get("fps") or pipe_fps) + 0.2 * pipe_fps
            view["fps_max"] = max(float(view.get("fps_max") or 0.0), pipe_fps)
            device_text = (
                f"CUDA:{resolved_device}"
                if str(resolved_device).lower() not in {"cpu"}
                else "CPU"
            )
            show_ms = max(0.0, work_s * 1000.0 - grab_ms - infer_ms - mask_ms)
            stages = {
                "grab": grab_ms,
                "infer": infer_ms,
                "mask": mask_ms,
                "show": show_ms,
            }
            for key, val in stages.items():
                ema_key = f"ema_{key}"
                view[ema_key] = 0.8 * float(view.get(ema_key) or val) + 0.2 * val
            limit = max(stages, key=lambda k: float(view.get(f"ema_{k}") or 0.0))
            extra = [
                "HQ" if not bool(globals().get("FAST_LIVE", False)) else "fast",
                f"grab {view['ema_grab']:.0f}ms",
                f"infer {view['ema_infer']:.0f}ms",
                f"mask {view['ema_mask']:.0f}ms",
                f"limit: {limit}",
            ]
            draw_fps_hud(shown, view["fps"], view["fps_max"], device_text, extra)
            controls.render()

            elapsed = time.perf_counter() - t0
            if elapsed >= 2.0:
                fps = frames / elapsed
                cv2.setWindowTitle(WINDOW_NAME, f"{WINDOW_NAME} — {fps:.1f} fps")
                frames = 0
                t0 = time.perf_counter()

            cv2.imshow(WINDOW_NAME, shown)
            remaining_ms = int((period - (time.perf_counter() - loop_start)) * 1000)
            key = cv2.waitKey(max(1, remaining_ms)) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("c"), ord("C")):
                roi_state["pt1"] = roi_state["pt2"] = roi_state["box"] = None
                roi_state["dragging"] = False
    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("Stopped.")




def main() -> None:
    run_live_view()


if __name__ == "__main__":
    main()
