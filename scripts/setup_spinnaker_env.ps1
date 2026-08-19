# Lean Python 3.10 env for live FLIR notebooks (PySpin + YOLO + OpenCV GUI).
# Intentionally does NOT install napari / SAM 2 / the full plugin — that is what
# made .venv-spinnaker huge last time.
# Run from repo root:  powershell -File scripts/setup_spinnaker_env.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $RepoRoot ".venv-spinnaker"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$Wheel = "C:\Program Files\Teledyne\Spinnaker\PySpin\spinnaker_python-4.2.0.88-cp310-cp310-win_amd64.whl"

Set-Location $RepoRoot

if (-not (Test-Path $Wheel)) {
    throw "Spinnaker wheel not found: $Wheel"
}

Write-Host "Installing Python 3.10 via uv..."
uv python install 3.10

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating .venv-spinnaker..."
    uv venv .venv-spinnaker --python 3.10
} else {
    Write-Host "Reusing existing .venv-spinnaker"
}

Write-Host "Installing CUDA PyTorch (for live YOLO)..."
uv pip install --python $VenvPython torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host "Installing live-inference packages..."
uv pip install --python $VenvPython ultralytics opencv-python ipykernel pillow numpy

Write-Host "Installing Spinnaker PySpin wheel..."
uv pip install --python $VenvPython $Wheel

# ultralytics may pull opencv-python-headless, which cannot open GUI windows.
$ErrorActionPreference = "Continue"
uv pip uninstall --python $VenvPython opencv-python-headless
$ErrorActionPreference = "Stop"
uv pip install --python $VenvPython opencv-python

# Unrelated PyPI package named 'pyspin' can shadow the real SDK.
$ErrorActionPreference = "Continue"
uv pip show --python $VenvPython pyspin | Out-Null
$hasFakePyspin = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = "Stop"
if ($hasFakePyspin) {
    Write-Host "Removing conflicting PyPI 'pyspin' package..."
    uv pip uninstall --python $VenvPython pyspin
    uv pip install --python $VenvPython --reinstall $Wheel
}

Write-Host "Registering Jupyter kernel 'Pecan PySpin (Python 3.10)'..."
& $VenvPython -m ipykernel install --user --name pecan-spinnaker --display-name "Pecan PySpin (Python 3.10)"

Write-Host "Verifying..."
& $VenvPython -c @"
import sys
print('Python', sys.version.split()[0])
import PySpin
s = PySpin.System.GetInstance()
print('PySpin OK, cameras:', s.GetCameras().GetSize())
s.ReleaseInstance()
import cv2
print('OpenCV', cv2.__version__, 'GUI' if hasattr(cv2, 'imshow') else 'NO GUI')
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
from ultralytics import YOLO
print('ultralytics OK')
"@

Write-Host ""
Write-Host "Done. In Cursor, switch this notebook kernel to: Pecan PySpin (Python 3.10)"
