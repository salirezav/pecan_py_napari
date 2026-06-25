# One-time setup: Python 3.10 env with Teledyne Spinnaker (PySpin) for live FLIR notebooks.
# Run from repo root:  powershell -File scripts/setup_spinnaker_env.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot ".venv-spinnaker\Scripts\python.exe"
$Wheel = "C:\Program Files\Teledyne\Spinnaker\PySpin\spinnaker_python-4.2.0.88-cp310-cp310-win_amd64.whl"

Set-Location $RepoRoot

if (-not (Test-Path $Wheel)) {
    throw "Spinnaker wheel not found: $Wheel"
}

Write-Host "Installing Python 3.10 via uv..."
uv python install 3.10

Write-Host "Creating .venv-spinnaker..."
uv venv .venv-spinnaker --python 3.10

Write-Host "Installing Spinnaker PySpin wheel..."
uv pip install --python $VenvPython $Wheel

Write-Host "Installing project + notebook dependencies..."
uv pip install --python $VenvPython -e . ipykernel opencv-python-headless

# A unrelated PyPI package named 'pyspin' can shadow the real SDK.
if (uv pip show --python $VenvPython pyspin 2>$null) {
    Write-Host "Removing conflicting PyPI 'pyspin' package..."
    uv pip uninstall --python $VenvPython pyspin
    uv pip install --python $VenvPython --reinstall $Wheel
}

Write-Host "Registering Jupyter kernel 'Pecan PySpin (Python 3.10)'..."
& $VenvPython -m ipykernel install --user --name pecan-spinnaker --display-name "Pecan PySpin (Python 3.10)"

Write-Host "Verifying PySpin..."
& $VenvPython -c "import PySpin; s=PySpin.System.GetInstance(); print('PySpin OK, cameras:', s.GetCameras().GetSize()); s.ReleaseInstance()"

Write-Host ""
Write-Host "Done. In Jupyter/Cursor, select kernel: Pecan PySpin (Python 3.10)"
