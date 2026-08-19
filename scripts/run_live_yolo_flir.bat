@echo off
title Live Pecan YOLO
cd /d "%~dp0.."

set "PY=%CD%\.venv-spinnaker\Scripts\python.exe"
set "SCRIPT=%CD%\scripts\live_yolo_flir.py"

if not exist "%PY%" (
    echo Could not find .venv-spinnaker at:
    echo   %PY%
    echo.
    echo From the repo folder run:
    echo   powershell -File scripts\setup_spinnaker_env.ps1
    pause
    exit /b 1
)

if not exist "%SCRIPT%" (
    echo Missing %SCRIPT%
    pause
    exit /b 1
)

echo Starting live FLIR + YOLO...
echo Close the OpenCV windows or press q / Esc to stop.
echo.
"%PY%" "%SCRIPT%"
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" (
    echo.
    echo Live view exited with error code %ERR%.
    pause
)
