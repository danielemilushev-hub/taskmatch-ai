@echo off
setlocal enabledelayedexpansion
title TaskMatch AI

cd /d "%~dp0"

echo ======================================================
echo    TaskMatch AI - Local Model Benchmarking Studio
echo ======================================================
echo.

:: Detect virtual environment if present
if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
) else if exist "env\Scripts\activate.bat" (
    echo [*] Activating virtual environment (env)...
    call env\Scripts\activate.bat
)

:: Check for Python
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PY_CMD=python"
) else (
    where py >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set "PY_CMD=py -3"
    ) else (
        echo [ERROR] Python was not found on your system PATH.
        echo Please install Python 3.10+ from python.org or the Microsoft Store.
        echo.
        pause
        exit /b 1
    )
)

echo [*] Starting TaskMatch AI server and opening browser...
echo [*] URL: http://127.0.0.1:8000
echo.
echo (Keep this window open while using TaskMatch AI. Press Ctrl+C to stop.)
echo ======================================================
echo.

%PY_CMD% cli.py serve

if %ERRORLEVEL% neq 0 (
    echo.
    echo [!] Server stopped or encountered an error.
    echo If packages are missing, run: pip install -r requirements.txt
    echo.
    pause
)
