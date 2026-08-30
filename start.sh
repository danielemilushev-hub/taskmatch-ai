#!/usr/bin/env bash
# TaskMatch AI launcher for macOS / Linux.
# POSIX counterpart to start.bat, so the repo isn't Windows-only at the
# double-click level. Everything it does is also achievable by hand with
# `python cli.py serve` -- this just adds venv auto-activation and a
# friendlier error when Python or the dependencies are missing.
set -euo pipefail

cd "$(dirname "$0")"

echo "======================================================"
echo "   TaskMatch AI - Local Model Benchmarking Studio"
echo "======================================================"
echo

# Activate a virtualenv if one is sitting in the usual place.
for venv in .venv venv env; do
    if [ -f "$venv/bin/activate" ]; then
        echo "[*] Activating virtual environment: $venv"
        # shellcheck disable=SC1090
        . "$venv/bin/activate"
        break
    fi
done

# python3 first: on most Linux distros and modern macOS, bare `python` is
# either absent or (historically) Python 2.
if command -v python3 >/dev/null 2>&1; then
    PY_CMD=python3
elif command -v python >/dev/null 2>&1; then
    PY_CMD=python
else
    echo "[ERROR] Python was not found on your PATH."
    echo "Install Python 3.11+ from python.org or your package manager."
    exit 1
fi

echo "[*] Starting TaskMatch AI server and opening browser..."
echo "[*] URL: http://127.0.0.1:8000"
echo
echo "[!] Keep this window open while using TaskMatch AI."
echo "[!] Press Ctrl+C when you want to stop the server."
echo "======================================================"
echo

if ! "$PY_CMD" cli.py serve; then
    echo
    echo "[!] Server stopped or encountered an error."
    echo "If packages are missing, run: $PY_CMD -m pip install -r requirements.txt"
    exit 1
fi
