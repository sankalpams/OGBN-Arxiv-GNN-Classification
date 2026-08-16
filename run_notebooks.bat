@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
title OGBN-Arxiv Jupyter Notebooks Launcher
cd /d "%~dp0"
echo =======================================================
echo   Starting Jupyter Notebooks for OGBN-Arxiv Project...
echo =======================================================
echo.
python run_notebooks.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start Jupyter with 'python'. Trying with 'py'...
    py run_notebooks.py %*
)
pause
