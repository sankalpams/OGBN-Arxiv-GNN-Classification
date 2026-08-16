@echo off
title 🧠 OGBN-Arxiv Deep GNN Intelligence Suite
chcp 65001 >nul
color 0B
cd /d "%~dp0"

:MENU
cls
echo  ==============================================================================
echo   ██████╗  ██████╗ ██████╗ ███╗   ██╗    █████╗ ██████╗ ██╗   ██╗██╗██╗   ██╗
echo  ██╔═══██╗██╔════╝ ██╔══██╗████╗  ██║   ██╔══██╗██╔══██╗██║   ██║██║██║   ██║
echo  ██║   ██║██║  ███╗██████╔╝██╔██╗ ██║   ███████║██████╔╝██║   ██║██║██║   ██║
echo  ██║   ██║██║   ██║██╔══██╗██║╚██╗██║   ██╔══██║██╔══██╗╚██╗ ██╔╝██║╚██╗ ██╔╝
echo  ╚██████╔╝╚██████╔╝██████╔╝██║ ╚████║   ██║  ██║██║  ██║ ╚████╔╝ ██║ ╚████╔╝ 
echo   ╚═════╝  ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═══╝  
echo  ==============================================================================
echo            ⚡ GRAPH NEURAL NETWORKS (GCN vs. GAT) EXPERIMENT WORKSPACE ⚡
echo  ==============================================================================
echo.
echo   [1] 📓  Launch Jupyter Notebook (Default Browser)
echo   [2] 🧪  Launch JupyterLab Workspace (Full Modern IDE)
echo   [3] 📊  Launch Streamlit Interactive Dashboard
echo   [4] 🔍  Run System & Environment Health Check
echo   [5] 📄  Compile Publication Technical Report (PDF)
echo   [0] ❌  Exit
echo.
echo  ==============================================================================
set /p choice="  👉 Enter your selection (0-5) [default: 1]: "

if "%choice%"=="" set choice=1
if "%choice%"=="1" goto NOTEBOOK
if "%choice%"=="2" goto JUPYTERLAB
if "%choice%"=="3" goto DASHBOARD
if "%choice%"=="4" goto HEALTHCHECK
if "%choice%"=="5" goto REPORT
if "%choice%"=="0" goto EXIT

echo   [!] Invalid selection. Please choose a number from 0 to 5.
timeout /t 2 >nul
goto MENU

:NOTEBOOK
cls
echo 🚀 Starting Jupyter Notebook Server...
python run_notebooks.py --notebook
goto PAUSE_MENU

:JUPYTERLAB
cls
echo 🚀 Starting JupyterLab Workspace...
python run_notebooks.py --lab
goto PAUSE_MENU

:DASHBOARD
cls
echo 🚀 Launching Streamlit Dashboard...
python run_dashboard.py
goto PAUSE_MENU

:HEALTHCHECK
cls
echo 🔍 Running Pre-Flight Diagnostics...
python run_notebooks.py --health
goto PAUSE_MENU

:REPORT
cls
echo 📄 Generating Publication-Grade PDF Report...
python build_full_report_pdf.py
goto PAUSE_MENU

:PAUSE_MENU
echo.
echo  ==============================================================================
echo   Process completed. Press any key to return to main menu...
pause >nul
goto MENU

:EXIT
cls
echo.
echo   👋 Exiting OGBN-Arxiv Intelligence Suite. Happy Researching!
echo.
timeout /t 2 >nul
exit /b 0
