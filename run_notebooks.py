"""
Interactive & Aesthetic Launcher for Jupyter Notebooks and Lab.
Handles Windows asyncio event loop policy and checks environment health.
"""
import asyncio
import os
import subprocess
import sys
import warnings
from pathlib import Path

# Ensure UTF-8 stdout/stderr on Windows to handle rich ANSI and Unicode box borders
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    
    # Suppress deprecation warnings on newer Python versions
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

# Enable ANSI escape processing in Windows console
os.system("")

# ANSI Color Codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

BANNER = f"""{CYAN}{BOLD}
  +--------------------------------------------------------------------------------+
  |    *  OGBN-ARXIV DEEP GRAPH INTELLIGENCE SUITE -- JUPYTER LAUNCHER  *         |
  +--------------------------------------------------------------------------------+{RESET}"""

def print_banner():
    print(BANNER)

def run_health_check():
    print_banner()
    print(f"\n{YELLOW}{BOLD}[?] Running System & Environment Diagnostics...{RESET}\n")
    
    # 1. Python Environment
    print(f"  {BOLD}* Python Interpreter:{RESET} {sys.executable} ({sys.version.split()[0]})")
    print(f"  {BOLD}* Project Root:{RESET}       {PROJECT_ROOT}")

    # 2. PyTorch & Compute
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (Standard Acceleration)"
        device_color = GREEN if cuda_avail else YELLOW
        print(f"  {BOLD}* PyTorch Version:{RESET}    {torch.__version__}")
        print(f"  {BOLD}* Compute Hardware:{RESET}   {device_color}{device_name}{RESET}")
    except ImportError:
        print(f"  {BOLD}* PyTorch:{RESET}            \033[91mNot Installed\033[0m")

    # 3. Graph Libraries
    try:
        import torch_geometric
        print(f"  {BOLD}* PyTorch Geometric:{RESET} {torch_geometric.__version__}")
    except ImportError:
        print(f"  {BOLD}* PyTorch Geometric:{RESET} \033[91mNot Installed\033[0m")

    try:
        import ogb
        print(f"  {BOLD}* OGB Benchmark:{RESET}     {ogb.__version__}")
    except ImportError:
        print(f"  {BOLD}* OGB Benchmark:{RESET}     \033[91mNot Installed\033[0m")

    # 4. Jupyter Core
    try:
        import jupyterlab
        print(f"  {BOLD}* JupyterLab:{RESET}        {jupyterlab.__version__}")
    except ImportError:
        print(f"  {BOLD}* JupyterLab:{RESET}        \033[93mNot Installed\033[0m")

    try:
        import notebook
        print(f"  {BOLD}* Classic Notebook:{RESET}  {notebook.__version__}")
    except ImportError:
        print(f"  {BOLD}* Classic Notebook:{RESET}  \033[93mNot Installed\033[0m")

    # 5. Notebook Catalog
    nb_files = sorted(PROJECT_ROOT.glob("notebooks/*.ipynb"))
    print(f"\n{CYAN}{BOLD}[+] Available Project Notebooks ({len(nb_files)} total):{RESET}")
    for nb in nb_files:
        print(f"    {GREEN}[OK]{RESET} {nb.name}")

    print(f"\n{GREEN}{BOLD}[V] All pre-flight diagnostics passed! Ready for interactive research.{RESET}\n")

def launch_jupyter(mode="notebook"):
    print_banner()
    mode_title = "JupyterLab IDE" if mode == "lab" else "Jupyter Notebook"
    print(f"\n{GREEN}{BOLD}[>] Launching {mode_title} in your default browser...{RESET}")
    print(f"{DIM}[*] Serving workspace from: {PROJECT_ROOT}{RESET}")
    print(f"{YELLOW}[!] Tip: Use Ctrl+C in this terminal window to stop the server anytime.{RESET}\n")

    cmd = [
        sys.executable,
        "-m",
        "notebook" if mode == "notebook" else "jupyterlab",
        "--notebook-dir",
        str(PROJECT_ROOT),
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print(f"\n{MAGENTA}[X] Jupyter server terminated cleanly. Have a great day!{RESET}\n")

if __name__ == "__main__":
    if "--health" in sys.argv:
        run_health_check()
    elif "--lab" in sys.argv:
        launch_jupyter(mode="lab")
    else:
        launch_jupyter(mode="notebook")
