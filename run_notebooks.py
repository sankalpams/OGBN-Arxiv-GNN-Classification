"""
Run Jupyter Notebooks / JupyterLab with proper Windows asyncio, UTF-8, and environment settings.
"""

import os
import sys
import warnings
import argparse
import subprocess
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Suppress event loop deprecation warnings on Windows Python 3.14
warnings.filterwarnings("ignore", category=DeprecationWarning)

if sys.platform == "win32":
    import asyncio
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def main():
    parser = argparse.ArgumentParser(description="Launch Jupyter Notebooks for OGBN-Arxiv Project")
    parser.add_argument("--lab", action="store_true", help="Launch JupyterLab instead of classic Notebook")
    parser.add_argument("--port", type=int, default=8888, help="Port to run Jupyter on (default: 8888)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    # Ensure src is in python path
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    app_type = "jupyterlab" if args.lab else "notebook"
    app_name = "JupyterLab" if args.lab else "Jupyter Notebook"

    print("=" * 65)
    print(f"[*] Launching {app_name} for OGBN-Arxiv Project")
    print(f"[*] Project Root: {PROJECT_ROOT}")
    print(f"[*] Notebooks Dir: {NOTEBOOKS_DIR}")
    print(f"[*] Python Executable: {sys.executable}")
    print("=" * 65)

    cmd = [
        sys.executable,
        "-m",
        app_type,
        f"--port={args.port}",
        f"--ServerApp.root_dir={PROJECT_ROOT}",
        f"--ServerApp.preferred_dir={NOTEBOOKS_DIR}",
    ]

    if args.no_browser:
        cmd.append("--no-browser")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[*] Jupyter server stopped cleanly.")


if __name__ == "__main__":
    main()
