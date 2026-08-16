"""
Convenience launcher for Streamlit Interactive Dashboard.
Run this from the project root:
    python run_dashboard.py
"""
import asyncio
import os
import subprocess
import sys
import warnings
from pathlib import Path

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Fix: Windows Proactor event loop doesn't support ZMQ add_reader;
# Use Selector policy to suppress the warning when streamlit uses asyncio on Windows.
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
dashboard = PROJECT_ROOT / "dashboard" / "app.py"

if __name__ == "__main__":
    print("🚀 Launching Streamlit Interactive Graph Intelligence Dashboard...")
    print(f"📂 Workspace: {PROJECT_ROOT}")
    print("🌐 Open URL in browser if it does not automatically launch.\n")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard)], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard server stopped.")
