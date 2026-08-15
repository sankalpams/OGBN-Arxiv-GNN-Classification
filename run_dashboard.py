"""
Convenience launcher — run this from the project root:
    python run_dashboard.py
"""
import asyncio
import subprocess
import sys
from pathlib import Path

# Fix: Windows Proactor event loop doesn't support ZMQ add_reader;
# use Selector policy to suppress the warning when streamlit uses asyncio.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

dashboard = Path(__file__).resolve().parent / "dashboard" / "app.py"
subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard)], check=True)
