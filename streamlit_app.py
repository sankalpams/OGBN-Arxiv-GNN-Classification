"""
Streamlit Cloud / Deployment Root Entrypoint.
Delegates execution to the main dashboard application in dashboard/app.py.
"""
from pathlib import Path
import runpy
import sys

_root = Path(__file__).resolve().parent
_dashboard_dir = _root / "dashboard"

if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_dashboard_dir) not in sys.path:
    sys.path.insert(0, str(_dashboard_dir))

dashboard_main = _dashboard_dir / "app.py"
runpy.run_path(str(dashboard_main), run_name="__main__")
