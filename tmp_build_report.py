"""
Builds the full, verified CCS4354 Technical Report PDF.
Usage:
    python tmp_build_report.py
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    builder = Path(__file__).resolve().parent / "build_full_report_pdf.py"
    subprocess.run([sys.executable, str(builder)], check=True)
