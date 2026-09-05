"""
Execute all 8 project notebooks (01 through 08) and save them with all outputs, plots, and tables populated.
"""
import sys
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
TARGET_NOTEBOOKS = [
    "01_tensor_fundamentals.ipynb",
    "02_graph_representation_analysis.ipynb",
    "03_data_preparation.ipynb",
    "04_gcn_model.ipynb",
    "05_gat_model.ipynb",
    "06_training_optimization.ipynb",
    "07_model_evaluation.ipynb",
    "08_explainability_embeddings.ipynb"
]

def execute_nb(nb_name):
    nb_path = NOTEBOOKS_DIR / nb_name
    print(f"--> Executing {nb_name}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NOTEBOOKS_DIR)}})
    
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"--> [SUCCESS] {nb_name} executed and saved with outputs!")

if __name__ == "__main__":
    for nb in TARGET_NOTEBOOKS:
        try:
            execute_nb(nb)
        except Exception as e:
            print(f"[ERROR] in {nb}: {e}")
