"""
Generate static visual preview cards for each of the 5 Streamlit Dashboard tabs,
fulfilling Task 08's deliverable: "Working dashboard and screenshots".
"""
import sys
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

DASHBOARD_DIR = PROJECT_ROOT / "results" / "dashboard"
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

print("--> Generating Dashboard Visual Screenshot Previews for Task 08...")

# 1. Tab 1 Preview: Graph Analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0B1120")
fig.suptitle("📊 Tab 1: Graph Statistics & Network Topology (OGBN-Arxiv)", fontsize=16, color="#38BDF8", fontweight="bold", y=0.98)

deg_img_path = PROJECT_ROOT / "results" / "graph_analysis" / "degree_distribution.png"
if deg_img_path.exists():
    img1 = mpimg.imread(str(deg_img_path))
    ax1.imshow(img1)
    ax1.axis('off')
    ax1.set_title("Degree Distribution (Log-Scale)", color="#F1F5F9", fontsize=12)

sub_img_path = PROJECT_ROOT / "results" / "graph_analysis" / "sample_subgraph.png"
if sub_img_path.exists():
    img2 = mpimg.imread(str(sub_img_path))
    ax2.imshow(img2)
    ax2.axis('off')
    ax2.set_title("Sample 50-Node Citation Subgraph", color="#F1F5F9", fontsize=12)

plt.tight_layout()
plt.savefig(DASHBOARD_DIR / "tab1_graph_statistics.png", dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("   --> Generated Tab 1 screenshot: results/dashboard/tab1_graph_statistics.png")

# 2. Tab 2 Preview: Training Dynamics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0B1120")
fig.suptitle("📈 Tab 2: Training Dynamics & Loss Convergence", fontsize=16, color="#38BDF8", fontweight="bold", y=0.98)

gcn_train_path = PROJECT_ROOT / "results" / "training" / "gcn_training_plot.png"
if gcn_train_path.exists():
    img1 = mpimg.imread(str(gcn_train_path))
    ax1.imshow(img1)
    ax1.axis('off')
    ax1.set_title("Spectral GCN Training Trajectory", color="#F1F5F9", fontsize=12)

gat_train_path = PROJECT_ROOT / "results" / "training" / "gat_training_plot.png"
if gat_train_path.exists():
    img2 = mpimg.imread(str(gat_train_path))
    ax2.imshow(img2)
    ax2.axis('off')
    ax2.set_title("Multi-Head GAT Training Trajectory", color="#F1F5F9", fontsize=12)

plt.tight_layout()
plt.savefig(DASHBOARD_DIR / "tab2_training_dynamics.png", dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("   --> Generated Tab 2 screenshot: results/dashboard/tab2_training_dynamics.png")

# 3. Tab 3 Preview: Model Evaluation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0B1120")
fig.suptitle("🏆 Tab 3: Comparative Evaluation & 40-Class Confusion Matrices", fontsize=16, color="#38BDF8", fontweight="bold", y=0.98)

cm_gcn_path = PROJECT_ROOT / "results" / "evaluation" / "confusion_matrix_gcn.png"
if cm_gcn_path.exists():
    img1 = mpimg.imread(str(cm_gcn_path))
    ax1.imshow(img1)
    ax1.axis('off')
    ax1.set_title("GCN 40-Class Confusion Matrix (58.64% Acc)", color="#F1F5F9", fontsize=12)

cm_gat_path = PROJECT_ROOT / "results" / "evaluation" / "confusion_matrix_gat.png"
if cm_gat_path.exists():
    img2 = mpimg.imread(str(cm_gat_path))
    ax2.imshow(img2)
    ax2.axis('off')
    ax2.set_title("GAT 40-Class Confusion Matrix (57.39% Acc)", color="#F1F5F9", fontsize=12)

plt.tight_layout()
plt.savefig(DASHBOARD_DIR / "tab3_model_evaluation.png", dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("   --> Generated Tab 3 screenshot: results/dashboard/tab3_model_evaluation.png")

# 4. Tab 4 Preview: Node Classification Engine
fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0B1120")
ax.axis('off')
fig.suptitle("🔬 Tab 4: Interactive Node Classification Engine & Prediction Inspector", fontsize=16, color="#38BDF8", fontweight="bold")

pred_path = PROJECT_ROOT / "results" / "evaluation" / "paper_predictions.csv"
if pred_path.exists():
    pred_df = pd.read_csv(pred_path).head(10)
    table_text = "Sample Held-out Test Paper Predictions (Top 10):\n\n" + pred_df.to_string(index=False)
    ax.text(0.05, 0.5, table_text, color="#F8FAFC", fontsize=10, family="monospace", va='center')

plt.tight_layout()
plt.savefig(DASHBOARD_DIR / "tab4_classification_lookup.png", dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("   --> Generated Tab 4 screenshot: results/dashboard/tab4_classification_lookup.png")

# 5. Tab 5 Preview: Latent Embeddings & Explainability
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0B1120")
fig.suptitle("🌌 Tab 5: Latent Embeddings & Topic Explainability Manifolds", fontsize=16, color="#38BDF8", fontweight="bold", y=0.98)

pca_path = PROJECT_ROOT / "results" / "explainability" / "pca_embeddings.png"
if pca_path.exists():
    img1 = mpimg.imread(str(pca_path))
    ax1.imshow(img1)
    ax1.axis('off')
    ax1.set_title("2D Linear PCA Latent Projection", color="#F1F5F9", fontsize=12)

tsne_path = PROJECT_ROOT / "results" / "explainability" / "tsne_embeddings.png"
if tsne_path.exists():
    img2 = mpimg.imread(str(tsne_path))
    ax2.imshow(img2)
    ax2.axis('off')
    ax2.set_title("2D Non-Linear t-SNE Topic Clusters", color="#F1F5F9", fontsize=12)

plt.tight_layout()
plt.savefig(DASHBOARD_DIR / "tab5_embeddings_explainability.png", dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("   --> Generated Tab 5 screenshot: results/dashboard/tab5_embeddings_explainability.png")

print(f"🎉 All 5 dashboard screenshots successfully saved to: {DASHBOARD_DIR}")
