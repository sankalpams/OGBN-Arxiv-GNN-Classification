# 🕸️ OGBN-Arxiv Graph Intelligence: Simple Project Guide & Task Breakdown

**Course:** CCS4354 — Tensors and Graphs (with Programming)  
**Institution:** Sri Lanka Technology Campus (SLTC)  
**Project Title:** Graph Neural Network Based Node Classification on the OGBN-Arxiv Citation Network  
**Target Dataset:** Open Graph Benchmark (`ogbn-arxiv`)  

---

## 📖 1. What is this Project About? (The Big Picture)

### The Core Problem
Imagine you have a digital library containing **169,343 computer science research papers** from arXiv. For each paper, you have two types of information:
1. **Paper Content:** A 128-dimensional mathematical summary (word embeddings) representing the words in the title and abstract.
2. **Citation Network:** Over **2.3 million citation links** showing which papers cite each other.

### The Objective
Build an AI system using **Graph Neural Networks (GNNs)** that predicts which of **40 Computer Science subject categories** (such as *Artificial Intelligence, Computer Vision, Cryptography, Data Structures, Robotics, etc.*) each paper belongs to.

```
       [ Paper A: "Deep Residual Learning for Image Recognition" ]
                                   │
                           (cites / cited by)
                                   ▼
             [ Paper B: "Convolutional Neural Networks" ]
                                   │
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │            Graph Neural Network (GNN)            │
          │  - Reads 128-dim word embeddings                 │
          │  - Aggregates clues from neighbor citations      │
          └──────────────────────────────────────────────────┘
                                   │
                                   ▼
              Predicted Category: [ cs.CV (Computer Vision) ]
```

### Why Graph Neural Networks (GNNs)?
Traditional machine learning algorithms evaluate each paper in isolation without knowing its context. However, scientific papers naturally form a **network (graph)**. Papers that cite each other are overwhelmingly likely to belong to the same research domain (**Homophily**). GNNs use a process called **Message Passing** — where each paper continuously aggregates information from its citation neighbors to make much more accurate predictions.

---

## 📊 2. Key Results & Findings

Two primary GNN architectures were implemented and benchmarked on **48,603 unseen test papers (published in 2019–2020)**:

| Model Architecture | Test Accuracy | Weighted F1 | Model Parameters | Epoch Speed (CPU) | Key Feature |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Graph Convolutional Network (GCN)** | **58.64%** *(Fine-tuned: 69.62%)* | **0.5389** | 43,816 | **~2.4 seconds** | Fast, stable isotropic averaging |
| **Graph Attention Network (GAT)** | **57.39%** *(Fine-tuned: 68.95%)* | **0.5220** | 43,624 | ~7.8 seconds | Learns dynamic neighbor importance |
| **Performance Difference** | **+1.25% (GCN leads)** | **+0.0169** | +192 | **3.25× Faster** | GCN outperforms with less compute |

### Why did GCN beat GAT?
1. **Smooth Citation Homophily:** In dense citation networks where neighboring papers share the same subject (>85% agreement), simple symmetric averaging ($\mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-\frac{1}{2}}$) acts as an ideal noise filter.
2. **Temporal Generalization:** GAT's multi-head attention mechanism learned historical citation patterns that slightly overfitted older citation styles and generalized less effectively to future papers.
3. **Computational Efficiency:** GCN trained **$3.25\times$ faster** per epoch.

---

## 🛠️ 3. Task-by-Task Explanation

Below is the simple, clear explanation of each task completed across the coursework:

---

### 🔹 Task 01 – Tensor Fundamentals (10 Marks)
* **Notebook:** [`notebooks/01_tensor_fundamentals.ipynb`](notebooks/01_tensor_fundamentals.ipynb)
* **What it is:** The mathematical foundation of deep learning in PyTorch.
* **Key Concepts Covered:**
  - **Tensor Creation:** Building 0D scalars, 1D vectors, 2D matrices, and converting from NumPy arrays.
  - **Indexing & Slicing:** Extracting specific node rows or feature columns.
  - **Reshaping & Flattening:** Transforming tensor dimensions for linear layers.
  - **Matrix Multiplication:** Computing $\mathbf{XW}$ transformations in neural network layers.
  - **Broadcasting:** Automatically expanding bias vectors across all $N$ nodes without extra memory.
  - **Aggregation Operations:** `mean()`, `sum()`, and `max()` operations that mirror GNN neighborhood pooling.
  - **GPU Acceleration:** Verifying CUDA device availability and transferring tensors to GPU memory.

---

### 🔹 Task 02 – Graph Representation and Analysis (10 Marks)
* **Notebook:** [`notebooks/02_graph_representation_analysis.ipynb`](notebooks/02_graph_representation_analysis.ipynb)
* **What it is:** Loading the large-scale graph and analyzing its structural topology.
* **Key Findings & Metrics:**
  - **Graph Size:** 169,343 papers (nodes) and 2,315,598 symmetrized citation links (edges).
  - **Extreme Sparsity:** Density $\rho = 8.075 \times 10^{-5}$ (~0.008%), meaning papers only cite a tiny fraction of all existing papers.
  - **Degree Statistics:** Average citation count is 13.67, median is 6.0, and maximum degree is 13,161 (Node 1353 is a super-hub paper).
  - **Scale-Free Property:** Plotted log-log degree histograms confirming power-law decay ($P(k) \sim k^{-\gamma}$).
  - **Homophily:** Visualized 2-hop local subgraphs demonstrating that citing papers cluster into identical subject colors.

---

### 🔹 Task 03 – Graph Data Preparation (10 Marks)
* **Notebook:** [`notebooks/03_data_preparation.ipynb`](notebooks/03_data_preparation.ipynb)
* **What it is:** Preparing the dataset for training without data leakage.
* **Key Steps:**
  - **Realistic Temporal Splitting:**
    - **Training Set ($\le 2017$):** 90,941 papers (53.7%)
    - **Validation Set ($2018$):** 29,799 papers (17.6%)
    - **Test Set ($2019\text{--}2020$):** 48,603 papers (28.7%)
  - **Strict Normalization:** Fitted `StandardScaler` strictly on the training set to prevent future data leakage into training features.
  - **Graph Symmetrization & Self-Loops:** Added reciprocal edges ($\mathbf{\tilde{A}} = \mathbf{A} + \mathbf{A}^\top + \mathbf{I}_N$) so information flows both forward and backward through citations, and nodes retain their own identity.

---

### 🔹 Task 04 – Graph Neural Network Development (25 Marks)
* **Notebooks:** [`notebooks/04_gcn_model.ipynb`](notebooks/04_gcn_model.ipynb), [`notebooks/05_gat_model.ipynb`](notebooks/05_gat_model.ipynb)
* **Source Files:** [`src/models/gcn.py`](src/models/gcn.py), [`src/models/gat.py`](src/models/gat.py)
* **What it is:** Building the GNN model architectures in PyTorch Geometric.
* **Models Implemented:**
  1. **Graph Convolutional Network (GCN):**
     - 2 Spectral `GCNConv` layers (128 input $\to$ 256 hidden $\to$ 40 output classes).
     - ReLU activation + Dropout ($p=0.5$).
     - Total Parameters: 43,816.
  2. **Graph Attention Network (GAT):**
     - 2 Spatial `GATConv` layers with 4 attention heads ($4 \times 64 = 256$ hidden channels).
     - ELU activation + Dropout ($p=0.5$).
     - Total Parameters: 43,624.
  3. **GraphSAGE (Alternative):** Inductive neighborhood sampling architecture (86,312 parameters).

---

### 🔹 Task 05 – Model Training and Optimization (10 Marks)
* **Notebook:** [`notebooks/06_training_optimization.ipynb`](notebooks/06_training_optimization.ipynb)
* **What it is:** Training models and executing systematic hyperparameter tuning.
* **Key Components:**
  - **Loss Function:** Negative Log-Likelihood / Multi-Class Cross-Entropy Loss.
  - **Optimizer:** Adam optimizer with weight decay ($L_2$ regularization).
  - **Hyperparameter Grid Search:** Evaluated hidden dimensions (128 vs. 256), dropout rates (0.3 vs. 0.5), and learning rates (0.01 vs. 0.005).
  - **Model Checkpoints:** Preserved best validation weights to [`models/best_gcn.pt`](models/best_gcn.pt) and [`models/best_gat.pt`](models/best_gat.pt).

---

### 🔹 Task 06 – Model Evaluation (10 Marks)
* **Notebook:** [`notebooks/07_model_evaluation.ipynb`](notebooks/07_model_evaluation.ipynb)
* **What it is:** Evaluating trained models on 48,603 unseen test papers.
* **Key Insights:**
  - **Metrics Evaluated:** Accuracy, Weighted Precision, Weighted Recall, Weighted F1, Macro F1.
  - **Class Imbalance:** High-support classes (`cs.CV`: 10,477 papers, `cs.LG`: 10,740 papers) reached $F_1 > 0.75$, while rare classes (`cs.OS`: 5 papers) had fewer samples.
  - **Confusion Matrices:** Confirmed strong diagonal accuracy with minor confusion between closely related subfields (e.g., `cs.AI` and `cs.LG`).

---

### 🔹 Task 07 – Graph Explainability and Embedding Analysis (10 Marks)
* **Notebook:** [`notebooks/08_explainability_embeddings.ipynb`](notebooks/08_explainability_embeddings.ipynb)
* **What it is:** Opening the AI "black box" to understand representations and predictions.
* **Techniques Used:**
  - **Feature Importance:** Ranked the most influential input embedding dimensions (Top features: #49, #115, #3, #38).
  - **Latent Embedding Projections (PCA & t-SNE):** Mapped 256-dimensional hidden representations into 2D, visually showing distinct topic clusters.
  - **Neighborhood Homophily Score:** Verified that over 85.7% of neighboring papers shared the predicted category, confirming neighborhood influence.

---

### 🔹 Task 08 – Graph Intelligence Dashboard (5 Marks)
* **Application Entry:** [`dashboard/app.py`](dashboard/app.py) / [`run_dashboard.py`](run_dashboard.py)
* **What it is:** A complete interactive Streamlit web dashboard accessible at `http://localhost:8501`.
* **5 Interactive Tabs:**
  1. 📊 **Graph Analysis:** Interactive topological KPIs, degree distribution charts, and force-directed subgraphs.
  2. 📈 **Training Dynamics:** Loss curves and validation accuracy trajectories over epochs.
  3. 🏆 **Model Evaluation:** Metric comparison tables, performance bar charts, and 40-class confusion matrices.
  4. 🔬 **Paper Lookup Demo:** Search papers by ID or category, inspect predictions, and filter by model agreement/disagreement.
  5. 🌌 **Embeddings & Manifolds:** Interactive 2D PCA and t-SNE manifold visualizer.

---

### 🔹 Task 09 – Technical Report & Presentation (10 Marks)
* **Artifacts:** [`report/CCS4354_Technical_Report.pdf`](report/CCS4354_Technical_Report.pdf) & [`presentation/README.md`](presentation/README.md)
* **What it is:** Formal academic deliverables.
* **Delivered:**
  - Complete 24-page SLTC-format academic PDF report.
  - 10–15 minute slide deck presentation plan and script.

---

### 🌟 Bonus Work (Up to 5 Extra Marks)
1. **Bonus A — Graph Transformers:** Attention-based long-range graph token routing.
2. **Bonus B — Relational GNNs (RGCN):** Multi-relational modeling for distinct citation types.
3. **Bonus C — Self-Supervised Learning (Deep Graph Infomax - DGI):** Contrastive pre-training with GCN fine-tuning reaching **69.92% test accuracy**.
4. **Bonus D — Advanced Optimization:** Focal Loss + 5-model weighted ensemble achieving top Macro F1 (**0.5199**).

---

## 📁 4. Project Directory Structure

```text
Tenso Project/
├── dashboard/
│   ├── app.py                         # Streamlit interactive application
│   └── components/                    # Modular UI cards & visualizations
├── models/
│   ├── best_gcn.pt                    # Trained GCN model weights
│   └── best_gat.pt                    # Trained GAT model weights
├── notebooks/
│   ├── 01_tensor_fundamentals.ipynb   # Task 01: Tensor operations
│   ├── 02_graph_representation_analysis.ipynb # Task 02: Graph topology
│   ├── 03_data_preparation.ipynb      # Task 03: Splitting & normalization
│   ├── 04_gcn_model.ipynb             # Task 04: GCN training
│   ├── 05_gat_model.ipynb             # Task 04: GAT training
│   ├── 06_training_optimization.ipynb # Task 05: Hyperparameter tuning
│   ├── 07_model_evaluation.ipynb      # Task 06: Metrics & benchmarking
│   └── 08_explainability_embeddings.ipynb # Task 07: PCA, t-SNE & homophily
├── report/
│   ├── CCS4354_Technical_Report.pdf     # 24-Page Formal Academic PDF Report
│   └── assets/                        # High-resolution generated figures
├── results/                           # Evaluation CSVs and plots
├── src/                               # Reusable Python pipeline modules
├── build_sltc_report_pdf.py           # Technical Report PDF generator
├── generate_report_assets.py          # High-DPI Plot generator
├── run_dashboard.py                   # One-click dashboard launcher
└── requirements.txt                   # Project Python dependencies
```

---

## 🚀 5. How to Run the Project

### 1. Run the Interactive Web Dashboard
```bash
python run_dashboard.py
```
*Open your browser and navigate to **`http://localhost:8501`**.*

### 2. Re-generate High-Resolution Plots & Figures
```bash
python generate_report_assets.py
```

### 3. Re-compile the 24-Page Academic Report PDF
```bash
python build_sltc_report_pdf.py
```
*Output generated at `report/CCS4354_Technical_Report.pdf`.*
