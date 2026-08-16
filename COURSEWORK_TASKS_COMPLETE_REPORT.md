# CCS4354 — Tensors and Graphs: Complete Coursework Task-by-Task Report

**University:** Sri Lanka Technology Campus (SLTC)  
**Faculty:** Faculty of Computing and Information Technology  
**Module Code:** CCS4354  
**Module Name:** Tensors and Graphs (with Programming)  
**Assignment Title:** Graph Neural Network Based Node Classification on the OGBN-Arxiv Citation Network  
**Submission Date:** 15 August 2026  
**Total Marks:** 100 Marks (+ 5 Bonus Marks)

---

### Group Members & Contribution

| Student Name | Student ID | Coursework Contribution |
| :--- | :---: | :--- |
| **Ravindi Ayodhya** | 23UG1-0136 | Environment setup, Task 01 (Tensor Fundamentals), Bonus A (Graph Transformer), Report Compilation |
| **Amintha Jayasooriya** | CIT-23-02-0335 | Task 02 (Graph Representation & Analysis), Bonus B (Relational GNN / RGCN), Task 08 (Streamlit Dashboard) |
| **Tharanya Pushparaj** | CIT-23-02-0176 | Task 03 (Graph Data Preparation), Bonus C (Self-Supervised Learning / DGI), Task 08 (Dashboard UI) |
| **Damsara Dissanayaka** | CIT-23-02-0163 | Task 04 (GNN Model Development), Task 05 (Training & Optimization), Bonus D (GNNExplainer), Performance Optimization |
| **Thamindu Kavinda** | CIT-23-02-0356 | Task 06 (Model Evaluation & Metrics), Task 07 (Graph Explainability & Embedding Analysis) |

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Task 01 – Tensor Fundamentals (10 Marks)](#task-01--tensor-fundamentals-10-marks)
3. [Task 02 – Graph Representation and Analysis (10 Marks)](#task-02--graph-representation-and-analysis-10-marks)
4. [Task 03 – Graph Data Preparation (10 Marks)](#task-03--graph-data-preparation-10-marks)
5. [Task 04 – Graph Neural Network Development (25 Marks)](#task-04--graph-neural-network-development-25-marks)
6. [Task 05 – Model Training and Optimization (10 Marks)](#task-05--model-training-and-optimization-10-marks)
7. [Task 06 – Model Evaluation (10 Marks)](#task-06--model-evaluation-10-marks)
8. [Task 07 – Graph Explainability and Embedding Analysis (10 Marks)](#task-07--graph-explainability-and-embedding-analysis-10-marks)
9. [Task 08 – Graph Intelligence Dashboard (5 Marks)](#task-08--graph-intelligence-dashboard-5-marks)
10. [Task 09 – Technical Report & Viva Presentation (10 Marks)](#task-09--technical-report--viva-presentation-10-marks)
11. [Bonus Work (Up to 5 Marks)](#bonus-work-up-to-5-marks)
12. [Repository Structure & Execution Guide](#repository-structure--execution-guide)

---

## Executive Summary

This coursework delivers an end-to-end, reproducible Graph Machine Learning pipeline for semi-supervised node classification on the Open Graph Benchmark (**OGBN-Arxiv**) citation network. 

- **Graph Scale:** 169,343 papers (nodes), 1,166,243 directed citations (2,315,598 symmetrized edges), 128-dimensional skip-gram text embeddings, and 40 fine-grained arXiv Computer Science subject categories.
- **Architectures Evaluated:** 2-layer Graph Convolutional Network (**GCN**) vs. 2-layer Graph Attention Network (**GAT**) / **GraphSAGE**.
- **Inductive Temporal Partitioning:** Train ($\le 2017$: 90,941 nodes), Validation ($2018$: 29,799 nodes), Test ($2019\text{--}2020$: 48,603 nodes).
- **Core Results:** GCN achieved **58.64% test accuracy (0.5389 weighted F1)** outperforming GAT (**57.39% test accuracy**) while training **$3.25\times$ faster** per epoch. Under hyperparameter-tuned full-batch setups, GCN reached **69.62% test accuracy**.
- **Explainability:** 2D PCA & t-SNE latent manifold embeddings, neighborhood homophily agreement ($>85\%$), and first-layer feature importance ranking.
- **Interactive Serving:** Tabbed **Streamlit** dashboard (`dashboard/app.py`) deployed on `http://localhost:8501`.

---

## Task 01 – Tensor Fundamentals (10 Marks)

**Deliverable Notebook:** [`notebooks/01_tensor_fundamentals.ipynb`](notebooks/01_tensor_fundamentals.ipynb)

### 1.1 Tensor Creation
We implemented and tested multiple tensor instantiation methods in PyTorch:
- **0D Scalars & 1D Vectors:** `torch.tensor(10)`, `torch.tensor([1, 2, 3, 4])`
- **2D Matrices:** `torch.tensor([[1, 2, 3], [4, 5, 6]])`
- **Random Initializations:** `torch.rand(2, 3)`
- **NumPy Interoperability:** `torch.from_numpy(np_array)` for zero-copy memory bridging.

```python
import torch
import numpy as np

# Scalar, Vector, Matrix creation
scalar = torch.tensor(10)
vector = torch.tensor([1, 2, 3, 4])
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
rand_mat = torch.rand(2, 3)
np_tensor = torch.from_numpy(np.array([10, 20, 30]))
print("Matrix Shape:", matrix.shape)  # torch.Size([2, 3])
```

### 1.2 Tensor Indexing & Slicing
Demonstrated row extraction, coordinate indexing, and batch chunking:
```python
x = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
node_row = x[0]          # tensor([1., 2., 3.])
feature_val = x[0, 1]    # tensor(2.)
batch_nodes = x[1:3, :]  # tensor([[4., 5., 6.], [7., 8., 9.]])
```

### 1.3 Tensor Reshaping & Flattening
Transformed 1D flat feature arrays into 2D matrices required for linear projections:
```python
flat_feat = torch.arange(12)
reshaped = flat_feat.reshape(3, 4)  # Shape: [3, 4]
flattened = reshaped.flatten()       # Shape: [12]
```

### 1.4 Matrix Multiplication
Vectorized linear transformations ($\mathbf{XW}$) performed in every GNN layer:
```python
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
result = torch.matmul(A, B)  # tensor([[19, 22], [43, 50]])
```

### 1.5 Tensor Broadcasting
Demonstrated dimension auto-expansion for adding layer bias vectors $\mathbf{b} \in \mathbb{R}^{d_{\text{out}}}$ across all $N$ node vectors:
```python
matrix_3x3 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
bias_vector = torch.tensor([10, 20, 30])
broadcasted_sum = matrix_3x3 + bias_vector
# Output: tensor([[11, 22, 33], [14, 25, 36], [17, 28, 39]])
```

### 1.6 Tensor Aggregation Operations
Implemented statistical reductions mimicking GNN neighborhood pooling:
- `torch.sum(x)` $\to$ `45`
- `torch.mean(x.float())` $\to$ `5.0`
- `torch.max(x)` $\to$ `9` | `torch.min(x)` $\to$ `1`
- `torch.sum(x, dim=1)` $\to$ `tensor([6, 15, 24])`

### 1.7 GPU Acceleration
Verified CUDA hardware availability and executed tensor computations on GPU:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t_gpu = torch.rand(3, 3).to(device)
out_gpu = t_gpu @ t_gpu
print("Device:", out_gpu.device)  # cuda:0
```

---

## Task 02 – Graph Representation and Analysis (10 Marks)

**Deliverable Notebook:** [`notebooks/02_graph_representation_analysis.ipynb`](notebooks/02_graph_representation_analysis.ipynb)  
**Artifacts Generated:** [`results/graph_analysis/`](results/graph_analysis)

### 2.1 Graph Representation
Loaded using `ogb.nodeproppred.PygNodePropPredDataset`. Represented as a PyG `Data` object:
```python
Data(num_nodes=169343, edge_index=[2, 1166243], x=[169343, 128], node_year=[169343, 1], y=[169343, 1])
```
- **Coordinate Format (COO):** Stored in `edge_index` of size `[2, 1166243]`.
- **Pandas Edge List:** Converted to a 2-column DataFrame `(source, target)` for network topology queries.

### 2.2 Topological Summary Statistics

| Topological Metric | Empirical Value | Machine Learning / Architectural Implication |
| :--- | :---: | :--- |
| **Node Count ($|\mathcal{V}|$)** | **169,343** | Large-scale academic benchmark requiring sparse tensor representation |
| **Directed Citation Edges** | **1,166,243** | Symmetrized to 2,315,598 directed edges for bidirectional message passing |
| **Feature Dimensionality** | **128** | Word2Vec skip-gram averaged title & abstract embeddings |
| **Number of Classes ($C$)** | **40** | Fine-grained arXiv Computer Science categories |
| **Graph Density ($\rho$)** | **$8.075 \times 10^{-5}$** | Extreme sparsity: $\rho = \frac{|\mathcal{E}|}{|\mathcal{V}|(|\mathcal{V}|-1)}$ (~0.008%) |
| **Mean Node Degree ($\bar{k}$)** | **13.67** | Average citation count per academic publication |
| **Median Node Degree** | **6.0** | Right-skewed distribution characteristic of scale-free networks |
| **Max Node Degree ($k_{\max}$)** | **13,161** | Foundational super-hub paper (Node 1353) |
| **Connected Components** | **1** | Single giant component ensuring graph-wide message reachability |

### 2.3 Degree Distribution & Scale-Free Characteristics
- Generated the **Log-Scaled Degree Histogram** showing exponential decay: $P(k) \sim k^{-\gamma}$.
- Plotted the **Degree-Rank Curve** illustrating heavy-tailed preferential attachment (*"rich get richer"* citation effect).
- Visualized a **2-Hop Local Ego Subgraph** colored by category, proving strong **graph homophily** ($\mathcal{H} > 0.65$).

---

## Task 03 – Graph Data Preparation (10 Marks)

**Deliverable Notebook:** [`notebooks/03_data_preparation.ipynb`](notebooks/03_data_preparation.ipynb)

### 3.1 Preprocessing Workflow
1. **Feature Loading:** Dense float tensor $\mathbf{X} \in \mathbb{R}^{169343 \times 128}$.
2. **Label Loading:** Ground truth category tensor $\mathbf{Y} \in \{0, \dots, 39\}^{169343 \times 1}$.
3. **Temporal Dataset Partitioning:**
   - **Training Set ($\le 2017$):** 90,941 papers ($53.7\%$)
   - **Validation Set ($2018$):** 29,799 papers ($17.6\%$)
   - **Test Set ($2019\text{--}2020$):** 48,603 papers ($28.7\%$)
4. **Leakage-Free Feature Normalization:**
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   # Fit strictly on training nodes to prevent data leakage from validation/test
   scaler.fit(data.x[train_idx])
   data.x = torch.from_numpy(scaler.transform(data.x)).float()
   ```
5. **Graph Symmetrization & Self-Loops:**
   $$\mathbf{\tilde{A}} = \mathbf{A} + \mathbf{A}^\top + \mathbf{I}_N$$
   Adding self-loops and reciprocal edges ensures citing papers receive context from referenced works, and referenced works aggregate context from citing papers.

---

## Task 04 – Graph Neural Network Development (25 Marks)

**Deliverable Notebooks:** [`notebooks/04_gcn_model.ipynb`](notebooks/04_gcn_model.ipynb), [`notebooks/05_gat_model.ipynb`](notebooks/05_gat_model.ipynb)  
**Source Code:** [`src/models/gcn.py`](src/models/gcn.py), [`src/models/gat.py`](src/models/gat.py)

### Model 1 – Graph Convolutional Network (GCN)
- **Spectral Propagation Rule:**
  $$\mathbf{H}^{(l+1)} = \sigma\left(\mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{H}^{(l)} \mathbf{W}^{(l)}\right)$$
- **Architecture Design:**
  - `GCNConv(in_channels=128, out_channels=256)`
  - Non-linear Activation: `ReLU`
  - Regularization: `Dropout(p=0.5)`
  - `GCNConv(in_channels=256, out_channels=40)`
  - Output Layer: `LogSoftmax(dim=-1)`
- **Total Trainable Parameters:** **43,304 / 43,816**

### Model 2 – Graph Attention Network (GAT)
- **Attention Mechanism Formulation:**
  $$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^\top [\mathbf{Wh}_i \,\|\, \mathbf{Wh}_j]\right), \quad \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$$
- **Multi-Head Spatial Aggregation ($K=4$ heads):**
  $$\mathbf{h}_i^{(l+1)} = \mathop{\Big\|}_{k=1}^K \text{ELU}\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k \mathbf{W}^k \mathbf{h}_j^{(l)}\right)$$
- **Architecture Design:**
  - `GATConv(in=128, out=64, heads=4, concat=True)` $\implies 256\text{-dim}$ hidden representation
  - Non-linear Activation: `ELU` + `Dropout(p=0.5)`
  - `GATConv(in=256, out=40, heads=1, concat=False)`
- **Total Trainable Parameters:** **43,624**

### Model 3 (Alternative) – GraphSAGE
- **Mean Neighborhood Aggregation:**
  $$\mathbf{h}_v^{(l+1)} = \sigma\left(\mathbf{W} \cdot \left[\mathbf{h}_v^{(l)} \,\|\, \frac{1}{|\mathcal{N}(v)|}\sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(l)}\right]\right)$$
- **Trainable Parameters:** **86,312**

---

## Task 05 – Model Training and Optimization (10 Marks)

**Deliverable Notebook:** [`notebooks/06_training_optimization.ipynb`](notebooks/06_training_optimization.ipynb)  
**Artifacts Generated:** [`results/training/`](results/training), [`models/`](models)

### 5.1 Objective Function & Optimizer
- **Loss Function:** Negative Log-Likelihood / Multi-Class Cross-Entropy Loss:
  $$\mathcal{L}(\Theta) = -\frac{1}{|\mathcal{V}_{\text{train}}|} \sum_{i \in \mathcal{V}_{\text{train}}} \ln \hat{y}_{i, y_i}$$
- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$) with learning rate $\eta \in [0.005, 0.01]$ and $L_2$ weight decay ($5\times 10^{-4} / 10^{-5}$).

### 5.2 Systematic Hyperparameter Tuning

| Trial | Architecture | Hidden Dim | Dropout | Learning Rate | Best Val Accuracy | Convergence Behavior |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1 (Best)** | **GCN** | **256** | **0.5** | **0.01** | **59.63% (100 ep: 70.51%)** | Optimal capacity, stable plateau |
| 2 | GCN | 256 | 0.3 | 0.01 | 58.42% | Minor overfitting after epoch 25 |
| 3 | GCN | 128 | 0.5 | 0.01 | 56.91% | Underfitting constrained representation |
| 4 | GCN | 128 | 0.3 | 0.01 | 56.15% | Lower capacity baseline |
| **5 (Best)** | **GAT** | **64 (4 heads)** | **0.5** | **0.005** | **58.11% (100 ep: 69.87%)** | Stable attention head optimization |
| 6 | GAT | 64 (4 heads) | 0.3 | 0.01 | 55.80% | Higher variance in attention logits |

- **Training Checkpoints:** Saved highest validation state dictionaries to `models/best_gcn.pt` and `models/best_gat.pt`.

---

## Task 06 – Model Evaluation (10 Marks)

**Deliverable Notebook:** [`notebooks/07_model_evaluation.ipynb`](notebooks/07_model_evaluation.ipynb)  
**Artifacts Generated:** [`results/evaluation/`](results/evaluation)

### 6.1 Test Split Comparative Evaluation (48,603 Held-Out Test Papers)

| Model Architecture | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Macro F1 | Epoch Compute Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GCN (2-layer, 256-dim)** | **58.64%** *(Fine-tuned: 69.62%)* | **0.5492** | **0.5864** | **0.5389** | **0.5169** | **~2.4s (CPU)** |
| **GAT (4-head, 64-dim)** | **57.39%** *(Fine-tuned: 68.95%)* | 0.5414 | 0.5739 | 0.5220 | 0.5112 | ~7.8s (CPU) |
| **Delta ($\Delta_{\text{GCN} - \text{GAT}}$)** | **+1.25%** | **+0.0078** | **+0.0125** | **+0.0169** | **+0.0057** | **$3.25\times$ Faster** |

### 6.2 Key Evaluation Findings & Discussion
1. **Why GCN Outperformed GAT:** In homophilous academic citation graphs, isotropic normalized aggregation provides a stronger smoothing inductive bias than anisotropic attention, which tended to overfit historical pre-2017 citation patterns under temporal drift.
2. **Class Imbalance Impact:** High-support classes (`cs.CV`: 10,477 papers, `cs.LG`: 10,740 papers) achieve $F_1 > 0.75$, whereas rare classes (`cs.OS`: 5 papers, `cs.MS`: 51 papers) score near 0, explaining the gap between overall accuracy (69.6%) and macro F1 (0.517).
3. **Confusion Matrix Analysis:** Heatmap plots show strong diagonal dominance with localized cross-citation confusion between adjacent disciplines (e.g., `cs.LG` and `cs.AI`).

---

## Task 07 – Graph Explainability and Embedding Analysis (10 Marks)

**Deliverable Notebook:** [`notebooks/08_explainability_embeddings.ipynb`](notebooks/08_explainability_embeddings.ipynb)  
**Artifacts Generated:** [`results/explainability/`](results/explainability)

### Option A — Feature Importance Analysis
Extracted the trained first-layer weight matrix $\mathbf{W}^{(0)} \in \mathbb{R}^{128 \times 256}$ and computed the mean absolute importance score for each input dimension:
$$\text{Importance}(d) = \frac{1}{256} \sum_{j=1}^{256} |W_{d, j}^{(0)}|$$
- **Top 10 Influential Features:** Feature indices **49, 115, 3, 38, 52, 110, 48, 70, 85, 10**.

### Option B — Latent Embedding Visualization (PCA & t-SNE)
Extracted 256-dimensional hidden representations $\mathbf{H}^{(1)}$ for 5,000 sampled test nodes:
- **PCA 2D Projection (Linear):** Highlights macroscopic separation between theoretical CS and applied machine learning.
- **t-SNE 2D Manifold (Non-Linear):** Resolves tight, isolated semantic clusters for specialized communities (Computer Vision, Cryptography, Robotics, Information Theory).

### Option D — Neighborhood Influence Analysis
Quantified local explainability via the **Neighborhood Homophily Agreement Score**:
$$\text{Agree}(v) = \frac{1}{|\mathcal{N}_v|} \sum_{u \in \mathcal{N}_v} \mathbb{I}(y_u = y_v)$$
- Evaluated on super-hub node `1353` (degree 13,161) and test node `346` ($\text{Agreement} = 85.71\%$), proving that neighbor category homophily directly justifies correct model predictions.

---

## Task 08 – Graph Intelligence Dashboard (5 Marks)

**Application Source:** [`dashboard/app.py`](dashboard/app.py)  
**Modular Components:** [`dashboard/components/`](dashboard/components)  
**Launcher Script:** [`run_dashboard.py`](run_dashboard.py) *(Active on `http://localhost:8501`)*

### Tabbed Dashboard Architecture (`st.tabs`)
The dashboard is designed as a streamlined, responsive single-page application using Streamlit's native **`st.tabs` container layout** and an **expanded persistent sidebar**:

1. **📌 Sidebar Overview:** Live project execution status, high-level dataset metrics (169,343 nodes, 2,315,598 edges, 128 features, 40 classes), and model architecture specifications.
2. **📊 Tab 1: Graph Analysis (`render_graph_stats`):** Displays real-time KPI metric cards, an interactive summary dataframe, the log-scaled node degree distribution plot, and the 2-hop force-directed sampled subgraph.
3. **📈 Tab 2: Training Dynamics:** Side-by-side interactive Streamlit line charts (`st.line_chart`) tracking training loss decay and validation accuracy convergence across epochs for both GCN and GAT models.
4. **🏆 Tab 3: Model Evaluation (`render_model_metrics`):** Displays test metric comparison tables (Accuracy, Precision, Recall, F1), comparative performance bar charts, and high-resolution 40-class confusion matrices.
5. **🔬 Tab 4: Node Classification Lookup (`render_classification_demo`):** Real-time interactive search engine allowing users to:
   - Search by **Node ID** or **Microsoft Academic Graph (MAG) Paper ID**.
   - Filter papers by ground-truth subject category.
   - Filter by prediction outcome: *Both Correct*, *Model Disagreements*, *GCN Correct Only*, *GAT Correct Only*, or *Both Incorrect*.
   - Inspect individual paper predictions and agreement scores via an interactive expandable inspector (`st.expander`).
6. **🌌 Tab 5: Embeddings & Explainability (`render_embedding_image`):** Dual-subtab view comparing 2D linear PCA projections and 2D non-linear t-SNE manifolds clustered by research category.

---

## Task 09 – Technical Report & Viva Presentation (10 Marks)

### Deliverables
- **Academic Technical Report (PDF):** [`report/CCS4354_Technical_Report.pdf`](report/CCS4354_Technical_Report.pdf) *(3.27 MB, 24-page complete PDF adhering to SLTC format)*.
- **Academic Technical Report (Markdown):** [`report/CCS4354_Technical_Report.md`](report/CCS4354_Technical_Report.md).
- **Presentation Deck Plan:** [`presentation/README.md`](presentation/README.md) (10–15 minute slide presentation script covering all 5 student contributions).

---

## Bonus Work (Up to 5 Marks)

1. **Bonus A — Graph Transformers:** Attention-based long-range message passing across topological distance.
2. **Bonus B — Relational GNNs (RGCN):** Multi-relational citation modelling with relation-specific weight matrices.
3. **Bonus C — Self-Supervised Learning (Deep Graph Infomax - DGI):** Contrastive graph pre-training with GCN fine-tuning achieving **69.92% test accuracy**.
4. **Bonus D — Advanced Optimization:** Focal loss for extreme 40-class imbalance + weighted 5-model ensemble achieving top test macro F1 (**0.5199**).

---

## Repository Structure & Execution Guide

```
Tenso Project/
├── dashboard/
│   ├── app.py                      # Multi-page Streamlit dashboard
│   └── components/                 # Reusable UI cards & charts
├── models/
│   ├── best_gcn.pt                 # Optimized GCN checkpoint
│   └── best_gat.pt                 # Optimized GAT checkpoint
├── notebooks/
│   ├── 01_tensor_fundamentals.ipynb
│   ├── 02_graph_representation_analysis.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_gcn_model.ipynb
│   ├── 05_gat_model.ipynb
│   ├── 06_training_optimization.ipynb
│   ├── 07_model_evaluation.ipynb
│   ├── 08_explainability_embeddings.ipynb
│   └── all_in_one_ogbn_arxiv.ipynb
├── report/
│   ├── CCS4354_Technical_Report.pdf  # 24-Page Formal Academic Report
│   ├── CCS4354_Technical_Report.md   # Markdown Report Source
│   └── assets/                       # High-DPI figures, plots, & mockups
├── results/
│   ├── evaluation/                 # Metrics CSVs, confusion matrices
│   ├── explainability/             # PCA, t-SNE, feature importance
│   ├── graph_analysis/             # Degree distributions, subgraphs
│   └── training/                   # Loss & accuracy history CSVs
├── src/                            # Modular PyTorch pipeline code
├── build_sltc_report_pdf.py        # 24-Page Report PDF Builder
├── generate_report_assets.py       # High-DPI Plot Generator
├── run_dashboard.py                # Streamlit launcher script
└── requirements.txt                # Python dependency definitions
```

### Command Quick-Start
```bash
# 1. Run the interactive Streamlit dashboard
python run_dashboard.py

# 2. Re-generate all high-resolution figures and plots
python generate_report_assets.py

# 3. Re-compile the 24-page formal Academic Technical Report PDF
python build_sltc_report_pdf.py
```
