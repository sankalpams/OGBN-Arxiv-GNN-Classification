# 🕸️ OGBN-Arxiv Graph Intelligence: GCN vs. GAT Node Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.5%2B-3C2179.svg?logo=pyg&logoColor=white)](https://pyg.org/)
[![OGB](https://img.shields.io/badge/OGB-1.3.6-green.svg)](https://ogb.stanford.edu/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Deep Graph Representation Learning project performing large-scale **node property prediction (subject classification)** on the **OGBN-Arxiv citation graph** (169,343 nodes, 2,315,598 edges, 40 categories). Features PyTorch Geometric implementations of **Graph Convolutional Networks (GCN)** and **Graph Attention Networks (GAT)**, inductive temporal evaluation, latent embedding explainability (PCA & t-SNE), a full **Streamlit interactive dashboard**, and a publication-grade **Technical Report**.
---

## 📌 Project Highlights

- **Scale & Benchmark:** 169,343 academic papers and 2.3M directed citation links from Microsoft Academic Graph (MAG) across 40 arXiv Computer Science categories.
- **Architectures:** Custom PyTorch Geometric implementations of Spectral Graph Convolutional Networks (GCN) and Spatial Multi-Head Graph Attention Networks (GAT).
- **Realistic Temporal Split:** Evaluated on strictly out-of-distribution temporal splits ($\le 2017$ train, $2018$ validation, $2019\text{--}2020$ test) to prevent chronological data leakage.
- **Empirical Results:** GCN achieved **58.64% test accuracy (53.89% weighted F1)**, outperforming GAT (**57.39% accuracy, 52.20% F1**) while training **$3.25\times$ faster**.
- **Representation Explainability:** 2D linear PCA and non-linear t-SNE manifold projections demonstrating semantic topic clustering and local neighborhood homophily ($85.71\%$ agreement).
- **Interactive Web App:** Multi-tab Streamlit dashboard for real-time topological analytics, dynamic prediction lookup, error inspection, and embedding exploration.
- **Full Technical Documentation:** Complete 4-page academic PDF report and comprehensive Markdown documentation.

---

## 🏛️ System Architecture

```
flowchart TD
    A[Raw OGBN-Arxiv Dataset\n169,343 Nodes | 2,315,598 Edges] --> B[Data Preparation & Preprocessing\nGraph Symmetrization Ã = A + Aᵀ + I]
    B --> C[Temporal Partitioning\nTrain: ≤2017 | Valid: 2018 | Test: 2019-2020]
    
    C --> D1[GCN Model\n2-Layer Spectral GCN\n256 Hidden Channels | ReLU | Dropout]
    C --> D2[GAT Model\n2-Layer Multi-Head GAT\n4 Heads × 64 Channels | ELU | Dropout]
    
    D1 --> E1[Training & Optimization\nAdam Optimizer | Cross-Entropy Loss\nCheckpoints: models/best_gcn.pt]
    D2 --> E2[Training & Optimization\nAdam Optimizer | Cross-Entropy Loss\nCheckpoints: models/best_gat.pt]
    
    E1 --> F[Model Evaluation & Benchmarking\nAccuracy | Weighted Precision | Recall | F1\n40-Class Confusion Matrices]
    E2 --> F
    
    E1 --> G[Latent Embeddings & Explainability\n256-dim Hidden Representations\n2D PCA & t-SNE Manifolds | Homophily Score]
    
    F --> H1[Streamlit Web Dashboard\nInteractive Analytics & Prediction Lookup\nlocalhost:8501]
    G --> H1
    F --> H2[Publication-Grade Technical Report\nPDF & Markdown Output]
    G --> H2
```

---

## 📊 Empirical Benchmark Results

Evaluated on the **48,603 held-out test papers (2019–2020)**:

| Model Architecture | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Model Parameters | Epoch Time (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Graph Convolutional Network (GCN)** | **58.64%** | **0.5492** | **0.5864** | **0.5389** | 43,816 | **$\approx 2.4\text{ s}$** |
| **Graph Attention Network (GAT)** | **57.39%** | **0.5414** | **0.5739** | **0.5220** | 43,624 | $\approx 7.8\text{ s}$ |
| **Performance Delta ($\Delta_{\text{GCN}-\text{GAT}}$)** | **+1.25%** | **+0.0078** | **+0.0125** | **+0.0169** | $+192$ | **$3.25\times\text{ faster}$** |

### Key Findings
1. **Isotropic Normalization Superiority:** In dense citation graphs with strong domain homophily ($\mathcal{H} > 0.65$), symmetric normalized aggregation ($\mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-\frac{1}{2}}$) provides an optimal smoothing prior.
2. **Temporal Generalization:** GAT's additional attention parameters are more susceptible to overfitting historical citation habits when generalizing to future publications.
3. **Efficiency:** GCN provides superior predictive accuracy while utilizing $3.25\times$ less compute per training epoch.

---

## 🚀 Quick Start Guide

### 1. Clone & Set Up Environment

```bash
# Clone the repository
git clone https://github.com/sankalpams/OGBN-Arxiv-GNN-Classification.git
cd OGBN-Arxiv-GNN-Classification

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute Jupyter Notebook Pipeline

Execute the sequentially numbered notebooks in `notebooks/` to reproduce all data preprocessing, model training, evaluation, and explainability plots:

```bash
python -m jupyter notebook
```

| # | Notebook | Focus Area | Outputs & Generated Artifacts |
| :---: | :--- | :--- | :--- |
| **01** | [`01_tensor_fundamentals.ipynb`](notebooks/01_tensor_fundamentals.ipynb) | Tensor algebra & GPU operations | Tensor indexing, broadcasting, matrix multiplications |
| **02** | [`02_graph_representation_analysis.ipynb`](notebooks/02_graph_representation_analysis.ipynb) | Network topology & degree analysis | `results/graph_analysis/degree_distribution.png`, `sample_subgraph.png` |
| **03** | [`03_data_preparation.ipynb`](notebooks/03_data_preparation.ipynb) | Data pipeline & temporal splitting | Processed graph tensors in `data/processed/` |
| **04** | [`04_gcn_model.ipynb`](notebooks/04_gcn_model.ipynb) | GCN model training & convergence | `models/best_gcn.pt`, `results/training/gcn_training_history.csv` |
| **05** | [`05_gat_model.ipynb`](notebooks/05_gat_model.ipynb) | GAT model training & attention | `models/best_gat.pt`, `results/training/gat_training_history.csv` |
| **06** | [`06_training_optimization.ipynb`](notebooks/06_training_optimization.ipynb) | Hyperparameter tuning grid | `results/training/hyperparameter_trials.csv` |
| **07** | [`07_model_evaluation.ipynb`](notebooks/07_model_evaluation.ipynb) | Comparative evaluation & confusion matrices | `results/evaluation/metrics.csv`, `confusion_matrix_gcn.png`, `confusion_matrix_gat.png` |
| **08** | [`08_explainability_embeddings.ipynb`](notebooks/08_explainability_embeddings.ipynb) | PCA, t-SNE & local homophily | `results/explainability/pca_embeddings.png`, `tsne_embeddings.png` |

---

### 3. Launch the Interactive Dashboard

Launch the Streamlit analytics and model inspection dashboard:

```bash
# Via convenience script
python run_dashboard.py

# Or directly via streamlit
python -m streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser to access:
- **📊 Graph Analysis:** Live structural KPIs, degree distribution, and interactive subgraphs.
- **📈 Training Dynamics:** Interactive GCN vs. GAT convergence and loss trajectories.
- **🏆 Model Evaluation:** Comparative scorecards and high-resolution 40-class confusion matrices.
- **🔬 Node Classification Lookup:** Real-time paper prediction search engine by Node ID or topic with model agreement filters.
- **🌌 Embeddings & Explainability:** Interactive 2D PCA linear projection and t-SNE manifold visualizations.

---

### 4. Build the Full Technical Report PDF

Generate the publication-grade PDF technical report with embedded figures and tables:

```bash
python tmp_build_report.py
# or
python build_full_report_pdf.py
```
Output compiled at: [`report/CCS4354_Technical_Report.pdf`](report/CCS4354_Technical_Report.pdf).

---

## 📂 Repository Structure

```text
├── dashboard/                     # Interactive Streamlit Web Application
│   ├── app.py                     # Main dashboard entrypoint & tab navigation
│   ├── requirements.txt           # Dashboard-specific dependencies
│   └── components/                # Modular UI visualization components
│       ├── classification.py      # Real-time node search & agreement inspector
│       ├── embeddings.py          # 2D PCA and t-SNE projection viewer
│       ├── graph_stats.py         # Topological metric cards & degree plots
│       └── model_metrics.py       # Metrics comparison table & confusion matrices
├── data/                          # Dataset directory (raw downloads & processed tensors)
│   ├── raw/                       # Automatic download directory for OGBN-Arxiv (~700 MB)
│   └── processed/                 # Processed PyTorch tensors (.pt)
├── models/                        # Saved model state checkpoints
│   ├── best_gcn.pt                # Checkpoint weights for trained GCN
│   └── best_gat.pt                # Checkpoint weights for trained GAT
├── notebooks/                     # End-to-end sequential Jupyter notebooks
│   ├── 01_tensor_fundamentals.ipynb
│   ├── 02_graph_representation_analysis.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_gcn_model.ipynb
│   ├── 05_gat_model.ipynb
│   ├── 06_training_optimization.ipynb
│   ├── 07_model_evaluation.ipynb
│   └── 08_explainability_embeddings.ipynb
├── report/                        # Technical report documentation
│   ├── CCS4354_Technical_Report.md # Full 12-section Markdown report
│   ├── CCS4354_Technical_Report.pdf # 4-page publication-grade compiled PDF
│   └── README.md
├── results/                       # Generated experimental figures & CSV metrics
│   ├── evaluation/                # Metrics CSV, comparison bar plots, confusion matrices
│   ├── explainability/            # PCA and t-SNE latent embedding plots
│   ├── graph_analysis/            # Summary stats, degree distributions, subgraph figures
│   └── training/                  # GCN/GAT training histories & hyperparameter trials
├── src/                           # Core reusable Python source library
│   ├── config.py                  # Project paths & directory configuration
│   ├── data/                      # Data loaders, split utilities, preprocessing
│   ├── evaluation/                # Classification metrics, evaluators, plotters
│   ├── explainability/            # Embedding extractors, PCA/t-SNE reducers
│   ├── graph/                     # Degree, density, component, and subgraph analysis
│   ├── models/                    # GCN and GAT PyTorch Geometric architectures
│   └── training/                  # Training loops, loss functions, hyperparameter grid
├── build_full_report_pdf.py       # Script to compile publication-grade PDF report
├── run_dashboard.py               # Convenience script to launch Streamlit dashboard
├── tmp_build_report.py            # Report build trigger script
├── requirements.txt               # Global Python package requirements
└── README.md                      # Project documentation
```

---

## 🔬 Mathematical Formulations

### Graph Convolutional Network (GCN)
$$\mathbf{H}^{(l+1)} = \sigma\left( \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right)$$
where $\mathbf{\tilde{A}} = \mathbf{A} + \mathbf{I}_N$ and $\mathbf{\tilde{D}}_{ii} = \sum_j \mathbf{\tilde{A}}_{ij}$.

### Graph Attention Network (GAT)
$$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_i \,\|\, \mathbf{W}\mathbf{h}_j]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_i \,\|\, \mathbf{W}\mathbf{h}_k]\right)\right)}$$

$$\mathbf{h}_i^{(l+1)} = \mathop{\Big\|}_{k=1}^K \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k \mathbf{W}^k \mathbf{h}_j^{(l)}\right)$$

---

## 📈 Latent Representations & Explainability

- **PCA 2D Projection:** Demonstrates macroscopic separation between broader academic disciplines (e.g., Theoretical Computer Science vs. Applied Deep Learning).
- **t-SNE 2D Manifold:** Resolves high-density non-linear semantic clusters corresponding to specialized communities (`cs.CV`, `cs.LG`, `cs.AI`, `cs.CR`).
- **Neighborhood Homophily Agreement:** Quantifies local prediction reliability by measuring label consistency across 1-hop neighbors ($\text{Agree}(v) = \frac{1}{|\mathcal{N}_v|} \sum_{u \in \mathcal{N}_v} \mathbb{I}(y_u = y_v)$).

---

## 💻 Tech Stack & Dependencies

- **Core:** Python 3.10+, PyTorch 2.2+, PyTorch Geometric 2.5+
- **Graph & ML:** Open Graph Benchmark (OGB 1.3.6), Scikit-Learn 1.4+, NetworkX 3.2+
- **Data & Visualization:** Pandas 2.1+, NumPy 1.26+, Matplotlib 3.8+, Seaborn 0.13+, Plotly 5.22+
- **Interactive UI:** Streamlit 1.35+
- **Report Generation:** ReportLab 4.0+

---

## 📚 References

1. **Kipf, T. N., & Welling, M. (2017).** *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.
2. **Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018).** *Graph Attention Networks.* ICLR.
3. **Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., & Leskovec, J. (2020).** *Open Graph Benchmark: Datasets for Machine Learning on Graphs.* NeurIPS, 33, 22118-22133.
4. **Hamilton, W. L., Ying, R., & Leskovec, J. (2017).** *Inductive Representation Learning on Large Graphs.* NeurIPS.
5. **Fey, M., & Lenssen, J. E. (2019).** *Fast Graph Representation Learning with PyTorch Geometric.* ICLR Workshop.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
