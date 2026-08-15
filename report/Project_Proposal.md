# Project Proposal: Deep Graph Neural Networks for Large-Scale Academic Paper Classification on the OGBN-Arxiv Citation Benchmark

---

## 📋 Project Metadata

| Field | Details |
| :--- | :--- |
| **Course Module** | **CCS4354 – Tensors and Graphs (with Programming)** |
| **Project Title** | Deep Graph Neural Networks for Academic Citation Networks: A Comparative Study of GCN and GAT on the OGBN-Arxiv Benchmark |
| **Target Dataset** | Open Graph Benchmark (`ogbn-arxiv`) |
| **Primary Architectures** | Graph Convolutional Networks (GCN) & Graph Attention Networks (GAT) |
| **Domain Area** | Graph Representation Learning, Tensor Algebra, Non-Euclidean Deep Learning, NLP |
| **Author(s)** | [Insert Student Name(s) / Student ID(s)] |
| **Submission Date** | August 2026 |
| **Status** | Approved Project Proposal |

---

## 1. Executive Summary

With the exponential growth of scientific literature published annually, automated academic paper classification is essential for semantic indexing, recommendation systems, literature discovery, and peer-review assignment. However, scientific papers do not exist in isolation; they are embedded within dense, non-Euclidean citation graphs where citing relationships encode vital contextual and categorical dependencies. Traditional Machine Learning and flat Natural Language Processing (NLP) models treat documents as independent and identically distributed ($i.i.d.$), entirely discarding structural graph topology.

This project proposes an end-to-end investigation into **Graph Representation Learning** for large-scale node property prediction on the **OGBN-Arxiv** benchmark dataset, comprising **169,343 research papers (nodes)**, **2,315,598 directed citation links (edges)**, **128-dimensional skip-gram word embeddings**, and **40 fine-grained subject categories**. We propose to design, implement, optimize, evaluate, and interpret two prominent Graph Neural Network (GNN) paradigms: **Spectral Graph Convolutional Networks (GCN)** and **Spatial Multi-Head Graph Attention Networks (GAT)** under a realistic **inductive temporal split** ($\le 2017$ train, $2018$ validation, $2019\text{--}2020$ test). The proposed pipeline encompasses tensor algebraic optimizations, structural network analysis, controlled hyperparameter search, latent embedding explainability (PCA/t-SNE), an interactive **Streamlit dashboard**, and a publication-grade technical report.

---

## 2. Problem Statement & Motivation

### 2.1 Limitations of Traditional Text Classification
Standard document classification pipelines extract textual features (e.g., TF-IDF, Word2Vec, BERT) and pass them through standard classifiers (e.g., Logistic Regression, Multi-Layer Perceptrons). While effective for isolated texts, this paradigm exhibits three severe shortcomings in academic networks:
1. **Neglect of Relational Context:** A paper with ambiguous abstract vocabulary often draws heavy citations from a specific foundational domain (e.g., Computer Vision), which immediately resolves the classification ambiguity.
2. **Failure under Out-of-Distribution Shift:** As new terminology emerges over time, text-only classifiers suffer severe degradation without topological grounding.
3. **Information Silos:** Text embeddings capture intra-document semantics but fail to propagate inter-document relational homophily across multi-hop neighborhoods.

### 2.2 Why Graph Neural Networks & Tensor Algebra?
Graph Neural Networks generalize deep learning operations (convolutions, attention, pooling) to irregular non-Euclidean domains via **message passing**. By formulating the graph as sparse coordinate tensors (`edge_index`) and dense feature matrices ($\mathbf{X} \in \mathbb{R}^{N \times 128}$), GNNs iteratively update node representations by aggregating transformed feature vectors from neighboring nodes:
$$\mathbf{h}_v^{(l+1)} = \text{UPDATE}\left(\mathbf{h}_v^{(l)}, \text{AGGREGATE}\left(\{\mathbf{h}_u^{(l)} : u \in \mathcal{N}(v)\}\right)\right)$$

### 2.3 Research Questions
This project aims to answer the following core research questions:
1. **RQ1:** How does isotropic symmetric spectral normalization (GCN) compare to anisotropic dynamic self-attention (GAT) in terms of classification accuracy, weighted F1-score, and computational efficiency on dense citation networks?
2. **RQ2:** How robust are GNN architectures when subjected to strict temporal distribution drift (predicting 2019–2020 papers from models trained on historical $\le 2017$ literature)?
3. **RQ3:** Can low-dimensional projections (PCA and t-SNE) of learned GNN hidden embeddings faithfully uncover the latent semantic modularity of 40 computer science disciplines?
4. **RQ4:** What is the empirical relationship between local citation homophily and GNN prediction reliability?

---

## 3. Project Objectives

### 3.1 Primary Technical Objectives
1. **Data Pipeline & Graph Engineering:** Ingest and preprocess the 169K-node OGBN-Arxiv network, construct sparse tensor representations, apply graph symmetrization with self-loops, and preserve official chronological temporal splits.
2. **Architecture Implementation:** Develop modular, vectorized PyTorch Geometric implementations of:
   - A 2-Layer **Graph Convolutional Network (GCN)** with 256 hidden channels, ReLU activations, and dropout regularization.
   - A 2-Layer **Graph Attention Network (GAT)** with 4 attention heads (64 channels per head), ELU activations, and attention dropout.
3. **Training & Hyperparameter Optimization:** Implement an automated training engine using Adam optimization, cross-entropy loss with log-softmax, learning rate schedules, and grid search tuning.
4. **Benchmarking & Evaluation:** Measure Test Top-1 Accuracy, Weighted Precision, Weighted Recall, Weighted F1, Macro F1, and generate 40-class confusion matrices.
5. **Latent Representation Explainability:** Extract 256-dimensional hidden layer representations and compute 2D linear PCA and non-linear t-SNE projections, accompanied by quantitative neighborhood label agreement metrics.
6. **Interactive Web Deployment:** Build and deploy a multi-tab Streamlit dashboard enabling real-time graph exploration, dual-model training comparison, and per-paper prediction lookup.

---

## 4. Proposed Technical Architecture & Methodology

```mermaid
flowchart TD
    subgraph Data Layer
        A1[OGBN-Arxiv Dataset\n169,343 Nodes | 2,315,598 Edges] --> A2[Data Preparation\nSparse COO Tensors & Symmetrization]
        A2 --> A3[Temporal Split\nTrain ≤2017 | Valid 2018 | Test 2019-2020]
    end

    subgraph Modeling & Computation Layer
        A3 --> B1[GCN Model\n2-Layer Spectral GCN\n256 Hidden Units]
        A3 --> B2[GAT Model\n2-Layer Spatial GAT\n4 Heads × 64 Hidden Units]
        B1 --> C1[Adam Optimization & Checkpointing\nmodels/best_gcn.pt]
        B2 --> C2[Adam Optimization & Checkpointing\nmodels/best_gat.pt]
    end

    subgraph Evaluation & Explainability Layer
        C1 --> D1[Metrics Evaluation\nAccuracy, Precision, Recall, F1]
        C2 --> D1
        C1 --> D2[Latent Embeddings\nPCA & t-SNE Manifolds]
        C1 --> D3[Homophily Scoring\nNeighborhood Agreement]
    end

    subgraph Serving & Presentation Layer
        D1 --> E1[Streamlit Web Dashboard\n5 Interactive Modules]
        D2 --> E1
        D1 --> E2[Publication-Grade Technical Report\nPDF & Markdown]
        D2 --> E2
    end
```

---

## 5. Mathematical Formulations of Proposed Models

### 5.1 Graph Convolutional Network (GCN)
The spectral graph convolution operates via localized first-order Chebyshev approximations:
$$\mathbf{H}^{(l+1)} = \sigma\left( \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right)$$
where:
- $\mathbf{\tilde{A}} = \mathbf{A} + \mathbf{I}_N$ is the adjacency matrix augmented with identity self-loops.
- $\mathbf{\tilde{D}}_{ii} = \sum_j \mathbf{\tilde{A}}_{ij}$ is the diagonal node degree matrix.
- $\mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-\frac{1}{2}}$ is the symmetric normalized adjacency matrix that bounds spectral eigenvalues and prevents exploding/vanishing gradients.
- $\mathbf{W}^{(l)}$ is the layer-specific trainable parameter tensor.
- $\sigma(\cdot)$ is the rectified linear unit ($\text{ReLU}$).

### 5.2 Graph Attention Network (GAT)
The spatial graph attention mechanism replaces uniform isotropic weights with parameterized self-attention:
$$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^\top \left[ \mathbf{W}\mathbf{h}_i \,\|\, \mathbf{W}\mathbf{h}_j \right]\right)$$
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$$
To stabilize attention learning across multi-modal citation habits, multi-head attention with $K=4$ independent heads is applied:
$$\mathbf{h}_i^{(l+1)} = \mathop{\Big\|}_{k=1}^K \text{ELU}\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k \mathbf{W}^k \mathbf{h}_j^{(l)}\right)$$

---

## 6. Dataset Characteristics & System Requirements

### 6.1 Benchmark Dataset Profile (`ogbn-arxiv`)

| Feature | Value / Specification |
| :--- | :--- |
| **Total Nodes ($|\mathcal{V}|$)** | **169,343** academic papers |
| **Total Edges ($|\mathcal{E}|$)** | **2,315,598** directed citation links |
| **Node Feature Dimension** | **128** (averaged word2vec skip-gram embeddings) |
| **Target Classes ($C$)** | **40** primary subject categories |
| **Graph Density ($\rho$)** | **$1.615 \times 10^{-4}$** (sparse network) |
| **Degree Statistics** | Min: $1.0$, Max: $13,161.0$, Mean: $13.67$, Median: $6.0$ |
| **Train Set ($\le 2017$)** | **90,941 nodes** ($53.70\%$) |
| **Validation Set ($2018$)** | **29,799 nodes** ($17.60\%$) |
| **Test Set ($2019\text{--}2020$)** | **48,603 nodes** ($28.70\%$) |

### 6.2 Hardware & Software Specifications
- **Programming Language:** Python 3.10+ / 3.14
- **Deep Learning Framework:** PyTorch 2.2+ (CUDA / CPU support)
- **Graph Neural Network Library:** PyTorch Geometric (PyG 2.5+)
- **Benchmark API:** Open Graph Benchmark (OGB 1.3.6)
- **Data Science & ML:** Scikit-Learn 1.4+, Pandas 2.1+, NumPy 1.26+, NetworkX 3.2+
- **Visualization & UI:** Matplotlib 3.8+, Seaborn 0.13+, Plotly 5.22+, Streamlit 1.35+
- **Document Generation:** ReportLab 4.0+

---

## 7. Work Breakdown Structure & Milestones

The project will be executed sequentially across 8 modular phases:

```text
Phase 1: Tensor Algebra & GPU Profiling (Notebook 01)
 ├── Tensor indexing, slicing, reshaping, broadcasting, and matrix multiplication
 └── Device memory management and autograd profiling

Phase 2: Graph Data Engineering & Topological Analysis (Notebook 02 & 03)
 ├── Ingestion of OGBN-Arxiv raw dataset (~700 MB)
 ├── Computation of degree distributions, power-law, density, and local ego-subgraphs
 └── Extraction and persistence of official chronological temporal splits

Phase 3: Deep GNN Modeling (Notebook 04 & 05)
 ├── Implementation of 2-layer GCN (256 hidden channels)
 ├── Implementation of 2-layer Multi-Head GAT (4 heads × 64 channels)
 └── Model checkpointing and loss/accuracy convergence logging

Phase 4: Hyperparameter Tuning & Optimization (Notebook 06)
 ├── Grid search over hidden dimensions (128, 256), dropout (0.3, 0.5), learning rates
 └── Empirical validation across candidate configurations

Phase 5: Evaluation & Error Analysis (Notebook 07)
 ├── Computation of test accuracy, weighted precision, recall, and F1 scores
 ├── Generation of 40-class confusion matrices
 └── Inter-model prediction agreement vs. disagreement analysis

Phase 6: Latent Representation Explainability (Notebook 08)
 ├── Extraction of 256-dim hidden representations
 ├── 2D PCA linear projection and 2D t-SNE non-linear manifold clustering
 └── Quantitative evaluation of neighborhood homophily agreement

Phase 7: Interactive Dashboard Engineering (dashboard/app.py)
 ├── 5-tab modular Streamlit web application
 └── Real-time paper search engine, KPI metric cards, and interactive visualization

Phase 8: Documentation & PDF Technical Report Compilation (report/)
 ├── Comprehensive 12-section technical Markdown documentation
 └── Compilation of publication-grade 4-page PDF technical report via ReportLab
```

---

## 8. Expected Deliverables & Impact

1. **Fully Modular Python Codebase:** Reusable, clean package in `src/` adhering to strict object-oriented design and typing.
2. **8 Executable Jupyter Notebooks:** Fully documented notebooks reproducing all tensor operations, graph analysis, training runs, evaluations, and visualizations.
3. **Saved Model Checkpoints:** Validated PyTorch state dictionaries in `models/best_gcn.pt` and `models/best_gat.pt`.
4. **Interactive Streamlit Web Dashboard:** Live analytics and lookup tool accessible on `localhost:8501`.
5. **Formal Technical Report:** 
   - Markdown report: `report/CCS4354_Technical_Report.md`
   - Publication-grade PDF report: `report/CCS4354_Technical_Report.pdf`
6. **Project Proposal Document:** This formal specification document in `report/Project_Proposal.md` (and compiled `report/Project_Proposal.pdf`).
7. **Comprehensive GitHub Repository:** Fully configured with badges, architecture diagrams, and quickstart documentation in `README.md`.

---

## 9. Risk Assessment & Mitigation Strategies

| Potential Risk | Severity | Mitigation Strategy |
| :--- | :---: | :--- |
| **High GPU/RAM Memory Consumption** | Medium | Utilize sparse COO tensor indexing (`edge_index`) instead of dense adjacency matrices; profile memory usage. |
| **Oversmoothing in Deep GNNs** | High | Constrain architecture to 2 message-passing layers; incorporate $50\%$ dropout and non-linear activations (ReLU/ELU). |
| **Temporal Data Leakage** | Critical | Strictly enforce OGB's official chronological split ($\le 2017$ train, $2018$ valid, $2019\text{--}2020$ test); avoid uniform random splitting. |
| **Class Imbalance Across 40 Classes** | Medium | Evaluate using **Weighted Precision, Weighted Recall, and Weighted F1-scores** in addition to Top-1 Accuracy. |
| **GAT Attention Instability** | Low | Apply multi-head attention ($K=4$) with normalized softmax and ELU non-linearities. |

---

## 10. References

1. **Kipf, T. N., & Welling, M. (2017).** Semi-Supervised Classification with Graph Convolutional Networks. *International Conference on Learning Representations (ICLR)*.
2. **Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018).** Graph Attention Networks. *International Conference on Learning Representations (ICLR)*.
3. **Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., & Leskovec, J. (2020).** Open Graph Benchmark: Datasets for Machine Learning on Graphs. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 22118-22133.
4. **Hamilton, W. L., Ying, R., & Leskovec, J. (2017).** Inductive Representation Learning on Large Graphs. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
5. **Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019).** How Powerful are Graph Neural Networks? *International Conference on Learning Representations (ICLR)*.
6. **Fey, M., & Lenssen, J. E. (2019).** Fast Graph Representation Learning with PyTorch Geometric. *ICLR Workshop on Representation Learning on Graphs and Manifolds*.

---
