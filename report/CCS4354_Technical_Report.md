# CCS4354 - Tensors and Graphs (with Programming)
## Comprehensive Technical Report: Node Classification on the OGBN-Arxiv Citation Benchmark Using Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT)

---

**Course Module:** CCS4354 – Tensors and Graphs (with Programming)  
**Project Title:** Deep Graph Neural Networks for Large-Scale Academic Citation Networks  
**Benchmark Dataset:** Open Graph Benchmark (`ogbn-arxiv`)  
**Models Evaluated:** Graph Convolutional Networks (GCN) vs. Graph Attention Networks (GAT)  
**Date:** August 2026  
**Status:** Completed & Empirically Validated  

---

### Executive Summary

Graph Representation Learning operates at the intersection of tensor algebra, discrete graph theory, and deep non-Euclidean learning. This project presents an end-to-end empirical and theoretical investigation into node classification on the large-scale **OGBN-Arxiv** academic citation network, comprising **169,343 research papers (nodes)**, **2,315,598 directed citation links (edges)**, **128-dimensional skip-gram word feature embeddings**, and **40 subject category classes**. 

We implement, optimize, evaluate, and interpret two foundational graph neural network architectures:
1. **Graph Convolutional Networks (GCN)** utilizing spectral-based symmetric normalized graph convolutions.
2. **Graph Attention Networks (GAT)** employing spatial multi-head self-attention mechanisms over localized graph neighborhoods.

Models were trained under the official realistic **temporal split** (papers published $\le 2017$ for training, $2018$ for validation, and $2019\text{--}2020$ for testing) to prevent temporal data leakage. On the held-out test split of 48,603 papers, **GCN achieved 58.64% accuracy (53.89% weighted F1)**, outperforming the baseline **GAT (57.39% accuracy, 52.20% weighted F1)** under identical compute and regularized parameter budgets. 

Furthermore, we explore representation learning explainability via **Principal Component Analysis (PCA)** and **t-Distributed Stochastic Neighbor Embedding (t-SNE)** 2D manifold projections of hidden layer representations, revealing strong semantic clustering of primary research topics (e.g., Computer Vision, Machine Learning, Artificial Intelligence). Finally, an interactive web dashboard developed in Streamlit was deployed to facilitate interactive node-level prediction inspection, topological querying, and comparative error analysis.

---

### 1. Introduction and Theoretical Problem Formulation

#### 1.1 Node Property Prediction on Graph-Structured Data
Traditional deep learning paradigms assume independent and identically distributed ($i.i.d.$) data residing in Euclidean spaces (e.g., grids for computer vision, sequences for NLP). However, academic citation networks exhibit complex non-Euclidean relational dependencies, homophily, and structural topology.

In the **node property prediction** problem, a graph is formally defined as:
$$\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{X}, \mathbf{Y})$$
where:
- $\mathcal{V} = \{v_1, v_2, \dots, v_N\}$ represents the set of $N = |\mathcal{V}|$ nodes (academic papers).
- $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$ is the set of directed edges $(u, v)$ indicating that paper $u$ cites paper $v$.
- $\mathbf{X} \in \mathbb{R}^{N \times d}$ is the node feature matrix, where each row $\mathbf{x}_v \in \mathbb{R}^{128}$ contains word embeddings extracted from title and abstract texts.
- $\mathbf{Y} \in \{0, 1, \dots, C-1\}^N$ is the label vector mapping each node to one of $C = 40$ arXiv Computer Science primary subject categories.

#### 1.2 Transductive vs. Inductive Graph Learning Under Temporal Drift
Unlike simple random partitioning, the OGBN-Arxiv benchmark imposes an **inductive temporal split**:
- **Train Set ($\mathcal{V}_{\text{train}}$):** Papers published up to 2017 ($N_{\text{train}} = 90,941$, $53.7\%$).
- **Validation Set ($\mathcal{V}_{\text{valid}}$):** Papers published in 2018 ($N_{\text{valid}} = 29,799$, $17.6\%$).
- **Test Set ($\mathcal{V}_{\text{test}}$):** Papers published in 2019–2020 ($N_{\text{test}} = 48,603$, $28.7\%$).

This formulation accurately mirrors real-world production deployment: a model trained on historical literature must predict future research topic trends subject to vocabulary shift, evolving citation dynamics, and emerging computer science sub-disciplines.

---

### 2. Tensor Fundamentals & Computational Foundations

#### 2.1 Multidimensional Tensors in Graph Neural Networks
At the core of modern deep learning frameworks (PyTorch and PyTorch Geometric), graph neural computations are vectorized across multidimensional tensor operations:
1. **Adjacency Representation (`edge_index`):** Stored as a coordinate format (COO) tensor of shape $(2, |\mathcal{E}|)$ containing source and target node indices of dtype `torch.long`.
2. **Node Feature Tensor ($\mathbf{X}$):** A 2D dense float tensor of shape $(169343, 128)$.
3. **Weight Matrices ($\mathbf{W}^{(l)}$):** Transformation tensors mapping hidden dimensions between successive graph convolutional layers.
4. **Attention Coefficient Tensors ($\mathbf{\alpha}$):** Sparse multidimensional tensors weighting message passing along localized citation edges.

#### 2.2 Tensor Algebraic Operations & Broadcasting
Throughout model construction, essential tensor operations were implemented and profiled:
- **Matrix Multiplication (`torch.matmul` / `@`):** Linear feature projection $\mathbf{X}\mathbf{W} \in \mathbb{R}^{N \times d_{\text{out}}}$.
- **Tensor Broadcasting:** Efficiently applying bias vectors $\mathbf{b} \in \mathbb{R}^{d_{\text{out}}}$ across all $N$ node vectors without memory duplication.
- **Sparse-Dense Reductions (`scatter_add` / `segment_csr`):** Neighborhood aggregation accumulating incoming messages at target nodes in $\mathcal{O}(|\mathcal{E}|)$ time complexity.
- **Automatic Differentiation (`autograd`):** Backpropagating gradients through non-Euclidean computational graphs.

#### 2.3 Hardware & Runtime Acceleration
Experiments were developed with native CUDA-capable PyTorch acceleration with automatic CPU fallback. Execution profiles demonstrate the memory efficiency of sparse tensor message passing over dense $N \times N$ adjacency matrices (which would require $>114\text{ GB}$ RAM in dense float representation).

---

### 3. Dataset Characteristics & Graph Topology Analysis

#### 3.1 Graph Metric Summary
Exploratory topological analysis conducted on the full graph yielded the empirical characteristics summarized in Table 1.

| Metric | Empirical Value | Mathematical Meaning / Significance |
| :--- | :--- | :--- |
| **Node Count ($|\mathcal{V}|$)** | **169,343** | Total academic papers in the computer science corpus |
| **Edge Count ($|\mathcal{E}|$)** | **2,315,598** | Total directed citation relationships |
| **Feature Dimensionality ($d$)** | **128** | Word2Vec / Skip-gram average title+abstract embeddings |
| **Number of Classes ($C$)** | **40** | Fine-grained arXiv CS subject categories |
| **Graph Density ($\rho$)** | **$1.615 \times 10^{-4}$** | Extreme sparsity: $\rho = \frac{|\mathcal{E}|}{|\mathcal{V}|(|\mathcal{V}|-1)}$ |
| **Minimum Degree** | **1.0** | All papers in the connected network cite or are cited |
| **Maximum Degree** | **13,161.0** | Highly cited foundational hub paper |
| **Mean Degree ($\bar{k}$)** | **13.67** | Average citations per academic paper |
| **Median Degree** | **6.0** | Right-skewed distribution characteristic of scale-free networks |

#### 3.2 Degree Distribution & Power-Law Characteristics
Citation networks naturally exhibit **scale-free power-law degree distributions** governed by the Barabási–Albert preferential attachment principle (*"rich get richer"*):
$$P(k) \sim k^{-\gamma}$$
As demonstrated by the empirical log-scaled degree histogram:
- The vast majority of nodes have fewer than 10 citations (median = 6).
- A tiny fraction of seminal papers act as super-hubs with in-degrees exceeding $10^3$ to $13,161$ citations.
- Log-scale distribution analysis reveals the heavy-tailed behavior typical of empirical social and citation networks.

#### 3.3 Sample Subgraph & Homophily Visualization
A 2-hop local ego-network was sampled and visualized using force-directed layout algorithms. Nodes colored by ground-truth subject category confirm strong **graph homophily** ($\mathcal{H} > 0.65$), indicating that papers within the same research domain (e.g., Machine Learning, Computer Vision, Distributed Computing) preferentially cite papers in the same or closely related categories.

---

### 4. Data Preparation & Split Strategy

#### 4.1 Temporal Partitioning Rationale
To reflect real-world predictive utility, the dataset adopts chronological indexing rather than uniform random sampling:
- **Training Set ($\le 2017$):** 90,941 nodes ($53.70\%$)
- **Validation Set ($2018$):** 29,799 nodes ($17.60\%$)
- **Test Set ($2019\text{--}2020$):** 48,603 nodes ($28.70\%$)

**Why Random Split Fails in Citation Graphs:**  
Random splitting introduces massive **temporal data leakage**, where a model trained on future 2020 papers predicts historical 2015 papers by exploiting backward citation links that could not exist in real deployment. Temporal splitting strictly prevents future-to-past information leakage.

#### 4.2 Graph Symmetry & Edge Directionality
In academic citation, an edge $u \to v$ strictly indicates that paper $u$ cites paper $v$ (pointing backward in time). In message passing GNNs, information propagates across citation links. To allow bidirectional semantic message flow between citing and cited works, the directed graph is symmetrized by adding self-loops and reverse edges:
$$\mathbf{\tilde{A}} = \mathbf{A} + \mathbf{A}^\top + \mathbf{I}_N$$
This formulation ensures that citing papers receive semantic context from their references while referenced papers receive contextual updates from citing works.

---

### 5. GNN Architectures & Mathematical Formulations

#### 5.1 Graph Convolutional Network (GCN)
The Graph Convolutional Network (Kipf & Welling, ICLR 2017) applies a first-order localized spectral approximation of graph Fourier transforms.

**Mathematical Propagation Rule:**
$$\mathbf{H}^{(l+1)} = \sigma\left( \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right)$$
where:
- $\mathbf{\tilde{A}} = \mathbf{A} + \mathbf{I}_N$ is the adjacency matrix with added self-loops.
- $\mathbf{\tilde{D}}_{ii} = \sum_j \mathbf{\tilde{A}}_{ij}$ is the diagonal degree matrix.
- $\mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-\frac{1}{2}}$ represents the symmetric normalized adjacency matrix ensuring numerical stability and preventing exploding/vanishing gradients.
- $\mathbf{W}^{(l)}$ is the trainable weight parameter matrix.
- $\sigma(\cdot)$ is the non-linear activation function (ReLU).

**Implemented GCN Architecture:**
- **Input Layer:** $\mathbf{X} \in \mathbb{R}^{N \times 128}$
- **Hidden Layer 1:** $\text{GCNConv}(128 \to 256) \to \text{ReLU} \to \text{Dropout}(p=0.5)$
- **Output Layer 2:** $\text{GCNConv}(256 \to 40) \to \text{Log-Softmax}$
- **Total Parameters:** $\approx 43,816$ trainable weights

#### 5.2 Graph Attention Network (GAT)
The Graph Attention Network (Veličković et al., ICLR 2018) replaces static symmetric normalization with an **anisotropic self-attention mechanism**, enabling nodes to assign dynamically learned importance weights to different neighbors.

**Mathematical Formulation:**
For a node pair $(i, j)$ where $j \in \mathcal{N}_i$, the unnormalized attention coefficient is computed as:
$$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^\top \left[ \mathbf{W}\mathbf{h}_i \,\|\, \mathbf{W}\mathbf{h}_j \right]\right)$$
where $\mathbf{a} \in \mathbb{R}^{2d'}$ is a learnable attention vector, $\|$ denotes vector concatenation, and $\text{LeakyReLU}$ uses negative slope $\alpha=0.2$.

Coefficients are normalized across the neighborhood $\mathcal{N}_i$ using the softmax function:
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$$

To stabilize learning and capture diverse citation dynamics, **multi-head attention** with $K=4$ independent heads is applied:
$$\mathbf{h}_i^{(l+1)} = \mathop{\Big\|}_{k=1}^K \text{ELU}\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k \mathbf{W}^k \mathbf{h}_j^{(l)}\right)$$

**Implemented GAT Architecture:**
- **Input Layer:** $\mathbf{X} \in \mathbb{R}^{N \times 128}$
- **Hidden Layer 1:** $\text{GATConv}(128 \to 64, \text{heads}=4, \text{concat}=\text{True}) \implies 256\text{-dim} \to \text{ELU} \to \text{Dropout}(p=0.5)$
- **Output Layer 2:** $\text{GATConv}(256 \to 40, \text{heads}=1, \text{concat}=\text{False}) \to \text{Log-Softmax}$
- **Total Parameters:** $\approx 43,624$ trainable weights

---

### 6. Training Dynamics & Hyperparameter Optimization

#### 6.1 Objective Function & Optimization
Both models were trained by minimizing the **Negative Log-Likelihood (NLL) / Multi-Class Cross-Entropy Loss** over the masked training node set $\mathcal{V}_{\text{train}}$:
$$\mathcal{L}(\Theta) = -\frac{1}{|\mathcal{V}_{\text{train}}|} \sum_{i \in \mathcal{V}_{\text{train}}} \sum_{c=0}^{C-1} y_{i, c} \ln(\hat{y}_{i, c})$$
where $\hat{y}_{i, c} = \text{Softmax}(\mathbf{z}_{i})_c$ is the predicted probability for class $c$.

Optimization was performed using the **Adam optimizer** ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$) with $L_2$ weight decay ($10^{-5}$) to constrain parameter norms. Training proceeded for 30 epochs with model checkpointing saving the state dictionary achieving the highest validation accuracy.

#### 6.2 Training Convergence Analysis
Table 2 details the training loss and accuracy trajectory for both models across selected epochs.

| Epoch | GCN Loss | GCN Train Acc | GCN Valid Acc | GAT Loss | GAT Train Acc | GAT Valid Acc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 3.7060 | 17.91% | 7.63% | 3.6985 | 18.24% | 8.12% |
| **5** | 3.0120 | 28.12% | 30.26% | 3.0542 | 27.89% | 29.85% |
| **10** | 2.6599 | 33.09% | 33.37% | 2.7120 | 32.45% | 33.10% |
| **15** | 2.3075 | 43.73% | 41.81% | 2.3815 | 42.10% | 40.95% |
| **20** | 2.0109 | 50.24% | 51.56% | 2.0890 | 49.50% | 50.12% |
| **25** | 1.7909 | 55.32% | 56.87% | 1.8654 | 54.12% | 55.40% |
| **30** | **1.6253** | **58.15%** | **59.63%** | **1.7012** | **56.89%** | **58.11%** |

Both architectures exhibit smooth monotonic loss decay without severe overfitting, stabilized by $50\%$ dropout regularization. GCN achieved faster initial convergence and superior asymptotic loss minimization compared to GAT.

#### 6.3 Hyperparameter Optimization Trials
A systematic grid search was executed to study the impact of hidden dimensions, dropout rates, and learning rates on validation accuracy. Table 3 presents the tuning trials.

| Trial | Architecture | Hidden Dim | Dropout ($p$) | Learning Rate ($\eta$) | Best Valid Accuracy | Convergence Behavior |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1 (Best)** | **GCN** | **256** | **0.5** | **0.01** | **59.63%** | Stable convergence, optimal capacity |
| **2** | GCN | 256 | 0.3 | 0.01 | 58.42% | Slight overfitting past epoch 20 |
| **3** | GCN | 128 | 0.5 | 0.01 | 56.91% | Underfitting; constrained representation |
| **4** | GCN | 128 | 0.3 | 0.01 | 56.15% | Lower capacity baseline |
| **5 (Best)** | **GAT** | **64 (4 heads)** | **0.5** | **0.005** | **58.11%** | Stable attention head optimization |
| **6** | GAT | 64 (4 heads) | 0.3 | 0.01 | 55.80% | Higher variance in attention logits |

---

### 7. Comprehensive Model Evaluation & Comparative Analysis

#### 7.1 Official OGB Test Split Performance
Models were evaluated on the 48,603 held-out test papers published in 2019–2020 using standard classification metrics: Top-1 Accuracy, Weighted Precision, Weighted Recall, and Weighted F1-score.

| Model Architecture | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | Total Parameters | Epoch Time (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Graph Convolutional Network (GCN)** | **58.64%** | **0.5492** | **0.5864** | **0.5389** | 43,816 | $\approx 2.4\text{s}$ |
| **Graph Attention Network (GAT)** | **57.39%** | **0.5414** | **0.5739** | **0.5220** | 43,624 | $\approx 7.8\text{s}$ |
| **Delta ($\Delta_{\text{GCN} - \text{GAT}}$)** | **+1.25%** | **+0.0078** | **+0.0125** | **+0.0169** | $+192$ | **$3.25\times\text{ faster}$** |

#### 7.2 Why GCN Outperformed GAT on OGBN-Arxiv
The empirical superiority of 2-layer GCN over GAT (+1.25% accuracy, +1.69% F1) on OGBN-Arxiv is rooted in key graph learning dynamics:
1. **High Graph Homophily:** In dense citation networks with high domain homophily, uniform isotropic degree normalization ($\mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-\frac{1}{2}}$) provides an exceptionally effective smoothing prior.
2. **Attention Over-parameterization on Temporal Drift:** GAT introduces extra attention projection parameters ($\mathbf{a} \in \mathbb{R}^{2d'}$). Under strict temporal out-of-distribution shifts ($2017 \to 2020$), learned attention weights tend to overfit historical citation patterns.
3. **Computational Efficiency:** GCN executes in $\mathcal{O}(|\mathcal{E}| d)$ while multi-head GAT requires $\mathcal{O}(K |\mathcal{E}| d + K |\mathcal{V}| d)$, making GCN $>3.2\times$ faster per epoch.

#### 7.3 40-Class Confusion Matrix & Per-Category Analysis
Evaluation across all 40 individual subject categories reveals distinct performance tiers:
- **High-Performance Categories ($F_1 > 0.72$):** Dominant classes such as `cs.CV` (Computer Vision), `cs.LG` (Machine Learning), `cs.AI` (Artificial Intelligence), and `cs.CR` (Cryptography & Security) achieved high accuracy due to abundant training representation and tight community citation clusters.
- **Challenging Categories ($F_1 < 0.35$):** Low-frequency or highly interdisciplinary categories such as `cs.OS` (Operating Systems), `cs.MS` (Mathematical Software), and `cs.GL` (General Literature) suffered from severe class imbalance and diffuse citation boundaries.
- **Inter-Model Agreement:** GCN and GAT agreed on **$81.4\%$** of test paper predictions. In disagreement cases, GCN demonstrated higher precision on border papers with high citation degrees.

---

### 8. Latent Representation Learning & Explainability

#### 8.1 Hidden Embedding Extraction
To understand how GNNs transform raw text features through message passing, 256-dimensional hidden representations $\mathbf{H}^{(1)} = \sigma(\mathbf{\tilde{A}}\mathbf{X}\mathbf{W}^{(0)})$ were extracted from the best-performing GCN model.

#### 8.2 PCA vs. t-SNE Manifold Projections
Dimensionality reduction was applied to project 5,000 randomly sampled node embeddings into 2D space:
- **PCA 2D Projection (Linear):** Captures the global variance axes. Demonstrates clear separation between macroscopic academic clusters (e.g., Theoretical Computer Science vs. Applied Machine Learning).
- **t-SNE 2D Manifold (Non-Linear):** Resolves fine-grained local topological manifolds. Shows tight, coherent semantic islands for specialized fields (e.g., Computer Vision, Information Theory, Robotics, Cryptography).

#### 8.3 Neighborhood Label Agreement Metric
To quantitatively explain predictions at the local node level, we define the **Neighborhood Homophily Agreement Score**:
$$\text{Agree}(v) = \frac{1}{|\mathcal{N}_v|} \sum_{u \in \mathcal{N}_v} \mathbb{I}(y_u = y_v)$$
For representative test nodes (e.g., Test Node 346), the neighborhood label agreement score reached **$85.71\%$**, demonstrating that correct GNN classification strongly correlates with local neighborhood homophily.

---

### 9. Interactive Dashboard Architecture & Web Deployment

An interactive web application was engineered using **Streamlit** to facilitate transparent model inspection and stakeholder demonstration.

#### 9.1 Dashboard Modular Architecture
The dashboard is structured into 5 dedicated analytics modules:
1. **Graph Analysis:** Displays real-time KPI metrics (169K nodes, 2.3M edges, 40 classes, density) alongside degree distribution histograms and 2-hop ego-network visualizations.
2. **Training Dynamics:** Interactive dual-column convergence charts tracking loss and train/validation accuracy across epochs for GCN and GAT.
3. **Model Evaluation:** Direct metric comparison tables, performance bar charts, and high-resolution 40-class confusion matrix viewers.
4. **Node Classification Lookup:** Real-time search and filter tool allowing users to query papers by Node ID, true category, and prediction status (Both Correct, Disagreements, GCN Correct Only, GAT Correct Only).
5. **Embeddings & Explainability:** Interactive tabbed interface comparing 2D PCA linear projections and non-linear t-SNE manifold visualizations.

#### 9.2 Execution & Verification
The dashboard operates locally via:
```bash
python run_dashboard.py
# or
python -m streamlit run dashboard/app.py
```
Verified operational on `http://localhost:8501`.

---

### 10. Discussion, Limitations & Future Work

#### 10.1 Key Technical Takeaways
1. **GNNs Leverage Graph Topology Effectively:** By combining text embeddings with relational citation links, GCN and GAT achieve substantial classification capability across 40 fine-grained categories.
2. **Inductive Temporal Generalization:** Evaluating under realistic temporal distribution shift demonstrates the importance of robust regularization and prevents misleading performance estimates.
3. **Simplicity vs. Expressive Power:** While GAT offers theoretical expressiveness through attention, isotropic GCN provides higher empirical generalization and computational efficiency on dense homophilous graphs.

#### 10.2 Limitations of the Current Study
- **Full-Batch In-Memory Execution:** Full-batch training requires loading the entire 169K-node graph into memory, limiting scaling to billion-edge graphs.
- **Shallow Architecture (2 Layers):** Deeper GNNs ($\ge 4$ layers) suffer from **oversmoothing** (node embeddings converge to identical averages) and **over-squashing** of distant messages.
- **Homogeneous Assumption:** Current models treat citation links as homogeneous, ignoring rich metadata such as author affiliations, publication venues, and citation timestamps.

#### 10.3 Future Research Directions
- **Scalable Mini-Batching:** Implementing **NeighborSampler**, **Cluster-GCN**, or **GraphSAINT** for sub-graph batch training.
- **Advanced GNN Backbones:** Exploring **GraphSAGE**, **Graph Isomorphism Networks (GIN)**, and **Graph Transformer** architectures with residual/skip connections.
- **Advanced Text Encoders:** Replacing static 128-dim word2vec features with contextual representations from pre-trained Language Models (e.g., DeBERTa, SciBERT, GIANT).

---

### 11. Reproducibility & Integrity Statement

To guarantee full reproducibility, all experimental artifacts, random seeds, and software dependencies are documented:
- **Random Seed:** Set to `42` across PyTorch, NumPy, and CUDA generators (`src/training/train.py`).
- **Software Dependencies:** Python 3.14, PyTorch 2.13.0, PyTorch Geometric 2.8.0, OGB 1.3.6, Scikit-Learn 1.9.0, Pandas 3.0.1, NumPy 2.4.3, Streamlit 1.61.1, ReportLab 5.0.0.
- **Execution Pipeline:** Sequential verification across notebooks 01 through 08.
- **Checkpoints:** Best weights stored in `models/best_gcn.pt` and `models/best_gat.pt`.

---

### 12. References

1. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *International Conference on Learning Representations (ICLR)*.
2. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph Attention Networks. *International Conference on Learning Representations (ICLR)*.
3. Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., & Leskovec, J. (2020). Open Graph Benchmark: Datasets for Machine Learning on Graphs. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 22118-22133.
4. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
5. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How Powerful are Graph Neural Networks? *International Conference on Learning Representations (ICLR)*.
6. Fey, M., & Lenssen, J. E. (2019). Fast Graph Representation Learning with PyTorch Geometric. *ICLR Workshop on Representation Learning on Graphs and Manifolds*.

---
