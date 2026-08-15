from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch_geometric.data import Data


@torch.no_grad()
def extract_embeddings(model: torch.nn.Module, data: Data) -> np.ndarray:
    model.eval()
    _, embeddings = model(data.x, data.edge_index, return_embeddings=True)
    return embeddings.detach().cpu().numpy()


def plot_embedding_projection(embeddings: np.ndarray, labels: np.ndarray, output_path: str | Path,
                              method: str = "pca", sample_size: int = 5000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(embeddings), size=min(sample_size, len(embeddings)), replace=False)
    sample_embeddings, sample_labels = embeddings[indices], labels[indices]
    reducer = PCA(n_components=2, random_state=seed) if method.lower() == "pca" else TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=seed)
    projection = reducer.fit_transform(sample_embeddings)
    plt.figure(figsize=(9, 7))
    plt.scatter(projection[:, 0], projection[:, 1], c=sample_labels, s=4, cmap="tab20", alpha=0.7)
    plt.title(f"{method.upper()} projection of learned node embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(label="Subject class")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
