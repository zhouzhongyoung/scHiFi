from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans


@dataclass
class GraphPartitions:
    parts: List[np.ndarray]
    method: str


def _membership_to_parts(membership: np.ndarray, n_parts: int) -> List[np.ndarray]:
    return [np.flatnonzero(membership == i).astype(np.int64) for i in range(n_parts)]


def _metis_partition(adjacency: sp.csr_matrix, n_parts: int) -> List[np.ndarray]:
    import pymetis  # optional dependency

    adjacency = adjacency.tocsr()
    adjacency_lists = [
        adjacency.indices[adjacency.indptr[i] : adjacency.indptr[i + 1]].tolist()
        for i in range(adjacency.shape[0])
    ]
    _, membership = pymetis.part_graph(n_parts, adjacency=adjacency_lists)
    return _membership_to_parts(np.asarray(membership), n_parts)


def _kmeans_partition(
    embedding: np.ndarray,
    n_parts: int,
    target_size: int,
    seed: int,
) -> List[np.ndarray]:
    labels = MiniBatchKMeans(
        n_clusters=n_parts,
        batch_size=min(4096, max(256, embedding.shape[0])),
        n_init=10,
        random_state=seed,
    ).fit_predict(embedding)
    rng = np.random.default_rng(seed)
    parts: List[np.ndarray] = []
    max_part_size = max(target_size * 2, target_size + 1)
    for part in _membership_to_parts(labels, n_parts):
        if part.size == 0:
            continue
        rng.shuffle(part)
        if part.size <= max_part_size:
            parts.append(part)
        else:
            parts.extend(np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(
                part, math.ceil(part.size / target_size)
            ))
    return parts


def partition_global_graph(
    adjacency: sp.csr_matrix,
    pca_embedding: np.ndarray,
    target_size: int = 1024,
    method: str = "auto",
    seed: int = 1111,
) -> GraphPartitions:
    """Partition one already-built global graph for Cluster-GCN-style training.

    ``metis`` is preferred when pymetis is installed. The deterministic fallback
    clusters the same global PCA space used to construct the KNN graph. In both
    cases, mini-batches use induced edges from the global graph and never rebuild
    local KNN graphs.
    """
    n = adjacency.shape[0]
    n_parts = max(1, math.ceil(n / max(1, int(target_size))))
    selected_method = method
    if method in {"auto", "metis"}:
        try:
            parts = _metis_partition(adjacency, n_parts)
            selected_method = "metis"
        except (ImportError, ModuleNotFoundError):
            if method == "metis":
                raise RuntimeError(
                    "--partition_method metis requires pymetis. Install it or use auto/kmeans."
                )
            parts = _kmeans_partition(pca_embedding, n_parts, target_size, seed)
            selected_method = "kmeans"
    elif method == "kmeans":
        parts = _kmeans_partition(pca_embedding, n_parts, target_size, seed)
    else:
        raise ValueError(f"Unknown partition method: {method}")

    parts = [np.sort(p) for p in parts if p.size > 0]
    covered = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    if np.unique(covered).size != n:
        missing = np.setdiff1d(np.arange(n), covered)
        if missing.size:
            parts.append(missing)
    print(
        f"Created {len(parts)} global-graph partitions with method={selected_method}; "
        f"median size={int(np.median([p.size for p in parts]))}"
    )
    return GraphPartitions(parts=parts, method=selected_method)
