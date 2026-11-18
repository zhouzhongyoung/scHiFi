from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph


@dataclass
class LocalBatchGraph:
    raw: List[sp.csr_matrix]
    normalized: List[sp.csr_matrix]
    dense_target: np.ndarray


def _degree_power(a: sp.spmatrix, power: float) -> sp.csr_matrix:
    degree = np.asarray(a.sum(axis=1)).reshape(-1)
    values = np.power(degree, power, where=degree > 0)
    values[degree <= 0] = 0.0
    values[~np.isfinite(values)] = 0.0
    return sp.diags(values.astype(np.float32), format="csr")


def norm_adj(a: sp.spmatrix) -> sp.csr_matrix:
    """Original-style D^{-1/2} A D^{-1/2} normalization.

    No extra self-loop is added here because local graphs are created with
    ``include_self=True`` to match the earliest batch-local implementation.
    """
    a = a.tocsr().astype(np.float32)
    d = _degree_power(a, -0.5)
    return (d @ a @ d).tocsr().astype(np.float32)


def build_local_batch_graph(
    x_graph_batch: np.ndarray,
    k_list: Sequence[int] = (10, 15, 20, 25),
    pca_dim: int = 50,
    metric: str = "euclidean",
) -> LocalBatchGraph:
    """Build four independent KNN graphs inside one mini-batch.

    This intentionally follows the user's earliest batch implementation:
    PCA is fit only on the current batch, KNN is computed only inside the
    current batch, ``include_self=True`` is used, and no symmetrization is
    imposed.
    """
    x = np.asarray(x_graph_batch, dtype=np.float32)
    n = int(x.shape[0])
    if n <= 0:
        raise ValueError("Empty local batch.")
    if n == 1:
        raw = [sp.eye(1, dtype=np.float32, format="csr") for _ in range(4)]
        norm = [sp.eye(1, dtype=np.float32, format="csr") for _ in range(4)]
        return LocalBatchGraph(raw=raw, normalized=norm, dense_target=np.ones((1, 1), dtype=np.float32))

    k_values = [max(1, min(int(k), n)) for k in k_list]
    if len(k_values) != 4:
        raise ValueError("scHiFi local mode expects exactly four K values.")

    pca_dim = min(int(pca_dim), x.shape[1], max(1, n - 1))
    if pca_dim > 0 and x.shape[1] > pca_dim and n > 1:
        emb = PCA(n_components=pca_dim).fit_transform(x).astype(np.float32)
    else:
        emb = x

    raw_graphs: List[sp.csr_matrix] = []
    norm_graphs: List[sp.csr_matrix] = []
    for k in k_values:
        a = kneighbors_graph(
            emb,
            n_neighbors=k,
            mode="connectivity",
            metric=metric,
            include_self=True,
        ).tocsr().astype(np.float32)
        raw_graphs.append(a)
        norm_graphs.append(norm_adj(a))
    return LocalBatchGraph(
        raw=raw_graphs,
        normalized=norm_graphs,
        dense_target=norm_graphs[0].toarray().astype(np.float32, copy=False),
    )


def scipy_to_torch_sparse(matrix: sp.spmatrix, device: Union[str, torch.device]) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32, copy=False))
    out = torch.sparse_coo_tensor(indices, values, coo.shape, dtype=torch.float32)
    return out.coalesce().to(device)


def graph_to_torch(graph: LocalBatchGraph, device: Union[str, torch.device]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    adjs = [scipy_to_torch_sparse(a, device) for a in graph.normalized]
    target = torch.from_numpy(graph.dense_target).to(device)
    return adjs, target
