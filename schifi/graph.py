from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


@dataclass
class GlobalGraphs:
    raw: List[sp.csr_matrix]
    normalized: List[sp.csr_matrix]
    pca_embedding: np.ndarray
    k_list: List[int]
    input_features: int

    @property
    def n_nodes(self) -> int:
        return int(self.raw[0].shape[0])


def normalize_adjacency(raw: sp.csr_matrix, add_self_loops: bool = True) -> sp.csr_matrix:
    a = raw.tocsr().astype(np.float32)
    if add_self_loops:
        a = a + sp.eye(a.shape[0], dtype=np.float32, format="csr")
    degree = np.asarray(a.sum(axis=1)).reshape(-1)
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    nz = degree > 0
    inv_sqrt[nz] = np.power(degree[nz], -0.5)
    d = sp.diags(inv_sqrt, format="csr")
    return (d @ a @ d).tocsr().astype(np.float32)


def _cache_prefix(cache_dir: Path, cache_key: str, k_list: Sequence[int], pca_dim: int) -> Path:
    k_text = "-".join(str(k) for k in k_list)
    safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in cache_key)
    return cache_dir / f"{safe_key}_pca{pca_dim}_k{k_text}"


def _load_cache(prefix: Path, expected_nodes: int, expected_features: int) -> Optional[GlobalGraphs]:
    metadata_path = prefix.with_suffix(".json")
    pca_path = prefix.parent / f"{prefix.name}_pca.npy"
    if not metadata_path.exists() or not pca_path.exists():
        return None
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(meta["n_nodes"]) != expected_nodes:
        return None
    if int(meta.get("n_features", -1)) != expected_features:
        return None
    k_list = [int(k) for k in meta["k_list"]]
    raw, normalized = [], []
    for i in range(len(k_list)):
        raw_path = prefix.parent / f"{prefix.name}_raw_{i}.npz"
        norm_path = prefix.parent / f"{prefix.name}_norm_{i}.npz"
        if not raw_path.exists() or not norm_path.exists():
            return None
        raw.append(sp.load_npz(raw_path).tocsr())
        normalized.append(sp.load_npz(norm_path).tocsr())
    return GlobalGraphs(raw, normalized, np.load(pca_path), k_list, int(meta["n_features"]))


def _save_cache(prefix: Path, graphs: GlobalGraphs) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "n_nodes": graphs.n_nodes,
        "n_features": int(graphs.input_features),
        "k_list": graphs.k_list,
    }
    prefix.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.save(prefix.parent / f"{prefix.name}_pca.npy", graphs.pca_embedding)
    for i, (raw, norm) in enumerate(zip(graphs.raw, graphs.normalized)):
        sp.save_npz(prefix.parent / f"{prefix.name}_raw_{i}.npz", raw)
        sp.save_npz(prefix.parent / f"{prefix.name}_norm_{i}.npz", norm)


def build_or_load_global_graphs(
    x_graph: np.ndarray,
    k_list: Sequence[int] = (10, 15, 20, 25),
    pca_dim: int = 50,
    metric: str = "euclidean",
    cache_dir: Union[str, Path] = "graph_cache",
    cache_key: str = "dataset",
    rebuild: bool = False,
    n_jobs: int = -1,
    seed: int = 1111,
) -> GlobalGraphs:
    """Build nested, symmetric, global KNN graphs without dense N x N matrices."""
    x_graph = np.asarray(x_graph, dtype=np.float32)
    n = x_graph.shape[0]
    if n < 2:
        raise ValueError("At least two cells are required to construct a graph.")
    k_list = sorted({min(int(k), n - 1) for k in k_list if int(k) > 0})
    if len(k_list) != 4:
        raise ValueError("scHiFi expects exactly four positive K values.")

    cache_dir = Path(cache_dir)
    prefix = _cache_prefix(cache_dir, cache_key, k_list, pca_dim)
    if not rebuild:
        cached = _load_cache(prefix, n, x_graph.shape[1])
        if cached is not None:
            print(f"Loaded global sparse graphs from {prefix.parent}")
            return cached

    pca_dim = min(int(pca_dim), x_graph.shape[1], max(1, n - 1))
    print(f"Running global PCA: {x_graph.shape} -> {pca_dim} dimensions")
    pca_embedding = PCA(
        n_components=pca_dim,
        svd_solver="randomized",
        random_state=seed,
    ).fit_transform(x_graph).astype(np.float32)

    max_k = max(k_list)
    print(f"Searching global {max_k}-nearest neighbors for {n} cells")
    nn = NearestNeighbors(
        n_neighbors=max_k + 1,
        metric=metric,
        n_jobs=n_jobs,
    ).fit(pca_embedding)
    neighbors = nn.kneighbors(return_distance=False)

    # Remove self robustly, then retain max_k neighbors.
    cleaned = np.empty((n, max_k), dtype=np.int64)
    for i in range(n):
        row = neighbors[i]
        row = row[row != i]
        if row.size < max_k:
            raise RuntimeError(f"Could not obtain {max_k} non-self neighbors for node {i}.")
        cleaned[i] = row[:max_k]

    raw_graphs: List[sp.csr_matrix] = []
    norm_graphs: List[sp.csr_matrix] = []
    rows_base = np.arange(n, dtype=np.int64)
    for k in k_list:
        rows = np.repeat(rows_base, k)
        cols = cleaned[:, :k].reshape(-1)
        data = np.ones(rows.shape[0], dtype=np.float32)
        a = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
        # Undirected union KNN graph. No self loops in raw graph.
        a = a.maximum(a.T).tocsr()
        a.setdiag(0)
        a.eliminate_zeros()
        raw_graphs.append(a)
        norm_graphs.append(normalize_adjacency(a, add_self_loops=True))
        print(f"Global KNN graph K={k}: {a.nnz // 2:,} undirected edges")

    graphs = GlobalGraphs(raw_graphs, norm_graphs, pca_embedding, list(k_list), int(x_graph.shape[1]))
    _save_cache(prefix, graphs)
    return graphs


def scipy_to_torch_sparse(
    matrix: sp.spmatrix,
    device: Union[torch.device, str],
) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32, copy=False))
    tensor = torch.sparse_coo_tensor(indices, values, coo.shape, dtype=torch.float32)
    return tensor.coalesce().to(device)


def induced_subgraphs(
    matrices: Iterable[sp.csr_matrix], nodes: np.ndarray
) -> List[sp.csr_matrix]:
    nodes = np.asarray(nodes, dtype=np.int64)
    return [m[nodes][:, nodes].tocsr() for m in matrices]


def expand_halo(
    core_nodes: np.ndarray,
    adjacency: sp.csr_matrix,
    hops: int = 0,
    max_nodes: Optional[int] = None,
) -> np.ndarray:
    core_nodes = np.unique(np.asarray(core_nodes, dtype=np.int64))
    if hops <= 0:
        return core_nodes
    selected = set(core_nodes.tolist())
    frontier = core_nodes
    for _ in range(hops):
        if frontier.size == 0:
            break
        neighbors = []
        for node in frontier:
            start, end = adjacency.indptr[node], adjacency.indptr[node + 1]
            neighbors.append(adjacency.indices[start:end])
        new_nodes = np.unique(np.concatenate(neighbors)) if neighbors else np.empty(0, dtype=np.int64)
        new_nodes = np.asarray([n for n in new_nodes if int(n) not in selected], dtype=np.int64)
        if max_nodes is not None and len(selected) + new_nodes.size > max_nodes:
            keep = max(0, int(max_nodes) - len(selected))
            new_nodes = new_nodes[:keep]
        selected.update(new_nodes.tolist())
        frontier = new_nodes
        if max_nodes is not None and len(selected) >= max_nodes:
            break
    return np.asarray(sorted(selected), dtype=np.int64)
