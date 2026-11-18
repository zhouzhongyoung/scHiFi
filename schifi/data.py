from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import h5py
import numpy as np
import scipy.sparse as sp
import pandas as pd


ArrayLike = Union[np.ndarray, sp.spmatrix]


@dataclass
class LoadedDataset:
    counts: ArrayLike
    labels: Optional[np.ndarray]
    cell_names: Optional[np.ndarray] = None
    gene_names: Optional[np.ndarray] = None


@dataclass
class ProcessedData:
    raw_counts: ArrayLike
    x_attr: np.ndarray
    x_graph: np.ndarray
    size_factors: np.ndarray
    labels: Optional[np.ndarray]
    cell_names: Optional[np.ndarray]
    gene_names: Optional[np.ndarray]
    hvg_indices: np.ndarray

    @property
    def n_cells(self) -> int:
        return int(self.x_attr.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.x_attr.shape[1])


def _decode(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O"}:
        return np.asarray([
            v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
            for v in values
        ])
    return values


def _read_h5_group(group: h5py.Group) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            out[key] = _read_h5_group(value)
        else:
            out[key] = value[...]
    return out


def _read_legacy_exprs(handle: h5py.File) -> ArrayLike:
    exprs = handle["exprs"]
    if isinstance(exprs, h5py.Group):
        return sp.csr_matrix(
            (exprs["data"][...], exprs["indices"][...], exprs["indptr"][...]),
            shape=tuple(exprs["shape"][...]),
        )
    return np.asarray(exprs[...], dtype=np.float32)


def load_dataset(path: Union[str, Path], label_key: str = "cell_type1") -> LoadedDataset:
    """Load the two HDF5 layouts used by the original scHiFi project.

    Supported layouts:
      1. root datasets ``X`` and ``Y``;
      2. legacy ``exprs`` + ``obs``/``obs_names`` + ``var_names`` layout.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".npz":
        obj = np.load(path, allow_pickle=True)
        labels = obj["Y"] if "Y" in obj else None
        return LoadedDataset(np.asarray(obj["X"], dtype=np.float32), labels)

    with h5py.File(path, "r") as f:
        if "X" in f:
            counts = np.asarray(f["X"][...], dtype=np.float32)
            labels = _decode(f["Y"][...]) if "Y" in f else None
            cell_names = _decode(f["cell_names"][...]) if "cell_names" in f else None
            gene_names = _decode(f["gene_names"][...]) if "gene_names" in f else None
            return LoadedDataset(counts, labels, cell_names, gene_names)

        if "exprs" not in f:
            raise ValueError(
                f"Unsupported HDF5 layout in {path}. Expected X/Y or exprs/obs."
            )
        counts = _read_legacy_exprs(f)
        cell_names = _decode(f["obs_names"][...]) if "obs_names" in f else None
        gene_names = _decode(f["var_names"][...]) if "var_names" in f else None
        labels = None
        if "obs" in f and label_key in f["obs"]:
            labels = _decode(f["obs"][label_key][...])
        elif "obs" in f:
            obs = _read_h5_group(f["obs"])
            if label_key in obs:
                labels = _decode(obs[label_key])
        return LoadedDataset(counts, labels, cell_names, gene_names)


def _row_sums(x: ArrayLike) -> np.ndarray:
    return np.asarray(x.sum(axis=1)).reshape(-1)


def _col_sums(x: ArrayLike) -> np.ndarray:
    return np.asarray(x.sum(axis=0)).reshape(-1)


def _subset(x: ArrayLike, row_mask: np.ndarray, col_mask: np.ndarray) -> ArrayLike:
    if sp.issparse(x):
        return x.tocsr()[row_mask][:, col_mask]
    return np.asarray(x)[np.ix_(row_mask, col_mask)]


def _normalize_log1p(counts: ArrayLike, target_sum: float) -> ArrayLike:
    lib = _row_sums(counts)
    scale = target_sum / np.maximum(lib, 1e-12)
    if sp.issparse(counts):
        out = sp.diags(scale.astype(np.float32)) @ counts.tocsr().astype(np.float32)
        out.data = np.log1p(out.data)
        return out.tocsr()
    out = np.asarray(counts, dtype=np.float32) * scale[:, None].astype(np.float32)
    return np.log1p(out).astype(np.float32, copy=False)


def _mean_var(x: ArrayLike, ddof: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    if sp.issparse(x):
        mean = np.asarray(x.mean(axis=0)).reshape(-1)
        mean_sq = np.asarray(x.multiply(x).mean(axis=0)).reshape(-1)
        var = np.maximum(mean_sq - np.square(mean), 0.0)
        if ddof == 1 and x.shape[0] > 1:
            var *= x.shape[0] / (x.shape[0] - 1)
        return mean, var
    arr = np.asarray(x, dtype=np.float32)
    return arr.mean(axis=0), arr.var(axis=0, ddof=ddof)


def _select_hvg_seurat(lognorm: ArrayLike, n_top: int, n_bins: int = 20) -> np.ndarray:
    """Approximate Scanpy's historical ``flavor='seurat'`` HVG ranking.

    The original project called ``sc.pp.highly_variable_genes`` after log1p.
    This implementation keeps the core project independent of Scanpy while
    retaining mean-bin normalized dispersion ranking.
    """
    if sp.issparse(lognorm):
        restored = lognorm.copy().tocsr()
        restored.data = np.expm1(restored.data)
    else:
        restored = np.expm1(np.asarray(lognorm, dtype=np.float32))
    mean, var = _mean_var(restored, ddof=1)
    safe_mean = np.maximum(mean, 1e-12)
    dispersion = var / safe_mean
    dispersion[dispersion <= 0] = np.nan
    log_disp = np.log(dispersion)
    log_mean = np.log1p(mean)

    frame = pd.DataFrame({"mean": log_mean, "dispersion": log_disp})
    # duplicates='drop' handles nearly constant small datasets.
    frame["mean_bin"] = pd.cut(frame["mean"], bins=n_bins, duplicates="drop")
    grouped = frame.groupby("mean_bin", observed=True)["dispersion"]
    bin_mean = grouped.transform("mean")
    bin_std = grouped.transform("std")
    single = ~np.isfinite(bin_std.to_numpy()) | (bin_std.to_numpy() < 1e-12)
    normalized = (frame["dispersion"].to_numpy() - bin_mean.to_numpy()) / np.maximum(
        bin_std.to_numpy(), 1e-12
    )
    # Scanpy treats singleton bins as informative rather than discarding them.
    normalized[single] = 1.0
    normalized[~np.isfinite(normalized)] = -np.inf
    n_top = min(int(n_top), normalized.size)
    indices = np.argsort(normalized, kind="stable")[-n_top:]
    indices.sort()
    return indices.astype(np.int64)


def _dense(x: ArrayLike) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def preprocess_counts(
    loaded: LoadedDataset,
    n_hvg: int = 2000,
    scale_attributes: bool = True,
    ceil_noninteger: bool = False,
) -> ProcessedData:
    """Filter, library-normalize, log-transform, scale and select HVGs.

    The graph branch receives log-normalized HVGs. The attribute branch receives
    all retained genes after log normalization and per-gene standardization.
    Raw filtered counts and size factors are retained for the ZINB objective.
    """
    counts: ArrayLike
    if sp.issparse(loaded.counts):
        counts = loaded.counts.tocsr().astype(np.float32)
        if counts.data.size and counts.data.min() < 0:
            raise ValueError("Count matrix contains negative values.")
        if ceil_noninteger:
            counts.data = np.ceil(counts.data)
    else:
        counts = np.asarray(loaded.counts, dtype=np.float32)
        if counts.ndim != 2:
            raise ValueError(f"Expected a 2-D count matrix, got {counts.shape}")
        if np.nanmin(counts) < 0:
            raise ValueError("Count matrix contains negative values.")
        if ceil_noninteger:
            counts = np.ceil(counts)

    cell_mask = _row_sums(counts) > 0
    gene_mask = _col_sums(counts) > 0
    counts = _subset(counts, cell_mask, gene_mask)

    labels = None if loaded.labels is None else np.asarray(loaded.labels)[cell_mask]
    if labels is not None:
        _, labels = np.unique(labels, return_inverse=True)
        labels = labels.astype(np.int64)
    cell_names = None if loaded.cell_names is None else np.asarray(loaded.cell_names)[cell_mask]
    gene_names = None if loaded.gene_names is None else np.asarray(loaded.gene_names)[gene_mask]

    lib = _row_sums(counts).astype(np.float32)
    target_sum = float(np.median(lib[lib > 0]))
    size_factors = lib / max(target_sum, 1e-12)
    lognorm = _normalize_log1p(counts, target_sum=target_sum)

    n_hvg = min(int(n_hvg), int(lognorm.shape[1]))
    hvg_indices = _select_hvg_seurat(lognorm, n_hvg)
    x_graph = _dense(lognorm[:, hvg_indices])

    x_attr = _dense(lognorm)
    if scale_attributes:
        attr_mean = x_attr.mean(axis=0, keepdims=True)
        attr_std = x_attr.std(axis=0, keepdims=True, ddof=1)
        attr_std[attr_std < 1e-6] = 1.0
        x_attr = (x_attr - attr_mean) / attr_std
    x_attr = np.asarray(x_attr, dtype=np.float32, order="C")

    raw_counts: ArrayLike
    if sp.issparse(counts):
        raw_counts = counts.tocsr().astype(np.float32)
    else:
        raw_counts = np.asarray(counts, dtype=np.float32, order="C")

    return ProcessedData(
        raw_counts=raw_counts,
        x_attr=x_attr,
        x_graph=np.asarray(x_graph, dtype=np.float32, order="C"),
        size_factors=np.asarray(size_factors, dtype=np.float32),
        labels=labels,
        cell_names=cell_names,
        gene_names=gene_names,
        hvg_indices=hvg_indices,
    )


def take_rows(x: ArrayLike, indices: np.ndarray) -> np.ndarray:
    block = x[indices]
    if sp.issparse(block):
        return block.toarray().astype(np.float32, copy=False)
    return np.asarray(block, dtype=np.float32)
