from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from .data import ProcessedData, take_rows
from .graph import (
    GlobalGraphs,
    expand_halo,
    induced_subgraphs,
    scipy_to_torch_sparse,
)
from .metrics import evaluate_clustering
from .model import ScHiFi
from .partition import GraphPartitions, partition_global_graph


@dataclass
class TrainerConfig:
    device: str = "cuda"
    inference_device: str = "auto"
    train_mode: str = "auto"  # auto, full, cluster
    full_batch_threshold: int = 6000
    partition_method: str = "auto"
    cluster_size: int = 1024
    halo_hops: int = 1
    max_halo_nodes: Optional[int] = 4096
    pretrain_epochs: int = 400
    train_epochs: int = 150
    pretrain_lr: float = 1e-3
    train_lr: float = 0.5
    update_interval: int = 1
    tol: float = 1e-3
    attr_batch_size: int = 256
    cluster_weight: float = 0.5
    zinb_weight: float = 1.0
    graph_weight: float = 0.1
    align_weight: float = 0.1
    negative_ratio: float = 1.0
    max_positive_edges: int = 20000
    seed: int = 1111
    log_interval: int = 10


@dataclass
class TrainResult:
    labels: np.ndarray
    latent: np.ndarray
    metrics: Dict[str, float]
    history: List[Dict[str, float]] = field(default_factory=list)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _sample_graph_reconstruction_loss(
    embeddings: torch.Tensor,
    raw_adjacency: sp.csr_matrix,
    negative_ratio: float,
    max_positive_edges: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Sampled link reconstruction; never materializes an N x N matrix."""
    upper = sp.triu(raw_adjacency, k=1).tocoo()
    n_pos_available = int(upper.nnz)
    if n_pos_available == 0:
        return embeddings.sum() * 0.0
    n_pos = min(n_pos_available, int(max_positive_edges))
    if n_pos < n_pos_available:
        chosen = rng.choice(n_pos_available, size=n_pos, replace=False)
        pos_u = upper.row[chosen]
        pos_v = upper.col[chosen]
    else:
        pos_u, pos_v = upper.row, upper.col

    n_nodes = raw_adjacency.shape[0]
    n_neg = max(1, int(round(n_pos * negative_ratio)))
    neg_u: List[int] = []
    neg_v: List[int] = []
    attempts = 0
    max_attempts = max(1000, n_neg * 30)
    while len(neg_u) < n_neg and attempts < max_attempts:
        batch = min((n_neg - len(neg_u)) * 3, 65536)
        u = rng.integers(0, n_nodes, size=batch, endpoint=False)
        v = rng.integers(0, n_nodes, size=batch, endpoint=False)
        valid = u != v
        u, v = u[valid], v[valid]
        # Sparse advanced indexing returns a matrix; A1 converts to a flat array.
        absent = np.asarray(raw_adjacency[u, v]).reshape(-1) == 0
        for a, b in zip(u[absent], v[absent]):
            neg_u.append(int(a))
            neg_v.append(int(b))
            if len(neg_u) >= n_neg:
                break
        attempts += batch
    if len(neg_u) < n_neg:
        raise RuntimeError("Negative-edge sampling failed; graph may be unexpectedly dense.")

    device = embeddings.device
    pos_u_t = torch.as_tensor(pos_u, dtype=torch.long, device=device)
    pos_v_t = torch.as_tensor(pos_v, dtype=torch.long, device=device)
    neg_u_t = torch.as_tensor(neg_u, dtype=torch.long, device=device)
    neg_v_t = torch.as_tensor(neg_v, dtype=torch.long, device=device)
    pos_logits = (embeddings[pos_u_t] * embeddings[pos_v_t]).sum(dim=1)
    neg_logits = (embeddings[neg_u_t] * embeddings[neg_v_t]).sum(dim=1)
    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
    return F.binary_cross_entropy_with_logits(logits, labels)


class ScHiFiTrainer:
    def __init__(
        self,
        model: ScHiFi,
        data: ProcessedData,
        graphs: GlobalGraphs,
        config: TrainerConfig,
        output_dir: Union[str, Path],
    ):
        _set_seed(config.seed)
        self.model = model
        self.data = data
        self.graphs = graphs
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if config.device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA is unavailable; falling back to CPU.")
            config.device = "cpu"
        self.device = torch.device(config.device)
        self.model.to(self.device)

        if config.train_mode == "auto":
            self.train_mode = "full" if data.n_cells <= config.full_batch_threshold else "cluster"
        elif config.train_mode in {"full", "cluster"}:
            self.train_mode = config.train_mode
        else:
            raise ValueError(f"Unknown train_mode: {config.train_mode}")
        print(f"Training mode: {self.train_mode}")

        self.partitions: Optional[GraphPartitions] = None
        if self.train_mode == "cluster":
            self.partitions = partition_global_graph(
                adjacency=graphs.raw[-1],
                pca_embedding=graphs.pca_embedding,
                target_size=config.cluster_size,
                method=config.partition_method,
                seed=config.seed,
            )

        self._full_torch_adjs: Optional[List[torch.Tensor]] = None
        self.history: List[Dict[str, float]] = []

    def _resolve_inference_device(self) -> torch.device:
        value = self.config.inference_device
        if value == "auto":
            return self.device
        if value.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(value)

    def _iter_core_parts(self, epoch: int) -> Iterable[np.ndarray]:
        if self.train_mode == "full":
            yield np.arange(self.data.n_cells, dtype=np.int64)
            return
        assert self.partitions is not None
        order = np.arange(len(self.partitions.parts))
        rng = np.random.default_rng(self.config.seed + epoch)
        rng.shuffle(order)
        for idx in order:
            yield self.partitions.parts[int(idx)]

    def _prepare_subgraph(
        self, core_nodes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[torch.Tensor], sp.csr_matrix]:
        if self.train_mode == "full" and core_nodes.size == self.data.n_cells:
            nodes = core_nodes
            if self._full_torch_adjs is None:
                self._full_torch_adjs = [
                    scipy_to_torch_sparse(a, self.device) for a in self.graphs.normalized
                ]
            adjs = self._full_torch_adjs
            raw = self.graphs.raw[0]
        else:
            # Cluster training operates on induced subgraphs from the same
            # global KNN graph. No local KNN is rebuilt.
            nodes = expand_halo(
                core_nodes,
                self.graphs.raw[-1],
                hops=self.config.halo_hops,
                max_nodes=self.config.max_halo_nodes,
            )
            sub_norm = induced_subgraphs(self.graphs.normalized, nodes)
            adjs = [scipy_to_torch_sparse(a, self.device) for a in sub_norm]
            raw = self.graphs.raw[0][nodes][:, nodes].tocsr()
        core_positions = np.searchsorted(nodes, core_nodes)
        return nodes, core_positions, adjs, raw

    def _batch_forward(
        self,
        core_nodes: np.ndarray,
        add_noise: bool,
    ):
        nodes, core_pos, adjs, raw_sub = self._prepare_subgraph(core_nodes)
        x_graph = torch.from_numpy(self.data.x_graph[nodes]).to(self.device)
        g1, g2, g3, g4 = self.model.encode_topology(x_graph, adjs)
        core_pos_t = torch.as_tensor(core_pos, dtype=torch.long, device=self.device)
        x_attr = torch.from_numpy(self.data.x_attr[core_nodes]).to(self.device)
        z = self.model.encode_attribute(
            x_attr,
            g1[core_pos_t],
            g2[core_pos_t],
            g3[core_pos_t],
            add_noise=add_noise,
        )
        return z, g3, g4[core_pos_t], raw_sub

    def _zinb_inputs(self, core_nodes: np.ndarray):
        raw = torch.from_numpy(take_rows(self.data.raw_counts, core_nodes)).to(self.device)
        sf = torch.from_numpy(self.data.size_factors[core_nodes]).to(self.device)
        return raw, sf

    def pretrain(self, checkpoint_path: Union[str, Path]) -> None:
        checkpoint_path = Path(checkpoint_path)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.pretrain_lr, amsgrad=True)
        print("Starting ZINB + sampled graph-link pretraining")
        for epoch in range(self.config.pretrain_epochs):
            self.model.train()
            total_loss = total_zinb = total_graph = 0.0
            n_steps = 0
            rng = np.random.default_rng(self.config.seed + epoch)
            for core_nodes in self._iter_core_parts(epoch):
                optimizer.zero_grad(set_to_none=True)
                z, g3_sub, _, raw_sub = self._batch_forward(core_nodes, add_noise=True)
                mean, disp, pi = self.model.decode_zinb(z)
                x_raw, sf = self._zinb_inputs(core_nodes)
                zinb = self.model.zinb_loss(x_raw, mean, disp, pi, sf)
                graph_loss = _sample_graph_reconstruction_loss(
                    g3_sub,
                    raw_sub,
                    self.config.negative_ratio,
                    self.config.max_positive_edges,
                    rng,
                )
                loss = self.config.zinb_weight * zinb + self.config.graph_weight * graph_loss
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())
                total_zinb += float(zinb.detach())
                total_graph += float(graph_loss.detach())
                n_steps += 1
            if (epoch + 1) % self.config.log_interval == 0 or epoch == 0:
                print(
                    f"Pretrain {epoch + 1:4d}/{self.config.pretrain_epochs}: "
                    f"loss={total_loss/n_steps:.6f}, zinb={total_zinb/n_steps:.6f}, "
                    f"graph={total_graph/n_steps:.6f}"
                )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self.model.state_dict()}, checkpoint_path)
        print(f"Saved pretrained model: {checkpoint_path}")

    def load_pretrained(self, checkpoint_path: Union[str, Path]) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state = checkpoint.get("model_state", checkpoint.get("ae_state_dict", checkpoint))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        # A previous checkpoint may include cluster centers; they are reinitialized below.
        missing = [key for key in missing if key != "mu"]
        unexpected = [key for key in unexpected if key != "mu"]
        if missing or unexpected:
            print(f"Checkpoint loaded with missing={missing}, unexpected={unexpected}")
        else:
            print(f"Loaded pretrained model: {checkpoint_path}")

    @torch.no_grad()
    def _exact_topology_all(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact full-global-graph GCN inference, independent of training partitions."""
        self.model.eval()
        device = self._resolve_inference_device()
        x = torch.from_numpy(self.data.x_graph).to(device)
        adjs = [scipy_to_torch_sparse(a, device) for a in self.graphs.normalized]

        # Use detached weights on the selected inference device. This permits CPU
        # layer-wise global inference even while training mini-batches on CUDA.
        outputs = []
        current = x
        for layer, adj in zip(
            [self.model.gnn.gnn1, self.model.gnn.gnn2, self.model.gnn.gnn3, self.model.gnn.gnn4],
            adjs,
        ):
            weight = layer.weight.detach().to(device)
            support = torch.tanh(F.linear(current, weight))
            current = torch.sparse.mm(adj, support)
            outputs.append(current.cpu())
        return tuple(outputs)  # type: ignore[return-value]

    @torch.no_grad()
    def encode_all(self) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        g1, g2, g3, g4 = self._exact_topology_all()
        latent_parts: List[torch.Tensor] = []
        topo_parts: List[torch.Tensor] = []
        bs = self.config.attr_batch_size
        for start in range(0, self.data.n_cells, bs):
            end = min(start + bs, self.data.n_cells)
            x = torch.from_numpy(self.data.x_attr[start:end]).to(self.device)
            z = self.model.encode_attribute(
                x,
                g1[start:end].to(self.device),
                g2[start:end].to(self.device),
                g3[start:end].to(self.device),
                add_noise=False,
            )
            latent_parts.append(z.cpu())
            topo_parts.append(g4[start:end])
        return torch.cat(latent_parts).numpy(), torch.cat(topo_parts).numpy()

    @torch.no_grad()
    def _q_and_target_all(self) -> Tuple[np.ndarray, torch.Tensor]:
        latent, _ = self.encode_all()
        q_parts = []
        bs = self.config.attr_batch_size
        for start in range(0, latent.shape[0], bs):
            z = torch.from_numpy(latent[start : start + bs]).to(self.device)
            q_parts.append(self.model.soft_assign(z).cpu())
        q = torch.cat(q_parts)
        p = self.model.target_distribution(q)
        return latent, p

    def initialize_clustering(self) -> np.ndarray:
        latent, _ = self.encode_all()
        kmeans = KMeans(
            n_clusters=self.model.n_clusters,
            init="k-means++",
            n_init=20,
            random_state=self.config.seed,
        )
        labels = kmeans.fit_predict(latent)
        centers = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(self.device)
        self.model.initialize_cluster_centers(centers)
        print("Initialized cluster centers with k-means++ on exact global embeddings")
        return labels

    def fit(self) -> TrainResult:
        y_pred_last = self.initialize_clustering()
        optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.config.train_lr, rho=0.9)
        p_full: Optional[torch.Tensor] = None
        final_metrics: Dict[str, float] = {}

        for epoch in range(self.config.train_epochs):
            if epoch % self.config.update_interval == 0 or p_full is None:
                latent_eval, p_full = self._q_and_target_all()
                with torch.no_grad():
                    q_parts = []
                    bs = self.config.attr_batch_size
                    for start in range(0, latent_eval.shape[0], bs):
                        z = torch.from_numpy(latent_eval[start : start + bs]).to(self.device)
                        q_parts.append(self.model.soft_assign(z).cpu())
                    q_full = torch.cat(q_parts)
                y_pred = q_full.argmax(dim=1).numpy()
                final_metrics = evaluate_clustering(self.data.labels, y_pred)
                metric_text = ", ".join(f"{k}={v:.4f}" for k, v in final_metrics.items())
                print(f"Epoch {epoch:4d} clustering: {metric_text or 'labels unavailable'}")
                delta = float(np.mean(y_pred != y_pred_last))
                y_pred_last = y_pred.copy()
                if epoch > 0 and delta < self.config.tol:
                    print(f"Early stopping: label change {delta:.6f} < tol {self.config.tol}")
                    break

            self.model.train()
            totals = {"loss": 0.0, "cluster": 0.0, "zinb": 0.0, "graph": 0.0, "align": 0.0}
            n_steps = 0
            rng = np.random.default_rng(self.config.seed + 100000 + epoch)
            assert p_full is not None
            for core_nodes in self._iter_core_parts(epoch):
                optimizer.zero_grad(set_to_none=True)
                z, g3_sub, topology_logits, raw_sub = self._batch_forward(core_nodes, add_noise=True)
                q = self.model.soft_assign(z)
                p = p_full[core_nodes].to(self.device)
                cluster_loss = F.kl_div(torch.log(q + 1e-8), p, reduction="batchmean")

                mean, disp, pi = self.model.decode_zinb(z)
                x_raw, sf = self._zinb_inputs(core_nodes)
                zinb = self.model.zinb_loss(x_raw, mean, disp, pi, sf)
                graph_loss = _sample_graph_reconstruction_loss(
                    g3_sub,
                    raw_sub,
                    self.config.negative_ratio,
                    self.config.max_positive_edges,
                    rng,
                )
                align = F.kl_div(
                    F.log_softmax(topology_logits, dim=1),
                    p,
                    reduction="batchmean",
                )
                loss = (
                    self.config.cluster_weight * cluster_loss
                    + self.config.zinb_weight * zinb
                    + self.config.graph_weight * graph_loss
                    + self.config.align_weight * align
                )
                loss.backward()
                optimizer.step()

                totals["loss"] += float(loss.detach())
                totals["cluster"] += float(cluster_loss.detach())
                totals["zinb"] += float(zinb.detach())
                totals["graph"] += float(graph_loss.detach())
                totals["align"] += float(align.detach())
                n_steps += 1

            row = {"epoch": float(epoch), **{k: v / n_steps for k, v in totals.items()}}
            row.update(final_metrics)
            self.history.append(row)
            if (epoch + 1) % self.config.log_interval == 0 or epoch == 0:
                print(
                    f"Train {epoch + 1:4d}/{self.config.train_epochs}: "
                    + ", ".join(f"{k}={row[k]:.6f}" for k in ["loss", "cluster", "zinb", "graph", "align"])
                )

        latent, _ = self.encode_all()
        q_parts = []
        for start in range(0, latent.shape[0], self.config.attr_batch_size):
            z = torch.from_numpy(latent[start : start + self.config.attr_batch_size]).to(self.device)
            q_parts.append(self.model.soft_assign(z).detach().cpu())
        labels = torch.cat(q_parts).argmax(dim=1).numpy()
        final_metrics = evaluate_clustering(self.data.labels, labels)
        self._save_outputs(latent, labels, final_metrics)
        return TrainResult(labels, latent, final_metrics, self.history)

    def _save_outputs(
        self,
        latent: np.ndarray,
        labels: np.ndarray,
        metrics: Dict[str, float],
    ) -> None:
        np.savetxt(self.output_dir / "latent.csv", latent, delimiter=",")
        np.savetxt(self.output_dir / "pred_labels.csv", labels, fmt="%d", delimiter=",")
        (self.output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        if self.history:
            pd.DataFrame(self.history).to_csv(self.output_dir / "history.csv", index=False)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "n_clusters": self.model.n_clusters,
            },
            self.output_dir / "final_model.pt",
        )
        print(f"Saved final outputs to {self.output_dir}")
