from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from .data import ProcessedData, take_rows
from .local_graph import build_local_batch_graph, graph_to_torch
from .metrics import evaluate_clustering
from .model import ScHiFi


@dataclass
class LocalTrainerConfig:
    device: str = "cuda"
    inference_device: str = "auto"
    local_batch_size: int = 256
    k_list: Tuple[int, int, int, int] = (10, 15, 20, 25)
    pca_dim: int = 50
    metric: str = "euclidean"
    pretrain_epochs: int = 400
    train_epochs: int = 150
    pretrain_lr: float = 1e-3
    train_lr: float = 0.5
    update_interval: int = 1
    tol: float = 1e-3
    cluster_weight: float = 0.5
    zinb_weight: float = 1.0
    graph_weight: float = 0.1
    align_weight: float = 0.1
    seed: int = 1111
    log_interval: int = 10
    cache_local_graphs: bool = True


@dataclass
class LocalTrainResult:
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


class BatchLocalScHiFiTrainer:
    """Batch-local KNN version built on top of the v3.1-compatible model.

    The global ``full`` and ``cluster`` modes still use ``ScHiFiTrainer``. This
    class is called only when ``--train_mode local`` is selected.
    """

    def __init__(
        self,
        model: ScHiFi,
        data: ProcessedData,
        config: LocalTrainerConfig,
        output_dir: Union[str, Path],
    ):
        _set_seed(config.seed)
        self.model = model
        self.data = data
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA is unavailable; falling back to CPU.")
            config.device = "cpu"
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.history: List[Dict[str, float]] = []
        self._graph_cache: Dict[Tuple[int, int], object] = {}
        print(f"Training mode: local; local_batch_size={config.local_batch_size}")

    def _iter_batches(self, epoch: int, shuffle: bool) -> Iterable[np.ndarray]:
        n = self.data.n_cells
        starts = np.arange(0, n, self.config.local_batch_size)
        if shuffle:
            # The earliest batch-local code used contiguous batches. We keep
            # deterministic contiguous batches by default to make results easier
            # to reproduce, so this branch is currently not used.
            rng = np.random.default_rng(self.config.seed + epoch)
            rng.shuffle(starts)
        for start in starts:
            end = min(int(start) + self.config.local_batch_size, n)
            yield np.arange(int(start), end, dtype=np.int64)

    def _get_local_graph(self, indices: np.ndarray):
        key = (int(indices[0]), int(indices[-1]) + 1)
        if self.config.cache_local_graphs and key in self._graph_cache:
            return self._graph_cache[key]
        graph = build_local_batch_graph(
            self.data.x_graph[indices],
            k_list=self.config.k_list,
            pca_dim=self.config.pca_dim,
            metric=self.config.metric,
        )
        if self.config.cache_local_graphs:
            self._graph_cache[key] = graph
        return graph

    def _batch_forward(self, indices: np.ndarray, add_noise: bool):
        graph = self._get_local_graph(indices)
        adjs, dense_adj0 = graph_to_torch(graph, self.device)
        x_graph = torch.from_numpy(self.data.x_graph[indices]).to(self.device)
        x_attr = torch.from_numpy(self.data.x_attr[indices]).to(self.device)
        g1, g2, g3, g4 = self.model.encode_topology(x_graph, adjs)
        z = self.model.encode_attribute(x_attr, g1, g2, g3, add_noise=add_noise)
        return z, g3, g4, dense_adj0

    def _zinb_inputs(self, indices: np.ndarray):
        raw = torch.from_numpy(take_rows(self.data.raw_counts, indices)).to(self.device)
        sf = torch.from_numpy(self.data.size_factors[indices]).to(self.device)
        return raw, sf

    def _dense_graph_mse(self, g3: torch.Tensor, dense_adj0: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(torch.mm(g3, g3.t()))
        return F.mse_loss(pred, dense_adj0)

    def pretrain(self, checkpoint_path: Union[str, Path]) -> None:
        checkpoint_path = Path(checkpoint_path)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.pretrain_lr, amsgrad=True)
        print("Starting local-batch ZINB + dense-MSE graph pretraining")
        for epoch in range(self.config.pretrain_epochs):
            self.model.train()
            total_loss = total_zinb = total_graph = 0.0
            n_steps = 0
            for indices in self._iter_batches(epoch, shuffle=False):
                optimizer.zero_grad(set_to_none=True)
                z, g3, _, dense_adj0 = self._batch_forward(indices, add_noise=True)
                mean, disp, pi = self.model.decode_zinb(z)
                x_raw, sf = self._zinb_inputs(indices)
                zinb = self.model.zinb_loss(x_raw, mean, disp, pi, sf)
                graph_loss = self._dense_graph_mse(g3, dense_adj0)
                loss = self.config.zinb_weight * zinb + self.config.graph_weight * graph_loss
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())
                total_zinb += float(zinb.detach())
                total_graph += float(graph_loss.detach())
                n_steps += 1
            if (epoch + 1) % self.config.log_interval == 0 or epoch == 0:
                print(
                    f"Local pretrain {epoch + 1:4d}/{self.config.pretrain_epochs}: "
                    f"loss={total_loss/n_steps:.6f}, zinb={total_zinb/n_steps:.6f}, "
                    f"graph={total_graph/n_steps:.6f}"
                )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self.model.state_dict()}, checkpoint_path)
        print(f"Saved local pretrained model: {checkpoint_path}")

    def load_pretrained(self, checkpoint_path: Union[str, Path]) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state = checkpoint.get("model_state", checkpoint.get("ae_state_dict", checkpoint))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        missing = [key for key in missing if key != "mu"]
        unexpected = [key for key in unexpected if key != "mu"]
        if missing or unexpected:
            print(f"Checkpoint loaded with missing={missing}, unexpected={unexpected}")
        else:
            print(f"Loaded local pretrained model: {checkpoint_path}")

    @torch.no_grad()
    def encode_all(self) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        latent_parts: List[torch.Tensor] = []
        topo_parts: List[torch.Tensor] = []
        for indices in self._iter_batches(epoch=0, shuffle=False):
            z, _, g4, _ = self._batch_forward(indices, add_noise=False)
            latent_parts.append(z.detach().cpu())
            topo_parts.append(g4.detach().cpu())
        return torch.cat(latent_parts).numpy(), torch.cat(topo_parts).numpy()

    @torch.no_grad()
    def _q_and_target_all(self) -> Tuple[np.ndarray, torch.Tensor]:
        latent, _ = self.encode_all()
        q_parts = []
        bs = self.config.local_batch_size
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
        print("Initialized cluster centers with k-means++ on local-batch embeddings")
        return labels

    def fit(self) -> LocalTrainResult:
        y_pred_last = self.initialize_clustering()
        optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.config.train_lr, rho=0.9)
        p_full: Optional[torch.Tensor] = None
        final_metrics: Dict[str, float] = {}
        for epoch in range(self.config.train_epochs):
            if epoch % self.config.update_interval == 0 or p_full is None:
                latent_eval, p_full = self._q_and_target_all()
                q_parts = []
                bs = self.config.local_batch_size
                with torch.no_grad():
                    for start in range(0, latent_eval.shape[0], bs):
                        z = torch.from_numpy(latent_eval[start : start + bs]).to(self.device)
                        q_parts.append(self.model.soft_assign(z).cpu())
                q_full = torch.cat(q_parts)
                y_pred = q_full.argmax(dim=1).numpy()
                final_metrics = evaluate_clustering(self.data.labels, y_pred)
                metric_text = ", ".join(f"{k}={v:.4f}" for k, v in final_metrics.items())
                print(f"Local epoch {epoch:4d} clustering: {metric_text or 'labels unavailable'}")
                delta = float(np.mean(y_pred != y_pred_last))
                y_pred_last = y_pred.copy()
                if epoch > 0 and delta < self.config.tol:
                    print(f"Early stopping: label change {delta:.6f} < tol {self.config.tol}")
                    break

            self.model.train()
            totals = {"loss": 0.0, "cluster": 0.0, "zinb": 0.0, "graph": 0.0, "align": 0.0}
            n_steps = 0
            assert p_full is not None
            for indices in self._iter_batches(epoch, shuffle=False):
                optimizer.zero_grad(set_to_none=True)
                z, g3, topology_logits, dense_adj0 = self._batch_forward(indices, add_noise=True)
                q = self.model.soft_assign(z)
                p = p_full[indices].to(self.device)
                cluster_loss = F.kl_div(torch.log(q + 1e-8), p, reduction="batchmean")
                mean, disp, pi = self.model.decode_zinb(z)
                x_raw, sf = self._zinb_inputs(indices)
                zinb = self.model.zinb_loss(x_raw, mean, disp, pi, sf)
                graph_loss = self._dense_graph_mse(g3, dense_adj0)
                align = F.kl_div(F.log_softmax(topology_logits, dim=1), p, reduction="batchmean")
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
                    f"Local train {epoch + 1:4d}/{self.config.train_epochs}: "
                    + ", ".join(f"{k}={row[k]:.6f}" for k in ["loss", "cluster", "zinb", "graph", "align"])
                )
        latent, _ = self.encode_all()
        q_parts = []
        for start in range(0, latent.shape[0], self.config.local_batch_size):
            z = torch.from_numpy(latent[start : start + self.config.local_batch_size]).to(self.device)
            q_parts.append(self.model.soft_assign(z).detach().cpu())
        labels = torch.cat(q_parts).argmax(dim=1).numpy()
        final_metrics = evaluate_clustering(self.data.labels, labels)
        self._save_outputs(latent, labels, final_metrics)
        return LocalTrainResult(labels, latent, final_metrics, self.history)

    def _save_outputs(self, latent: np.ndarray, labels: np.ndarray, metrics: Dict[str, float]) -> None:
        np.savetxt(self.output_dir / "latent.csv", latent, delimiter=",")
        np.savetxt(self.output_dir / "pred_labels.csv", labels, fmt="%d", delimiter=",")
        (self.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if self.history:
            pd.DataFrame(self.history).to_csv(self.output_dir / "history.csv", index=False)
        torch.save(
            {"model_state": self.model.state_dict(), "n_clusters": self.model.n_clusters},
            self.output_dir / "final_model.pt",
        )
        print(f"Saved final local outputs to {self.output_dir}")
