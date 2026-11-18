# scHiFi

Progressive structure-attribute fusion for single-cell RNA-seq clustering.
scHiFi combines expression reconstruction, multi-scale cell graphs, progressive
feature fusion and cluster-level cross-view alignment.

## Installation

Python 3.9 is used for the pinned environment. The PyTorch build in
`requirements.txt` uses CUDA 11.8. CUDA execution requires a compatible NVIDIA
GPU and driver; CPU execution is available with `--device cpu`.

```bash
conda env create -f environment.yml
conda activate schifi
```

Install the additional dependencies for the biological analyses and figures:

```bash
python -m pip install -r requirements-analysis.txt
```

## Data and checkpoint

The [v1.0.2 release](https://github.com/zhouzhongyoung/scHiFi/releases/tag/v1.0.2) contains:

* `scHiFi_datasets_13.tar.gz.part01` through `.part13`: all 13 benchmark
  expression matrices and annotations, stored as checksum-verified archive parts;
* `scHiFi_Klein_checkpoint_1111.tar.gz`: the saved Klein model, configuration,
  predictions, embedding and training history;
* `SHA256SUMS.txt`: checksums for the release attachments.

Download the attachments, assemble the dataset archive and extract it into
this directory. The script performs these steps automatically:

```bash
python scripts/download_data.py
```

`metadata/datasets.csv` records the original-study accessions, dimensions and
checksums of the supplied benchmark files. Data layout and preprocessing are
described in [docs/DATASETS.md](docs/DATASETS.md). Reference labels are used to
specify the cluster count for benchmark comparison and to calculate evaluation
metrics; they are not training targets.

## Training

Train Klein with the benchmark configuration:

```bash
python scripts/train.py --dataset Klein --device cuda
```

Train every benchmark dataset:

```bash
python scripts/train.py --all --device cuda
```

The direct entry provides all model and optimization options:

```bash
python main.py --dataset Klein --n_clusters 4 --train_mode full --device cuda
```

The three largest datasets (Tosches_turtle, Chen and Bach) use batch-local graphs
in the supplied benchmark configurations. The remaining datasets use global
graphs and full-batch topology propagation. Global graph partitioning is also
available through `--train_mode cluster`.

Each run saves `final_model.pt`, `latent.csv`, `pred_labels.csv`, `metrics.json`
and `history.csv`. The checkpoint contains `model_state` and `n_clusters`.
The `--pretrained` option expects a representation-pretraining checkpoint, not
the bundled final Klein clustering checkpoint. Evaluation of saved predictions:

```bash
python scripts/evaluate.py --dataset Klein --predictions checkpoints/Klein/seed_1111/predictions.npy
```

See [docs/PARAMETERS.md](docs/PARAMETERS.md) for defaults and
`configs/benchmarks.json` for dataset-specific settings.

## Analyses and figures

`analysis/qx` contains differential expression, GO/KEGG enrichment, the statistical
inputs and the plotting scripts for Figures 5 and 6. `analysis/performance`
contains the means and plotting scripts for Figure 2. `analysis/biology.py`
calculates marker consistency and class-level precision, recall and F1 from the
provided cell groups. Commands and input formats are in
[analysis/README.md](analysis/README.md).

## Comparison methods

The eleven baseline repositories are listed in
[docs/BASELINES.md](docs/BASELINES.md) and `metadata/baselines.csv`.

## License

Original scHiFi code is distributed under the MIT License. Benchmark data retain
the terms of their original studies and repositories; the software license does
not relicense these data. The original studies are identified in the dataset
table. Baseline implementations are distributed by their respective authors.
