# Command-line parameters

Paths are resolved from the working directory. `--n_clusters 0` uses the number
of reference categories when labels exist; provide an explicit positive count
for unlabelled data. `full`, `cluster` and `local` select full global-graph,
partitioned global-graph and batch-local training, respectively. `auto` selects
global-graph training based on cell count.

| Option | Default | Description |
| --- | --- | --- |
| `--dataset` | `Klein` | Dataset identifier and output/cache prefix. |
| `--data_file` | `None` | Explicit expression HDF5 file; overrides the data directory lookup. |
| `--data_root` | `./data` | Directory containing the dataset subdirectories. |
| `--label_key` | `cell_type1` | Reference-label field in a legacy HDF5 obs group. |
| `--n_clusters` | `0` | Number of clusters; zero infers the count from reference labels. |
| `--n_hvg` | `2000` | Number of highly variable genes used by the graph branch. |
| `--ceil_noninteger` | `False` | Round supplied expression upward before preprocessing when enabled. |
| `--k_list` | `[10, 15, 20, 25]` | Four neighborhood sizes for the successive graph layers. |
| `--pca_dim` | `50` | PCA dimension for graph neighbor search. |
| `--metric` | `euclidean` | Distance metric for neighbor search. |
| `--graph_cache_dir` | `./graph_cache` | Directory for cached graph matrices and PCA coordinates. |
| `--rebuild_graph` | `False` | Rebuild graph files instead of reusing cache entries. |
| `--train_mode` | `auto` | Global full-batch, global partitioned, batch-local, or automatic selection. |
| `--local_batch_size` | `0` | Number of cells per local graph batch; zero uses the attribute batch size. |
| `--full_batch_threshold` | `6000` | Cell-count threshold used by automatic global-graph mode selection. |
| `--partition_method` | `auto` | Partitioning method for cluster mode; auto uses METIS when available. |
| `--cluster_size` | `1024` | Target number of core cells per graph partition. |
| `--halo_hops` | `1` | Number of neighborhood hops added around partition core cells. |
| `--max_halo_nodes` | `4096` | Maximum halo nodes; zero disables the cap. |
| `--encoder_dims` | `[512, 256, 128]` | Three expression/topology encoder widths. |
| `--decoder_dims` | `[128, 256]` | Two expression decoder widths. |
| `--z_dim` | `32` | Dimension of the fused latent representation. |
| `--sigma` | `1.5` | Gaussian noise standard deviation for the expression input. |
| `--alpha` | `1.0` | Student-t clustering kernel parameter. |
| `--pretrain_epochs` | `400` | Number of representation-pretraining epochs. |
| `--maxiter` | `150` | Maximum number of clustering-stage epochs. |
| `--pretrain_lr` | `0.001` | Pretraining learning rate. |
| `--lr_fit`, `--lr` | `0.5` | Clustering-stage learning rate. |
| `--update_interval` | `1` | Clustering target update interval in epochs. |
| `--tol` | `0.001` | Early-stopping threshold for the changed-label fraction. |
| `--attr_batch_size`, `--batch_size` | `256` | Attribute reconstruction mini-batch size. |
| `--cluster_weight` | `0.5` | Fused-view clustering loss coefficient. |
| `--zinb_weight` | `1.0` | Expression reconstruction loss coefficient. |
| `--graph_weight` | `0.1` | Graph reconstruction loss coefficient. |
| `--align_weight` | `0.1` | Cross-view alignment loss coefficient. |
| `--negative_ratio` | `1.0` | Negative-to-positive pair ratio for sampled global graph reconstruction. |
| `--max_positive_edges` | `20000` | Maximum positive edges sampled for global graph reconstruction. |
| `--device` | `cuda` | Training device, such as cuda or cpu. |
| `--inference_device` | `auto` | Device for inference; auto selects according to the training mode. |
| `--seed` | `1111` | Random initialization and sampling parameter. |
| `--log_interval` | `10` | Epoch interval for progress logging. |
| `--output_dir`, `--save_dir` | `None` | Directory for trained weights, predictions, embedding and metrics. |
| `--pretrained`, `--ae_weights` | `None` | Representation-pretraining checkpoint to load. |
| `--pretrained_output`, `--ae_weight_file` | `None` | Representation-pretraining checkpoint save/load location. |
| `--skip_pretrain` | `False` | Skip representation pretraining if no checkpoint is loaded. |
