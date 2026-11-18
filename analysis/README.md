# Analyses and figures

Run these commands from the repository root after installing
`requirements-analysis.txt` and extracting the dataset attachment.

## Figures 5 and 6

The included statistical tables use the saved scHiFi-predicted QX Limb Muscle
clusters. Reference annotations enter only the cluster-to-type matching step.
Differential expression uses predicted clusters versus the remaining cells,
tie-corrected Wilcoxon tests and Benjamini-Hochberg correction. GO and KEGG
overrepresentation uses the expressed and annotated gene background for each
library. Database snapshots and the resulting term-to-gene sets are included.

```bash
python analysis/qx/analyze.py
python analysis/qx/figures.py
```

The plotting command can also use the included `analysis/qx/statistics` tables
directly. It writes PNG, PDF and editable SVG files to `analysis/qx/figures`.
GO annotations and KEGG reference files retain their upstream terms of use.
Database download locations are recorded in `analysis/qx/databases/sources.csv`.

## Figure 2

```bash
python analysis/performance/figure2.py
```

This plots the supplied benchmark means without running model training.

## Marker consistency and rare-cell recognition

```bash
python analysis/biology.py
```

`analysis/inputs/groups` contains cell-level predictions for the four-method
Adam, Plasschaert and bladder comparisons and the QX scHiFi analysis. The marker
panel includes its literature references. For each dataset, the calculation uses
the intersection of cell indices across methods and one-to-one Hungarian
matching between clusters and reference categories. Marker scores average the
within-gene percentile ranks of the two marker genes for each cell type. AUROC
contrasts the matched cluster with other cells. Rare-cell precision, recall and
F1 are calculated per reference category; macro averages use categories with
at most 5% of bladder cells. Outputs are written to
`analysis/biological_outputs`. Existing marker summaries are also included in
`analysis/marker_results`.
