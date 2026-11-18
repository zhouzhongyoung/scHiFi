# Benchmark datasets

The release includes the exact supplied HDF5 benchmark files. Study accessions
identify the original public studies; the release files define the processed
inputs used here. QX_Limb_Muscle and QX_Spleen are separate tissue datasets
from the same Tabula Muris study.

| Dataset | Cells | Genes | Classes | Accession | Study |
| --- | ---: | ---: | ---: | --- | --- |
| Klein | 2,717 | 24,047 | 4 | [GSE65525](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65525) | [Article](https://doi.org/10.1016/j.cell.2015.04.044) |
| Adam | 3,660 | 23,797 | 8 | [GSE94333](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE94333) | [Article](https://doi.org/10.1242/dev.151142) |
| mouse_bladder_cell | 2,746 | 20,670 | 16 | [GSE108097](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE108097) | [Article](https://doi.org/10.1016/j.cell.2018.02.001) |
| Zeisel | 3,005 | 19,972 | 9 | [GSE60361](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE60361) | [Article](https://doi.org/10.1126/science.aaa1934) |
| 10X_PBMC | 4,271 | 16,653 | 8 | [SRP073767](https://www.ncbi.nlm.nih.gov/sra/?term=SRP073767) | [Article](https://doi.org/10.1038/ncomms14049) |
| QX_Limb_Muscle | 3,909 | 23,341 | 6 | [GSE109774](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109774) | [Article](https://doi.org/10.1038/s41586-018-0590-4) |
| Tosches_turtle | 18,664 | 23,500 | 15 | [PRJNA408230](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA408230) | [Article](https://doi.org/10.1126/science.aar4237) |
| Plasschaert | 6,977 | 28,205 | 8 | [GSE102580](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE102580) | [Article](https://doi.org/10.1038/s41586-018-0394-6) |
| Chen | 12,089 | 23,284 | 46 | [GSE87544](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87544) | [Article](https://doi.org/10.1016/j.celrep.2017.03.004) |
| QX_Spleen | 9,552 | 23,341 | 5 | [GSE109774](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109774) | [Article](https://doi.org/10.1038/s41586-018-0590-4) |
| Romanov | 2,881 | 21,143 | 7 | [GSE74672](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE74672) | [Article](https://doi.org/10.1038/nn.4462) |
| Wang_Lung | 9,519 | 14,561 | 2 | [GSE106960](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE106960) | [Article](https://doi.org/10.1073/pnas.1719474115) |
| Bach | 23,184 | 19,965 | 8 | [GSE106273](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE106273) | [Article](https://doi.org/10.1038/s41467-017-02001-5) |

## File format

Each HDF5 input, such as `data/Klein/data.h5`, uses one of the two layouts accepted by
`schifi.data.load_dataset`:

* `X` and `Y`: cell-by-gene expression and reference labels;
* `exprs`, `obs`, `obs_names`, `var_names`: expression, cell annotations and identifiers.

`exprs` may be a dense dataset or a CSR group containing `data`, `indices`,
`indptr` and `shape`. `cells.csv` and `genes.csv` retain available annotation.
For files without gene symbols, `genes.csv` retains the input column indices.

## Preprocessing

Remove cells and genes with zero total expression. Normalize each cell to the
median library total within its dataset, then apply log1p. The attribute
branch uses gene-centered values divided by the sample standard deviation;
standard deviations below 1e-6 are replaced by 1. The graph branch uses the
log-normalized expression of 2000 highly variable genes, selected by
mean-bin-normalized dispersion with 20 mean bins. The reconstruction target
is the filtered supplied expression, with library-size factors retained.

No additional gene-count or mitochondrial-percentage cutoff and no batch
correction are applied at this model-input stage. Wang_Lung contains
fractional supplied expression and is retained without rounding. Its
log-gamma reconstruction expression serves as an optimization surrogate.

Downstream marker and differential-expression analyses separately use
library totals of 10,000 followed by log1p.
