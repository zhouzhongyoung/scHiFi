"""Marker-panel AUROC and class-level precision, recall and F1."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def match(truth, predicted):
    labels = np.unique(truth)
    clusters = np.unique(predicted)
    counts = pd.crosstab(truth, predicted).reindex(index=labels, columns=clusters, fill_value=0)
    rows, cols = linear_sum_assignment(-counts.to_numpy())
    mapping = {clusters[c]: labels[r] for r, c in zip(rows, cols)}
    return np.asarray([mapping.get(c, -1) for c in predicted]), mapping


def class_metrics(truth, matched, names):
    rows = []
    for label in np.unique(truth):
        support = int((truth == label).sum())
        n_pred = int((matched == label).sum())
        tp = int(((truth == label) & (matched == label)).sum())
        rows.append(dict(reference_label=int(label), reference_type=names[label],
                         reference_cells=support, predicted_cells=n_pred, true_positives=tp,
                         precision=tp / n_pred if n_pred else 0., recall=tp / support,
                         F1=2 * tp / (support + n_pred), fraction=support / len(truth)))
    return rows


def marker_expression(dataset, genes):
    with h5py.File(ROOT / 'data' / dataset / 'data.h5') as h:
        node = h['exprs']
        if isinstance(node, h5py.Group):
            x = sparse.csr_matrix((node['data'][:], node['indices'][:], node['indptr'][:]), shape=node['shape'][:])
        else:
            x = node[:]
        symbols = h['var_names'][:].astype(str)
        columns = pd.Index(symbols).get_indexer(genes)
        if (columns < 0).any():
            raise ValueError('A marker gene is absent from ' + dataset)
        totals = np.asarray(x.sum(axis=1)).ravel().astype(float)
        raw = x[:, columns].toarray() if sparse.issparse(x) else x[:, columns]
    return np.log1p(raw * (10000. / np.maximum(totals, 1.))[:, None])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis/biological_outputs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(ROOT / 'analysis/inputs/marker_panel.csv')
    class_rows, marker_rows, expression_rows = [], [], []
    for dataset in ['Adam', 'Plasschaert', 'mouse_bladder_cell']:
        files = sorted((ROOT / 'analysis/inputs/groups' / dataset).glob('*.csv'))
        groups = {p.stem: pd.read_csv(p) for p in files}
        common = sorted(set.intersection(*[set(g.input_row) for g in groups.values()]))
        annotation = pd.read_csv(ROOT / 'data' / dataset / 'cells.csv').set_index('input_row')
        truth = annotation.loc[common, 'reference_label'].to_numpy()
        names = annotation.groupby('reference_label').reference_type.first().to_dict()
        if dataset != 'mouse_bladder_cell':
            selected = panel[panel.dataset == dataset].reset_index(drop=True)
            genes = selected.gene.tolist()
            log = marker_expression(dataset, genes)[common]
            rounded = np.round(log, 12)
            ranks = np.column_stack([rankdata(rounded[:, j], method='average') for j in range(len(genes))])
            z = (log - log.mean(axis=0)) / np.where(log.std(axis=0) > 0, log.std(axis=0), 1.)
        for method, group in groups.items():
            predicted = group.set_index('input_row').loc[common, 'predicted_cluster'].to_numpy()
            matched, mapping = match(truth, predicted)
            for row in class_metrics(truth, matched, names):
                class_rows.append(dict(dataset=dataset, method=method, **row))
            if dataset == 'mouse_bladder_cell':
                continue
            for label in np.unique(truth):
                cell_type = names[label]
                columns = np.flatnonzero(selected.reference_type.to_numpy() == cell_type)
                scores = (ranks[:, columns].mean(axis=1) - .5) / len(common)
                mask = matched == label
                auc = float(roc_auc_score(mask, scores)) if mask.any() and (~mask).any() else np.nan
                marker_rows.append(dict(dataset=dataset, method=method, reference_label=int(label),
                                        reference_type=cell_type, marker_panel_AUROC=auc))
            for cluster in np.unique(predicted):
                mask = predicted == cluster
                label = mapping.get(cluster, -1)
                for j, gene in enumerate(genes):
                    expression_rows.append(dict(dataset=dataset, method=method, predicted_cluster=int(cluster),
                                                matched_reference_type=names.get(label, 'unmatched'), gene=gene,
                                                mean_log_expression=float(log[mask, j].mean()),
                                                mean_z_expression=float(z[mask, j].mean()),
                                                detection_fraction=float((log[mask, j] > 0).mean())))
    classes = pd.DataFrame(class_rows)
    classes.to_csv(args.output / 'class_metrics.csv', index=False)
    aucs = pd.DataFrame(marker_rows)
    aucs.to_csv(args.output / 'marker_panel_AUROC.csv', index=False)
    aucs.groupby(['dataset', 'method']).marker_panel_AUROC.mean().to_csv(args.output / 'marker_macro_AUROC.csv')
    pd.DataFrame(expression_rows).to_csv(args.output / 'cluster_marker_expression.csv', index=False)
    rare = classes[(classes.dataset == 'mouse_bladder_cell') & (classes.fraction <= .05)]
    rare.groupby('method')[['precision', 'recall', 'F1']].mean().to_csv(args.output / 'bladder_rare_macro.csv')


if __name__ == '__main__':
    main()
