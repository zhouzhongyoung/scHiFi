"""Calculate ACC, NMI, ARI and AMI from saved cell predictions."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from schifi.data import load_dataset, preprocess_counts
from schifi.metrics import evaluate_clustering


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = preprocess_counts(load_dataset(ROOT / 'data' / args.dataset / 'data.h5'), n_hvg=2000)
    if data.labels is None:
        raise ValueError('Reference annotations are required for evaluation.')
    prediction = np.load(args.predictions) if args.predictions.suffix == '.npy' else np.loadtxt(args.predictions, delimiter=',', dtype=int)
    prediction = np.asarray(prediction).reshape(-1)
    if len(prediction) != data.n_cells:
        raise ValueError('Prediction length does not match the retained cell count.')
    result = evaluate_clustering(data.labels, prediction)
    body = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body + '\n')
    print(body)


if __name__ == '__main__':
    main()
