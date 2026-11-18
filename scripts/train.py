"""Train scHiFi with the benchmark configurations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    configs = json.loads((root / 'configs/benchmarks.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--dataset', choices=list(configs))
    selection.add_argument('--all', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--output', type=Path, default=root / 'results')
    args = parser.parse_args()
    for name in configs if args.all else [args.dataset]:
        config = configs[name]
        directory = args.output.resolve() / name / f'seed_{args.seed}'
        if directory.exists():
            raise FileExistsError(f'Output directory already exists: {directory}')
        directory.mkdir(parents=True)
        command = [sys.executable, str(root / 'main.py'), '--dataset', name,
                   '--device', args.device, '--seed', str(args.seed),
                   '--output_dir', str(directory), '--pretrained_output', str(directory / 'pretrained.pt'),
                   '--graph_cache_dir', str(directory / 'graph_cache'), '--rebuild_graph']
        for key, value in config.items():
            if key == 'data_file':
                value = str(root / value)
            command.append('--' + key)
            command.extend(str(x) for x in value) if isinstance(value, list) else command.append(str(value))
        (directory / 'configuration.json').write_text(json.dumps(dict(dataset=name, seed=args.seed, **config), indent=2))
        # Set the initialization state before constructing the model.
        bootstrap = (
            'import runpy,sys,random,numpy as np,torch; '
            f'seed={args.seed}; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); '
            'torch.cuda.manual_seed_all(seed); '
            'torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False; '
            'torch.use_deterministic_algorithms(True,warn_only=True); '
            'sys.path.insert(0,sys.argv[1]); sys.argv=sys.argv[2:]; '
            'runpy.run_path(sys.argv[0],run_name="__main__")'
        )
        subprocess.run([sys.executable, '-c', bootstrap, str(root), *command[1:]], cwd=root, check=True)


if __name__ == '__main__':
    main()
