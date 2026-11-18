"""Download and extract the versioned benchmark data and Klein checkpoint."""
import argparse
import hashlib
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://github.com/zhouzhongyoung/scHiFi/releases/download/v1.0.0/'
ASSETS = ['scHiFi_datasets_13.tar.gz', 'scHiFi_Klein_checkpoint_1111.tar.gz']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-only', action='store_true')
    parser.add_argument('--checkpoint-only', action='store_true')
    parser.add_argument('--keep-archives', action='store_true')
    args = parser.parse_args()
    if args.data_only and args.checkpoint_only:
        parser.error('Choose either --data-only or --checkpoint-only.')
    selected = ASSETS[:1] if args.data_only else ASSETS[1:] if args.checkpoint_only else ASSETS
    with urllib.request.urlopen(BASE + 'SHA256SUMS.txt') as f:
        hashes = dict(reversed(line.split()) for line in f.read().decode().splitlines() if line.strip())
    for name in selected:
        archive = ROOT / name
        if not archive.exists():
            print('Downloading ' + name, flush=True)
            urllib.request.urlretrieve(BASE + name, archive)
        if digest(archive) != hashes[name]:
            raise ValueError('Checksum mismatch: ' + name)
        with tarfile.open(archive, 'r:gz') as handle:
            for member in handle.getmembers():
                target = (ROOT / member.name).resolve()
                if ROOT.resolve() not in target.parents or member.issym() or member.islnk():
                    raise ValueError('Invalid archive member: ' + member.name)
            handle.extractall(ROOT)
        if not args.keep_archives:
            archive.unlink()
        print('Extracted ' + name, flush=True)


if __name__ == '__main__':
    main()
