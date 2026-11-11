"""Download version-pinned public PhysioNet files and verify official SHA-256 sums.

SDDB uses only records with audited .atr annotations. Unaudited .ari files
are never substituted as labels. Data licences remain those of PhysioNet.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

DBS = ('mitdb', 'incartdb', 'svdb', 'sddb')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def fetch(url, path, expected=None):
    if path.exists() and expected and digest(path) == expected: return
    temporary = path.with_suffix(path.suffix + '.part')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as target:
                while block := response.read(8*1024*1024): target.write(block)
            if expected and digest(temporary) != expected:
                raise ValueError('Official checksum mismatch: ' + path.name)
            temporary.replace(path)
            return
        except Exception:
            if attempt == 2: raise


def download(db, out='data', workers=4):
    root = Path(out)/db; root.mkdir(parents=True, exist_ok=True)
    base = f'https://physionet.org/files/{db}/1.0.0/'
    fetch(base+'SHA256SUMS.txt', root/'SHA256SUMS.txt')
    checksums = {}
    for line in (root/'SHA256SUMS.txt').read_text().splitlines():
        value, name = line.split(maxsplit=1); checksums[name.lstrip('*')] = value
    fetch(base+'RECORDS', root/'RECORDS', checksums.get('RECORDS'))
    records = (root/'RECORDS').read_text().split()
    if any('/' in r or '..' in r for r in records): raise ValueError('Unexpected record path')
    selected = [r for r in records if db != 'sddb' or r+'.atr' in checksums]
    excluded = [r for r in records if r not in selected]
    names = [r+ext for r in selected for ext in ('.hea', '.dat', '.atr')]
    for name in names:
        if name not in checksums: raise ValueError('Missing official checksum: '+name)
    def get(name):
        fetch(base+name, root/name, checksums[name]); return name
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, name in enumerate(pool.map(get, names), 1):
            if i % 15 == 0 or i == len(names): print(db, i, '/', len(names), flush=True)
    report = {'database':db,'version':'1.0.0','source':base,'records':selected,
              'excluded_records':excluded,'exclusion_reason':'No audited .atr annotation' if excluded else None,
              'all_downloaded_files_match_official_sha256':True,
              'files':{name:checksums[name] for name in names}}
    (root/'download_provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--db', choices=DBS, required=True); ap.add_argument('--out', default='data')
    ap.add_argument('--workers', type=int, default=4); a=ap.parse_args()
    download(a.db, a.out, a.workers)
