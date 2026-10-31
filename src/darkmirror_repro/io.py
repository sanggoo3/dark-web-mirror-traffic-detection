from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METRICS = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
SEEDS = [11, 23, 71, 101, 202]
KINDS = ['group_aware', 'session_stratified']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def csv(path):
    return pd.read_csv(path, float_precision='round_trip')


def dataset(root=ROOT):
    with np.load(Path(root) / 'data/processed/direction_sequences.npz', allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def parts(root, kind, fold, inner=None):
    name = 'split_assignments.csv' if inner is None else 'inner_split_assignments.csv'
    frame = csv(Path(root) / 'data/splits' / kind / name)
    frame = frame[frame.fold == fold]
    if inner is not None:
        frame = frame[frame.inner_fold == inner]
    return {role: frame.loc[frame.split == role, 'index'].to_numpy(dtype=np.int64)
            for role in (['fit', 'validation', 'test'] if inner is None else ['fit', 'validation'])}


def sequence(data, condition):
    prefix = data['prefix']
    if condition == 'post506_control':
        return prefix[:, None, 20:526].astype(np.float32)
    if condition == 'only20_padded526':
        result = np.zeros((len(prefix), 1, 526), dtype=np.float32)
        result[:, 0, :20] = prefix[:, :20]
        return result
    if condition.startswith('included_'):
        length = int(condition.split('_')[1])
        if length not in [137, 300, 526, 905, 1276]:
            raise ValueError(condition)
        return np.ascontiguousarray(prefix[:, None, :length], dtype=np.float32)
    raise ValueError(condition)


def verify(root=ROOT):
    root = Path(root)
    manifest = read_json(root / 'release_manifest.json')
    for relative, digest in manifest['files'].items():
        if sha(root / relative) != digest:
            raise ValueError('Checksum mismatch: ' + relative)
    d = dataset(root)
    assert d['prefix'].shape == (4600, 1276)
    assert np.isin(d['prefix'], [-1, 0, 1]).all()
    assert np.isin(d['prefix'][:, :526], [-1, 1]).all()
    assert np.bincount(d['y']).tolist() == [2300, 2300]
    assert len(np.unique(d['groups'])) == 44
    for i, count in enumerate(d['retained_count']):
        stop = min(int(count), 1276)
        assert (d['prefix'][i, :stop] != 0).all() and (d['prefix'][i, stop:] == 0).all()
        assert np.isfinite(d['relative_times_s'][i, :stop]).all()
        assert np.isnan(d['relative_times_s'][i, stop:]).all()
    rows = []
    for kind in KINDS:
        tested = []
        for fold in range(1, 6):
            p = parts(root, kind, fold)
            assert np.array_equal(np.sort(np.concatenate(list(p.values()))), np.arange(4600))
            tested.extend(p['test'])
            for a, b in [('fit', 'validation'), ('fit', 'test'), ('validation', 'test')]:
                assert not set(p[a]) & set(p[b])
                if kind == 'group_aware':
                    assert not set(d['groups'][p[a]]) & set(d['groups'][p[b]])
            for inner in range(1, 4):
                q = parts(root, kind, fold, inner)
                assert set(np.concatenate(list(q.values()))) == set(p['fit'])
                assert not set(q['fit']) & set(q['validation'])
                if kind == 'group_aware':
                    assert not set(d['groups'][q['fit']]) & set(d['groups'][q['validation']])
            rows.append(dict(split=kind, fold=fold, **{k: len(v) for k, v in p.items()}))
        assert np.array_equal(np.sort(tested), np.arange(4600))
    return dict(status='passed', dataset_id=manifest['dataset_id'], checked_files=len(manifest['files']),
                sessions=4600, sites=46, domain_groups=44, split_counts=rows)
