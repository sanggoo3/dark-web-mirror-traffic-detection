"""Optional fresh PCAPNG measurements; private captures must be provided separately."""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .direct_parser import read_directions
from .engine import PaperEngine
from .features import light_features
from .io import ROOT, csv, parts, read_json, save_json, sha
from .preprocessing import validate_direct
from .reproduce import latency_summary


def benchmark(out, private_manifest, root=ROOT, tshark='tshark'):
    out, root = Path(out), Path(root)
    jobs = read_json(out / 'test_lock.json')['jobs']
    manifest = csv(private_manifest).set_index('index', drop=False)
    samples = csv(root / 'data/metadata/latency_sample_assignments.csv')
    required = set(samples['index'])
    assert required <= set(manifest.index)
    # Verify all supplied captures against the dissector and public cache before timing.
    validate_direct(manifest.reset_index(drop=True), root, tshark)
    engine = PaperEngine(root, out)
    rows = []
    for job in jobs:
        kind, model, condition = job['split'], job['Model'], job['Condition']
        selected = samples[(samples.split == kind) & (samples.fold == job['fold'])]
        if kind == 'group_aware' and model not in ['TCN', 'ResNet1D']:
            continue
        if condition not in ['included_526', 'light_fusion526']:
            continue
        if condition == 'light_fusion526':
            selected = selected[selected.Role == 'test']
        p = parts(root, kind, job['fold'])
        engine.configure(condition, p['fit'])
        directory = out / job['folder']
        checkpoint = directory / ('model.joblib' if model == 'ROCKET-style' else 'model.pt')
        assert sha(checkpoint) == job['checkpoint_sha256']
        probability, loaded = engine.runtime(model, job['config'], directory)
        for _ in range(50):
            probability(engine.packed[p['fit'][0]:p['fit'][0] + 1])
        torch.cuda.synchronize()
        for sample in selected.itertuples():
            index = sample.index
            source = manifest.loc[index]
            for repeat in range(3):
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                direction, timestamps = read_directions(Path(source.source_path), source.client_ip, 526)
                parsed = time.perf_counter_ns()
                value = direction.astype(np.float32).reshape(1, 1, 526)
                if condition == 'light_fusion526':
                    light = light_features(direction, timestamps)
                    scaled = engine.scaler.transform(light[None, :]).astype(np.float32)
                    value = np.concatenate([value, scaled[:, None, :]], axis=2)
                packed = time.perf_counter_ns()
                score = probability(value)
                predicted = int(score >= job['threshold'])
                torch.cuda.synchronize()
                end = time.perf_counter_ns()
                assert np.array_equal(value, engine.packed[index:index + 1])
                rows.append(dict(split=kind, protocol='cv', fold=job['fold'], seed=job['seed'], Model=model,
                    Condition=condition, Role=sample.Role, index=index, group=str(engine.groups[index]), repeat=repeat,
                    Parse_ms=(parsed - start) / 1e6, Feature_and_pack_ms=(packed - parsed) / 1e6,
                    Preprocess_ms=(packed - start) / 1e6, Inference_ms=(end - packed) / 1e6, Total_ms=(end - start) / 1e6,
                    probability=score, y_pred=predicted))
        del probability, loaded
        torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'new_latency_calls.csv.gz', index=False)
    latency_summary(frame).to_csv(out / 'new_latency_summary.csv', index=False)
    save_json(out / 'new_latency_environment.json', dict(torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), batch=1, warmup=50, repeats=3,
        includes='File read, parse, packing, H2D, forward, D2H, class decision and final CUDA synchronization.',
        excludes='Loading, collection wait, validation assertions, result output; no OS cache flush.'))
