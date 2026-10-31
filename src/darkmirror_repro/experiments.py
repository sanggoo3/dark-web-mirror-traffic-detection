"""HPO, validation-only decisions, then locked singleton test evaluation."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .engine import PaperEngine
from .engine_core import calculate, choose_threshold
from .io import ROOT, KINDS, SEEDS, csv, parts, read_json, save_json, sha


def tune(out, root=ROOT, kinds=None, models=None, folds=None):
    root, out = Path(root), Path(out)
    spec = read_json(root / 'configs/experiment.json')
    engine = PaperEngine(root, out)
    lock_path = out / 'selected_configurations.json'
    choices = read_json(lock_path) if lock_path.exists() else {}
    records = []
    for kind in kinds or KINDS:
        for fold in folds or range(1, 6):
            for model in models or spec['models']:
                trials = []
                for config in spec['grids'][model]:
                    for inner in range(1, 4):
                        p = parts(root, kind, fold, inner)
                        for seed in spec['pilot_seeds']:
                            directory = out / 'tuning' / kind / str(fold) / spec['model_keys'][model] / config['config_id'] / str(inner) / str(seed)
                            result = engine.fit(model, config, seed, p['fit'], p['validation'], directory)
                            trials.append(dict(split=kind, protocol='cv', fold=fold, Model=model,
                                               config_id=config['config_id'], inner_fold=inner, seed=seed,
                                               validation_auc=result['validation_auc'], fit_auc=result['fit_auc'],
                                               epochs_run=result['epochs_run'], seconds=result['seconds']))
                            print('HPO', kind, fold, model, config['config_id'], inner, seed, flush=True)
                rank = pd.DataFrame(trials).groupby('config_id').validation_auc.agg(['mean', 'std']).reset_index()
                winner = rank.sort_values(['mean', 'std', 'config_id'], ascending=[False, True, True]).iloc[0]
                config = next(c for c in spec['grids'][model] if c['config_id'] == winner.config_id)
                choices[f'{kind}/cv_{fold}/{model}'] = dict(configuration=config,
                    mean_validation_auc=float(winner['mean']), sd_validation_auc=float(winner['std']), selection_input_length=526)
                save_json(lock_path, choices)
                records.extend(trials)
    # Full immutable per-fit outputs remain available even when this invocation is filtered.
    path = out / 'hpo_trials.csv'
    combined = pd.concat(([csv(path)] if path.exists() else []) + [pd.DataFrame(records)], ignore_index=True)
    combined.drop_duplicates(['split', 'fold', 'Model', 'config_id', 'inner_fold', 'seed'], keep='last').to_csv(path, index=False)


def paper_conditions(kind, model):
    result = ['included_526']
    if kind == 'session_stratified':
        if model == 'TCN':
            result += ['included_137', 'included_300', 'included_905', 'included_1276',
                       'post506_control', 'only20_padded526']
        if model in ['TCN', 'ResNet1D']:
            result += ['light_fusion526']
    return result


def train(out, root=ROOT, recorded_choices=False, kinds=None, models=None, folds=None, seeds=None):
    root, out = Path(root), Path(out)
    spec = read_json(root / 'configs/experiment.json')
    source = root / 'configs/selected_configurations.json' if recorded_choices else out / 'selected_configurations.json'
    choices = read_json(source)
    engine = PaperEngine(root, out)
    path = out / 'test_lock.json'
    existing = read_json(path)['jobs'] if path.exists() else []
    jobs = {(j['split'], j['fold'], j['Model'], j['Condition'], j['seed']): j for j in existing}
    for kind in kinds or KINDS:
        for fold in folds or range(1, 6):
            p = parts(root, kind, fold)
            for model in models or spec['models']:
                config = choices[f'{kind}/cv_{fold}/{model}']['configuration']
                for condition in paper_conditions(kind, model):
                    engine.configure(condition, p['fit'])
                    for seed in seeds or SEEDS:
                        directory = out / 'evaluation' / kind / str(fold) / spec['model_keys'][model] / condition / str(seed)
                        engine.fit(model, config, seed, p['fit'], p['validation'], directory, condition)
                        validation = csv(directory / 'validation_scores.csv')
                        threshold, search = choose_threshold(validation.y_true, validation.probability)
                        search.to_csv(directory / 'threshold_search.csv', index=False)
                        if engine.scaler is not None:
                            save_json(directory / 'feature_scaler.json', dict(mean=engine.scaler.mean_.tolist(),
                                      scale=engine.scaler.scale_.tolist(), n_samples_seen=int(engine.scaler.n_samples_seen_)))
                        checkpoint = directory / ('model.joblib' if model == 'ROCKET-style' else 'model.pt')
                        job = dict(split=kind, fold=fold, Model=model, Condition=condition, seed=seed, config=config,
                                   folder=directory.relative_to(out).as_posix(), threshold=threshold,
                                   checkpoint_sha256=sha(checkpoint))
                        jobs[kind, fold, model, condition, seed] = job
                        print('TRAIN', kind, fold, model, condition, seed, flush=True)
    # No test score is computed before every requested training/threshold decision is frozen.
    save_json(path, dict(choices_sha256=sha(source), recorded_choices=recorded_choices, jobs=list(jobs.values())))


def evaluate(out, root=ROOT):
    out, root = Path(out), Path(root)
    lock_path = out / 'test_lock.json'
    digest = sha(lock_path)
    jobs = read_json(lock_path)['jobs']
    engine = PaperEngine(root, out)
    records, predictions = [], {}
    spec = read_json(root / 'configs/experiment.json')
    for job in jobs:
        kind, model, condition = job['split'], job['Model'], job['Condition']
        p = parts(root, kind, job['fold'])
        engine.configure(condition, p['fit'])
        directory = out / job['folder']
        checkpoint = directory / ('model.joblib' if model == 'ROCKET-style' else 'model.pt')
        assert sha(checkpoint) == job['checkpoint_sha256']
        probability, loaded = engine.runtime(model, job['config'], directory)
        scores = np.asarray([probability(engine.packed[i:i + 1]) for i in p['test']])
        frame = pd.DataFrame(dict(index=p['test'], group=engine.groups[p['test']], y_true=engine.y[p['test']],
            probability=scores, threshold=job['threshold'], y_pred=(scores >= job['threshold']).astype(int),
            fold=job['fold'], seed=job['seed']))
        predictions.setdefault((kind, model, condition), []).append(frame)
        records.append(dict(split=kind, protocol='cv', fold=job['fold'], seed=job['seed'], Method=model,
                            Condition=condition, Threshold=job['threshold'], **calculate(frame.y_true, scores, job['threshold'])))
        del probability, loaded
        torch.cuda.empty_cache()
        print('TEST', kind, job['fold'], model, condition, job['seed'], flush=True)
    assert sha(lock_path) == digest
    for (kind, model, condition), frames in predictions.items():
        directory = out / 'predictions' / kind
        directory.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(directory / f'{spec["model_keys"][model]}__{condition}.csv.gz', index=False)
    pd.DataFrame(records).to_csv(out / 'run_metrics.csv', index=False)
    save_json(out / 'evaluation_identity.json', dict(test_lock_sha256=digest, evaluated_jobs=len(jobs), batch_size=1))
