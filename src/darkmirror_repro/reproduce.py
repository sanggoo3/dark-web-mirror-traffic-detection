"""Recalculate manuscript tables from released data and recorded probabilities."""
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .engine_core import calculate
from .features import light_features
from .io import ROOT, KINDS, METRICS, SEEDS, csv, dataset, parts, read_json, save_json
from .statistics import seed_statistics, cluster_plan, domain_draws, paired


def cutoff(y, probabilities):
    thresholds = np.round(np.arange(.1, .901, .01), 2)
    pred = np.asarray(probabilities)[:, None] >= thresholds
    truth = np.asarray(y)[:, None] == 1
    tp, fp, fn = (pred & truth).sum(0), (pred & ~truth).sum(0), (~pred & truth).sum(0)
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(thresholds)), where=denom > 0)
    accuracy = (pred == truth).mean(0)
    winner = sorted(range(len(thresholds)), key=lambda i: (-f1[i], -accuracy[i], thresholds[i]))[0]
    return float(thresholds[winner])


def audit_hpo(root):
    trials = csv(root / 'results/recorded/hpo_trials.csv.gz')
    choices = read_json(root / 'configs/selected_configurations.json')
    spec = read_json(root / 'configs/experiment.json')
    assert len(trials) == 7200
    for (kind, fold, model), block in trials.groupby(['split', 'fold', 'Model']):
        assert len(block) == 72 and block.config_id.nunique() == 12
        assert sorted(block.seed.unique()) == [11, 23]
        assert not block.duplicated(['config_id', 'seed', 'inner_fold']).any()
        rank = block.groupby('config_id').validation_auc.agg(['mean', 'std']).reset_index()
        winner = rank.sort_values(['mean', 'std', 'config_id'], ascending=[False, True, True]).iloc[0]
        choice = choices[f'{kind}/cv_{fold}/{model}']
        assert choice['configuration']['config_id'] == winner.config_id
        config = next(c for c in spec['grids'][model] if c['config_id'] == winner.config_id)
        assert choice['configuration'] == config
        assert abs(choice['mean_validation_auc'] - winner['mean']) < 1e-12
        assert abs(choice['sd_validation_auc'] - winner['std']) < 1e-12
    return choices


def audit_predictions(root, d):
    spec = read_json(root / 'configs/experiment.json')
    runs = csv(root / 'results/recorded/run_metrics.csv.gz')
    frames, recalculated = {}, []
    for (kind, model, condition), part in runs.groupby(['split', 'Method', 'Condition'], sort=False):
        name = f'{spec["model_keys"][model]}__{condition}.csv.gz'
        frame = csv(root / 'results/recorded/predictions' / kind / name)
        validation = csv(root / 'results/recorded/validation' / kind / name)
        assert len(frame) == 23000 and not frame.duplicated(['index', 'seed']).any()
        assert np.isfinite(frame.probability).all() and frame.probability.between(0, 1).all()
        assert np.array_equal(frame.y_true, d['y'][frame['index']])
        assert np.array_equal(frame.group, d['groups'][frame['index']])
        for row in part.itertuples():
            test = frame[(frame.fold == row.fold) & (frame.seed == row.seed)]
            val = validation[(validation.fold == row.fold) & (validation.seed == row.seed)]
            p = parts(root, kind, row.fold)
            assert np.array_equal(test['index'], p['test']) and np.array_equal(val['index'], p['validation'])
            assert np.array_equal(val.y_true, d['y'][val['index']])
            threshold = cutoff(val.y_true, val.probability)
            assert abs(threshold - row.Threshold) < 1e-12
            assert np.allclose(test.threshold, threshold, atol=1e-12, rtol=0)
            assert np.array_equal(test.y_pred, (test.probability >= threshold).astype(int))
            values = calculate(test.y_true, test.probability, threshold)
            assert all(abs(values[m] - getattr(row, m.replace('-', '_'), values[m])) < 1e-12
                       for m in ['Accuracy', 'Precision', 'Recall'])
            original = part[(part.fold == row.fold) & (part.seed == row.seed)].iloc[0]
            assert max(abs(values[m] - original[m]) for m in METRICS) < 1e-12
            recalculated.append(dict(split=kind, protocol='cv', fold=row.fold, seed=row.seed,
                                     Method=model, Condition=condition, Threshold=threshold, **values))
        frames[kind, model, condition] = frame
    assert len(recalculated) == 700
    with np.load(root / 'data/processed/light_statistics_526.npz') as features:
        recomputed = np.stack([light_features(q[:526], t[:526])
                              for q, t in zip(d['prefix'], d['relative_times_s'])])
        assert np.array_equal(recomputed, features['light'])
    return pd.DataFrame(recalculated), frames


def latency_summary(calls):
    metrics = ['Preprocess_ms', 'Inference_ms', 'Total_ms', 'Parse_ms', 'Feature_and_pack_ms']
    keys = ['split', 'protocol', 'fold', 'seed', 'Model', 'Condition', 'Role', 'index', 'group']
    assert np.allclose(calls.Preprocess_ms + calls.Inference_ms, calls.Total_ms, atol=1e-10, rtol=0)
    assert not calls.duplicated(keys + ['repeat']).any()
    assert (calls.groupby(keys).size() == 3).all()
    means = calls.groupby(keys, sort=False)[metrics].mean().reset_index()
    run_keys = ['split', 'protocol', 'fold', 'seed', 'Model', 'Condition', 'Role']
    per_run = means.groupby(run_keys)[metrics].mean().reset_index()
    seeds = per_run.groupby([k for k in run_keys if k != 'fold'])[metrics].mean().reset_index()
    rows = []
    for key, block in means.groupby(['split', 'protocol', 'Model', 'Condition', 'Role'], sort=False):
        kind, protocol, model, condition, role = key
        selected = seeds[(seeds.split == kind) & (seeds.Model == model) & (seeds.Condition == condition) & (seeds.Role == role)]
        row = dict(split=kind, protocol=protocol, Model=model, Condition=condition, Role=role,
                   Unique_sessions=block['index'].nunique(), Session_seed_observations=len(block),
                   Timed_calls=len(block) * 3, Batch=1, Repeats=3, Warmup=50)
        for metric in metrics:
            row[metric + '_mean'] = block[metric].mean()
            row[metric + '_SD_session_seed'] = block[metric].std(ddof=1)
            row[metric + '_p95'] = block[metric].quantile(.95)
            row[metric + '_SD_seed'] = selected[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def wide(stats):
    result = stats[['Model', 'Condition']].drop_duplicates().copy()
    for metric in METRICS:
        block = stats[stats.Metric == metric][['Model', 'Condition', 'Mean', 'SD_seed']]
        block[metric] = [f'{a:.4f} +/- {b:.4f}' for a, b in zip(block.Mean, block.SD_seed)]
        result = result.merge(block[['Model', 'Condition', metric]], on=['Model', 'Condition'])
    return result


def reproduce(root=ROOT, out=None, bootstrap=True):
    root = Path(root)
    out = Path(out or root / 'artifacts/reproduced')
    out.mkdir(parents=True, exist_ok=True)
    threadpool_limits(2)
    d = dataset(root)
    choices = audit_hpo(root)
    runs, frames = audit_predictions(root, d)
    runs.to_csv(out / 'recalculated_run_metrics.csv', index=False)
    stats_by_kind = {}
    draws_by_kind = {}
    for kind in KINDS:
        stats, seeds = seed_statistics(runs[runs.split == kind])
        reference = csv(root / f'results/recorded/{kind}_statistics_reference.csv')
        merged = stats.merge(reference, on=['Model', 'Condition', 'Metric'], suffixes=('', '_ref'))
        assert len(merged) == len(stats)
        for column in ['Mean', 'SD_seed', 'Seed_CI_lower', 'Seed_CI_upper']:
            assert np.allclose(merged[column], merged[column + '_ref'], atol=1e-12, rtol=0)
        draws_all = {}
        if bootstrap:
            plan = cluster_plan(frames[kind, 'ResNet1D', 'included_526'], kind)
            for (model, condition), _ in stats.groupby(['Model', 'Condition'], sort=False):
                draws = domain_draws(frames[kind, model, condition], plan)
                draws_all[model, condition] = draws
                for metric in METRICS:
                    low, high = np.quantile(draws[metric], [.025, .975])
                    select = (stats.Model == model) & (stats.Condition == condition) & (stats.Metric == metric)
                    stats.loc[select, ['Domain_CI_lower', 'Domain_CI_upper']] = [low, high]
                print('BOOTSTRAP', kind, model, condition, flush=True)
            check = stats.merge(reference, on=['Model', 'Condition', 'Metric'], suffixes=('', '_ref'))
            for column in ['Domain_CI_lower', 'Domain_CI_upper']:
                assert np.allclose(check[column], check[column + '_ref'], atol=1e-12, rtol=0)
            contrast = paired(seeds, draws_all['ResNet1D', 'included_526'], draws_all['TCN', 'included_526'])
            contrast.to_csv(out / f'{kind}_paired_resnet_minus_tcn.csv', index=False)
            if kind == 'group_aware':
                expected = csv(root / 'results/recorded/paired_resnet_minus_tcn_reference.csv')
                check = contrast.merge(expected, on='Metric', suffixes=('', '_ref'))
                for col in ['Mean_delta', 'Seed_CI_lower', 'Seed_CI_upper', 'Domain_CI_lower', 'Domain_CI_upper']:
                    assert np.allclose(check[col], check[col + '_ref'], atol=1e-12, rtol=0)
        seeds.to_csv(out / f'{kind}_per_seed.csv', index=False)
        stats.to_csv(out / f'{kind}_full_statistics.csv', index=False)
        stats_by_kind[kind], draws_by_kind[kind] = stats, draws_all
    corrected = pd.DataFrame([dict(Window_s=w, Min=int(n.min()), Max=int(n.max()), Mean=n.mean(),
                                   Median=np.median(n), Fixed_length=l)
                              for n, w, l in zip(d['window_counts'].T, [1, 3, 5, 10], [137, 526, 905, 1276])])
    corrected.to_csv(out / 'table3a_corrected_packet_counts.csv', index=False)
    historical = csv(root / 'results/recorded/table3a_reported_in_submitted_pdf.csv')
    corrected.merge(historical, on='Window_s', suffixes=('_corrected', '_pdf')).to_csv(out / 'table3a_correction_comparison.csv', index=False)
    pad = pd.DataFrame([dict(Sequence_length=l, Class=c, Sessions_with_padding=float((d['retained_count'][d['y'] == c] < l).mean()),
                            Mean_zero_fraction=float((np.maximum(0, l - d['retained_count'][d['y'] == c]) / l).mean()))
                        for l in [137, 526, 905, 1276] for c in [0, 1]])
    pad.to_csv(out / 'figure3_padding.csv', index=False)
    ss, ga = stats_by_kind['session_stratified'], stats_by_kind['group_aware']
    for name, block in [
        ('table3b_tcn_fixed_lengths', ss[(ss.Model == 'TCN') & ss.Condition.isin(['included_137', 'included_526', 'included_905', 'included_1276'])]),
        ('table4a_session_models', ss[ss.Condition == 'included_526']),
        ('table5a_handshake', ss[(ss.Model == 'TCN') & ss.Condition.isin(['included_526', 'post506_control', 'only20_padded526'])]),
        ('table5b_reduction', ss[(ss.Model == 'TCN') & ss.Condition.isin(['included_526', 'included_300'])]),
        ('table6_fusion16', ss[ss.Model.isin(['TCN', 'ResNet1D']) & ss.Condition.isin(['included_526', 'light_fusion526'])]),
        ('table7_group_models', ga[ga.Condition == 'included_526'])]:
        wide(block).to_csv(out / f'{name}.csv', index=False)
    for number, kind in [(8, 'session_stratified'), (7, 'group_aware')]:
        stats_by_kind[kind][(stats_by_kind[kind].Condition == 'included_526') & (stats_by_kind[kind].Metric == 'ROC-AUC')].to_csv(
            out / f'table{number}_auc_intervals.csv', index=False)
    timing = latency_summary(csv(root / 'results/recorded/latency_calls.csv.gz'))
    timing.to_csv(out / 'latency_summary.csv', index=False)
    expected = csv(root / 'results/recorded/latency_summary_reference.csv')
    keys = ['split', 'protocol', 'Model', 'Condition', 'Role']
    merged = timing.merge(expected, on=keys, suffixes=('', '_ref'))
    assert len(merged) == len(timing)
    for col in [c for c in timing if c not in keys]:
        assert np.allclose(merged[col], merged[col + '_ref'], atol=1e-10, rtol=0)
    timing[(timing.split == 'session_stratified') & (timing.Condition == 'included_526') & (timing.Role == 'test')].to_csv(out / 'table4b_session_test_time.csv', index=False)
    selection = []
    for model in ['TCN', 'ResNet1D']:
        values = [choices[f'session_stratified/cv_{fold}/{model}']['mean_validation_auc'] for fold in range(1, 6)]
        row = dict(Model=model, Selection_AUC=np.mean(values), SD_outer_fold=np.std(values, ddof=1))
        entry = timing[(timing.split == 'session_stratified') & (timing.Model == model) & (timing.Condition == 'included_526') & (timing.Role == 'validation')].iloc[0]
        for metric in ['Preprocess_ms', 'Inference_ms', 'Total_ms']:
            row[metric + '_mean'], row[metric + '_SD'] = entry[metric + '_mean'], entry[metric + '_SD_session_seed']
        selection.append(row)
    pd.DataFrame(selection).to_csv(out / 'table4c_validation_selection_time.csv', index=False)
    sessions = csv(root / 'data/metadata/sessions.csv')
    sessions.groupby('label').agg(Sessions=('index', 'size'), Sites=('site_id', 'nunique'), Domains=('domain_id', 'nunique')).to_csv(out / 'table1_dataset.csv')
    traffic = csv(root / 'data/metadata/full20s_direction_descriptives.csv').merge(sessions[['index', 'label']], on='index')
    traffic.groupby('label')[['Packet_count', 'Direction_run_count']].agg(['mean', 'std', 'min', 'median', 'max']).to_csv(out / 'full20s_class_descriptives.csv')
    save_json(out / 'table2_environment.json', read_json(root / 'environment_versions.json'))
    plot(ss, pad, out)
    result = dict(status='passed', evaluated_runs=700, hpo_trials=7200, selected_configurations=100,
                  seed_intervals_recalculated=True, domain_intervals_recalculated=bootstrap,
                  matched_features_exact=True, thresholds_validation_only=True, latency_reaggregated=True)
    save_json(out / 'verification.json', result)
    return result


def plot(stats, pad, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lengths = [137, 526, 905, 1276]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    secondary = ax.twinx()
    for c, offset, color, label in [(0, -.18, '#86b5d5', 'Clear web'), (1, .18, '#ae5039', 'Dark mirror web')]:
        secondary.bar(x + offset, pad[pad.Class == c].set_index('Sequence_length').loc[lengths].Sessions_with_padding,
                      width=.34, color=color, label=label)
    for metric, color, style, marker in [('Accuracy', '#222222', '-', 'o'), ('F1-score', '#184b74', '--', 's'), ('ROC-AUC', '#6f184b', '-.', '^')]:
        part = stats[(stats.Model == 'TCN') & (stats.Metric == metric)].set_index('Condition').loc[[f'included_{l}' for l in lengths]]
        ax.errorbar(x, part.Mean, yerr=part.SD_seed, color=color, linestyle=style, marker=marker, capsize=3, label=metric)
    ax.set_zorder(secondary.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xticks(x, lengths)
    ax.set_xlabel('Sequence length')
    ax.set_ylabel('Performance (mean +/- seed SD)')
    secondary.set_ylabel('Proportion of sessions requiring padding')
    ax.set_ylim(0, 1.05)
    secondary.set_ylim(0, 1.05)
    h, labels = ax.get_legend_handles_labels()
    h2, labels2 = secondary.get_legend_handles_labels()
    fig.legend(h + h2, labels + labels2, loc='upper right', bbox_to_anchor=(.99, .99), ncol=3)
    fig.subplots_adjust(top=.84, right=.88, bottom=.13, left=.09)
    fig.savefig(out / 'figure3_performance_padding.png', dpi=240, bbox_inches='tight')
    plt.close(fig)


def summarize_new(out, root=ROOT):
    """Apply the same statistical procedure to a complete newly retrained run."""
    out = Path(out)
    runs = csv(out / 'run_metrics.csv')
    assert len(runs) == 700, 'Complete the 700 final paper fits/evaluations before full reporting.'
    spec = read_json(Path(root) / 'configs/experiment.json')
    for kind in KINDS:
        block = runs[runs.split == kind]
        assert (block.groupby(['Method', 'Condition', 'seed']).size() == 5).all()
        stats, seeds = seed_statistics(block)
        directory = out / 'predictions' / kind
        plan = cluster_plan(csv(directory / 'resnet1d__included_526.csv.gz'), kind)
        all_draws = {}
        for model, condition in block[['Method', 'Condition']].drop_duplicates().itertuples(index=False, name=None):
            frame = csv(directory / f'{spec["model_keys"][model]}__{condition}.csv.gz')
            draws = domain_draws(frame, plan)
            all_draws[model, condition] = draws
            for metric in METRICS:
                low, high = np.quantile(draws[metric], [.025, .975])
                mask = (stats.Model == model) & (stats.Condition == condition) & (stats.Metric == metric)
                stats.loc[mask, ['Domain_CI_lower', 'Domain_CI_upper']] = [low, high]
        stats.to_csv(out / f'{kind}_new_full_statistics.csv', index=False)
        seeds.to_csv(out / f'{kind}_new_per_seed.csv', index=False)
        paired(seeds, all_draws['ResNet1D', 'included_526'], all_draws['TCN', 'included_526']).to_csv(
            out / f'{kind}_new_paired_resnet_minus_tcn.csv', index=False)
    return dict(status='completed', rerun_evaluation_rows=len(runs), no_comparison_to_reference_numbers_required=True)
