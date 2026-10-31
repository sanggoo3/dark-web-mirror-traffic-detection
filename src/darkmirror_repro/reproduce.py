"""Summaries of newly executed experiments; no archived-result dependency."""
from pathlib import Path
import numpy as np
import pandas as pd
from .io import ROOT, KINDS, METRICS, csv, dataset, read_json
from .statistics import seed_statistics, cluster_plan, domain_draws, paired


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
        row = dict(split=kind, protocol=protocol, Model=model, Condition=condition, Role=role, Unique_sessions=block['index'].nunique(), Session_seed_observations=len(block), Timed_calls=len(block) * 3, Batch=1, Repeats=3, Warmup=50)
        for metric in metrics:
            row[metric + '_mean'] = block[metric].mean()
            row[metric + '_SD_session_seed'] = block[metric].std(ddof=1)
            row[metric + '_p95'] = block[metric].quantile(0.95)
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

def plot(stats, pad, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lengths = [137, 526, 905, 1276]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    secondary = ax.twinx()
    for c, offset, color, label in [(0, -0.18, '#86b5d5', 'Clear web'), (1, 0.18, '#ae5039', 'Dark mirror web')]:
        secondary.bar(x + offset, pad[pad.Class == c].set_index('Sequence_length').loc[lengths].Sessions_with_padding, width=0.34, color=color, label=label)
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
    fig.legend(h + h2, labels + labels2, loc='upper right', bbox_to_anchor=(0.99, 0.99), ncol=3)
    fig.subplots_adjust(top=0.84, right=0.88, bottom=0.13, left=0.09)
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
            frame = csv(directory / f'{spec['model_keys'][model]}__{condition}.csv.gz')
            draws = domain_draws(frame, plan)
            all_draws[model, condition] = draws
            for metric in METRICS:
                low, high = np.quantile(draws[metric], [0.025, 0.975])
                mask = (stats.Model == model) & (stats.Condition == condition) & (stats.Metric == metric)
                stats.loc[mask, ['Domain_CI_lower', 'Domain_CI_upper']] = [low, high]
        stats.to_csv(out / f'{kind}_new_full_statistics.csv', index=False)
        seeds.to_csv(out / f'{kind}_new_per_seed.csv', index=False)
        paired(seeds, all_draws['ResNet1D', 'included_526'], all_draws['TCN', 'included_526']).to_csv(out / f'{kind}_new_paired_resnet_minus_tcn.csv', index=False)
    paper_tables(out, root)
    return dict(status='completed', rerun_evaluation_rows=len(runs))


def descriptives(root=ROOT, out=None):
    root = Path(root)
    out = Path(out or root / 'artifacts/descriptives')
    out.mkdir(parents=True, exist_ok=True)
    d = dataset(root)
    rows = [dict(Window_s=w, Minimum=int(n.min()), Maximum=int(n.max()), Mean=float(n.mean()),
                 Median=float(np.median(n)), Fixed_length=l)
            for n, w, l in zip(d['window_counts'].T, [1, 3, 5, 10], [137, 526, 905, 1276])]
    pd.DataFrame(rows).to_csv(out / 'table3a_packet_counts.csv', index=False)
    padding = pd.DataFrame([dict(Sequence_length=l, Class=c,
        Sessions_with_padding=float((d['retained_count'][d['y'] == c] < l).mean()),
        Mean_zero_fraction=float((np.maximum(0, l - d['retained_count'][d['y'] == c]) / l).mean()))
        for l in [137, 526, 905, 1276] for c in [0, 1]])
    padding.to_csv(out / 'figure3_padding.csv', index=False)
    sessions = csv(root / 'data/metadata/sessions.csv')
    sessions.groupby('label').agg(Sessions=('index', 'size'), Sites=('site_id', 'nunique')).to_csv(out / 'table1_dataset.csv')
    traffic = csv(root / 'data/metadata/full20s_direction_descriptives.csv').merge(sessions[['index', 'label']], on='index')
    traffic.groupby('label')[['Packet_count', 'Direction_run_count']].agg(
        ['mean', 'std', 'min', 'median', 'max']).to_csv(out / 'traffic_descriptives.csv')
    return padding


def paper_tables(out, root=ROOT):
    out, root = Path(out), Path(root)
    padding = descriptives(root, out)
    ss = csv(out / 'session_stratified_new_full_statistics.csv')
    ga = csv(out / 'group_aware_new_full_statistics.csv')
    for name, block in [
        ('table3b_fixed_lengths', ss[(ss.Model == 'TCN') & ss.Condition.isin(['included_137', 'included_526', 'included_905', 'included_1276'])]),
        ('table4a_session_models', ss[ss.Condition == 'included_526']),
        ('table5a_handshake', ss[(ss.Model == 'TCN') & ss.Condition.isin(['included_526', 'post506_control', 'only20_padded526'])]),
        ('table5b_reduction', ss[(ss.Model == 'TCN') & ss.Condition.isin(['included_526', 'included_300'])]),
        ('table6_fusion16', ss[ss.Model.isin(['TCN', 'ResNet1D']) & ss.Condition.isin(['included_526', 'light_fusion526'])]),
        ('table7_group_models', ga[ga.Condition == 'included_526'])]:
        wide(block).to_csv(out / f'{name}.csv', index=False)
    for number, stats in [(7, ga), (8, ss)]:
        stats[(stats.Condition == 'included_526') & (stats.Metric == 'ROC-AUC')].to_csv(
            out / f'table{number}_auc_intervals.csv', index=False)
    choices_path = out / 'selected_configurations.json'
    if (out / 'test_lock.json').exists() and read_json(out / 'test_lock.json')['recorded_choices']:
        choices_path = root / 'configs/selected_configurations.json'
    if not choices_path.exists():
        choices_path = root / 'configs/selected_configurations.json'
    records = []
    for key, choice in read_json(choices_path).items():
        kind, unit, model = key.split('/')
        records.append(dict(split=kind, fold=int(unit.split('_')[1]), Model=model,
                            Selection_AUC=choice['mean_validation_auc']))
    selection = pd.DataFrame(records).groupby(['split', 'Model']).Selection_AUC.agg(['mean', 'std']).reset_index()
    selection.to_csv(out / 'validation_model_selection.csv', index=False)
    plot(ss, padding, out)
