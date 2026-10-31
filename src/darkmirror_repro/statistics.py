"""Seed-level summaries and conditional domain-cluster bootstrap."""
import numpy as np
import pandas as pd
from scipy.stats import t

from .io import METRICS, SEEDS


def seed_statistics(runs):
    seeds = runs.groupby(['Method', 'Condition', 'seed'], sort=False)[METRICS].mean().reset_index()
    rows = []
    for (model, condition), block in seeds.groupby(['Method', 'Condition'], sort=False):
        assert sorted(block.seed) == SEEDS
        for metric in METRICS:
            mean, sd = block[metric].mean(), block[metric].std(ddof=1)
            radius = t.ppf(.975, 4) * sd / np.sqrt(5)
            rows.append(dict(Model=model, Condition=condition, Metric=metric, Mean=mean, SD_seed=sd,
                             Seed_CI_lower=max(0., mean - radius), Seed_CI_upper=min(1., mean + radius)))
    return pd.DataFrame(rows), seeds


def cluster_plan(template, kind, repetitions=2000):
    rng = np.random.default_rng(20261022)
    shared = None
    if kind == 'session_stratified':
        first = template[template.seed == 11]
        names = [sorted(first.loc[first.y_true == c, 'group'].unique()) for c in [0, 1]]
        weights = [rng.multinomial(len(g), np.ones(len(g)) / len(g), size=repetitions) for g in names]
        shared = names, weights
    plan = []
    for fold in range(1, 6):
        first = template[(template.fold == fold) & (template.seed == 11)]
        if shared is None:
            names = [sorted(first.loc[first.y_true == c, 'group'].unique()) for c in [0, 1]]
            weights = [rng.multinomial(len(g), np.ones(len(g)) / len(g), size=repetitions) for g in names]
        else:
            names, weights = shared
        sizes = [np.array([(first.group == g).sum() for g in groups]) for groups in names]
        plan.append((fold, names, weights, sizes))
    return plan


def domain_draws(frame, plan):
    n = len(plan[0][2][0])
    draws = {metric: np.zeros(n) for metric in METRICS}
    for fold, names, weights, sizes in plan:
        negative, positive = names
        w0, w1 = weights
        matrices, confusion = [], []
        for seed in SEEDS:
            block = frame[(frame.fold == fold) & (frame.seed == seed)]
            values = {g: part.probability.to_numpy() for g, part in block.groupby('group')}
            matrix = np.zeros((len(positive), len(negative)))
            for i, p in enumerate(positive):
                for j, q in enumerate(negative):
                    if p in values and q in values:
                        delta = values[p][:, None] - values[q][None, :]
                        matrix[i, j] = (delta > 0).sum() + .5 * (delta == 0).sum()
            matrices.append(matrix)
            counts = []
            for group in negative + positive:
                part = block[block.group == group]
                y, pred = part.y_true.to_numpy(), part.y_pred.to_numpy()
                counts.append([((y == 1) & (pred == 1)).sum(), ((y == 0) & (pred == 1)).sum(),
                               ((y == 1) & (pred == 0)).sum(), ((y == 0) & (pred == 0)).sum()])
            confusion.append(counts)
        denominator = (w0 @ sizes[0]) * (w1 @ sizes[1])
        assert (denominator > 0).all()
        draws['ROC-AUC'] += np.einsum('bi,ij,bj->b', w1, np.mean(matrices, axis=0), w0) / denominator / 5
        counts = np.einsum('bg,sgc->bsc', np.concatenate([w0, w1], axis=1), np.asarray(confusion, dtype=np.float64))
        tp, fp, fn, tn = [counts[:, :, i] for i in range(4)]
        draws['Accuracy'] += ((tp + tn) / (tp + fp + fn + tn)).mean(axis=1) / 5
        for metric, numerator, denom in [('Precision', tp, tp + fp), ('Recall', tp, tp + fn),
                                          ('F1-score', 2 * tp, 2 * tp + fp + fn)]:
            draws[metric] += np.divide(numerator, denom, out=np.zeros_like(tp), where=denom > 0).mean(axis=1) / 5
    return draws


def paired(seeds, a_draws, b_draws):
    a = seeds[(seeds.Method == 'ResNet1D') & (seeds.Condition == 'included_526')].sort_values('seed')
    b = seeds[(seeds.Method == 'TCN') & (seeds.Condition == 'included_526')].sort_values('seed')
    result = []
    for metric in METRICS:
        delta = a[metric].to_numpy() - b[metric].to_numpy()
        radius = t.ppf(.975, 4) * delta.std(ddof=1) / np.sqrt(5)
        low, high = np.quantile(a_draws[metric] - b_draws[metric], [.025, .975])
        result.append(dict(Metric=metric, Mean_delta=delta.mean(), Seed_CI_lower=delta.mean() - radius,
                           Seed_CI_upper=delta.mean() + radius, Domain_CI_lower=low, Domain_CI_upper=high))
    return pd.DataFrame(result)
