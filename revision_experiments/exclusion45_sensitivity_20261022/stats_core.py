"""Unchanged statistical functions extracted from the completed remote analysis."""
import numpy as np
from scipy.stats import binomtest
REPS=20000
SEED=20260711
MODELS={"qwen":"Qwen3-VL-8B","gemma":"Gemma-3-12B-it","internvl":"InternVL3-8B"}

def interval(values, alpha=0.05):
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return (None, None)
    return tuple((float(v) for v in np.quantile(finite, [alpha / 2, 1 - alpha / 2])))

def safe_div(a, b):
    a, b = (np.asarray(a, float), np.asarray(b, float))
    result = np.full(np.broadcast_shapes(a.shape, b.shape), np.nan)
    return np.divide(a, b, out=result, where=b != 0)

def exact(a_only, b_only):
    return float(binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue) if a_only + b_only else 1.0

def adjust_holm(rows):
    families = {}
    for row in rows:
        if row['model'] != 'Pooled fixed-model strata':
            families.setdefault((row['population'], row['holm_family']), []).append(row)
    for group in families.values():
        running = 0.0
        for rank, row in enumerate(sorted(group, key=lambda r: r['mcnemar_exact_aux_p'])):
            running = max(running, min(1.0, (len(group) - rank) * row['mcnemar_exact_aux_p']))
            row['mcnemar_holm_aux_p'] = running
            row['holm_family_size'] = len(group)

def contrast(rows, left, right, name, model, population, family, seed=SEED, simultaneous=1):
    tasks = sorted({r['task_cluster'] for r in rows})
    counts, sizes = ([], [])
    for task in tasks:
        selected = [r for r in rows if r['task_cluster'] == task]
        a = np.array([r[left] for r in selected], bool)
        b = np.array([r[right] for r in selected], bool)
        counts.append([a.sum(), b.sum(), (a & b).sum(), (a | b).sum(), (a & ~b).sum(), (~a & b).sum(), (~a & ~b).sum()])
        sizes.append(len(selected))
    count_array = np.asarray(counts, float)
    samples = np.random.default_rng(seed).integers(0, len(tasks), size=(REPS, len(tasks)))
    boot_counts = count_array[samples].sum(axis=1)
    boot_den = np.asarray(sizes)[samples].sum(axis=1)
    boot_rates = boot_counts[:, :2] / boot_den[:, None]
    boot_diff = 100 * (boot_rates[:, 0] - boot_rates[:, 1])
    totals = count_array.sum(axis=0).astype(int)
    a, b, both, union, a_only, b_only, neither = map(int, totals)
    n = len(rows)
    record = {'population': population, 'model': model, 'contrast': name, 'left_endpoint': left, 'right_endpoint': right, 'n': n, 'task_clusters': len(tasks), 'left_success_n': a, 'right_success_n': b, 'left_rate_pct': 100 * a / n, 'right_rate_pct': 100 * b / n, 'left_ci_low_pct': interval(100 * boot_rates[:, 0])[0], 'left_ci_high_pct': interval(100 * boot_rates[:, 0])[1], 'right_ci_low_pct': interval(100 * boot_rates[:, 1])[0], 'right_ci_high_pct': interval(100 * boot_rates[:, 1])[1], 'difference_left_minus_right_pp': 100 * (a - b) / n, 'difference_ci_low_pp': interval(boot_diff)[0], 'difference_ci_high_pp': interval(boot_diff)[1], 'both': both, 'left_only': a_only, 'right_only': b_only, 'neither': neither, 'jaccard_pct': 100 * both / union if union else None, 'jaccard_ci_low_pct': interval(100 * safe_div(boot_counts[:, 2], boot_counts[:, 3]))[0], 'jaccard_ci_high_pct': interval(100 * safe_div(boot_counts[:, 2], boot_counts[:, 3]))[1], 'left_success_right_failure_share_pct': 100 * a_only / a if a else None, 'left_failure_share_ci_low_pct': interval(100 * safe_div(boot_counts[:, 4], boot_counts[:, 0]))[0], 'left_failure_share_ci_high_pct': interval(100 * safe_div(boot_counts[:, 4], boot_counts[:, 0]))[1], 'left_failure_share_finite_bootstrap_replicates': int(np.isfinite(safe_div(boot_counts[:, 4], boot_counts[:, 0])).sum()), 'holm_family': family, 'mcnemar_exact_aux_p': exact(a_only, b_only) if model != 'Pooled fixed-model strata' else None}
    if simultaneous > 1:
        record['simultaneous_difference_ci_low_pp'], record['simultaneous_difference_ci_high_pp'] = interval(boot_diff, 0.05 / simultaneous)
        record['simultaneous_family_size'] = simultaneous
    return record

def scaffold_simultaneous(rows, population, tables):
    """Use the original joint six-contrast max-|T| method, not marginal CIs."""
    tasks = list(dict.fromkeys((r['task_cluster'] for r in rows if r['model_key'] == 'qwen')))
    samples = np.random.default_rng(SEED).integers(0, len(tasks), size=(REPS, len(tasks)))
    boot, observed, labels = ([], [], [])
    for key, display in MODELS.items():
        selected = [r for r in rows if r['model_key'] == key]
        sums = np.asarray([[sum((r[k] for r in selected if r['task_cluster'] == t)) for k in ('full_procedure_success', 'phase_scaffold_success', 'target_scaffold_success')] for t in tasks], float)
        sizes = np.asarray([sum((r['task_cluster'] == t for r in selected)) for t in tasks], float)
        rates = sums.sum(axis=0) / sizes.sum()
        draws = sums[samples].sum(axis=1) / sizes[samples].sum(axis=1)[:, None]
        for col, name in ((1, 'Phase Scaffold versus Full'), (2, 'Target Scaffold versus Full')):
            boot.append(draws[:, col] - draws[:, 0])
            observed.append(rates[col] - rates[0])
            labels.append((display, name))
    boot = np.column_stack(boot)
    observed = np.asarray(observed)
    se = boot.std(axis=0, ddof=1)
    assert np.all(se > 0)
    critical = float(np.quantile(np.max(np.abs((boot - observed) / se), axis=1), 0.95))
    for i, (model, name) in enumerate(labels):
        record = next((r for r in tables if r['population'] == population and r['model'] == model and (r['contrast'] == name)))
        record['simultaneous_difference_ci_low_pp'] = 100 * (observed[i] - critical * se[i])
        record['simultaneous_difference_ci_high_pp'] = 100 * (observed[i] + critical * se[i])
        record['simultaneous_method'] = 'joint_task_cluster_bootstrap_max_abs_t'
        record['simultaneous_family_size'] = 6
        record['max_abs_t_critical_value'] = critical
    if population == 'original_2133':
        assert abs(critical - 2.615006247585567) < 1e-10, critical
