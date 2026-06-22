import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import ast
import os
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from itertools import combinations

# =========================
# LOAD DATA
# =========================
trial_info = pd.read_excel('../data/new_trial_final.xlsx')
subject_trials = trial_info[trial_info['subject_id'] == 14][['trial_id_final', 'num_images_presented', 'stimulus_index_enc1', 'stimulus_index_enc2', 'stimulus_index_enc3', 'response_accuracy']]
print(subject_trials)
y_matrix = subject_trials.reset_index(drop=True)

df_delay_filtered = pd.read_excel('../graph_data/graph_delay.xlsx')
df_fixation = pd.read_excel('../clean_data/cleaned_Fixation.xlsx')
df_clustering = pd.read_excel('../Neuron_Check_Significant_All.xlsx')

bin_sizes = [0.25, 0.5, 0.75, 1.0]
step_size = 0.025
durations = {1: 2.5, 2: 2.5, 3: 2.5}

n_splits = 3
n_repeats = 10              # repeated CV
n_permutations_null = 1000  # real vs shuffled
n_permutations_pair = 10000 # paired sign-flip test for bin-vs-bin
random_state = 42

folder_path = './'
os.makedirs(folder_path, exist_ok=True)

def parse_spike_times(spike_str):
    if pd.isna(spike_str):
        return []
    try:
        parsed = ast.literal_eval(spike_str)
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return list(parsed)
        return []
    except Exception:
        return []

def create_time_bins(start_time, end_time, bin_size, step_size):
    return np.arange(start_time, end_time - bin_size + step_size, step_size)

def count_spikes_in_bins(spike_times, time_bins, bin_size):
    spike_times = np.asarray(spike_times, dtype=float)
    return np.array([
        np.sum((spike_times >= t) & (spike_times < t + bin_size))
        for t in time_bins
    ], dtype=float)

def compute_sem(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) <= 1:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(len(x))

def repeated_stratified_cv_accuracies(X, y, n_splits=3, n_repeats=10, seed=42):
    """
    Returns one mean CV accuracy per repeat.
    Each repeat uses a new StratifiedKFold split seed.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    unique_classes, class_counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2:
        return np.full(n_repeats, np.nan)

    min_class_count = np.min(class_counts)
    if min_class_count < n_splits:
        return np.full(n_repeats, np.nan)

    repeat_accuracies = []

    for rep in range(n_repeats):
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + rep
        )

        fold_accs = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(np.unique(y_train)) < 2:
                fold_accs.append(np.nan)
                continue

            clf = SVC(kernel='linear', random_state=seed + rep)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = np.mean(y_pred == y_test) * 100
            fold_accs.append(acc)

        repeat_accuracies.append(np.nanmean(fold_accs))

    return np.array(repeat_accuracies, dtype=float)

def permutation_test_cv_repeats(X, y, n_permutations=1000, n_splits=3, n_repeats=10, seed=42):
    """
    Null distribution for real vs shuffled:
    each permutation returns one scalar = mean accuracy across repeats.
    """
    rng = np.random.default_rng(seed)
    null_dist = []

    for perm_idx in range(n_permutations):
        y_shuffled = rng.permutation(y)
        rep_accs = repeated_stratified_cv_accuracies(
            X, y_shuffled,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed + 10000 + perm_idx
        )
        null_dist.append(np.nanmean(rep_accs))

    return np.array(null_dist, dtype=float)

def paired_sign_flip_test(x, y, n_permutations=10000, seed=42):
    """
    Paired permutation test for bin-vs-bin comparison.
    Tests whether mean(x - y) differs from 0 using sign flips.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 5:
        return np.nan, np.nan, n

    diffs = x - y
    observed = np.mean(diffs)

    # remove exact zeros only if all are zero
    if np.allclose(diffs, 0):
        return 1.0, observed, n

    rng = np.random.default_rng(seed)

    null_vals = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        null_vals[i] = np.mean(diffs * signs)

    p = (np.sum(np.abs(null_vals) >= abs(observed)) + 1) / (n_permutations + 1)
    return p, observed, n

filtered_clustering = df_delay_filtered[
    (df_delay_filtered['Signi'] == 'Y')
].copy()

candidate_neurons = filtered_clustering['Neuron_ID_3'].dropna().unique()
delay_neurons = df_delay_filtered['Neuron_ID_3'].dropna().unique()
fix_neurons = df_fixation['Neuron_ID_3'].dropna().unique()

final_neurons = np.intersect1d(candidate_neurons, np.intersect1d(delay_neurons, fix_neurons))
final_neurons = np.array(sorted(final_neurons))

trial_count = y_matrix.shape[0]
neuron_count = len(final_neurons)

design_matrix = np.empty((trial_count, neuron_count), dtype=object)
design_matrix[:] = None

delay_lookup = {}
for _, row in df_delay_filtered.iterrows():
    key = (row['trial_id_final'], row['Neuron_ID_3'])
    delay_lookup[key] = np.array(parse_spike_times(row['Standardized_Spikes_in_Delay']), dtype=float)

neuron_to_col = {nid: idx for idx, nid in enumerate(final_neurons)}

for trial_idx, row in y_matrix.iterrows():
    trial_id = row['trial_id_final']
    for neuron_id in final_neurons:
        col_idx = neuron_to_col[neuron_id]
        spikes = delay_lookup.get((trial_id, neuron_id), np.array([], dtype=float))
        design_matrix[trial_idx, col_idx] = spikes

df_fixation_filtered = df_fixation[df_fixation['Neuron_ID_3'].isin(final_neurons)].copy()

fixation_means = (
    df_fixation_filtered.groupby('Neuron_ID_3')['Spikes_rate_Fixation']
    .mean()
    .to_dict()
)

fixation_stds = (
    df_fixation_filtered.groupby('Neuron_ID_3')['Spikes_rate_Fixation']
    .std()
    .to_dict()
)

y_labels = []
for _, row in y_matrix.iterrows():
    load = row['num_images_presented']
    if load == 1:
        y_labels.append(row['stimulus_index_enc1'])
    elif load == 2:
        y_labels.append(row['stimulus_index_enc2'])
    elif load == 3:
        y_labels.append(row['stimulus_index_enc3'])
    else:
        y_labels.append(np.nan)

y_labels = np.array(y_labels)

valid_trials = ~pd.isna(y_labels)
y_matrix = y_matrix.loc[valid_trials].reset_index(drop=True)
y_labels = y_labels[valid_trials]
design_matrix = design_matrix[valid_trials]

load_trials = {
    1: np.where(y_matrix['num_images_presented'].values == 1)[0],
    2: np.where(y_matrix['num_images_presented'].values == 2)[0],
    3: np.where(y_matrix['num_images_presented'].values == 3)[0],
}

def build_matrix(trials, bin_size, end_time):
    time_bins = create_time_bins(0, end_time, bin_size, step_size)
    n_trials = len(trials)
    n_bins = len(time_bins)

    matrix = np.zeros((n_trials, neuron_count, n_bins), dtype=float)

    for i, trial_idx in enumerate(trials):
        for j, neuron_id in enumerate(final_neurons):
            spikes = design_matrix[trial_idx, j]
            binned = count_spikes_in_bins(spikes, time_bins, bin_size)

            mean_fix = fixation_means.get(neuron_id, 0)
            std_fix = fixation_stds.get(neuron_id, 1)

            if pd.isna(std_fix) or std_fix <= 0:
                matrix[i, j, :] = binned
            else:
                matrix[i, j, :] = (binned - mean_fix) / std_fix

    return matrix.reshape(n_trials, -1)

regions = [f'{int(b * 1000)} ms' for b in bin_sizes]
categories = ['Load1', 'Load2', 'Load3']

results = []
repeat_accs_by_bin = {b: {} for b in bin_sizes}

for bin_size in tqdm(bin_sizes, desc="Bin sizes"):
    real_acc_means = []
    real_acc_sems = []
    null_means = []
    null_sems = []
    nulls_for_this_bin = []

    for load in [1, 2, 3]:
        trial_idxs = load_trials[load]
        y_load = y_labels[trial_idxs]

        print(f"\nBin size {bin_size}, Load {load}")
        print("Class counts:", dict(zip(*np.unique(y_load, return_counts=True))))

        X = build_matrix(trial_idxs, bin_size, durations[load])

        # repeated CV accuracies: one scalar per repeat
        rep_accs = repeated_stratified_cv_accuracies(
            X, y_load,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=random_state
        )

        repeat_accs_by_bin[bin_size][load] = rep_accs.copy()

        acc_mean = np.nanmean(rep_accs)
        acc_sem = compute_sem(rep_accs)

        real_acc_means.append(acc_mean)
        real_acc_sems.append(acc_sem)

        # null distribution for real vs shuffled
        nulls = permutation_test_cv_repeats(
            X, y_load,
            n_permutations=n_permutations_null,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=random_state
        )

        null_means.append(np.nanmean(nulls))
        null_sems.append(compute_sem(nulls))
        nulls_for_this_bin.append(nulls)

    avg_shuffled = np.nanmean(null_means)
    nulls_stack = np.vstack(nulls_for_this_bin)
    avg_shuffled_distribution = np.nanmean(nulls_stack, axis=0)
    avg_shuffled_sem = compute_sem(avg_shuffled_distribution)

    results.append({
        'bin_size': bin_size,
        'real_acc_means': real_acc_means,
        'real_acc_sems': real_acc_sems,
        'null_means': null_means,
        'null_sems': null_sems,
        'nulls_for_this_bin': nulls_for_this_bin,
        'avg_shuffled': avg_shuffled,
        'avg_shuffled_distribution': avg_shuffled_distribution,
        'avg_shuffled_sem': avg_shuffled_sem
    })

p_values = np.zeros((len(bin_sizes), 3))

for bin_idx, r in enumerate(results):
    for load_idx in range(3):
        real_acc = r['real_acc_means'][load_idx]
        null_dist = r['nulls_for_this_bin'][load_idx]

        p = (np.sum(null_dist >= real_acc) + 1) / (len(null_dist) + 1)
        p_values[bin_idx, load_idx] = p

p_values_flat = p_values.flatten()
rejected, pvals_fdr, _, _ = multipletests(p_values_flat, alpha=0.05, method='fdr_bh')
pvals_fdr = pvals_fdr.reshape(p_values.shape)
rejected = rejected.reshape(p_values.shape)

bin_pairs = list(combinations(bin_sizes, 2))
bin_compare_rows = []

for load in [1, 2, 3]:
    for b1, b2 in bin_pairs:
        rep1 = repeat_accs_by_bin[b1][load]
        rep2 = repeat_accs_by_bin[b2][load]

        p, observed_diff, n_used = paired_sign_flip_test(
            rep1, rep2,
            n_permutations=n_permutations_pair,
            seed=random_state + int(b1 * 1000) + int(b2 * 1000) + load
        )

        mean1 = np.nanmean(rep1)
        mean2 = np.nanmean(rep2)
        sem1 = compute_sem(rep1)
        sem2 = compute_sem(rep2)

        bin_compare_rows.append({
            'Load': f'Load{load}',
            'Bin 1 (ms)': int(b1 * 1000),
            'Bin 2 (ms)': int(b2 * 1000),
            'Mean Accuracy Bin 1 (%)': mean1,
            'SEM Bin 1': sem1,
            'Mean Accuracy Bin 2 (%)': mean2,
            'SEM Bin 2': sem2,
            'Mean Difference (Bin1-Bin2)': observed_diff,
            'n repeats used': n_used,
            'Paired sign-flip p-value': p
        })

bin_compare_df = pd.DataFrame(bin_compare_rows)

# FDR for bin-width comparisons
valid_mask = bin_compare_df['Paired sign-flip p-value'].notna()
if valid_mask.sum() > 0:
    rej_bin, pvals_bin_fdr, _, _ = multipletests(
        bin_compare_df.loc[valid_mask, 'Paired sign-flip p-value'],
        alpha=0.05,
        method='fdr_bh'
    )
    bin_compare_df.loc[valid_mask, 'p-value (FDR)'] = pvals_bin_fdr
    bin_compare_df.loc[valid_mask, 'Significant (FDR)'] = np.where(rej_bin, 'Yes', 'No')
else:
    bin_compare_df['p-value (FDR)'] = np.nan
    bin_compare_df['Significant (FDR)'] = ''

# =========================
# PLOT: REAL VS SHUFFLED
# =========================

plt.figure(figsize=(14, 8))
bar_width = 0.18
x = np.arange(len(regions))
chance_level = 20  # assuming 5-way decoding

for i in range(3):
    offsets = x + i * bar_width - (bar_width * 3 / 2) + bar_width / 2
    means = [results[j]['real_acc_means'][i] for j in range(len(regions))]
    sems = [results[j]['real_acc_sems'][i] for j in range(len(regions))]

    plt.bar(offsets, means, bar_width, yerr=sems, label=categories[i], capsize=5)

    for j in range(len(regions)):
        p_fdr = pvals_fdr[j, i]

        if p_fdr < 0.001:
            stars = '***'
        elif p_fdr < 0.01:
            stars = '**'
        elif p_fdr < 0.05:
            stars = '*'
        else:
            stars = ''

        if stars:
            plt.text(
                offsets[j],
                means[j] + (sems[j] if not np.isnan(sems[j]) else 0) + 3,
                stars,
                ha='center',
                fontsize=16,
                fontweight='bold',
                color='r'
            )

shuffled_offsets = x + 3 * bar_width - (bar_width * 3 / 2) + bar_width / 2
shuf_means = [results[j]['avg_shuffled'] for j in range(len(regions))]
shuf_sems = [results[j]['avg_shuffled_sem'] for j in range(len(regions))]

plt.bar(
    shuffled_offsets, shuf_means, bar_width, yerr=shuf_sems,
    label='Shuffled (avg)', color='gray', capsize=5
)

plt.axhline(y=chance_level, color='gray', linestyle='--', label=f'Chance ({chance_level}%)')
plt.title('SVM Decoding Accuracy by Load and Bin Size (Repeated CV)', fontsize=16)
plt.ylabel('Accuracy (%)')
plt.xticks(x, regions)
plt.ylim(0, 100)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

plot_save_path = os.path.join(folder_path, "decoding_timebins_repeatedcv.eps")
plt.savefig(plot_save_path, format='eps', dpi=300)
plt.show()

# =========================
# SAVE RESULTS TABLE: REAL VS SHUFFLED
# =========================

comparison_results = []
for i, region in enumerate(regions):
    for k in range(3):
        comparison_results.append({
            'Time Bin': region,
            'Condition': categories[k],
            'Mean Accuracy (%)': results[i]['real_acc_means'][k],
            'SEM (Repeated CV)': results[i]['real_acc_sems'][k],
            'Mean Shuffled Accuracy (%)': results[i]['null_means'][k],
            'SEM (Shuffled)': results[i]['null_sems'][k],
            'p-value': p_values[i, k],
            'Significant (uncorrected)': 'Yes' if p_values[i, k] < 0.05 else 'No',
            'p-value (FDR)': pvals_fdr[i, k],
            'Significant (FDR)': 'Yes' if rejected[i, k] else 'No'
        })

    comparison_results.append({
        'Time Bin': region,
        'Condition': 'Shuffled (avg)',
        'Mean Accuracy (%)': results[i]['avg_shuffled'],
        'SEM (Repeated CV)': '',
        'Mean Shuffled Accuracy (%)': '',
        'SEM (Shuffled)': results[i]['avg_shuffled_sem'],
        'p-value': '',
        'Significant (uncorrected)': '',
        'p-value (FDR)': '',
        'Significant (FDR)': ''
    })

df_comparisons = pd.DataFrame(comparison_results)

table_save_path = os.path.join(folder_path, "decoding_comparisons_repeatedcv.xlsx")
df_comparisons.to_excel(table_save_path, index=False)

# =========================
# SAVE TABLE: BIN-WIDTH COMPARISONS
# =========================

bin_compare_save_path = os.path.join(folder_path, "bin_width_pairwise_comparisons_repeatedcv.xlsx")
bin_compare_df.to_excel(bin_compare_save_path, index=False)

print("\nReal vs Shuffled Comparisons:")
print(df_comparisons)

print("\nPairwise Bin-Width Comparisons:")
print(bin_compare_df)

print(f"\nSaved real-vs-shuffled table to: {table_save_path}")
print(f"Saved bin-width comparison table to: {bin_compare_save_path}")
print(f"Saved plot to: {plot_save_path}")
