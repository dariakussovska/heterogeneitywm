import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import ast
import os
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Patch

# =========================
# LOAD DATA
# =========================
trial_info = pd.read_excel('../data/new_trial_final.xlsx')
subject_trials = trial_info[trial_info['subject_id'] == 14][['trial_id_final', 'num_images_presented', 'stimulus_index_enc1', 'stimulus_index_enc2', 'stimulus_index_enc3', 'response_accuracy']]
y_matrix = subject_trials

df_delay_filtered = pd.read_excel('../clean_data/graph_dela.xlsx')
df_fixation = pd.read_excel('../clean_data/cleaned_Fixation.xlsx')
df_clustering = pd.read_excel('../revision_clustering_waveform_labels.xlsx')

# =========================
# SETTINGS
# =========================

bin_sizes = [0.25, 0.5, 0.75, 1.0]
step_size = 0.025
durations = {1: 2.5, 2: 2.5, 3: 2.5}
n_bootstrap = 100
n_splits = 3
n_repeats = 10
chance_level = 20
rng = np.random.default_rng(42)

# =========================
# HELPERS
# =========================

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
    ])

def repeated_stratified_cv_accuracy(X, y, n_splits=3, n_repeats=10, seed=42):
    """
    Returns mean accuracy (%) across n_repeats of stratified k-fold CV.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    unique_classes, class_counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2 or np.min(class_counts) < n_splits:
        return np.nan

    fold_accs = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(np.unique(y_train)) < 2:
                continue

            clf = SVC(kernel='linear', random_state=seed + rep)
            clf.fit(X_train, y_train)
            fold_accs.append(np.mean(clf.predict(X_test) == y_test) * 100)

    return np.nanmean(fold_accs) if fold_accs else np.nan

def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) <= 1:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(len(x))

# =========================
# FILTER PY / IN NEURONS
# =========================

# Keep only significant neurons and only PY / IN
df_clustering_filtered = df_clustering[
    (df_clustering['Cell_Type_New'].isin(['PY', 'IN'])) &
    (df_clustering['Signi'] == 'Y')
].copy()

# Unique neuron pools
py_neurons = df_clustering_filtered.loc[
    df_clustering_filtered['Cell_Type_New'] == 'PY', 'Neuron_ID_3'
].dropna().unique()

in_neurons = df_clustering_filtered.loc[
    df_clustering_filtered['Cell_Type_New'] == 'IN', 'Neuron_ID_3'
].dropna().unique()

# Keep only neurons that are actually present in delay and fixation
delay_neurons = df_delay_filtered['Neuron_ID_3'].dropna().unique()
fix_neurons = df_fixation['Neuron_ID_3'].dropna().unique()
available_neurons = np.intersect1d(delay_neurons, fix_neurons)

py_neurons = np.intersect1d(py_neurons, available_neurons)
in_neurons = np.intersect1d(in_neurons, available_neurons)

print(f"Number of available significant PY neurons: {len(py_neurons)}")
print(f"Number of available significant IN neurons: {len(in_neurons)}")

if len(py_neurons) == 0 or len(in_neurons) == 0:
    raise ValueError("After filtering, one of the neuron groups is empty.")

sample_size = min(len(py_neurons), len(in_neurons))
print(f"Matched bootstrap sample size: {sample_size}")

# Master neuron list for building the design matrix once
master_neurons = np.array(sorted(np.union1d(py_neurons, in_neurons)))
neuron_to_col = {nid: idx for idx, nid in enumerate(master_neurons)}

trial_count = y_matrix.shape[0]
neuron_count = len(master_neurons)

design_matrix = np.empty((trial_count, neuron_count), dtype=object)
design_matrix[:] = None

# Build a quick lookup dictionary from delay file
delay_lookup = {}
for _, row in df_delay_filtered.iterrows():
    key = (row['trial_id_final'], row['Neuron_ID_3'])
    delay_lookup[key] = parse_spike_times(row['Standardized_Spikes'])

for trial_idx, row in y_matrix.iterrows():
    trial_id = row['trial_id_final']
    for neuron_id in master_neurons:
        col_idx = neuron_to_col[neuron_id]
        spikes = delay_lookup.get((trial_id, neuron_id), [])
        design_matrix[trial_idx, col_idx] = np.array(spikes, dtype=float)

df_fixation_filtered = df_fixation[df_fixation['Neuron_ID_3'].isin(master_neurons)].copy()

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

# Valid trials only
valid_trials = ~pd.isna(y_labels)
y_matrix = y_matrix.loc[valid_trials].reset_index(drop=True)
y_labels = y_labels[valid_trials]
design_matrix = design_matrix[valid_trials]

load_trials = {
    1: np.where(y_matrix['num_images_presented'].values == 1)[0],
    2: np.where(y_matrix['num_images_presented'].values == 2)[0],
    3: np.where(y_matrix['num_images_presented'].values == 3)[0],
}

def build_matrix_subset(trials, sampled_neurons, bin_size, end_time):
    """
    sampled_neurons can include duplicates because sampling is with replacement.
    Returns X of shape (n_trials, n_sampled_neurons * n_time_bins)
    """
    time_bins = create_time_bins(0, end_time, bin_size, step_size)
    n_trials = len(trials)
    n_neurons = len(sampled_neurons)
    n_bins = len(time_bins)

    matrix = np.zeros((n_trials, n_neurons, n_bins), dtype=float)

    for i, trial_idx in enumerate(trials):
        for j, neuron_id in enumerate(sampled_neurons):
            master_idx = neuron_to_col[neuron_id]
            spikes = design_matrix[trial_idx, master_idx]

            binned = count_spikes_in_bins(spikes, time_bins, bin_size)

            mean_fix = fixation_means.get(neuron_id, 0)
            std_fix = fixation_stds.get(neuron_id, 1)

            if pd.isna(std_fix) or std_fix == 0:
                matrix[i, j, :] = binned
            else:
                matrix[i, j, :] = (binned - mean_fix) / std_fix

    return matrix.reshape(n_trials, -1)

# =========================
# BOOTSTRAP + SHUFFLE DECODING
# =========================

n_shuffle = 100   # label-permutation iterations

results = {}   # results[load][bin_size]

for load in [1, 2, 3]:
    trial_idxs = load_trials[load]
    y_load     = y_labels[trial_idxs]
    results[load] = {}

    if len(trial_idxs) == 0:
        print(f"Skipping Load {load}: no trials found.")
        continue

    for bin_size in tqdm(bin_sizes, desc=f"Load {load}"):
        py_real, in_real = [], []
        py_shuf, in_shuf = [], []

        # ── Real: n_bootstrap samples with replacement (unchanged) ──────────
        for b in range(n_bootstrap):
            sampled_py = rng.choice(py_neurons, size=sample_size, replace=True)
            sampled_in = rng.choice(in_neurons, size=sample_size, replace=True)
            X_py = build_matrix_subset(trial_idxs, sampled_py, bin_size, durations[load])
            X_in = build_matrix_subset(trial_idxs, sampled_in, bin_size, durations[load])
            py_real.append(repeated_stratified_cv_accuracy(
                X_py, y_load, n_splits=n_splits, n_repeats=n_repeats, seed=42 + b))
            in_real.append(repeated_stratified_cv_accuracy(
                X_in, y_load, n_splits=n_splits, n_repeats=n_repeats, seed=42 + b))

        # ── Shuffle: n_shuffle permutations, each with one bootstrap draw ───
        for s in range(n_shuffle):
            y_shuf     = rng.permutation(y_load)
            sampled_py = rng.choice(py_neurons, size=sample_size, replace=True)
            sampled_in = rng.choice(in_neurons, size=sample_size, replace=True)
            X_py = build_matrix_subset(trial_idxs, sampled_py, bin_size, durations[load])
            X_in = build_matrix_subset(trial_idxs, sampled_in, bin_size, durations[load])
            py_shuf.append(repeated_stratified_cv_accuracy(
                X_py, y_shuf, n_splits=n_splits, n_repeats=n_repeats, seed=42 + s))
            in_shuf.append(repeated_stratified_cv_accuracy(
                X_in, y_shuf, n_splits=n_splits, n_repeats=n_repeats, seed=42 + s))

        results[load][bin_size] = {
            'PY_real': np.array(py_real, dtype=float),
            'IN_real': np.array(in_real, dtype=float),
            'PY_shuf': np.array(py_shuf, dtype=float),
            'IN_shuf': np.array(in_shuf, dtype=float),
        }

# =========================
# BOOTSTRAP + PAIRED SHUFFLE DECODING
# =========================

n_shuffle = 100
n_perm_stats = 1000

results = {}

for load in [1, 2, 3]:
    trial_idxs = load_trials[load]
    y_load = y_labels[trial_idxs]
    results[load] = {}

    if len(trial_idxs) == 0:
        print(f"Skipping Load {load}: no trials found.")
        continue

    for bin_size in tqdm(bin_sizes, desc=f"Load {load}"):

        py_real, in_real = [], []
        py_shuf, in_shuf = [], []

        for b in range(n_bootstrap):

            sampled_py = rng.choice(py_neurons, size=sample_size, replace=True)
            sampled_in = rng.choice(in_neurons, size=sample_size, replace=True)

            X_py = build_matrix_subset(trial_idxs, sampled_py, bin_size, durations[load])
            X_in = build_matrix_subset(trial_idxs, sampled_in, bin_size, durations[load])

            real_py_acc = repeated_stratified_cv_accuracy(
                X_py, y_load, n_splits=n_splits, n_repeats=n_repeats, seed=42 + b
            )

            real_in_acc = repeated_stratified_cv_accuracy(
                X_in, y_load, n_splits=n_splits, n_repeats=n_repeats, seed=142 + b
            )

            y_shuf = rng.permutation(y_load)

            shuf_py_acc = repeated_stratified_cv_accuracy(
                X_py, y_shuf, n_splits=n_splits, n_repeats=n_repeats, seed=242 + b
            )

            shuf_in_acc = repeated_stratified_cv_accuracy(
                X_in, y_shuf, n_splits=n_splits, n_repeats=n_repeats, seed=342 + b
            )

            py_real.append(real_py_acc)
            in_real.append(real_in_acc)
            py_shuf.append(shuf_py_acc)
            in_shuf.append(shuf_in_acc)

        results[load][bin_size] = {
            'PY_real': np.array(py_real, dtype=float),
            'IN_real': np.array(in_real, dtype=float),
            'PY_shuf': np.array(py_shuf, dtype=float),
            'IN_shuf': np.array(in_shuf, dtype=float),
        }


# =========================
# STATISTICS + EXCEL EXPORT
# Paired sign-flip permutation tests + FDR
# =========================

from statsmodels.stats.multitest import multipletests

folder_path = './'
os.makedirs(folder_path, exist_ok=True)

n_perm_stats = 1000

def clean_array(x):
    x = np.asarray(x, dtype=float)
    return x[~np.isnan(x)]

def clean_paired(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    return a[mask], b[mask]

def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) <= 1:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(len(x))

def stars_from_p(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

def paired_sign_flip_test(a, b, n_permutations=1000, seed=42, alternative='two-sided'):
    """
    Paired sign-flip permutation test.

    Tests mean(a - b).

    alternative='greater': mean(a - b) > 0
    alternative='less': mean(a - b) < 0
    alternative='two-sided': mean(a - b) != 0
    """

    rng_local = np.random.default_rng(seed)

    a, b = clean_paired(a, b)
    n_used = len(a)

    if n_used < 2:
        return np.nan, np.nan, n_used

    diffs = a - b
    observed_diff = np.mean(diffs)

    null_diffs = np.zeros(n_permutations)

    for i in range(n_permutations):
        signs = rng_local.choice([-1, 1], size=n_used, replace=True)
        null_diffs[i] = np.mean(diffs * signs)

    if alternative == 'greater':
        p_value = (np.sum(null_diffs >= observed_diff) + 1) / (n_permutations + 1)
    elif alternative == 'less':
        p_value = (np.sum(null_diffs <= observed_diff) + 1) / (n_permutations + 1)
    elif alternative == 'two-sided':
        p_value = (np.sum(np.abs(null_diffs) >= np.abs(observed_diff)) + 1) / (n_permutations + 1)
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")

    return p_value, observed_diff, n_used

def apply_fdr(df, p_col='p_raw', alpha=0.05):
    df = df.copy()
    df['p_fdr'] = np.nan
    df['significant_fdr'] = False

    valid = df[p_col].notna().values

    if valid.sum() > 0:
        reject, p_corr, _, _ = multipletests(
            df.loc[valid, p_col].values,
            alpha=alpha,
            method='fdr_bh'
        )
        df.loc[valid, 'p_fdr'] = p_corr
        df.loc[valid, 'significant_fdr'] = reject

    df['stars_fdr'] = df['p_fdr'].apply(stars_from_p)
    return df


# =========================
# REAL VS SHUFFLED STATS
# =========================

real_vs_shuffle_rows = []

for load in [1, 2, 3]:
    for bin_size in bin_sizes:
        for cell_type in ['PY', 'IN']:

            r = results[load][bin_size]

            real = r[f'{cell_type}_real']
            shuf = r[f'{cell_type}_shuf']

            real_clean, shuf_clean = clean_paired(real, shuf)

            p_raw, observed_diff, n_pairs = paired_sign_flip_test(
                real_clean,
                shuf_clean,
                n_permutations=n_perm_stats,
                seed=10000 + load * 1000 + int(bin_size * 1000) + (0 if cell_type == 'PY' else 1),
                alternative='greater'
            )

            real_vs_shuffle_rows.append({
                'Load': load,
                'Bin_ms': int(bin_size * 1000),
                'Cell_Type': cell_type,
                'n_pairs': n_pairs,

                'Real_mean_accuracy': np.nanmean(real_clean),
                'Real_sem_accuracy': sem(real_clean),
                'Real_std_accuracy': np.nanstd(real_clean, ddof=1),

                'Shuffled_mean_accuracy': np.nanmean(shuf_clean),
                'Shuffled_sem_accuracy': sem(shuf_clean),
                'Shuffled_std_accuracy': np.nanstd(shuf_clean, ddof=1),

                'Observed_diff_real_minus_shuffled': observed_diff,
                'Test': 'Paired sign-flip permutation test; one-sided real > shuffled',
                'p_raw': p_raw
            })

real_vs_shuffle_df = pd.DataFrame(real_vs_shuffle_rows)
real_vs_shuffle_df = apply_fdr(real_vs_shuffle_df, p_col='p_raw', alpha=0.05)


# =========================
# PY VS IN STATS PER WINDOW
# =========================

py_vs_in_rows = []

for load in [1, 2, 3]:
    for bin_size in bin_sizes:

        r = results[load][bin_size]

        in_real = r['IN_real']
        py_real = r['PY_real']

        in_clean, py_clean = clean_paired(in_real, py_real)

        p_raw, observed_diff, n_pairs = paired_sign_flip_test(
            in_clean,
            py_clean,
            n_permutations=n_perm_stats,
            seed=20000 + load * 1000 + int(bin_size * 1000),
            alternative='greater'
        )

        py_vs_in_rows.append({
            'Load': load,
            'Bin_ms': int(bin_size * 1000),
            'n_pairs': n_pairs,

            'PY_mean_accuracy': np.nanmean(py_clean),
            'PY_sem_accuracy': sem(py_clean),
            'PY_std_accuracy': np.nanstd(py_clean, ddof=1),

            'IN_mean_accuracy': np.nanmean(in_clean),
            'IN_sem_accuracy': sem(in_clean),
            'IN_std_accuracy': np.nanstd(in_clean, ddof=1),

            'Observed_diff_IN_minus_PY': observed_diff,
            'Test': 'Paired sign-flip permutation test; one-sided IN > PY',
            'p_raw': p_raw
        })

py_vs_in_df = pd.DataFrame(py_vs_in_rows)
py_vs_in_df = apply_fdr(py_vs_in_df, p_col='p_raw', alpha=0.05)


# =========================
# TEMPORAL-WINDOW COMPARISONS
# =========================

temporal_rows = []

for load in [1, 2, 3]:
    for cell_type in ['PY', 'IN']:
        for i, bin_a in enumerate(bin_sizes):
            for bin_b in bin_sizes[i + 1:]:

                acc_a = results[load][bin_a][f'{cell_type}_real']
                acc_b = results[load][bin_b][f'{cell_type}_real']

                acc_a_clean, acc_b_clean = clean_paired(acc_a, acc_b)

                p_raw, observed_diff, n_pairs = paired_sign_flip_test(
                    acc_b_clean,
                    acc_a_clean,
                    n_permutations=n_perm_stats,
                    seed=30000 + load * 1000 + int(bin_a * 1000) + int(bin_b * 1000) + (0 if cell_type == 'PY' else 1),
                    alternative='two-sided'
                )

                temporal_rows.append({
                    'Load': load,
                    'Cell_Type': cell_type,
                    'Bin_A_ms': int(bin_a * 1000),
                    'Bin_B_ms': int(bin_b * 1000),
                    'n_pairs': n_pairs,

                    'Mean_A_accuracy': np.nanmean(acc_a_clean),
                    'SEM_A_accuracy': sem(acc_a_clean),
                    'STD_A_accuracy': np.nanstd(acc_a_clean, ddof=1),

                    'Mean_B_accuracy': np.nanmean(acc_b_clean),
                    'SEM_B_accuracy': sem(acc_b_clean),
                    'STD_B_accuracy': np.nanstd(acc_b_clean, ddof=1),

                    'Observed_diff_B_minus_A': observed_diff,
                    'Test': 'Paired sign-flip permutation test; two-sided bin comparison',
                    'p_raw': p_raw
                })

temporal_window_df = pd.DataFrame(temporal_rows)
temporal_window_df = apply_fdr(temporal_window_df, p_col='p_raw', alpha=0.05)

long_rows = []

for load in [1, 2, 3]:
    for bin_size in bin_sizes:
        r = results[load][bin_size]

        for cell_type in ['PY', 'IN']:
            for i, val in enumerate(r[f'{cell_type}_real']):
                long_rows.append({
                    'Load': load,
                    'Bin_ms': int(bin_size * 1000),
                    'Cell_Type': cell_type,
                    'Condition': 'Real',
                    'Iteration': i,
                    'Accuracy': val
                })

            for i, val in enumerate(r[f'{cell_type}_shuf']):
                long_rows.append({
                    'Load': load,
                    'Bin_ms': int(bin_size * 1000),
                    'Cell_Type': cell_type,
                    'Condition': 'Shuffled',
                    'Iteration': i,
                    'Accuracy': val
                })

df_long = pd.DataFrame(long_rows)

average_accuracy_df = (
    df_long
    .groupby(['Load', 'Bin_ms', 'Cell_Type', 'Condition'], as_index=False)
    .agg(
        Mean_accuracy=('Accuracy', 'mean'),
        SEM_accuracy=('Accuracy', sem),
        STD_accuracy=('Accuracy', lambda x: np.nanstd(x, ddof=1)),
        N_iterations=('Accuracy', lambda x: np.sum(~pd.isna(x)))
    )
)

real_average_py_in_df = average_accuracy_df[
    average_accuracy_df['Condition'] == 'Real'
].copy()

excel_path = os.path.join(
    folder_path,
    'PY_IN_decoding_paired_signflip_FDR_statistics.xlsx'
)

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df_long.to_excel(writer, sheet_name='raw_long_values', index=False)
    average_accuracy_df.to_excel(writer, sheet_name='average_accuracy_all', index=False)
    real_average_py_in_df.to_excel(writer, sheet_name='real_avg_PY_IN_by_bin', index=False)
    real_vs_shuffle_df.to_excel(writer, sheet_name='real_vs_shuffled_stats', index=False)
    py_vs_in_df.to_excel(writer, sheet_name='PY_vs_IN_stats', index=False)
    temporal_window_df.to_excel(writer, sheet_name='temporal_window_stats', index=False)
