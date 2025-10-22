import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.utils import shuffle
import ast
import os
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests

trial_info = pd.read_excel('../new_trial_final.xlsx')
subject_trials = trial_info[trial_info['subject_id'] == 14][['trial_id_final', 'num_images_presented', 'stimulus_index_enc1', 'stimulus_index_enc2', 'stimulus_index_enc3', 'response_accuracy']]
print(subject_trials)

y_matrix = subject_trials
df_delay_filtered = pd.read_excel('../graph_data/graph_delay.xlsx')
df_fixation = pd.read_excel('../clean_data/cleaned_Fixation.xlsx')

y_matrix = y_matrix.reset_index(drop=True)

# Filter PY or IN neurons from df_clustering
#filtered_clustering = df_clustering[df_clustering['Cell_Type_New'].isin(['IN'])]

# Get list of neuron IDs that are PY or IN
#py_in_neurons = filtered_clustering['Neuron_ID_3'].unique()

#df_delay_filtered = df_delay_filtered[df_delay_filtered['Neuron_ID_3'].isin(py_in_neurons)]

final_neurons = sorted(df_delay_filtered[df_delay_filtered['Signi'] == "Y"]['Neuron_ID_3'].unique())

trial_count = y_matrix.shape[0]
neuron_count = len(final_neurons)

design_matrix = np.empty((trial_count, neuron_count), dtype=object)

def parse_spike_times(spike_str):
    try:
        return ast.literal_eval(spike_str)
    except:
        return []

for trial_idx, row in y_matrix.iterrows():
    trial_id = row['trial_id_final']
    for neuron_idx, neuron_id in enumerate(final_neurons):
        match = df_delay_filtered[
            (df_delay_filtered['trial_id_final'] == trial_id) &
            (df_delay_filtered['Neuron_ID_3'] == neuron_id)
        ]
        if not match.empty:
            spikes = parse_spike_times(match['Standardized_Spikes'].values[0])
            design_matrix[trial_idx, neuron_idx] = np.array(spikes)
        else:
            design_matrix[trial_idx, neuron_idx] = np.array([])


df_fixation_filtered = df_fixation[df_fixation['Neuron_ID_3'].isin(final_neurons)]
fixation_means = df_fixation_filtered.groupby('Neuron_ID_3')['Spikes_rate_Fixation'].mean().to_dict()
fixation_stds = df_fixation_filtered.groupby('Neuron_ID_3')['Spikes_rate_Fixation'].std().to_dict()


def create_time_bins(start_time, end_time, bin_size, step_size):
    return np.arange(start_time, end_time - bin_size + step_size, step_size)

def count_spikes_in_bins(spike_times, time_bins, bin_size):
    return np.array([np.sum((spike_times >= t) & (spike_times < t + bin_size)) for t in time_bins])

def leave_one_out_cv_vec(X, y, global_indices, scramble_labels=False, random_state=None):
    results = []
    for i in range(len(global_indices)):
        test_global_idx = global_indices[i]
        test_local_idx = i

        train_local_idx = np.delete(np.arange(len(global_indices)), i)
        train_global_idx = [global_indices[j] for j in train_local_idx]

        X_train = X[train_local_idx]
        y_train = y[train_global_idx]
        if scramble_labels:
            y_train = shuffle(y_train, random_state=random_state)
        clf = SVC(kernel='linear', random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X[[test_local_idx]])
        results.append(int(y_pred[0] == y[test_global_idx]))
    return np.array(results)  # 1D array: length n_trials, entries 0/1

rng = np.random.default_rng(42)

def permutation_test_cv(X, y, global_indices, n_permutations=1000):
    permuted_accuracies = []
    for _ in range(n_permutations):
        y_shuffled = rng.permutation(y[global_indices])
        accs_vec = leave_one_out_cv_vec(X, y_shuffled, np.arange(len(global_indices)))
        acc = accs_vec.mean() * 100
        permuted_accuracies.append(acc)
    return np.array(permuted_accuracies)

# Construct y_labels for each trial (1-3 images)
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

load_trials = [
    y_matrix[y_matrix['num_images_presented'] == 1].index.values,
    y_matrix[y_matrix['num_images_presented'] == 2].index.values,
    y_matrix[y_matrix['num_images_presented'] == 3].index.values
]


bin_sizes = [0.25, 0.5, 0.75, 1.0]
step_size = 0.025
durations = {1: 2.5, 2: 2.5, 3: 2.5}
n_permutations = 1000

regions = [f'{int(b * 1000)} ms' for b in bin_sizes]
categories = ['Load1', 'Load2', 'Load3']

results = []
perm_distributions = []

def build_matrix(trials, bin_size, end_time):
    time_bins = create_time_bins(0, end_time, bin_size, step_size)
    matrix = np.zeros((len(trials), neuron_count, len(time_bins)))
    for i, trial_idx in enumerate(trials):
        for j, neuron_id in enumerate(final_neurons):
            spikes = design_matrix[trial_idx, j]
            binned = count_spikes_in_bins(spikes, time_bins, bin_size)
            mean = fixation_means.get(neuron_id, 0)
            std = fixation_stds.get(neuron_id, 1)
            matrix[i, j, :] = (binned - mean) / std if std > 0 else binned
    return matrix.reshape(len(trials), -1)

all_load_results = {b: {} for b in bin_sizes}
all_nulls = {b: {} for b in bin_sizes}

for bin_size in tqdm(bin_sizes, desc="Bin sizes"):
    real_acc_means = []
    real_acc_sems = []
    null_means = []
    null_sems = []
    nulls_for_this_bin = []

    for lid, trial_idxs in enumerate(load_trials):  # Load1, 2, 3
        X = build_matrix(trial_idxs, bin_size, durations[lid + 1])
        accs_vec = leave_one_out_cv_vec(X, y_labels, trial_idxs)
        acc_mean = accs_vec.mean() * 100
        acc_sem = accs_vec.std(ddof=1) / np.sqrt(len(accs_vec)) * 100
        real_acc_means.append(acc_mean)
        real_acc_sems.append(acc_sem)

        nulls = permutation_test_cv(X, y_labels, trial_idxs, n_permutations=n_permutations)
        null_means.append(nulls.mean())
        null_sems.append(nulls.std(ddof=1) / np.sqrt(n_permutations))
        nulls_for_this_bin.append(nulls)
        all_load_results[bin_size][categories[lid]] = {'mean': acc_mean, 'sem': acc_sem}
        all_nulls[bin_size][categories[lid]] = nulls

    # Shuffled "bar" for this time bin 
    avg_shuffled = np.mean(null_means)
    nulls_stack = np.vstack(nulls_for_this_bin)
    avg_shuffled_distribution = nulls_stack.mean(axis=0)
    avg_shuffled_sem = avg_shuffled_distribution.std(ddof=1) / np.sqrt(n_permutations)
    results.append({
        'real_acc_means': real_acc_means,
        'real_acc_sems': real_acc_sems,
        'avg_shuffled': avg_shuffled,
        'avg_shuffled_sem': avg_shuffled_sem,
        'null_means': null_means,
        'null_sems': null_sems,
        'nulls_for_this_bin': nulls_for_this_bin,
        'avg_shuffled_distribution': avg_shuffled_distribution
    })
    perm_distributions.append(nulls_for_this_bin)

p_values = np.zeros((len(bin_sizes), 3))
for bin_idx, r in enumerate(results):
    for load in range(3):
        real_acc = r['real_acc_means'][load]
        null_dist = r['nulls_for_this_bin'][load]
        p = (np.sum(null_dist >= real_acc) + 1) / (len(null_dist) + 1)
        p_values[bin_idx, load] = p

p_values_flat = p_values.flatten()
rejected, pvals_fdr, _, _ = multipletests(p_values_flat, alpha=0.05, method='fdr_bh')
pvals_fdr = pvals_fdr.reshape(p_values.shape)
rejected = rejected.reshape(p_values.shape)

plt.figure(figsize=(14, 8))
bar_width = 0.18
x = np.arange(len(regions))
bars_positions = []
p_threshold = 0.05
chance_level = 20

for i in range(3):
    offsets = x + i * bar_width - (bar_width * 3 / 2) + bar_width / 2
    means = [results[j]['real_acc_means'][i] for j in range(len(regions))]
    sems = [results[j]['real_acc_sems'][i] for j in range(len(regions))]
    bars = plt.bar(offsets, means, bar_width, yerr=sems, label=categories[i], capsize=5)
    bars_positions.append(offsets)
    for j, (p_val, p_fdr) in enumerate(zip(p_values[:, i], pvals_fdr[:, i])):

        if p_fdr < 0.005:
            stars = '***'
        elif p_fdr < 0.01:
            stars = '**'
        elif p_fdr < 0.05:
            stars = '*'
        else:
            stars = ''
        if stars:
            plt.text(offsets[j], means[j] + sems[j] + 7, stars,
                    ha='center', fontsize=16, fontweight='bold', color='r')

shuffled_offsets = x + 3 * bar_width - (bar_width * 3 / 2) + bar_width / 2
shuf_means = [results[j]['avg_shuffled'] for j in range(len(regions))]
shuf_sems = [results[j]['avg_shuffled_sem'] for j in range(len(regions))]
plt.bar(shuffled_offsets, shuf_means, bar_width, yerr=shuf_sems,
        label='Shuffled (avg)', color='gray', capsize=5)
bars_positions.append(shuffled_offsets)

plt.axhline(y=chance_level, color='gray', linestyle='--', label=f'Chance ({chance_level}%)')
plt.title('SVM Decoding Accuracy by Load and Bin Size', fontsize=16)
plt.ylabel('Accuracy (%)')
plt.xticks(x, regions)
plt.ylim(0, 100)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

folder_path = './'
os.makedirs(folder_path, exist_ok=True)
save_path = os.path.join(folder_path, "decoding_timebins_new.eps")
plt.savefig(save_path, format='eps', dpi=300)
plt.show()

comparison_results = []
for i, region in enumerate(regions):
    for k in range(3):
        comparison_results.append({
            'Time Bin': region,
            'Condition': categories[k],
            'Accuracy (%)': results[i]['real_acc_means'][k],
            'SEM (Real)': results[i]['real_acc_sems'][k],
            'Mean Shuffled Accuracy (%)': results[i]['null_means'][k],
            'SEM (Shuffled)': results[i]['null_sems'][k],
            'p-value': p_values[i, k],
            'Significant (p<0.05)': 'Yes' if p_values[i, k] < p_threshold else 'No',
            'p-value (FDR)': pvals_fdr[i, k],
            'Significant (FDR)': 'Yes' if rejected[i, k] else 'No'
        })
    comparison_results.append({
        'Time Bin': region,
        'Condition': 'Shuffled (avg)',
        'Accuracy (%)': results[i]['avg_shuffled'],
        'SEM (Real)': '',
        'Mean Shuffled Accuracy (%)': '',
        'SEM (Shuffled)': results[i]['avg_shuffled_sem'],
        'p-value': '',
        'Significant (p<0.05)': '',
        'p-value (FDR)': '',
        'Significant (FDR)': ''
    })

df_comparisons = pd.DataFrame(comparison_results)
print("\nReal vs Shuffled Comparisons (with SEMs and FDR):")
print(df_comparisons)
