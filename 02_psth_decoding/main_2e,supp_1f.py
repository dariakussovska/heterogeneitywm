import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import ast
from statsmodels.stats.multitest import multipletests

# Load data
df_delay = pd.read_excel('../graph_data/graph_delay.xlsx')
df_fixation = pd.read_excel('../clean_data/cleaned_Fixation.xlsx')
df_regions = pd.read_excel('../data/all_neuron_brain_regions_cleaned.xlsx')
trial_info = pd.read_excel('../trial_info.xlsx')

region_groups = {
    'amygdala': ['amygdala_left', 'amygdala_right'],
    'hippocampus': ['hippocampus_left', 'hippocampus_right'],
    'dACC': ['dorsal_anterior_cingulate_cortex_right', 'dorsal_anterior_cingulate_cortex_left'],
    'preSMA': ['pre_supplementary_motor_area_left', 'pre_supplementary_motor_area_right'],
    'amygdala_hippocampus': ['amygdala_left', 'amygdala_right', 'hippocampus_left', 'hippocampus_right'],
    'dACC_preSMA': ['dorsal_anterior_cingulate_cortex_right', 'dorsal_anterior_cingulate_cortex_left',
                    'pre_supplementary_motor_area_left', 'pre_supplementary_motor_area_right']
}

bin_size = 1
step_size = 0.25
durations = {1: 2.5, 2: 2.5, 3: 2.5}
time_bins_by_load = {l: np.arange(0, durations[l] - bin_size + step_size, step_size) for l in durations}
n_shuffles = 1000
rng = np.random.default_rng(42)  # reproducible random generator

# Subject color map
all_subject_ids = sorted(trial_info['subject_id'].unique())
cmap = plt.colormaps.get_cmap('tab20')
subject_color_map = {subj: cmap(i % 20) for i, subj in enumerate(all_subject_ids)}

def parse_spike_times(spike_str):
    try:
        return ast.literal_eval(spike_str)
    except:
        return []

def count_spikes(spikes, bins, bin_size):
    return np.array([np.sum((spikes >= t) & (spikes < t + bin_size)) for t in bins])

def leave_one_out_cv(X, y):
    correct = 0
    for i in range(len(X)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i].reshape(1, -1)
        y_test = y[i]
        clf = SVC(kernel='linear', random_state=42)
        clf.fit(X_train, y_train)
        correct += (clf.predict(X_test)[0] == y_test)
    return (correct / len(X)) * 100

results = {}

for region, locations in region_groups.items():
    per_subject_obs = []
    per_subject_shuf = []
    per_subject_ids = []

    for subj in all_subject_ids:
        neuron_ids = df_regions[
            (df_regions['Location'].isin(locations)) &
            (df_regions['subject_id'] == subj) &
            (df_regions['Signi'] == 'Y')
        ]['Neuron_ID_3'].unique()
        if len(neuron_ids) <= 3:
            continue

        y_matrix = trial_info[trial_info['subject_id'] == subj][[
            'trial_id', 'num_images_presented', 'stimulus_index_enc1',
            'stimulus_index_enc2', 'stimulus_index_enc3']].reset_index(drop=True)

        df_r = df_delay[(df_delay['subject_id'] == subj) & (df_delay['Neuron_ID_3'].isin(neuron_ids))]
        df_f = df_fixation[(df_fixation['subject_id'] == subj) & (df_fixation['Neuron_ID_3'].isin(neuron_ids))]
        means = df_f.groupby('Neuron_ID_3')['Spikes_rate_Fixation'].mean().to_dict()
        stds = df_f.groupby('Neuron_ID_3')['Spikes_rate_Fixation'].std().to_dict()

        # Get labels (same as before)
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

        # Design matrix
        design_matrix = np.empty((len(y_matrix), len(neuron_ids)), dtype=object)
        for i, row in y_matrix.iterrows():
            tid = row['trial_id']
            for j, n in enumerate(neuron_ids):
                match = df_r[(df_r['trial_id'] == tid) & (df_r['Neuron_ID_3'] == n)]
                spikes = parse_spike_times(match['Standardized_Spikes'].values[0]) if not match.empty else []
                design_matrix[i, j] = np.array(spikes)

        obs_accs = []
        shuf_accs = []
        for load in [1, 2, 3]:
            idxs = y_matrix[y_matrix['num_images_presented'] == load].index.values
            if len(idxs) < 2:
                obs_accs.append(np.nan)
                shuf_accs.append(np.full(n_shuffles, np.nan))
                continue

            bins = time_bins_by_load[load]
            X = np.zeros((len(idxs), len(neuron_ids) * len(bins)))
            for i, idx in enumerate(idxs):
                for j, n in enumerate(neuron_ids):
                    spikes = design_matrix[idx, j]
                    binned = count_spikes(spikes, bins, bin_size)
                    mean = means.get(n, 0)
                    std = stds.get(n, 1)
                    X[i, j*len(bins):(j+1)*len(bins)] = (binned - mean) / std if std > 0 else binned
            y = y_labels[idxs]
            acc_real = leave_one_out_cv(X, y)
            obs_accs.append(acc_real)
            accs_shuf = [leave_one_out_cv(X, rng.permutation(y)) for _ in range(n_shuffles)]
            shuf_accs.append(np.array(accs_shuf))

        per_subject_obs.append(obs_accs)
        per_subject_shuf.append(shuf_accs)
        per_subject_ids.append(subj)

    # Force all to regular arrays
    num_subjects = len(per_subject_obs)
    obs_arr = np.full((num_subjects, 3), np.nan)
    shuf_arr = np.full((num_subjects, 3, n_shuffles), np.nan)
    for s in range(num_subjects):
        for l in range(3):
            obs_arr[s, l] = per_subject_obs[s][l]
            shuf_arr[s, l, :] = per_subject_shuf[s][l]
    # Save
    results[region] = {
        'observed': obs_arr,            # (n_subjects, 3)
        'shuffled': shuf_arr,           # (n_subjects, 3, n_shuffles)
        'subject_ids': per_subject_ids  # list
    }

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.multitest import multipletests

region_names = list(results.keys())
bar_width = 0.2
x = np.arange(len(region_names))
load_labels = ['Load 1', 'Load 2', 'Load 3']

fig, ax = plt.subplots(figsize=(14, 6))

# (1) Plot REAL LOADS (bars + subject dots)
for load_idx in range(3):
    pos = x + load_idx * bar_width - bar_width
    heights = []
    errors = []
    for region in region_names:
        # Check if there are any subjects
        obs = results[region]['observed']
        if obs.shape[0] == 0:
            heights.append(np.nan)
            errors.append(np.nan)
        else:
            heights.append(np.nanmean(obs[:, load_idx]))
            valid = ~np.isnan(obs[:, load_idx])
            n = np.sum(valid)
            if n > 0:
                errors.append(np.nanstd(obs[:, load_idx]) / np.sqrt(n))
            else:
                errors.append(np.nan)
    ax.bar(pos, heights, width=bar_width, color=plt.cm.tab10(load_idx), 
           label=load_labels[load_idx], yerr=errors, capsize=4)
    for reg_idx, region in enumerate(region_names):
        obs = results[region]['observed']
        if obs.shape[0] == 0:
            continue
        for subj_idx, subj_id in enumerate(results[region]['subject_ids']):
            val = obs[subj_idx, load_idx]
            if not np.isnan(val):
                ax.scatter(pos[reg_idx] + np.random.normal(0, 0.02), val,
                          color=subject_color_map[subj_id], edgecolor='black', s=40)

# (2) Shuffled: average per subject across all loads and shuffles
for region in region_names:
    subj_shuffles = results[region]['shuffled']
    if subj_shuffles.shape[0] == 0:
        results[region]['avg_shuffled_per_subject'] = np.full(0, np.nan)
    else:
        results[region]['avg_shuffled_per_subject'] = np.nanmean(subj_shuffles, axis=(1, 2)) # (subjects,)

avg_shuffled = []
shuffled_err = []
for region in region_names:
    avg_subj = results[region]['avg_shuffled_per_subject']
    if avg_subj.shape[0] == 0:
        avg_shuffled.append(np.nan)
        shuffled_err.append(np.nan)
    else:
        avg_shuffled.append(np.nanmean(avg_subj))
        valid = ~np.isnan(avg_subj)
        n = np.sum(valid)
        if n > 0:
            shuffled_err.append(np.nanstd(avg_subj) / np.sqrt(n))
        else:
            shuffled_err.append(np.nan)

ax.bar(x + 3 * bar_width - bar_width, avg_shuffled, width=bar_width, 
       color='gray', label='Shuffled (avg)', yerr=shuffled_err, capsize=4, alpha=0.7)

for reg_idx, region in enumerate(region_names):
    avg_subj = results[region]['avg_shuffled_per_subject']
    if avg_subj.shape[0] == 0:
        continue
    for subj_idx, subj_id in enumerate(results[region]['subject_ids']):
        val = avg_subj[subj_idx]
        if not np.isnan(val):
            ax.scatter(x[reg_idx] + 3 * bar_width - bar_width + np.random.normal(0, 0.02), 
                  val, color=subject_color_map[subj_id], edgecolor='black', s=40)

# (3) Permutation test stats (real vs shuffled mean across subjects)
p_values = np.full((len(region_names), 3), np.nan)
for reg_idx, region in enumerate(region_names):
    obs = results[region]['observed']
    shuf = results[region]['shuffled']
    if obs.shape[0] == 0 or (shuf.shape[0] == 0):
        continue
    for load_idx in range(3):
        # Get real values (across subjects)
        real = obs[:, load_idx]
        shuf_vals = shuf[:, load_idx, :]  # shape (n_subjects, n_shuffles)
        # Remove nan subjects
        mask = ~np.isnan(real)
        if np.sum(mask) == 0:
            continue
        real_mean = np.mean(real[mask])
        # Null distribution: for each shuffle, mean across valid subjects
        shuf_means = np.mean(shuf_vals[mask, :], axis=0)
        # One-sided p-value: How often is null >= real?
        p_val = (np.sum(shuf_means >= real_mean) + 1) / (len(shuf_means) + 1)
        p_values[reg_idx, load_idx] = p_val

# FDR-corrected significance per load
for load_idx in range(3):
    load_pvals = p_values[:, load_idx]
    valid = ~np.isnan(load_pvals)
    if np.sum(valid) == 0:
        continue
    _, pvals_fdr, _, _ = multipletests(load_pvals[valid], method='fdr_bh')
    # Annotate significant bars
    # Annotate significant bars with 1/2/3 stars based on corrected p-value
    for reg_idx, pval in zip(np.where(valid)[0], pvals_fdr):
        mean_val = np.nanmean(results[region_names[reg_idx]]['observed'][:, load_idx])
        if np.isnan(mean_val):
            continue
        if pval < 0.005:
            stars = '***'
        elif pval < 0.01:
            stars = '**'
        elif pval < 0.05:
            stars = '*'
        else:
            stars = ''
        if stars:
            ax.text(x[reg_idx] + load_idx * bar_width - bar_width,
                  mean_val + 5, stars, ha='center',
                  fontsize=14, fontweight='bold')


ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(region_names)
ax.axhline(20, linestyle='--', color='gray', label='Chance (20%)')
ax.set_ylabel("Decoding Accuracy (%)")
ax.set_title("Decoding Accuracy by Region & Load vs. Average Shuffled Baseline")
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig("./brain_regions_decoding.eps", format='eps', bbox_inches='tight', dpi=300)
plt.show()

import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests

# Collect stats
rows = []

for region in results:
    observed = np.atleast_2d(results[region]['observed'])  # shape: (n_subjects, 3)
    shuffled_all = np.atleast_3d(results[region]['shuffled'])  # shape: (n_subjects, 3, n_shuffles)

    if observed.shape[0] == 0 or observed.shape[1] != 3:
        print(f"Skipping region: {region} due to missing or malformed data.")
        continue

    for load_idx in range(3):  # Load 1–3
        real_vals = observed[:, load_idx]
        shuf_vals = shuffled_all[:, load_idx, :]  # shape: (n_subjects, n_shuffles)

        # Drop NaNs
        valid_mask = ~np.isnan(real_vals)
        real_vals = real_vals[valid_mask]
        shuf_vals = shuf_vals[valid_mask]

        if len(real_vals) == 0:
            continue

        real_mean = np.mean(real_vals)
        shuf_means = np.mean(shuf_vals, axis=0)

        # One-sided permutation p-value
        p_val = (np.sum(shuf_means >= real_mean) + 1) / (len(shuf_means) + 1)

        rows.append({
            'Region': region,
            'Load': load_idx + 1,
            'Real_Accuracy': round(real_mean, 2),
            'Shuffled_Mean': round(np.mean(shuf_means), 2),
            'p-value': p_val
        })

# Create DataFrame
stats_table = pd.DataFrame(rows)

# Apply FDR correction per Load
fdr_corrected_pvals = []
for load in stats_table['Load'].unique():
    mask = stats_table['Load'] == load
    pvals = stats_table.loc[mask, 'p-value'].values
    _, fdr_pvals, _, _ = multipletests(pvals, method='fdr_bh')
    stats_table.loc[mask, 'FDR_corrected_p'] = np.round(fdr_pvals, 4)

# Round p-values
stats_table['p-value'] = np.round(stats_table['p-value'], 4)

# Filter significant results
significant_fdr = stats_table[stats_table['FDR_corrected_p'] < 0.05]

# Display
print(stats_table)
print(significant_fdr)

# Save to CSV
stats_table.to_csv('./load_vs_shuffled_stats_corrected_group.csv', index=False)
