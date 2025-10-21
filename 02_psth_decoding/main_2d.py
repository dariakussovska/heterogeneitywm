import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from scipy.stats import binom
import ast
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import label, find_objects

trial_info = pd.read_feather('/./new_trial_final.feather')
subject_trials = trial_info[trial_info['subject_id'] == 14][['trial_id_final', 'num_images_presented', 'stimulus_index_enc1', 'stimulus_index_enc2', 'stimulus_index_enc3', 'response_accuracy']]
print(subject_trials)

y_matrix = subject_trials

# ===============================
# 1. Load Spike Data from All Periods
# ===============================
df_enc1 = pd.read_feather('/./clean_data/cleaned_Encoding1.feather')
df_enc2 = pd.read_feather('/./clean_data/cleaned_Encoding2.feather')
df_enc3 = pd.read_feather('/./clean_data/cleaned_Encoding3.feather')
df_delay = pd.read_feather('/./clean_data/cleaned_Delay.feather')
df_probe = pd.read_feather('/./clean_data/cleaned_Probe.feather')
df_fixation = pd.read_feather('/./clean_data/cleaned_Fixation.feather')

y_matrix = y_matrix.reset_index(drop=True)
neuron_ids = df_enc1[df_enc1['Signi'] == 'Y']['Neuron_ID_3'].unique()
trial_count = y_matrix.shape[0]
neuron_count = len(neuron_ids)
design_matrix = np.empty((trial_count, neuron_count), dtype=object)

def parse_spike_times(spike_str):
    try:
        return ast.literal_eval(spike_str)
    except:
        return []

for trial_idx, row in y_matrix.iterrows():
    trial_id = row['trial_id_final']
    num_images = row['num_images_presented']
    for neuron_idx, neuron in enumerate(neuron_ids):
        all_spikes = []
        for df, col in [
            (df_enc1, 'Standardized_Spikes'),
            (df_enc2, 'Standardized_Spikes'),
            (df_enc3, 'Standardized_Spikes'),
            (df_delay, 'Standardized_Spikes'),
            (df_probe, 'Standardized_Spikes')
        ]:
            if (df is df_enc2 and num_images < 2) or (df is df_enc3 and num_images < 3):
                continue
            match = df[(df['trial_id_final'] == trial_id) & (df['Neuron_ID_3'] == neuron)]
            if not match.empty:
                all_spikes.extend(parse_spike_times(match[col].values[0]))
        design_matrix[trial_idx, neuron_idx] = np.array(all_spikes)

bin_size = 0.25
step_size = 0.1
load_durations = {1: 5, 2: 6, 3: 7}

fixation_means = df_fixation.groupby('Neuron_ID_3')['Spikes_rate_Fixation'].mean().to_dict()
fixation_stds = df_fixation.groupby('Neuron_ID_3')['Spikes_rate_Fixation'].std().to_dict()

def create_time_bins(duration):
    return np.arange(0, duration - bin_size + step_size, step_size)

def count_spikes(spikes, bins, size):
    return np.array([np.sum((spikes >= t) & (spikes < t + size)) for t in bins])

def build_binned_matrix(duration):
    bins = create_time_bins(duration)
    binned_matrix = np.zeros((trial_count, neuron_count, len(bins)))
    for i in range(trial_count):
        for j in range(neuron_count):
            spikes = design_matrix[i, j]
            binned = count_spikes(spikes, bins, bin_size)
            mean = fixation_means.get(neuron_ids[j], 0)
            std = fixation_stds.get(neuron_ids[j], 1)
            binned_matrix[i, j, :] = (binned - mean) / std if std > 0 else binned
    return binned_matrix

def cross_temporal_decoding(X, y):
    n_bins = X.shape[2]
    acc_matrix = np.zeros((n_bins, n_bins))
    for train_bin in range(n_bins):
        for test_bin in range(n_bins):
            correct = 0
            for i in range(len(X)):
                X_train = np.delete(X, i, axis=0)[:, :, train_bin]
                y_train = np.delete(y, i)
                X_test = X[i, :, test_bin].reshape(1, -1)
                y_test = y[i]
                clf = SVC(kernel='linear')
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
                correct += (pred[0] == y_test)
            acc_matrix[train_bin, test_bin] = correct / len(X)
    return acc_matrix

def add_epoch_lines(ax, enc_bins, delay_bins, total_bins):
    ax.axvline(enc_bins, color='k', linewidth=1)
    ax.axvline(enc_bins + delay_bins, color='k', linewidth=1)
    ax.axhline(enc_bins, color='k', linewidth=1)
    ax.axhline(enc_bins + delay_bins, color='k', linewidth=1)
    ax.text(-2, enc_bins / 2, "Enc", va='center', ha='right', fontsize=10)
    ax.text(-2, enc_bins + delay_bins / 2, "Delay", va='center', ha='right', fontsize=10)
    ax.text(-2, enc_bins + delay_bins + (total_bins - enc_bins - delay_bins) / 2, "Probe", va='center', ha='right', fontsize=10)
    ax.text(enc_bins / 2, total_bins + 1, "Encoding", ha='center', va='bottom', fontsize=10)
    ax.text(enc_bins + delay_bins / 2, total_bins + 1, "Delay", ha='center', va='bottom', fontsize=10)
    ax.text(enc_bins + delay_bins + (total_bins - enc_bins - delay_bins) / 2, total_bins + 1, "Probe", ha='center', va='bottom', fontsize=10)

colors = [
    (44/255, 63/255, 122/255),
    (120/255, 150/255, 200/255),
    (227/255, 215/255, 207/255),
    (255/255, 160/255, 160/255),
    (200/255, 70/255, 70/255),
    (138/255, 31/255, 34/255)
]
custom_cmap = LinearSegmentedColormap.from_list("custom_red_blue", colors)

from scipy.stats import binom
from statsmodels.stats.multitest import fdrcorrection
from skimage import measure
import os

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

chance_level = 1 / len(np.unique(y_matrix['stimulus_index_enc1']))
alpha = 0.05

for idx, load in enumerate([1, 2, 3]):
    duration = load_durations[load]
    bins = create_time_bins(duration)
    n_bins = len(bins)

    enc_dur = load + 0.2
    delay_dur = 2.8
    probe_dur = 1.2

    enc_bins = len(create_time_bins(enc_dur))
    delay_bins = len(create_time_bins(delay_dur))
    probe_bins = n_bins - enc_bins - delay_bins

    trials = y_matrix[y_matrix['num_images_presented'] == load].index.values
    if load == 1:
        y = y_matrix.loc[trials, 'stimulus_index_enc1'].values
    elif load == 2:
        y = y_matrix.loc[trials, 'stimulus_index_enc2'].values
    elif load == 3:
        y = y_matrix.loc[trials, 'stimulus_index_enc3'].values

    N = len(y)
    X_all = build_binned_matrix(duration)
    X = X_all[trials, :, :]

    acc_matrix = cross_temporal_decoding(X, y)

    p_vals = 1 - binom.cdf((acc_matrix * N).astype(int), N, chance_level)
    p_vals_flat = p_vals.flatten()
    _, p_vals_fdr = fdrcorrection(p_vals_flat, alpha=alpha)
    sig_mask = p_vals_fdr.reshape(acc_matrix.shape) < alpha

    ax = axes[idx]
    im = ax.imshow(
       acc_matrix * 100,
       cmap=custom_cmap, vmin=0, vmax=100,
       interpolation='nearest',
       aspect='equal'
    )

    ax.invert_yaxis()  

    contours = measure.find_contours(sig_mask.astype(float), 0.5)
    for contour in contours:
        if len(contour) > 13: 
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color=(138/255, 31/255, 34/255))

    add_epoch_lines(ax, enc_bins, delay_bins, n_bins)
    ax.set_title(f"Load {load} Decoding", fontsize=14)
    ax.set_xlabel("Test Time Bin")
    ax.set_ylabel("Train Time Bin")
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="Accuracy (%)")
plt.suptitle("Cross-Temporal Decoding Accuracy by Load", fontsize=16)
folder_path = "/./02_psth_decoding/
os.makedirs(folder_path, exist_ok=True)
save_path = os.path.join(folder_path, "cross_temporal_decoding.eps")
plt.savefig(save_path, format='eps', dpi=300)
plt.show()
