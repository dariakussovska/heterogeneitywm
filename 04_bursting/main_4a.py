import numpy as np
import pandas as pd
import ast
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

df_enc1_filtered = pd.read_excel('/./graph_data/graph_encoding1.xlsx')

subject_id = 14
selected_neurons = [11, 13, 14, 15, 16, 17, 19, 20, 24, 26, 27, 34, 38]
df_subject = df_enc1_filtered[(df_enc1_filtered['subject_id'] == subject_id) & 
                              (df_enc1_filtered['num_images_presented'] == 1)]
keep_trials = sorted(df_subject['trial_id'].unique())[:45]
df_subject = df_subject[df_subject['trial_id'].isin(keep_trials)]
if selected_neurons:
    df_subject = df_subject[df_subject['Neuron_ID'].isin(selected_neurons)]

TRIAL_DUR = 1
N_TRIALS = len(keep_trials)
BIN_SIZE = 0.07
SIGMA_SEC = 0.04
sigma_bins = SIGMA_SEC / BIN_SIZE
total_duration = N_TRIALS * TRIAL_DUR
time_bins = np.arange(0, total_duration + BIN_SIZE, BIN_SIZE)
bin_centers = time_bins[:-1]

# ----------- Build real population activity -----------
spike_times_all = []
neuron_indices = []
neuron_spike_counts = {}  # (neuron_id, trial_idx) -> spikes count

for trial_idx, trial_id in enumerate(sorted(keep_trials)):
    df_trial = df_subject[df_subject['trial_id'] == trial_id]
    for neuron_idx, neuron_id in enumerate(sorted(df_trial['Neuron_ID'].unique())):
        ser = df_trial[df_trial['Neuron_ID'] == neuron_id]['Standardized_Spikes_New'].dropna()
        spike_list = []
        for s in ser:
            if s != '[]':
                try:
                    t = np.array(ast.literal_eval(s), dtype=float)
                except Exception:
                    continue
                t = t[(t >= 0) & (t <= TRIAL_DUR)]
                if t.size:
                    t = t + trial_idx * TRIAL_DUR
                    spike_list.extend(t)
        if spike_list:
            neuron_spike_counts[(neuron_id, trial_idx)] = neuron_spike_counts.get((neuron_id, trial_idx), 0) + len(spike_list)
        spike_times_all.append(np.array(spike_list))
        neuron_indices.append(neuron_idx)

valid_real = [x for x in spike_times_all if x.size > 0]
if valid_real:
    real_counts, _ = np.histogram(np.concatenate(valid_real), bins=time_bins)
else:
    real_counts = np.zeros(len(time_bins) - 1, dtype=int)

smoothed_real = gaussian_filter1d(real_counts.astype(float), sigma=sigma_bins, mode='nearest')

# Threshold based on REAL data (kept constant for Poisson runs)
common_threshold = np.percentile(smoothed_real, 95) if smoothed_real.size else 0.0
real_peaks, _ = signal.find_peaks(smoothed_real, prominence=common_threshold)
num_real_bursts = len(real_peaks)

# Poisson surrogates
num_poisson_repeats = 100
poisson_burst_counts = []
poisson_smoothed_traces = []

# Firing rate per neuron-trial (spikes/sec)
neuron_firing_rates = {k: v / TRIAL_DUR for k, v in neuron_spike_counts.items()}

for _ in range(num_poisson_repeats):
    poi_spike_lists = []
    # Note: if a neuron-trial had 0 spikes in real data, it's not in neuron_firing_rates.
    # If you want to include those as 0 Hz, add them explicitly (see note below).
    for (neuron_id, trial_idx), fr in neuron_firing_rates.items():
        n_spikes = np.random.poisson(fr * TRIAL_DUR)
        if n_spikes > 0:
            t = np.sort(np.random.uniform(0, TRIAL_DUR, n_spikes)) + trial_idx * TRIAL_DUR
            poi_spike_lists.append(t)

    poi_concat = np.concatenate(poi_spike_lists) if poi_spike_lists else np.array([])
    poi_counts, _ = np.histogram(poi_concat, bins=time_bins)
    smoothed_poi = gaussian_filter1d(poi_counts.astype(float), sigma=sigma_bins, mode='nearest')
    poisson_smoothed_traces.append(smoothed_poi)

    poi_peaks, _ = signal.find_peaks(smoothed_poi, prominence=common_threshold)
    poisson_burst_counts.append(len(poi_peaks))

poisson_smoothed_traces = np.vstack(poisson_smoothed_traces) if poisson_smoothed_traces else np.zeros((1, len(bin_centers)))
mean_poi = poisson_smoothed_traces.mean(axis=0)
std_poi = poisson_smoothed_traces.std(axis=0)

mean_poisson_bursts = float(np.mean(poisson_burst_counts)) if poisson_burst_counts else 0.0
std_poisson_bursts = float(np.std(poisson_burst_counts)) if poisson_burst_counts else 0.0

# Plot: real + Poisson mean (with ±1 SD band) 
plt.figure(figsize=(15, 5))
plt.plot(bin_centers, smoothed_real, label='Real Data')
plt.scatter(bin_centers[real_peaks], smoothed_real[real_peaks],
            label=f'Bursts (Real, {num_real_bursts})')

plt.plot(bin_centers, mean_poi, linestyle='dashed', label='Poisson (mean of repeats)')
plt.fill_between(bin_centers, mean_poi - std_poi, mean_poi + std_poi, alpha=0.2, label='Poisson ±1 SD')

plt.xlabel("Time (s)")
plt.ylabel("Smoothed spike count (a.u.)")
plt.title(f"Burst Detection: Real vs Poisson (N={num_poisson_repeats})\n"
          f"Poisson bursts: {mean_poisson_bursts:.2f} ± {std_poisson_bursts:.2f}")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Number of bursts in real data: {num_real_bursts}")
print(f"Poisson burst count distribution: Mean = {mean_poisson_bursts:.2f}, Std = {std_poisson_bursts:.2f}")

### ONE POISSON MODEL ONLY 

valid_real = [sp for sp in spike_times_all if len(sp) > 0]
real_concat = np.concatenate(valid_real) if valid_real else np.array([])
real_counts, _ = np.histogram(real_concat, bins=time_bins)
smoothed_real = gaussian_filter1d(real_counts.astype(float), sigma=sigma_bins, mode='nearest')

#ONE Poisson run, using per-(neuron,trial) rates 
# neuron_spike_counts is {(neuron_id, trial_idx): count_in_that_trial}
neuron_firing_rates = {k: v / TRIAL_DUR for k, v in neuron_spike_counts.items()}  # spikes/sec

poi_spike_lists = []
for (neuron_id, trial_idx), fr in neuron_firing_rates.items():
    n_spikes = np.random.poisson(fr * TRIAL_DUR)
    if n_spikes > 0:
        t = np.sort(np.random.uniform(0, TRIAL_DUR, n_spikes)) + trial_idx * TRIAL_DUR
        poi_spike_lists.append(t)

poi_concat = np.concatenate(poi_spike_lists) if poi_spike_lists else np.array([])
poi_counts, _ = np.histogram(poi_concat, bins=time_bins)          # <-- same bins
smoothed_poi = gaussian_filter1d(poi_counts.astype(float), sigma=sigma_bins, mode='nearest')  # <-- same smoothing
poi_peaks, _ = signal.find_peaks(smoothed_poi, prominence=common_threshold)
num_poi_bursts = len(poi_peaks)

# Plot both traces + peaks ----
plt.figure(figsize=(15, 5))
plt.plot(bin_centers, smoothed_real, label='Real')
plt.scatter(bin_centers[real_peaks], smoothed_real[real_peaks], label=f'Real bursts ({num_real_bursts})')

plt.plot(bin_centers, smoothed_poi, linestyle='dashed', label='Poisson (1 run)')
plt.scatter(bin_centers[poi_peaks], smoothed_poi[poi_peaks], label=f'Poisson bursts ({num_poi_bursts})')

plt.xlabel("Time (s)")
plt.ylabel("Smoothed spike count (a.u.)")
plt.title("Burst detection: Real vs. single Poisson surrogate")
plt.legend()
plt.tight_layout()
save_path = "/./04_bursting/main_4a.eps"
plt.savefig(save_path, format='eps', dpi=300)
plt.show()

print(f"Real bursts: {num_real_bursts}")
print(f"Poisson bursts (1 run): {num_poi_bursts}")
