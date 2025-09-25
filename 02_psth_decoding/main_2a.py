import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from ipywidgets import interact, widgets, SelectMultiple

# Load the data
fixation_data_full = pd.read_excel('/home/daria/PROJECT/clean_data/cleaned_Fixation.xlsx')
enc_data_full = pd.read_excel('/home/daria/PROJECT/graph_data/graph_encoding1.xlsx')
delay_data_full = pd.read_excel('/home/daria/PROJECT/graph_data/graph_delay.xlsx')
probe_data_full = pd.read_excel('/home/daria/PROJECT/graph_data/graph_probe.xlsx')

from ast import literal_eval

# === SET NEURON OF INTEREST ===
subject_id = 4
neuron_id = 52

def preprocess_period_data(df, subject_id, neuron_id):
    df = df[(df["subject_id"] == subject_id) & (df["Neuron_ID"] == neuron_id)]
    if "num_images_presented" in df.columns:
        df = df[df["num_images_presented"] == 1]
    return df

time_bins_enc = np.arange(0, 1.2, 0.05)
time_bins_delay = np.arange(0, 2.8, 0.05)
time_bins_probe = np.arange(0, 1.2, 0.05)

def calculate_baseline_stats(fixation_data, spike_column):
    if fixation_data.empty or spike_column not in fixation_data.columns:
        return 0, 1
    firing_rates = fixation_data[spike_column].dropna().values
    return np.mean(firing_rates), np.std(firing_rates)

def calculate_firing_rates_matrix(trials, spike_column, time_bins):
    firing_rates = []
    for _, row in trials.iterrows():
        spike_times = literal_eval(str(row[spike_column])) if isinstance(row[spike_column], str) else []
        if not isinstance(spike_times, list):
            continue
        spike_counts, _ = np.histogram(spike_times, bins=time_bins)
        fr = spike_counts / (time_bins[1] - time_bins[0])
        firing_rates.append(fr)
    return np.array(firing_rates)

def calculate_z_scores(firing_rates, mean_baseline, std_baseline):
    if std_baseline == 0:
        return np.zeros_like(firing_rates)
    return (firing_rates - mean_baseline) / std_baseline

# Z-SCORE FUNCTIONS FOR EACH PERIOD
def construct_z_encoding(enc_data, fixation_data, time_bins):
    baseline_mean, baseline_std = calculate_baseline_stats(fixation_data, "Spikes_rate_Fixation")
    pref = enc_data[enc_data["Category"] == "Preferred"]
    nonpref = enc_data[enc_data["Category"] == "Non-Preferred"]
    fr_pref = calculate_firing_rates_matrix(pref, "Standardized_Spikes_New", time_bins)
    fr_nonpref = calculate_firing_rates_matrix(nonpref, "Standardized_Spikes_New", time_bins)
    return calculate_z_scores(fr_pref, baseline_mean, baseline_std), calculate_z_scores(fr_nonpref, baseline_mean, baseline_std)

def construct_z_delay(delay_data, fixation_data, time_bins):
    baseline_mean, baseline_std = calculate_baseline_stats(fixation_data, "Spikes_rate_Fixation")
    pref = delay_data[delay_data["Category"] == "Preferred"]
    nonpref = delay_data[delay_data["Category"] == "Non-Preferred"]
    fr_pref = calculate_firing_rates_matrix(pref, "Standardized_Spikes_in_Delay", time_bins)
    fr_nonpref = calculate_firing_rates_matrix(nonpref, "Standardized_Spikes_in_Delay", time_bins)
    return calculate_z_scores(fr_pref, baseline_mean, baseline_std), calculate_z_scores(fr_nonpref, baseline_mean, baseline_std)

def construct_z_probe_all_categories(probe_data, fixation_data, time_bins):
    baseline_mean, baseline_std = calculate_baseline_stats(fixation_data, "Spikes_rate_Fixation")
    trial_types = [
        "Preferred Encoded",
        "Preferred Nonencoded",
        "Nonpreferred Encoded",
        "Nonpreferred Nonencoded"
    ]
    z_scores_by_type = {}
    for trial_type in trial_types:
        trials = probe_data[probe_data["Trial_Type"] == trial_type]
        fr = calculate_firing_rates_matrix(trials, "Standardized_Spikes_in_Probe", time_bins)
        z_scores_by_type[trial_type] = calculate_z_scores(fr, baseline_mean, baseline_std)
    return z_scores_by_type

enc_data = preprocess_period_data(enc_data_full.copy(), subject_id, neuron_id)
delay_data = preprocess_period_data(delay_data_full.copy(), subject_id, neuron_id)
probe_data = preprocess_period_data(probe_data_full.copy(), subject_id, neuron_id)
fixation_data = fixation_data_full.copy()
fixation_data = fixation_data[(fixation_data["subject_id"] == subject_id) & (fixation_data["Neuron_ID"] == neuron_id)]

# CONSTRUCT Z-SCORE MATRICES =
z_enc_pref, z_enc_nonpref = construct_z_encoding(enc_data, fixation_data, time_bins_enc)
z_delay_pref, z_delay_nonpref = construct_z_delay(delay_data, fixation_data, time_bins_delay)
z_probe_all = construct_z_probe_all_categories(probe_data, fixation_data, time_bins_probe)

print(f"[Encoding] Preferred: {z_enc_pref.shape}, Non-Preferred: {z_enc_nonpref.shape}")
print(f"[Delay] Preferred: {z_delay_pref.shape}, Non-Preferred: {z_delay_nonpref.shape}")
for label, z in z_probe_all.items():
    print(f"[Probe] {label}: {z.shape}")


# Convert to DataFrames for compatibility with smoothing + plotting
z1 = pd.DataFrame(z_enc_pref.T)
z2 = pd.DataFrame(z_enc_nonpref.T)
z3 = pd.DataFrame(z_delay_pref.T)
z4 = pd.DataFrame(z_delay_nonpref.T)

# Extract the 4 probe matrices
zp1 = pd.DataFrame(z_probe_all["Preferred Encoded"].T)
zp2 = pd.DataFrame(z_probe_all["Preferred Nonencoded"].T)
zp3 = pd.DataFrame(z_probe_all["Nonpreferred Encoded"].T)
zp4 = pd.DataFrame(z_probe_all["Nonpreferred Nonencoded"].T)

from scipy.ndimage import gaussian_filter1d

def smooth_z_scores(z_score_table, sigma=1.0):
    """
    Smooth Z-scores using a Gaussian filter along the time (row) axis.

    Parameters:
    z_score_table (pd.DataFrame): DataFrame of Z-scores (rows=time bins, columns=neurons).
    sigma (float): Standard deviation for Gaussian kernel.

    Returns:
    pd.DataFrame: Smoothed Z-score table.
    """
    smoothed = z_score_table.apply(
        lambda col: gaussian_filter1d(col.fillna(0).values, sigma=sigma), axis=0
    )
    return pd.DataFrame(smoothed.values, index=z_score_table.index, columns=z_score_table.columns)

z1_smooth = smooth_z_scores(z1, sigma=1.0)
z2_smooth = smooth_z_scores(z2, sigma=1.0)
z3_smooth = smooth_z_scores(z3, sigma=1.0)
z4_smooth = smooth_z_scores(z4, sigma=1.0)

zp1_smooth = smooth_z_scores(zp1, sigma=1.0)
zp2_smooth = smooth_z_scores(zp2, sigma=1.0)
zp3_smooth = smooth_z_scores(zp3, sigma=1.0)
zp4_smooth = smooth_z_scores(zp4, sigma=1.0)

# Display a smoothed Z-score table 
print("Smoothed Enc1 Preferred:")
display(z1_smooth)

from joblib import Parallel, delayed
from statsmodels.stats.multitest import fdrcorrection

def fnc_time_bootstrap_optimized(time_vec1, time_vec2, nboot, CI_int, n_jobs=-1, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    nt = time_vec1.shape[1]
    n_num = min(time_vec1.shape[0], time_vec2.shape[0])
    bootstrap_indices = np.random.randint(0, n_num, size=(nboot, n_num))

    def compute_bootstrap_for_timepoint(ti):
        t1_temp = time_vec1[:, ti]
        t2_temp = time_vec2[:, ti]
        t1_resampled = t1_temp[bootstrap_indices]
        t2_resampled = t2_temp[bootstrap_indices]
        t1_avg = np.nanmean(t1_resampled, axis=1)
        t2_avg = np.nanmean(t2_resampled, axis=1)
        diff_avg = t1_avg - t2_avg
        t1_ci = np.percentile(t1_avg, CI_int)
        t2_ci = np.percentile(t2_avg, CI_int)
        diff_ci = np.percentile(diff_avg, CI_int)
        pos_diff = np.sum(diff_avg > 0) / nboot
        neg_diff = np.sum(diff_avg < 0) / nboot
        p_diff_value = min(2 * min(pos_diff, neg_diff), 1)
        return t1_ci, t2_ci, diff_ci, p_diff_value, t1_avg, t2_avg

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_bootstrap_for_timepoint)(ti) for ti in range(nt)
    )
    t1_CI, t2_CI, diff_CI, p_diff, t1_means, t2_means = zip(*results)
    return (
        np.array(t1_CI),
        np.array(t2_CI),
        np.array(diff_CI),
        np.array(p_diff),
        np.array(t1_means).T,  # shape = (nboot, nt)
        np.array(t2_means).T,
    )

enc1_preferred_data = z1_smooth.values.T
enc1_non_preferred_data = z2_smooth.values.T
delay_preferred_data = z3_smooth.values.T
delay_non_preferred_data = z4_smooth.values.T
zp1_data = zp1_smooth.values.T
zp2_data = zp2_smooth.values.T
zp3_data = zp3_smooth.values.T
zp4_data = zp4_smooth.values.T

enc1_preferred_mean = z1_smooth.mean(axis=1)
enc1_non_preferred_mean = z2_smooth.mean(axis=1)
delay_preferred_mean = z3_smooth.mean(axis=1)
delay_non_preferred_mean = z4_smooth.mean(axis=1)
zp1_mean = zp1_smooth.mean(axis=1)
zp2_mean = zp2_smooth.mean(axis=1)
zp3_mean = zp3_smooth.mean(axis=1)
zp4_mean = zp4_smooth.mean(axis=1)

CI_int = (2.5, 97.5)
nboot = 1000

# Encoding
enc1_t1_CI, enc1_t2_CI, _, enc1_p_diff, enc1_boot_pref, enc1_boot_nonpref = fnc_time_bootstrap_optimized(
    enc1_preferred_data, enc1_non_preferred_data, nboot, CI_int, random_seed=42
)

# Delay
delay_t1_CI, delay_t2_CI, _, delay_p_diff, delay_boot_pref, delay_boot_nonpref = fnc_time_bootstrap_optimized(
    delay_preferred_data, delay_non_preferred_data, nboot, CI_int, random_seed=42
)

# Probe 
zp1_ci, _, _, _, zp1_boot, _ = fnc_time_bootstrap_optimized(zp1_data, zp1_data, nboot, CI_int, random_seed=42)
zp2_ci, _, _, _, zp2_boot, _ = fnc_time_bootstrap_optimized(zp2_data, zp2_data, nboot, CI_int, random_seed=42)
zp3_ci, _, _, _, zp3_boot, _ = fnc_time_bootstrap_optimized(zp3_data, zp3_data, nboot, CI_int, random_seed=42)
zp4_ci, _, _, _, zp4_boot, _ = fnc_time_bootstrap_optimized(zp4_data, zp4_data, nboot, CI_int, random_seed=42)

# Probe significance
_, _, _, probe_p_diff, _, _ = fnc_time_bootstrap_optimized(zp2_data, zp4_data, nboot, CI_int, random_seed=42)

# FDR CORRECTION 
enc1_fdr_mask, _ = fdrcorrection(enc1_p_diff, alpha=0.05)
delay_fdr_mask, _ = fdrcorrection(delay_p_diff, alpha=0.05)
probe_fdr_mask, _ = fdrcorrection(probe_p_diff, alpha=0.05)

# GLOBAL Y LIMITS 
all_values = np.concatenate([
    enc1_boot_pref.mean(axis=0), enc1_boot_nonpref.mean(axis=0),
    delay_boot_pref.mean(axis=0), delay_boot_nonpref.mean(axis=0),
    zp1_boot.mean(axis=0), zp2_boot.mean(axis=0),
    zp3_boot.mean(axis=0), zp4_boot.mean(axis=0),
    zp1_ci.flatten(), zp2_ci.flatten(), zp3_ci.flatten(), zp4_ci.flatten()
])
ymin, ymax = all_values.min() - 1, all_values.max() + 1

bin_size = 0.05
fig, axs = plt.subplots(1, 3, figsize=(24, 6), sharey=True)
fig.suptitle("Z-Scores with 95% Bootstrapped Confidence Intervals", fontsize=18)

# Encoding 1 
x_enc = np.arange(enc1_t1_CI.shape[0]) * bin_size
axs[0].fill_between(x_enc, enc1_t1_CI[:, 0], enc1_t1_CI[:, 1], color='lightgreen', alpha=0.3)
axs[0].fill_between(x_enc, enc1_t2_CI[:, 0], enc1_t2_CI[:, 1], color='plum', alpha=0.3)
axs[0].plot(x_enc, enc1_boot_pref.mean(axis=0), color='darkgreen', label="Preferred Mean", linewidth=2.5)
axs[0].plot(x_enc, enc1_boot_nonpref.mean(axis=0), color='purple', label="Non-Preferred Mean", linewidth=2.5)
axs[0].axhline(0, color='black', linestyle='--')
axs[0].set_title("Encoding 1")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Z-score")
axs[0].legend()
axs[0].grid(True)
axs[0].scatter(x_enc[enc1_fdr_mask], [ymax - 0.5] * np.sum(enc1_fdr_mask), color='black', marker='*', s=100)
axs[0].set_xlim(0, 1.0)
axs[0].set_xticks(np.linspace(0, 1.0, 5))

# Delay
x_delay = np.arange(delay_t1_CI.shape[0]) * bin_size
axs[1].fill_between(x_delay, delay_t1_CI[:, 0], delay_t1_CI[:, 1], color='lightgreen', alpha=0.3)
axs[1].fill_between(x_delay, delay_t2_CI[:, 0], delay_t2_CI[:, 1], color='plum', alpha=0.3)
axs[1].plot(x_delay, delay_boot_pref.mean(axis=0), color='darkgreen', label="Preferred Mean", linewidth=2.5)
axs[1].plot(x_delay, delay_boot_nonpref.mean(axis=0), color='purple', label="Non-Preferred Mean", linewidth=2.5)
axs[1].axhline(0, color='black', linestyle='--')
axs[1].set_title("Delay")
axs[1].set_xlabel("Time (s)")
axs[1].legend()
axs[1].grid(True)
axs[1].scatter(x_delay[delay_fdr_mask], [ymax - 0.5] * np.sum(delay_fdr_mask), color='black', marker='*', s=100)
axs[1].set_xlim(0, 2.5)
axs[1].set_xticks(np.linspace(0, 2.5, 6))

# Probe 
x_probe = np.arange(zp1_ci.shape[0]) * bin_size
axs[2].fill_between(x_probe, zp1_ci[:, 0], zp1_ci[:, 1], color='red', alpha=0.2)
axs[2].fill_between(x_probe, zp2_ci[:, 0], zp2_ci[:, 1], color='orange', alpha=0.2)
axs[2].fill_between(x_probe, zp3_ci[:, 0], zp3_ci[:, 1], color='green', alpha=0.2)
axs[2].fill_between(x_probe, zp4_ci[:, 0], zp4_ci[:, 1], color='blue', alpha=0.2)
axs[2].plot(x_probe, zp1_boot.mean(axis=0), color='red', label="Pref at probe and in memory", linewidth=2.5)
axs[2].plot(x_probe, zp2_boot.mean(axis=0), color='orange', label="Pref not at probe and in memory", linewidth=2.5)
axs[2].plot(x_probe, zp3_boot.mean(axis=0), color='green', label="Pref at probe and not in memory", linewidth=2.5)
axs[2].plot(x_probe, zp4_boot.mean(axis=0), color='blue', label="Pref not at probe and not in memory", linewidth=2.5)
axs[2].axhline(0, color='black', linestyle='--')
axs[2].set_title("Probe")
axs[2].set_xlabel("Time (s)")
axs[2].legend(loc="upper right", ncol=2)
axs[2].grid(True)
axs[2].scatter(x_probe[probe_fdr_mask], [ymax - 0.5] * np.sum(probe_fdr_mask), color='black', marker='*', s=100)
axs[2].set_xlim(0, 1)
axs[2].set_xticks(np.linspace(0, 1, 5))

for ax in axs:
    ax.set_ylim(ymin, ymax)
plt.tight_layout(rect=[0, 0, 1, 0.95])

save_path = "/home/daria/PROJECT/single_PSTH_4052_load1.eps"
plt.savefig(save_path, format='eps', dpi=300)
print(f"✅ Saved EPS figure with FDR-corrected significance: {save_path}")
plt.show()

import matplotlib.pyplot as plt
from ast import literal_eval

subject_id = 4
neuron_id = 52

def plot_ordered_raster(ax, df_top, df_bottom, spike_column, color_top, color_bottom, title):
    trial_idx = 0
    # Top condition (Preferred)
    for _, row in df_top.iterrows():
        spike_times = literal_eval(str(row[spike_column])) if isinstance(row[spike_column], str) else []
        ax.plot(spike_times, [trial_idx + 1] * len(spike_times), 'o', color=color_top, markersize=4)
        trial_idx += 1
    # Bottom condition (Non-Preferred)
    for _, row in df_bottom.iterrows():
        spike_times = literal_eval(str(row[spike_column])) if isinstance(row[spike_column], str) else []
        ax.plot(spike_times, [trial_idx + 1] * len(spike_times), 'o', color=color_bottom, markersize=4)
        trial_idx += 1

    ax.set_title(title)
    ax.set_ylabel("Trial")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-1, trial_idx + 1)
    ax.grid(False)

def plot_probe_raster(ax, groups, spike_column, colors, labels, title):
    trial_idx = 0
    for group, color, label in zip(groups, colors, labels):
        for _, row in group.iterrows():
            spike_times = literal_eval(str(row[spike_column])) if isinstance(row[spike_column], str) else []
            ax.plot(spike_times, [trial_idx + 1] * len(spike_times), 'o', color=color, markersize=4, label=label if trial_idx == 0 else "")
            trial_idx += 1

    ax.set_title(title)
    ax.set_ylabel("Trial")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-1, trial_idx + 1)
    ax.grid(False)

def preprocess_period_data(df, subject_id, neuron_id):
    return df[
        (df["subject_id"] == subject_id) &
        (df["Neuron_ID"] == neuron_id) &
        (df["num_images_presented"] == 1)
    ]

enc_df = preprocess_period_data(enc_data_full.copy(), subject_id, neuron_id)
delay_df = preprocess_period_data(delay_data_full.copy(), subject_id, neuron_id)
probe_df = preprocess_period_data(probe_data_full.copy(), subject_id, neuron_id)

# Encoding
enc_pref = enc_df[enc_df["Category"] == "Preferred"]
enc_nonpref = enc_df[enc_df["Category"] == "Non-Preferred"]

# Delay
delay_pref = delay_df[delay_df["Category"] == "Preferred"]
delay_nonpref = delay_df[delay_df["Category"] == "Non-Preferred"]

# Probe: only Nonencoded
probe_pe = probe_df[probe_df["Trial_Type"] == "Preferred Encoded"]
probe_pn = probe_df[probe_df["Trial_Type"] == "Preferred Nonencoded"]
probe_ne = probe_df[probe_df["Trial_Type"] == "Nonpreferred Encoded"]
probe_np = probe_df[probe_df["Trial_Type"] == "Nonpreferred Nonencoded"]

fig, axs = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(f"Raster Plots — Subject {subject_id}, Neuron {neuron_id}", fontsize=18)

# Encoding Raster
plot_ordered_raster(
    axs[0], enc_pref, enc_nonpref,
    spike_column="Standardized_Spikes_New",
    color_top="darkgreen", color_bottom="purple",
    title="Encoding 1"
)
axs[0].set_xlim(0, 1)
# Delay Raster 
plot_ordered_raster(
    axs[1], delay_pref, delay_nonpref,
    spike_column="Standardized_Spikes_in_Delay",
    color_top="darkgreen", color_bottom="purple",
    title="Delay"
)
axs[1].set_xlim(0, 2.5)
plot_probe_raster(
    axs[2],
    groups=[probe_pe, probe_pn, probe_ne, probe_np],
    spike_column="Standardized_Spikes_in_Probe",
    colors=["red", "orange", "green", "blue"],
    labels=["Pref at probe and in memory", "Pref not at probe and in memory", "Pref at probe and not in memory", "Pref not at probe and not in memory"],
    title="Probe (All Trials)"
)
axs[2].set_xlim(0, 1)
axs[2].legend(
    handles=[
        plt.Line2D([], [], color='red', marker='o', linestyle='None', label='P1'),
        plt.Line2D([], [], color='orange', marker='o', linestyle='None', label='P2'),
        plt.Line2D([], [], color='green', marker='o', linestyle='None', label='P3'),
        plt.Line2D([], [], color='blue', marker='o', linestyle='None', label='P4')
    ],
    loc='upper right',
    fontsize=9,
    frameon=True
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
save_path = "/home/daria/PROJECT/raster_plots_4052.eps"
plt.savefig(save_path, format='eps', dpi=300)
plt.show()
