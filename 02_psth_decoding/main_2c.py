import pandas as pd
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt

# === Load Data ===
fixation_data = pd.read_feather('../clean_data/cleaned_Fixation.feather')
enc1 = pd.read_feather('../graph_data/graph_encoding1.feather')
enc2 = pd.read_feather('../graph_data/graph_encoding2.feather')
enc3 = pd.read_feather('../graph_data/graph_encoding3.feather')
delay = pd.read_feather('../graph_data/graph_delay.feather')
probe = pd.read_feather('../graph_data/graph_probe.feather')

classification_df = pd.read_feather('../all_neuron_brain_regions_cleaned.feather')
classification_df = classification_df[
    (classification_df["Signi"] == "Y")] # & (classification_df["Cell_Type_New"] == "IN")]
valid_neuron_ids = classification_df["Neuron_ID_3"].dropna().unique().tolist()

# === Time Bins ===
time_bins_enc = np.arange(0, 1.2, 0.05)
time_bins_delay = np.arange(0, 3, 0.05)
time_bins_probe = np.arange(0, 1.2, 0.05)

# === Helper Functions ===
def filter_data_by_neuron_id_3(data, valid_ids):
    return data[data["Neuron_ID_3"].isin(valid_ids)]

def calculate_baseline_stats(fixation_data, spike_column, neuron_id_3):
    neuron_data = fixation_data[fixation_data["Neuron_ID_3"] == neuron_id_3]
    if neuron_data.empty or spike_column not in neuron_data.columns:
        return 0, 1
    firing_rates = neuron_data[spike_column].dropna().values
    return np.mean(firing_rates), np.std(firing_rates)

def calculate_firing_rates(trials, spike_column, time_bins):
    all_firing_rates = []
    for _, row in trials.iterrows():
        spike_times = literal_eval(str(row[spike_column])) if isinstance(row[spike_column], str) else []
        if not isinstance(spike_times, list):
            continue
        spike_counts, _ = np.histogram(spike_times, bins=time_bins)
        firing_rates = spike_counts / (time_bins[1] - time_bins[0])
        all_firing_rates.append(firing_rates)
    return np.mean(all_firing_rates, axis=0) if all_firing_rates else np.zeros(len(time_bins) - 1)

def calculate_z_scores(firing_rates, mean_baseline, std_baseline):
    return (firing_rates - mean_baseline) / std_baseline if std_baseline != 0 else np.zeros_like(firing_rates)

def construct_z_score_tables_encoding_only(enc_data, fixation_data, time_bins, spike_column="Standardized_Spikes"):
    neuron_ids = enc_data["Neuron_ID_3"].dropna().unique()
    z_scores_pref = pd.DataFrame(index=[f"Bin_{i}" for i in range(len(time_bins)-1)], columns=neuron_ids)
    z_scores_nonpref = z_scores_pref.copy()

    for neuron_id in neuron_ids:
        baseline_mean, baseline_std = calculate_baseline_stats(fixation_data, "Spikes_rate_Fixation", neuron_id)
        pref_trials = enc_data[(enc_data["Category"] == "Preferred") & (enc_data["Neuron_ID_3"] == neuron_id)]
        nonpref_trials = enc_data[(enc_data["Category"] == "Non-Preferred") & (enc_data["Neuron_ID_3"] == neuron_id)]

        z_scores_pref[neuron_id] = calculate_z_scores(
            calculate_firing_rates(pref_trials, spike_column, time_bins), baseline_mean, baseline_std
        )
        z_scores_nonpref[neuron_id] = calculate_z_scores(
            calculate_firing_rates(nonpref_trials, spike_column, time_bins), baseline_mean, baseline_std
        )

    return z_scores_pref, z_scores_nonpref

def construct_z_probe_all_categories(probe_data, fixation_data, time_bins, spike_column="Standardized_Spikes"):
    neuron_ids = probe_data["Neuron_ID_3"].dropna().unique()
    bin_labels = [f"Bin_{i}" for i in range(len(time_bins) - 1)]

    z_pref_encoded = pd.DataFrame(index=bin_labels, columns=neuron_ids)
    z_pref_nonencoded = pd.DataFrame(index=bin_labels, columns=neuron_ids)
    z_nonpref_encoded = pd.DataFrame(index=bin_labels, columns=neuron_ids)
    z_nonpref_nonencoded = pd.DataFrame(index=bin_labels, columns=neuron_ids)

    for neuron_id in neuron_ids:
        baseline_mean, baseline_std = calculate_baseline_stats(fixation_data, "Spikes_rate_Fixation", neuron_id)
        neuron_trials = probe_data[probe_data["Neuron_ID_3"] == neuron_id]

        z_pref_encoded[neuron_id] = calculate_z_scores(
            calculate_firing_rates(neuron_trials[neuron_trials["Probe_Category"] == "Preferred Encoded"], spike_column, time_bins),
            baseline_mean, baseline_std
        )
        z_pref_nonencoded[neuron_id] = calculate_z_scores(
            calculate_firing_rates(neuron_trials[neuron_trials["Probe_Category"] == "Preferred Nonencoded"], spike_column, time_bins),
            baseline_mean, baseline_std
        )
        z_nonpref_encoded[neuron_id] = calculate_z_scores(
            calculate_firing_rates(neuron_trials[neuron_trials["Probe_Category"] == "Nonpreferred Encoded"], spike_column, time_bins),
            baseline_mean, baseline_std
        )
        z_nonpref_nonencoded[neuron_id] = calculate_z_scores(
            calculate_firing_rates(neuron_trials[neuron_trials["Probe_Category"] == "Nonpreferred Nonencoded"], spike_column, time_bins),
            baseline_mean, baseline_std
        )

    return z_pref_encoded, z_pref_nonencoded, z_nonpref_encoded, z_nonpref_nonencoded


fixation_filtered = filter_data_by_neuron_id_3(fixation_data, valid_neuron_ids)

enc1_L1 = filter_data_by_neuron_id_3(enc1[enc1["num_images_presented"] == 1], valid_neuron_ids)
delay_L1 = filter_data_by_neuron_id_3(delay[delay["num_images_presented"] == 1], valid_neuron_ids)
probe_L1 = filter_data_by_neuron_id_3(probe[probe["num_images_presented"] == 1], valid_neuron_ids)

z_enc1_L1_pref, z_enc1_L1_nonpref = construct_z_score_tables_encoding_only(enc1_L1, fixation_filtered, time_bins_enc)
z_delay_L1_pref, z_delay_L1_nonpref = construct_z_score_tables_encoding_only(delay_L1, fixation_filtered, time_bins_delay, spike_column="Standardized_Spikes")
zp1_L1, zp2_L1, zp3_L1, zp4_L1 = construct_z_probe_all_categories(probe_L1, fixation_filtered, time_bins_probe)

enc1_L2 = filter_data_by_neuron_id_3(enc1[enc1["num_images_presented"] == 2], valid_neuron_ids)
enc2_L2 = filter_data_by_neuron_id_3(enc2[enc2["num_images_presented"] == 2], valid_neuron_ids)
delay_L2 = filter_data_by_neuron_id_3(delay[delay["num_images_presented"] == 2], valid_neuron_ids)

z_enc1_L2_pref, z_enc1_L2_nonpref = construct_z_score_tables_encoding_only(enc1_L2, fixation_filtered, time_bins_enc)
z_enc2_L2_pref, z_enc2_L2_nonpref = construct_z_score_tables_encoding_only(enc2_L2, fixation_filtered, time_bins_enc)
z_delay_L2_pref, z_delay_L2_nonpref = construct_z_score_tables_encoding_only(delay_L2, fixation_filtered, time_bins_delay, spike_column="Standardized_Spikes")

enc1_L3 = filter_data_by_neuron_id_3(enc1[enc1["num_images_presented"] == 3], valid_neuron_ids)
enc2_L3 = filter_data_by_neuron_id_3(enc2[enc2["num_images_presented"] == 3], valid_neuron_ids)
enc3_L3 = filter_data_by_neuron_id_3(enc3[enc3["num_images_presented"] == 3], valid_neuron_ids)
delay_L3 = filter_data_by_neuron_id_3(delay[delay["num_images_presented"] == 3], valid_neuron_ids)

z_enc1_L3_pref, z_enc1_L3_nonpref = construct_z_score_tables_encoding_only(enc1_L3, fixation_filtered, time_bins_enc)
z_enc2_L3_pref, z_enc2_L3_nonpref = construct_z_score_tables_encoding_only(enc2_L3, fixation_filtered, time_bins_enc)
z_enc3_L3_pref, z_enc3_L3_nonpref = construct_z_score_tables_encoding_only(enc3_L3, fixation_filtered, time_bins_enc)
z_delay_L3_pref, z_delay_L3_nonpref = construct_z_score_tables_encoding_only(delay_L3, fixation_filtered, time_bins_delay, spike_column="Standardized_Spikes")

probe_L2 = filter_data_by_neuron_id_3(probe[probe["num_images_presented"] == 2], valid_neuron_ids)
zp1_L2, zp2_L2, zp3_L2, zp4_L2 = construct_z_probe_all_categories(probe_L2, fixation_filtered, time_bins_probe)

probe_L3 = filter_data_by_neuron_id_3(probe[probe["num_images_presented"] == 3], valid_neuron_ids)
zp1_L3, zp2_L3, zp3_L3, zp4_L3 = construct_z_probe_all_categories(probe_L3, fixation_filtered, time_bins_probe)

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

z_enc1_L1_pref_smooth = smooth_z_scores(z_enc1_L1_pref)
z_enc1_L1_nonpref_smooth = smooth_z_scores(z_enc1_L1_nonpref)
z_delay_L1_pref_smooth = smooth_z_scores(z_delay_L1_pref)
z_delay_L1_nonpref_smooth = smooth_z_scores(z_delay_L1_nonpref)
zp1_L1_smooth = smooth_z_scores(zp1_L1)
zp2_L1_smooth = smooth_z_scores(zp2_L1)
zp3_L1_smooth = smooth_z_scores(zp3_L1)
zp4_L1_smooth = smooth_z_scores(zp4_L1)

z_enc1_L2_pref_smooth = smooth_z_scores(z_enc1_L2_pref)
z_enc1_L2_nonpref_smooth = smooth_z_scores(z_enc1_L2_nonpref)

z_enc2_L2_pref_smooth = smooth_z_scores(z_enc2_L2_pref)
z_enc2_L2_nonpref_smooth = smooth_z_scores(z_enc2_L2_nonpref)

z_delay_L2_pref_smooth = smooth_z_scores(z_delay_L2_pref)
z_delay_L2_nonpref_smooth = smooth_z_scores(z_delay_L2_nonpref)

z_enc1_L3_pref_smooth = smooth_z_scores(z_enc1_L3_pref)
z_enc1_L3_nonpref_smooth = smooth_z_scores(z_enc1_L3_nonpref)

z_enc2_L3_pref_smooth = smooth_z_scores(z_enc2_L3_pref)
z_enc2_L3_nonpref_smooth = smooth_z_scores(z_enc2_L3_nonpref)

z_enc3_L3_pref_smooth = smooth_z_scores(z_enc3_L3_pref)
z_enc3_L3_nonpref_smooth = smooth_z_scores(z_enc3_L3_nonpref)

z_delay_L3_pref_smooth = smooth_z_scores(z_delay_L3_pref)
z_delay_L3_nonpref_smooth = smooth_z_scores(z_delay_L3_nonpref)

zp1_L2_smooth = smooth_z_scores(zp1_L2)
zp2_L2_smooth = smooth_z_scores(zp2_L2)
zp3_L2_smooth = smooth_z_scores(zp3_L2)
zp4_L2_smooth = smooth_z_scores(zp4_L2)

zp1_L3_smooth = smooth_z_scores(zp1_L3)
zp2_L3_smooth = smooth_z_scores(zp2_L3)
zp3_L3_smooth = smooth_z_scores(zp3_L3)
zp4_L3_smooth = smooth_z_scores(zp4_L3) 

from joblib import Parallel, delayed
import numpy as np

def fnc_time_bootstrap_optimized(time_vec1, time_vec2, nboot, CI_int, n_jobs=-1, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    min_timepoints = min(time_vec1.shape[1], time_vec2.shape[1])
    time_vec1 = time_vec1[:, :min_timepoints]
    time_vec2 = time_vec2[:, :min_timepoints]

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
        return t1_ci, t2_ci, diff_ci, p_diff_value

    results = Parallel(n_jobs=n_jobs)(delayed(compute_bootstrap_for_timepoint)(ti) for ti in range(min_timepoints))
    t1_CI, t2_CI, diff_CI, p_diff = zip(*results)
    return np.array(t1_CI), np.array(t2_CI), np.array(diff_CI), np.array(p_diff)

from statsmodels.stats.multitest import fdrcorrection

bin_size = 0.05  

bootstrap_inputs = [
    # Load 1
    ("Enc1 (L1)", z_enc1_L1_pref_smooth, z_enc1_L1_nonpref_smooth),
    ("Delay (L1)", z_delay_L1_pref_smooth, z_delay_L1_nonpref_smooth),

    # Load 2
    ("Enc1 (L2)", z_enc1_L2_pref_smooth, z_enc1_L2_nonpref_smooth),
    ("Enc2 (L2)", z_enc2_L2_pref_smooth, z_enc2_L2_nonpref_smooth),
    ("Delay (L2)", z_delay_L2_pref_smooth, z_delay_L2_nonpref_smooth),

    # Load 3
    ("Enc1 (L3)", z_enc1_L3_pref_smooth, z_enc1_L3_nonpref_smooth),
    ("Enc2 (L3)", z_enc2_L3_pref_smooth, z_enc2_L3_nonpref_smooth),
    ("Enc3 (L3)", z_enc3_L3_pref_smooth, z_enc3_L3_nonpref_smooth),
    ("Delay (L3)", z_delay_L3_pref_smooth, z_delay_L3_nonpref_smooth),
]

ci_results = []
all_pvals = []
for label, z_pref, z_nonpref in bootstrap_inputs:
    ci = fnc_time_bootstrap_optimized(
        z_pref.values.T, z_nonpref.values.T, nboot=1000, CI_int=(2.5, 97.5), random_seed=42
    )
    ci_results.append((label, z_pref.mean(axis=1), z_nonpref.mean(axis=1), *ci))
    all_pvals.append(ci[3])  

probe_lines = {
    "P1": ("red", zp1_L1_smooth),
    "P2": ("orange", zp2_L1_smooth),
    "P3": ("green", zp3_L1_smooth),
    "P4": ("blue", zp4_L1_smooth),
}
probe_ci_results = {}
for label, (_, df) in probe_lines.items():
    mean_vals = df.mean(axis=1)
    ci = fnc_time_bootstrap_optimized(df.values.T, df.values.T, nboot=1000, CI_int=(2.5, 97.5), random_seed=42)
    probe_ci_results[label] = (mean_vals, ci[0])  

# Load 1
_, _, _, p_diff_probe_L1 = fnc_time_bootstrap_optimized(
    zp2_L1_smooth.values.T, zp4_L1_smooth.values.T, nboot=1000, CI_int=(5, 90), random_seed=42
)

# Load 2
_, _, _, p_diff_probe_L2 = fnc_time_bootstrap_optimized(
    zp2_L2_smooth.values.T, zp4_L2_smooth.values.T, nboot=1000, CI_int=(5, 90), random_seed=42
)

# Load 3
_, _, _, p_diff_probe_L3 = fnc_time_bootstrap_optimized(
    zp2_L3_smooth.values.T, zp4_L3_smooth.values.T, nboot=1000, CI_int=(5, 90), random_seed=42
)

all_pvals.extend([p_diff_probe_L1, p_diff_probe_L2, p_diff_probe_L3])
flat_pvals = np.concatenate(all_pvals)
fdr_pass, fdr_corrected_pvals = fdrcorrection(flat_pvals, alpha=0.05, method='indep')

reshaped_fdr_pvals = []
start = 0
for p_array in all_pvals:
    end = start + len(p_array)
    reshaped_fdr_pvals.append(fdr_corrected_pvals[start:end])
    start = end

fig, axs = plt.subplots(3, 5, figsize=(30, 18), sharey=True)
fig.suptitle("Z-Scores with Bootstrapped Confidence Intervals + FDR-corrected Significance", fontsize=26)

layout_map = {
    "Enc1 (L1)": (0, 0), "Delay (L1)": (0, 2), "Probe (L1)": (0, 3),
    "Enc1 (L2)": (1, 0), "Enc2 (L2)": (1, 1), "Delay (L2)": (1, 2), "Probe (L2)": (1, 4),
    "Enc1 (L3)": (2, 0), "Enc2 (L3)": (2, 1), "Enc3 (L3)": (2, 2), "Delay (L3)": (2, 3), "Probe (L3)": (2, 4),
}

for i, (label, m_pref, m_nonpref, ci_pref, ci_nonpref, ci_diff, p_diff) in enumerate(ci_results):
    row, col = layout_map[label]
    ax = axs[row, col]

    num_bins = len(m_pref)
    x = np.arange(num_bins) * bin_size
    if "Delay" in label:
        ax.set_xlim(0, 2.5)
        ax.set_ylim(-2, 8)
        ax.set_xticks(np.linspace(0, 2.5, 6))
    else:
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-2, 8)
        ax.set_xticks(np.linspace(0, 1.0, 5))

    ax.fill_between(x, ci_pref[:, 0], ci_pref[:, 1], color="lightgreen", alpha=0.3)
    ax.fill_between(x, ci_nonpref[:, 0], ci_nonpref[:, 1], color="plum", alpha=0.3)

    ax.plot(x, m_pref.values, color="darkgreen", label="Preferred", linewidth=2.5)
    ax.plot(x, m_nonpref.values, color="purple", linestyle="--", label="Non-Preferred", linewidth=2.5)

    corrected_pvals = reshaped_fdr_pvals[i]
    sig_bins = np.where(corrected_pvals < 0.05)[0]
    if len(sig_bins) > 0:
        star_y = max(m_pref.max(), m_nonpref.max()) + 0.5
        ax.scatter(x[sig_bins], [star_y]*len(sig_bins), color='black', marker='*', s=100)

    ax.axhline(0, color="black", linestyle="--")
    ax.set_title(label)
    ax.set_xlabel("Time (s)")
    ax.grid(True)
    if col == 0:
        ax.set_ylabel("Z-score")
    if row == 0 and col == 0:
        ax.legend()

ax = axs[0, 3]
num_bins = probe_ci_results["P1"][0].shape[0]
x = np.arange(num_bins) * bin_size

for probe_label, (color, df) in probe_lines.items():
    mean_vals, ci = probe_ci_results[probe_label]
    ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.25)
    ax.plot(x, mean_vals, color=color, label=probe_label, linewidth=2)
corrected_pvals_L1 = reshaped_fdr_pvals[-3]
sig_bins_L1 = np.where(corrected_pvals_L1 < 0.05)[0]
if len(sig_bins_L1) > 0:
    star_y = max(zp2_L1_smooth.mean().max(), zp4_L1_smooth.mean().max()) + 0.5
    axs[0, 3].scatter(x[sig_bins_L1], [star_y]*len(sig_bins_L1), color='black', marker='*', s=100)

ax.set_title("Probe (L1)")
ax.set_xlabel("Time (s)")
ax.set_xlim(0, 1.0)
ax.set_ylim(-2, 8)
ax.set_xticks(np.linspace(0, 1.0, 5))
ax.grid(True)
ax.legend()

ax = axs[1, 4]
for probe_label, df, color in zip(["P1", "P2", "P3", "P4"],
                                  [zp1_L2_smooth, zp2_L2_smooth, zp3_L2_smooth, zp4_L2_smooth],
                                  ["red", "orange", "green", "blue"]):
    mean_vals, ci = fnc_time_bootstrap_optimized(df.values.T, df.values.T, nboot=1000, CI_int=(2.5, 97.5), random_seed=42)[0:2]
    ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.25)
    ax.plot(x, df.mean(axis=1), color=color, label=probe_label, linewidth=2)
corrected_pvals_L2 = reshaped_fdr_pvals[-2]
sig_bins_L2 = np.where(corrected_pvals_L2 < 0.05)[0]
if len(sig_bins_L2) > 0:
    star_y = max(zp2_L2_smooth.mean().max(), zp4_L2_smooth.mean().max()) + 0.5
    axs[1, 4].scatter(x[sig_bins_L2], [star_y]*len(sig_bins_L2), color='black', marker='*', s=100)

ax.set_title("Probe (L2)")
ax.set_xlabel("Time (s)")
ax.set_xlim(0, 1.0)
ax.set_ylim(-2, 8)
ax.set_xticks(np.linspace(0, 1.0, 5))
ax.grid(True)
ax.legend()

ax = axs[2, 4]
for probe_label, df, color in zip(["P1", "P2", "P3", "P4"],
                                  [zp1_L3_smooth, zp2_L3_smooth, zp3_L3_smooth, zp4_L3_smooth],
                                  ["red", "orange", "green", "blue"]):
    mean_vals, ci = fnc_time_bootstrap_optimized(df.values.T, df.values.T, nboot=1000, CI_int=(2.5, 97.5), random_seed=42)[0:2]
    ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.25)
    ax.plot(x, df.mean(axis=1), color=color, label=probe_label, linewidth=2)
corrected_pvals_L3 = reshaped_fdr_pvals[-1]
sig_bins_L3 = np.where(corrected_pvals_L3 < 0.05)[0]
if len(sig_bins_L3) > 0:
    star_y = max(zp2_L3_smooth.mean().max(), zp4_L3_smooth.mean().max()) + 0.5
    axs[2, 4].scatter(x[sig_bins_L3], [star_y]*len(sig_bins_L3), color='black', marker='*', s=100)
ax.set_title("Probe (L3)")
ax.set_xlabel("Time (s)")
ax.set_xlim(0, 1.0)
ax.set_ylim(-2, 8)
ax.set_xticks(np.linspace(0, 1.0, 5))
ax.grid(True)
ax.legend()

axs[1, 3].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
save_path = "./PSTH_concept.eps"
plt.savefig(save_path, format='eps', dpi=300)
print(f"Saved EPS figure with bootstrapped confidence intervals + FDR stars to: {save_path}")
plt.show()
