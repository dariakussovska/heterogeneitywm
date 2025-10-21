import pandas as pd
import numpy as np
import scipy.signal as signal
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# Parameters
bin_sizes = np.arange(0.05, 0.111, 0.01)  # 50-110 ms in 10 ms steps
sigma_values = np.arange(0.01, 0.101, 0.01)  # 10-100 ms in 10 ms steps

TRIAL_WIN = 1                       # seconds per trial window (delay epoch length)
prominence_threshold_percentile = 90 # percentile on smoothed rate
min_inter_burst_interval = 0.14      # seconds (distance between peaks)
PRINT_DEBUG = True
path_all_meta   = '../data/all_neuron_brain_regions_cleaned.feather'
path_trials     = '../graph_data/graph_encoding1.feather'

df_metadata_all_cells = pd.read_feather(path_all_meta)
df_enc1_filtered      = pd.read_feather(path_trials)

def parse_spike_entry(val):
    ""Parse a spike string/list and keep spikes within [0, TRIAL_WIN]."""
    if val is None:
        return np.array([], dtype=float)
    if isinstance(val, str):
        if val.strip() in ("", "[]"):
            return np.array([], dtype=float)
        try:
            arr = np.array(ast.literal_eval(val), dtype=float)
        except Exception:
            return np.array([], dtype=float)
    elif isinstance(val, (list, tuple, np.ndarray)):
        arr = np.array(val, dtype=float)
    else:
        return np.array([], dtype=float)
    if arr.ndim == 0:
        arr = np.array([float(arr)])
    return arr[(arr >= 0.0) & (arr <= TRIAL_WIN)]

def first_nonnull_series(series):
    """Return at most one spike-list cell to avoid double-counting duplicates."""
    for v in series:
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return [v]
    return []

def subjects_for_all_cells():
    """Subjects eligible for All_cells category."""
    counts = df_metadata_all_cells["subject_id"].value_counts()
    return counts[counts >= 10].index.tolist()

# =========================
# Grid Search Function
# =========================
def run_grid_search():
    """Run grid search over bin sizes and sigma values for All_cells condition."""
    
    # Get eligible subjects
    subj_ids = subjects_for_all_cells()

    # Initialize results storage
    results_grid = np.zeros((len(bin_sizes), len(sigma_values)))
    subject_counts = np.zeros((len(bin_sizes), len(sigma_values)))
    
    # Progress tracking
    total_combinations = len(bin_sizes) * len(sigma_values)
    current_combination = 0
    
    for bin_idx, bin_size in enumerate(bin_sizes):
        for sigma_idx, sigma_sec in enumerate(sigma_values):
            current_combination += 1
            
            # Derived parameters for this combination
            min_bin_separation = int(np.ceil(min_inter_burst_interval / bin_size))
            
            # Store burst counts for this parameter combination
            all_burst_counts = []
            valid_subjects = 0
            
            for subject_id in sorted(subj_ids):
                # Trials for this subject (only memory load == 1)
                df_subject_trials = df_enc1_filtered[
                    (df_enc1_filtered['subject_id'] == subject_id) &
                    (df_enc1_filtered['num_images_presented'] == 1)
                ]
                if df_subject_trials.empty:
                    continue

                # Get all neurons for this subject
                neuron_ids = df_metadata_all_cells[
                    df_metadata_all_cells['subject_id'] == subject_id
                ]["Neuron_ID_3"].dropna().tolist()
                
                if len(neuron_ids) == 0:
                    continue

                # Restrict trials to those neurons
                df_cat = df_subject_trials[df_subject_trials['Neuron_ID_3'].isin(neuron_ids)]
                trial_ids = sorted(df_cat['trial_id'].unique())
                n_trials = len(trial_ids)
                if n_trials == 0:
                    continue

                # REAL BURSTS 
                all_spikes = []
                for trial_idx, trial_id in enumerate(trial_ids):
                    df_trial = df_cat[df_cat['trial_id'] == trial_id]
                    for neuron_id in sorted(df_trial['Neuron_ID_3'].unique()):
                        spikes_series = df_trial.loc[
                            df_trial['Neuron_ID_3'] == neuron_id, 
                            'Standardized_Spikes_New'
                        ].dropna()
                        for s in first_nonnull_series(spikes_series):
                            arr = parse_spike_entry(s)
                            if arr.size:
                                all_spikes.extend(arr + trial_idx * TRIAL_WIN)

                if len(all_spikes) == 0:
                    continue

                total_duration = n_trials * TRIAL_WIN
                time_bins = np.arange(0.0, total_duration + bin_size, bin_size)

                spike_counts, _ = np.histogram(all_spikes, bins=time_bins)

                # Smoothing kernel for current sigma
                kernel = gaussian(len(spike_counts), std=sigma_sec / bin_size)
                kernel = kernel / np.sum(kernel)

                smoothed = np.convolve(spike_counts, kernel, mode='same')
                threshold = np.percentile(smoothed, prominence_threshold_percentile)

                real_peaks, _ = signal.find_peaks(
                    smoothed, prominence=threshold, distance=min_bin_separation
                )
                real_burst_count = int(len(real_peaks))
                
                all_burst_counts.append(real_burst_count)
                valid_subjects += 1

            # Store mean burst count for this parameter combination
            if all_burst_counts:
                mean_bursts = np.mean(all_burst_counts)
                results_grid[bin_idx, sigma_idx] = mean_bursts
                subject_counts[bin_idx, sigma_idx] = valid_subjects

            else:
                results_grid[bin_idx, sigma_idx] = np.nan
                subject_counts[bin_idx, sigma_idx] = 0

    return results_grid, subject_counts

if __name__ == "__main__":
    results_grid, subject_counts = run_grid_search()
    
    results_df = pd.DataFrame(
        results_grid,
        index=[f"{bs*1000:.0f}ms" for bs in bin_sizes],
        columns=[f"{sig*1000:.0f}ms" for sig in sigma_values]
    )
  
    plt.figure(figsize=(12, 8))
    sns.heatmap(results_df, 
                annot=True, 
                fmt=".1f", 
                cmap="RdYlBu_r",  # Red-Yellow-Blue reversed for better contrast
                cbar_kws={'label': 'Mean Burst Count', 'shrink': 0.8},
                linewidths=0.5,
                linecolor='white',
                annot_kws={"size": 10})
    plt.title('Grid Search: Mean Burst Counts Across Subjects\nBin Size vs Gaussian Kernel Width', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Gaussian Kernel Width (σ, ms)', fontsize=14)
    plt.ylabel('Bin Size (ms)', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./supp_3a.eps",
                format='eps', dpi=300)
    plt.show()
