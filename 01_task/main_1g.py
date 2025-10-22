import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem

def plot_raster_and_psth_grouped_by_stimulus(
    data, neuron_id_3, subject_id, spike_column='Standardized_Spikes',
    stimulus_column='stimulus_index', time_range=(0, 1), bin_size=0.05,
    smooth_sigma=1, save_dir=None
):
    """
    Plot raster + smoothed PSTH with confidence intervals for a single neuron, grouped by stimulus identity.

    Parameters:
    - data: DataFrame with 'Neuron_ID_3', 'subject_id', 'trial_id', spike_column (list), stimulus_column
    - neuron_id_3: ID of the neuron to plot
    - subject_id: ID of the subject
    - time_range: tuple (start, end) time in seconds
    - bin_size: bin size in seconds 
    - smooth_sigma: standard deviation for Gaussian smoothing (in bins)
    - save_dir: path to save .eps files 
    """
    df = data[(data['Neuron_ID_3'] == neuron_id_3) & (data['subject_id'] == subject_id)].copy()
    if df.empty:
        print(f"No data found for Neuron_ID_3 = {neuron_id_3}, Subject {subject_id}.")
        return

    df = df.sort_values(by=[stimulus_column, 'trial_id']).reset_index(drop=True)
    stimulus_ids = df[stimulus_column].unique()
    color_map = plt.cm.get_cmap('tab10', len(stimulus_ids))
    bins = np.arange(time_range[0], time_range[1] + bin_size * 1.5, bin_size)
    bin_centers = bins[:-1]
    
    plt.figure(figsize=(12, 6))
    for i, row in df.iterrows():
        spike_times = row[spike_column]
        if isinstance(spike_times, str):
            spike_times = eval(spike_times)
        stim = row[stimulus_column]
        stim_color = color_map(np.where(stimulus_ids == stim)[0][0])
        spike_times = [s for s in spike_times if time_range[0] <= s <= time_range[1]]
        if spike_times:
            plt.scatter(spike_times, [i + 0.5] * len(spike_times), color=stim_color, s=20)

    plt.title(f'Raster — Neuron {neuron_id_3}, Subject {subject_id} (grouped by stimulus)')
    plt.xlabel('Time (s)')
    plt.ylabel('Trials (grouped by stimulus)')
    plt.xlim(time_range)
    plt.xticks(np.arange(time_range[0], time_range[1] + 0.001, 0.1))
    plt.ylim(0, len(df))
    plt.gca().set_xlim(time_range)
    plt.gca().set_xbound(lower=time_range[0], upper=time_range[1])
    handles = [Patch(color=color_map(i), label=f'Stim {s}') for i, s in enumerate(stimulus_ids)]
    plt.legend(handles=handles, title='Stimulus ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        raster_path = os.path.join(save_dir, f'neuron_{neuron_id_3}_raster.eps')
        plt.savefig(raster_path, format='eps')
        print(f"Raster plot saved to {raster_path}")

    plt.show()
    plt.figure(figsize=(12, 6))

    for i, stim in enumerate(stimulus_ids):
        stim_df = df[df[stimulus_column] == stim]
        hist_trials = []

        for _, row in stim_df.iterrows():
            spike_times = row[spike_column]
            if isinstance(spike_times, str):
                spike_times = eval(spike_times)
            spike_times = [s for s in spike_times if time_range[0] <= s <= time_range[1]]
            hist, _ = np.histogram(spike_times, bins=bins)
            hist_trials.append(hist)

        hist_trials = np.array(hist_trials) / bin_size  
        mean_hist = hist_trials.mean(axis=0)
        ci = np.full_like(mean_hist, 1.96 * sem(hist_trials.flatten())) 

        if smooth_sigma:
            pad_width = int(3 * smooth_sigma)
            mean_padded = np.pad(mean_hist, pad_width, mode='constant')
            ci_padded = np.pad(ci, pad_width, mode='constant')
            mean_hist = gaussian_filter1d(mean_padded, sigma=smooth_sigma)[pad_width:-pad_width]
            ci = gaussian_filter1d(ci_padded, sigma=smooth_sigma)[pad_width:-pad_width]
            
        mask = (bin_centers >= time_range[0]) & (bin_centers <= time_range[1])
        x = bin_centers[mask]
        mean_hist = mean_hist[mask]
        ci = ci[mask]

        color = color_map(i)
        plt.plot(x, mean_hist, label=f'Stim {stim}', color=color)
        plt.fill_between(x, mean_hist - ci, mean_hist + ci, color=color, alpha=0.3)

    plt.title(f'Smoothed PSTH + CI — Neuron {neuron_id_3}, Subject {subject_id}')
    plt.xlabel('Time (s)')
    plt.ylabel('Firing Rate (Hz)')
    plt.xlim(time_range)
    plt.xticks(np.arange(time_range[0], time_range[1] + 0.001, 0.1))
    plt.gca().set_xlim(time_range)
    plt.gca().set_xbound(lower=time_range[0], upper=time_range[1])
    plt.legend(title='Stimulus ID')
    plt.grid(True)
    plt.tight_layout()

    if save_dir:
        psth_path = os.path.join(save_dir, f'neuron_{neuron_id_3}_psth.eps')
        plt.savefig(psth_path, format='eps')
        print(f"PSTH plot with CI saved to {psth_path}")
    plt.show()

import pandas as pd

df_spikes = pd.read_excel('../clean_data/cleaned_Encoding1.xlsx')

plot_raster_and_psth_grouped_by_stimulus(
    data=df_spikes,
    neuron_id_3=14016,
    subject_id=14,
    spike_column='Standardized_Spikes',
    stimulus_column='stimulus_index',
    time_range=(0, 1),
    bin_size=0.05,
    smooth_sigma=1,
    save_dir='./'
)
