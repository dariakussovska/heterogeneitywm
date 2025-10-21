import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="hdmf.utils")  # To suppress unnecessary warnings

def read_nwb_file(filepath):
    """
    Reads an NWB file and extracts unit and electrode data as DataFrames.
    """
    io = NWBHDF5IO(filepath, mode='r', load_namespaces=True)
    nwbfile = io.read()

    units = nwbfile.units
    electrodes = nwbfile.electrodes

    units_data = units.to_dataframe()
    electrodes_data = electrodes.to_dataframe()

    io.close()

    return units_data, electrodes_data

def calculate_metrics(units_data):
    """
    Calculate metrics for each unit based on spike times and waveform properties.
    """
    def calculate_isi(spike_times):
        if len(spike_times) < 2:
            return np.nan
        isi = np.diff(spike_times)
        return np.sum(isi < 0.003) / len(isi)

    def calculate_firing_rate(spike_times):
        if len(spike_times) < 2:
            return np.nan
        return len(spike_times) / (spike_times[-1] - spike_times[0])

    def calculate_cv2(spike_times):
        if len(spike_times) < 2:
            return np.nan
        isi = np.diff(spike_times)
        return np.std(isi) / np.mean(isi)

    metrics = pd.DataFrame({
        'ISI_3ms': units_data['spike_times'].apply(calculate_isi),
        'firing_rate': units_data['spike_times'].apply(calculate_firing_rate),
        'CV2': units_data['spike_times'].apply(calculate_cv2),
        'waveform_peak_snr': units_data['waveforms_peak_snr'],
        'waveform_mean_snr': units_data['waveforms_mean_snr'],
        'projection_distance': units_data['waveforms_mean_proj_dist'],
        'isolation_distance': units_data['waveforms_isolation_distance'].apply(lambda x: np.log10(x) if x > 0 else np.nan)
    })

    return metrics

def plot_metrics(metrics):
    """
    Plot quality metrics (excluding electrode-specific plots).
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Spike Sorting Quality Metrics')

    titles = [
        'Percent of ISI < 3 ms', 'Firing Rate (Hz)', 'CV2',
        'Waveform Peak SNR', 'Waveform Mean SNR', 'Isolation Distance (log10)'
    ]

    data = [
        metrics['ISI_3ms'], metrics['firing_rate'], metrics['CV2'],
        metrics['waveform_peak_snr'], metrics['waveform_mean_snr'], metrics['isolation_distance']
    ]

    for ax, title, values in zip(axes.flatten(), titles, data):
        ax.hist(values.dropna(), bins=30, color='blue', alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel('Count')
        ax.set_xlabel(title)

    plt.tight_layout()
    plt.show()

filepaths = [f"/./000469/sub-{i+1}/sub-{i+1}_ses-2_ecephys+image.nwb" for i in range(21)]

all_metrics = pd.DataFrame()

for filepath in filepaths:
    if os.path.exists(filepath):
        units_data, electrodes_data = read_nwb_file(filepath)
        if units_data is not None and electrodes_data is not None:
            metrics = calculate_metrics(units_data)
            metrics['file_id'] = os.path.basename(filepath)  
            all_metrics = pd.concat([all_metrics, metrics], ignore_index=True)

if not all_metrics.empty:
    plot_metrics(all_metrics) 
    all_metrics.to_feather('/./all_spike_rate_metrics.feather', index=False)
    print("Data has been saved to all_spike_rate_metrics.feather")
else:
    print("Failed to read data from NWB files.")

import pandas as pd

# Function to determine the number of images presented based on stimulus indexes
def determine_num_images_presented(stimulus_index_enc2, stimulus_index_enc3):
    count_fives = [stimulus_index_enc2, stimulus_index_enc3].count(5)
    return 3 - count_fives

def ensure_consistent_dtypes(df):
    df['subject_id'] = df['subject_id'].astype(str)  
    df['trial_id'] = df['trial_id'].astype(int) 
    return df

def create_trial_info_df(all_data_enc1, all_data_enc2, all_data_enc3):
    all_data_enc1 = ensure_consistent_dtypes(all_data_enc1)
    all_data_enc2 = ensure_consistent_dtypes(all_data_enc2)
    all_data_enc3 = ensure_consistent_dtypes(all_data_enc3)

    all_data_enc1 = all_data_enc1.drop_duplicates(subset=['subject_id', 'trial_id'])
    all_data_enc2 = all_data_enc2.drop_duplicates(subset=['subject_id', 'trial_id'])
    all_data_enc3 = all_data_enc3.drop_duplicates(subset=['subject_id', 'trial_id'])

    merged_df = all_data_enc1.merge(all_data_enc2, on=['subject_id', 'trial_id'], suffixes=('_enc1', '_enc2'))
    merged_df = merged_df.merge(all_data_enc3, on=['subject_id', 'trial_id'])
    merged_df.rename(columns={'image_id': 'image_id_enc3', 'stimulus_index': 'stimulus_index_enc3'}, inplace=True)

    merged_df['num_images_presented'] = merged_df.apply(
        lambda row: determine_num_images_presented(row['stimulus_index_enc2'], row['stimulus_index_enc3']),
        axis=1
    )

    trial_info_df = merged_df[['subject_id', 'trial_id', 'image_id_enc1', 'stimulus_index_enc1',
                               'image_id_enc2', 'stimulus_index_enc2', 'image_id_enc3',
                               'stimulus_index_enc3', 'num_images_presented']]

    return trial_info_df

all_data_enc1 = pd.read_feather(f'/./all_spike_rate_data_encoding1.feather')
all_data_enc2 = pd.read_feather(f'/./all_spike_rate_data_encoding2.feather')
all_data_enc3 = pd.read_feather(f'/./all_spike_rate_data_encoding3.feather')

trial_info_df = create_trial_info_df(all_data_enc1, all_data_enc2, all_data_enc3)
output_path = '/./trial_info.feather'
trial_info_df.to_feather(output_path, index=False)

print(f"Trial info has been saved to {output_path}")
