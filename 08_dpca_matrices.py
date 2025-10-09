import pandas as pd 
import numpy as np
import ast
from dPCA import dPCA
import matplotlib.pyplot as plt
from ast import literal_eval
from scipy.ndimage import gaussian_filter1d
from dPCA.dPCA import dPCA
from collections import defaultdict
from scipy.spatial.distance import pdist
import scipy.stats as sps
import scikit_posthocs as sp

trial_info = pd.read_excel('/home/daria/PROJECT/new_trial_info.xlsx')
subject_trials = trial_info[(trial_info['subject_id'] == 14) & (trial_info['num_images_presented'] == 1)][['new_trial_id', 'num_images_presented', 'stimulus_index']]
y_matrix = subject_trials

target_trials_per_stimulus = 6  # Desired number of trials per stimulus

# Initialize an empty DataFrame for the balanced y_matrix
y_matrix_balanced = pd.DataFrame()

# Iterate through each stimulus index and sample the desired number of trials
for stimulus in y_matrix['stimulus_index'].unique():
    stimulus_trials = y_matrix[y_matrix['stimulus_index'] == stimulus]
    sampled_trials = stimulus_trials.sample(n=target_trials_per_stimulus, random_state=42)  # Set random_state for reproducibility
    y_matrix_balanced = pd.concat([y_matrix_balanced, sampled_trials]) 

y_matrix_balanced.reset_index(drop=True, inplace=True)
balanced_distribution = y_matrix_balanced['stimulus_index'].value_counts().sort_index()
print("\nBalanced trial distribution across stimuli:")
print(balanced_distribution)

balanced_y_matrix = y_matrix_balanced

fixation_data_with_brain_region = pd.read_excel('/home/daria/PROJECT/graph_data/graph_fixation.xlsx')
enc1_data_with_brain_region = pd.read_excel('/home/daria/PROJECT/clean_data/cleaned_Encoding1.xlsx')
delay_data_with_brain_region = pd.read_excel('/home/daria/PROJECT/graph_data/graph_delay.xlsx')

fixation_data = fixation_data_with_brain_region

def filter_data(data, subject_ids=None, brain_regions=None, neuron_ids_3=None):
    filtered_data = data[data['Signi'] == 'Y']  # Start with all significant neurons
    if subject_ids:
        filtered_data = filtered_data[filtered_data['subject_id'].isin(subject_ids)]
    if brain_regions:
        filtered_data = filtered_data[filtered_data['Location'].isin(brain_regions)]
    if neuron_ids_3:  
        filtered_data = filtered_data[filtered_data['Neuron_ID_3'].isin(neuron_ids_3)]
    return filtered_data

def calculate_baseline_stats(fixation_data, spike_column, subject_id, neuron_id):
    neuron_data = fixation_data[(fixation_data['subject_id'] == subject_id) & (fixation_data['Neuron_ID'] == neuron_id)]
    if neuron_data.empty or spike_column not in neuron_data.columns:
        return 0, 1 
    firing_rates = neuron_data[spike_column].dropna().values
    return np.mean(firing_rates), np.std(firing_rates)

def calculate_firing_rates(trials, spike_column, time_bins):
    trial_firing_rates = []
    for _, row in trials.iterrows():
        spike_times = literal_eval(str(row[spike_column])) if isinstance(row[spike_column], str) else []
        if not isinstance(spike_times, list):
            trial_firing_rates.append(np.zeros(len(time_bins) - 1))
            continue
        spike_counts, _ = np.histogram(spike_times, bins=time_bins)
        firing_rates = spike_counts / (time_bins[1] - time_bins[0]) 
        trial_firing_rates.append(firing_rates)
    return np.array(trial_firing_rates)  

def calculate_z_scores(firing_rates, mean_baseline, std_baseline):
    if std_baseline == 0:
        return np.zeros_like(firing_rates) 
    return (firing_rates - mean_baseline) / std_baseline

def construct_stimulus_tables_no_avg_balanced(enc1_data, fixation_data, balanced_y_matrix, time_bins_enc1, smoothing_sigma):
    subjects_in_data = enc1_data['subject_id'].unique()
    print(f"Subjects included in analysis: {subjects_in_data}")

    neuron_subject_combinations = enc1_data[['Neuron_ID', 'subject_id', 'Neuron_ID_3']].drop_duplicates()
    num_neurons_analyzed = neuron_subject_combinations.shape[0]
    print(f"Number of neurons analyzed: {num_neurons_analyzed}")

    stimulus_ids = balanced_y_matrix['stimulus_index'].unique()
    z_scores_by_stimulus = {stimulus: {} for stimulus in stimulus_ids}

    for _, row in neuron_subject_combinations.iterrows():
        neuron_id, subject_id, neuron_id_3 = row['Neuron_ID'], row['subject_id'], row['Neuron_ID_3']
        label = f"{subject_id}_{neuron_id}_{neuron_id_3}" 

        baseline_mean, baseline_std = calculate_baseline_stats(
            fixation_data, 'Spikes_rate_Fixation', subject_id, neuron_id
        )

        for stimulus in stimulus_ids:
            trial_ids = balanced_y_matrix[balanced_y_matrix['stimulus_index'] == stimulus]['new_trial_id']
            stimulus_trials = enc1_data[
                (enc1_data['new_trial_id'].isin(trial_ids)) &
                (enc1_data['Neuron_ID'] == neuron_id) &
                (enc1_data['subject_id'] == subject_id)
            ]

            trial_firing_rates = calculate_firing_rates(stimulus_trials, 'Standardized_Spikes', time_bins_enc1)
            trial_z_scores = np.array([calculate_z_scores(fr, baseline_mean, baseline_std) for fr in trial_firing_rates])
            smoothed_z_scores = np.array([gaussian_filter1d(z, sigma=smoothing_sigma) for z in trial_z_scores])

            if label not in z_scores_by_stimulus[stimulus]:
                z_scores_by_stimulus[stimulus][label] = smoothed_z_scores
            else:
                z_scores_by_stimulus[stimulus][label] = np.vstack(
                    [z_scores_by_stimulus[stimulus][label], smoothed_z_scores]
                )

    return z_scores_by_stimulus

subject_ids_to_include = []  
brain_regions_to_include = []  
neuron_ids_3_to_include = []  

fixation_filtered = filter_data(fixation_data_with_brain_region, subject_ids_to_include, brain_regions_to_include, neuron_ids_3_to_include)
enc1_filtered = filter_data(enc1_data_with_brain_region, subject_ids_to_include, brain_regions_to_include, neuron_ids_3_to_include)

time_bins_enc1 = np.arange(0, 1, 0.002) 
smoothing_sigma = 60 

z_scores_by_stimulus_balanced = construct_stimulus_tables_no_avg_balanced(enc1_filtered, fixation_filtered, balanced_y_matrix, time_bins_enc1, smoothing_sigma)

neurons = enc1_filtered[['Neuron_ID', 'subject_id', 'Neuron_ID_3']].drop_duplicates()
neuron_labels = neurons.apply(lambda x: f"{x['subject_id']}_{x['Neuron_ID']}_{x['Neuron_ID_3']}", axis=1).tolist()
num_neurons = len(neuron_labels) 
print(f"Number of neurons: {num_neurons}")

stimuli = balanced_y_matrix['stimulus_index'].unique()
num_stimuli = len(stimuli)
num_samples_per_stimulus = {stimulus: balanced_y_matrix[balanced_y_matrix['stimulus_index'] == stimulus].shape[0]
                            for stimulus in stimuli}
max_samples = max(num_samples_per_stimulus.values()) 

num_time_bins = len(time_bins_enc1) - 1
Xtrial = np.zeros((max_samples, num_neurons, num_stimuli, num_time_bins))

for stimulus_idx, stimulus in enumerate(stimuli):
    stimulus_trials = balanced_y_matrix[balanced_y_matrix['stimulus_index'] == stimulus]['new_trial_id'].tolist()
    for sample_idx, trial_id in enumerate(stimulus_trials):
        for neuron_idx, neuron_label in enumerate(neuron_labels):
            if neuron_label in z_scores_by_stimulus_balanced[stimulus]:
                trial_data = z_scores_by_stimulus_balanced[stimulus][neuron_label]
                if sample_idx < trial_data.shape[0]:
                    Xtrial[sample_idx, neuron_idx, stimulus_idx, :] = trial_data[sample_idx]
                else:
                    Xtrial[sample_idx, neuron_idx, stimulus_idx, :] = np.zeros(num_time_bins)

# check the shape of the constructed Xtrial matrix
print(f"Xtrial matrix shape: {Xtrial.shape}")  

### DELAY 

# Define the time bins for the delay period
time_bins_delay = np.arange(0, 2.5, 0.002)  
subject_ids_to_include = []  
brain_regions_to_include = [] 
neuron_ids_3_to_include = []

delay_filtered = filter_data(delay_data_with_brain_region, subject_ids_to_include, brain_regions_to_include, neuron_ids_3_to_include)
delay_filtered = delay_filtered[delay_filtered['Signi'] == 'Y']

time_bins_delay = np.arange(0, 2.5, 0.002)  

def construct_stimulus_tables_no_avg_delay(delay_data, fixation_data, balanced_y_matrix, time_bins_delay, smoothing_sigma):
    subjects_in_data = delay_data['subject_id'].unique()
    print(f"Subjects included in analysis: {subjects_in_data}")

    neuron_subject_combinations = delay_data[['Neuron_ID', 'subject_id', 'Neuron_ID_3']].drop_duplicates()
    num_neurons_analyzed = neuron_subject_combinations.shape[0]
    print(f"Number of neurons analyzed: {num_neurons_analyzed}")

    stimulus_ids = balanced_y_matrix['stimulus_index'].unique()
    z_scores_by_stimulus = {stimulus: {} for stimulus in stimulus_ids}

    for _, row in neuron_subject_combinations.iterrows():
        neuron_id, subject_id, neuron_id_3 = row['Neuron_ID'], row['subject_id'], row['Neuron_ID_3']
        label = f"{subject_id}_{neuron_id}_{neuron_id_3}" 

        # Baseline statistics from fixation period
        baseline_mean, baseline_std = calculate_baseline_stats(
            fixation_data, 'Spikes_rate_Fixation', subject_id, neuron_id
        )

        for stimulus in stimulus_ids:
            trial_ids = balanced_y_matrix[balanced_y_matrix['stimulus_index'] == stimulus]['new_trial_id']
            stimulus_trials = delay_data[
                (delay_data['trial_id'].isin(trial_ids)) &
                (delay_data['Neuron_ID'] == neuron_id) &
                (delay_data['subject_id'] == subject_id)
            ]

            trial_firing_rates = calculate_firing_rates(stimulus_trials, 'Standardized_Spikes_in_Delay', time_bins_delay)
            trial_z_scores = np.array([calculate_z_scores(fr, baseline_mean, baseline_std) for fr in trial_firing_rates])
            smoothed_z_scores = np.array([gaussian_filter1d(z, sigma=smoothing_sigma) for z in trial_z_scores])
            if label not in z_scores_by_stimulus[stimulus]:
                z_scores_by_stimulus[stimulus][label] = smoothed_z_scores
            else:
                z_scores_by_stimulus[stimulus][label] = np.vstack(
                    [z_scores_by_stimulus[stimulus][label], smoothed_z_scores]
                )

    return z_scores_by_stimulus

z_scores_by_stimulus_delay = construct_stimulus_tables_no_avg_delay(delay_filtered, fixation_filtered, balanced_y_matrix, time_bins_delay, smoothing_sigma)
neurons = delay_filtered[['Neuron_ID', 'subject_id', 'Neuron_ID_3']].drop_duplicates()
neuron_labels = neurons.apply(lambda x: f"{x['subject_id']}_{x['Neuron_ID']}_{x['Neuron_ID_3']}", axis=1).tolist()
num_neurons = len(neuron_labels) 
print(f"Number of neurons: {num_neurons}")

num_time_bins_delay = len(time_bins_delay) - 1
Dtrial = np.zeros((max_samples, num_neurons, num_stimuli, num_time_bins_delay))

# Populate the Dtrial matrix
for stimulus_idx, stimulus in enumerate(stimuli):
    stimulus_trials = balanced_y_matrix[balanced_y_matrix['stimulus_index'] == stimulus]['new_trial_id'].tolist()
    for sample_idx, trial_id in enumerate(stimulus_trials):
        for neuron_idx, neuron_label in enumerate(neuron_labels):
            if neuron_label in z_scores_by_stimulus_delay[stimulus]:
                trial_data = z_scores_by_stimulus_delay[stimulus][neuron_label]
                if sample_idx < trial_data.shape[0]:
                    Dtrial[sample_idx, neuron_idx, stimulus_idx, :] = trial_data[sample_idx]
                else:
                    Dtrial[sample_idx, neuron_idx, stimulus_idx, :] = np.zeros(num_time_bins_delay)

print(f"Dtrial matrix shape: {Dtrial.shape}")  

trialE = Xtrial
trialD = Dtrial 
print(trialE.shape)
print(trialD.shape)

import numpy as np

# Define your directory path
save_directory = "/home/daria/PROJECT/"  # Replace with your desired path

# Save individual files
np.save(f"{save_directory}trialE.npy", trialE)
np.save(f"{save_directory}trialD.npy", trialD)

print(f"Matrices saved to {save_directory}")
