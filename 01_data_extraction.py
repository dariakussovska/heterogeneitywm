import pandas as pd
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt 
import re
import numpy as np
import os
from typing import Tuple, Dict, List

def extract_subject_id(file_path: str) -> int:
    """Extracts the subject_id from the file path."""
    match = re.search(r'sub-(\d+)', file_path)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Subject ID not found in file path: {file_path}")

def read_nwb_file(filepath: str):
    """Read NWB file with proper error handling."""
    io = NWBHDF5IO(filepath, mode='r', load_namespaces=True)
    nwbfile = io.read()
    return nwbfile, io

def visualize_and_rank_stimulus_images(nwbfile) -> Dict:
    """Visualize and rank stimulus images."""
    stim_templates = nwbfile.stimulus_template['StimulusTemplates']
    sorted_keys = sorted(stim_templates.images.keys())
    num_images = len(sorted_keys)
    
    # Create visualization
    num_columns = 6
    num_rows = (num_images + num_columns - 1) // num_columns 
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 3 * num_rows))
    axes = axes.flatten() 
    
    image_order = {}
    for rank, key in enumerate(sorted_keys):
        image_data = stim_templates.images[key].data[:]
        axes[rank].imshow(image_data, cmap='gray')
        axes[rank].set_title(f"ID: {key}\nRank: {rank}")
        axes[rank].axis('off')  
        image_order[key] = rank
        
    # Hide unused axes
    for ax in axes[len(sorted_keys):]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return image_order

def map_image_ids_to_encoding_periods(nwbfile, image_order: Dict, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map image IDs to encoding periods."""
    subject_id = extract_subject_id(file_path)
    stim_pres = nwbfile.stimulus['StimulusPresentation']
    trial_data = nwbfile.intervals['trials']
    
    n_trials = len(trial_data)
    image_keys = list(image_order.keys())
    
    # Pre-allocate data lists for better performance
    enc1_data, enc2_data, enc3_data = [], [], []
    
    for i in range(n_trials):
        idx_base = i * 4
        stimulus_indices = [
            stim_pres.data[idx_base],
            stim_pres.data[idx_base + 1], 
            stim_pres.data[idx_base + 2]
        ]
        
        image_ids = [image_keys[idx] for idx in stimulus_indices]
        
        trial_record = {
            'subject_id': subject_id,
            'trial_id': i + 1,
        }
        
        # Encoding 1
        enc1_data.append({
            **trial_record,
            'start_time': trial_data['timestamps_Encoding1'][i],
            'stop_time': trial_data['timestamps_Encoding1_end'][i],
            'image_id': image_ids[0],
            'stimulus_index': stimulus_indices[0],
            'image_rank': image_order[image_ids[0]]
        })
        
        # Encoding 2
        enc2_data.append({
            **trial_record,
            'start_time': trial_data['timestamps_Encoding2'][i],
            'stop_time': trial_data['timestamps_Encoding2_end'][i],
            'image_id': image_ids[1],
            'stimulus_index': stimulus_indices[1],
            'image_rank': image_order[image_ids[1]]
        })
        
        # Encoding 3
        enc3_data.append({
            **trial_record,
            'start_time': trial_data['timestamps_Encoding3'][i],
            'stop_time': trial_data['timestamps_Encoding3_end'][i],
            'image_id': image_ids[2],
            'stimulus_index': stimulus_indices[2],
            'image_rank': image_order[image_ids[2]]
        })
    
    return pd.DataFrame(enc1_data), pd.DataFrame(enc2_data), pd.DataFrame(enc3_data)

def calculate_spike_rate(spike_times: np.ndarray, start_time: float, stop_time: float) -> Tuple[List, float]:
    """Calculate spikes and spike rate for a given time window."""
    if pd.isna(start_time) or pd.isna(stop_time) or (stop_time - start_time) <= 0:
        return [], 0.0
    
    spikes_in_window = [spike for spike in spike_times if start_time <= spike < stop_time]
    duration = stop_time - start_time
    spike_rate = len(spikes_in_window) / duration if duration > 0 else 0.0
    
    return spikes_in_window, spike_rate

def get_spike_rate_single_neuron(nwbfile, neuron_id: int, df_encoding: pd.DataFrame) -> pd.DataFrame:
    """Calculate spike rates for a single neuron across encoding periods."""
    spike_times = nwbfile.units['spike_times'][neuron_id]
    
    spikes = []
    spikes_rate = []
    
    for _, row in df_encoding.iterrows():
        spike_list, rate = calculate_spike_rate(spike_times, row['start_time'], row['stop_time'])
        spikes.append(spike_list)
        spikes_rate.append(rate)
    
    df_result = df_encoding.copy()
    df_result['Spikes'] = spikes
    df_result['Spikes_rate'] = spikes_rate
    df_result['Neuron_ID'] = neuron_id
    
    return df_result

def get_spike_rate_all_neurons(nwbfile, df_enc1: pd.DataFrame, df_enc2: pd.DataFrame, df_enc3: pd.DataFrame, subject_id: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute spike rates for all neurons across encoding periods."""
    total_neurons = len(nwbfile.units['spike_times'])
    
    # Use list comprehension for better performance
    enc1_results = []
    enc2_results = [] 
    enc3_results = []
    
    for neuron_id in range(total_neurons):
        enc1_results.append(get_spike_rate_single_neuron(nwbfile, neuron_id, df_enc1.copy()))
        enc2_results.append(get_spike_rate_single_neuron(nwbfile, neuron_id, df_enc2.copy()))
        enc3_results.append(get_spike_rate_single_neuron(nwbfile, neuron_id, df_enc3.copy()))
    
    # Concatenate all results at once
    df_final_enc1 = pd.concat(enc1_results, ignore_index=True)
    df_final_enc2 = pd.concat(enc2_results, ignore_index=True)
    df_final_enc3 = pd.concat(enc3_results, ignore_index=True)
    
    # Add subject_id (already present from mapping function)
    return df_final_enc1, df_final_enc2, df_final_enc3

def process_encoding_file(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process an NWB file and extract encoding period data."""
    nwbfile, io = read_nwb_file(filepath)
    subject_id = extract_subject_id(filepath)
    
    try:
        image_order = visualize_and_rank_stimulus_images(nwbfile)
        df_enc1, df_enc2, df_enc3 = map_image_ids_to_encoding_periods(nwbfile, image_order, filepath)
        df_final_enc1, df_final_enc2, df_final_enc3 = get_spike_rate_all_neurons(nwbfile, df_enc1, df_enc2, df_enc3, subject_id)
    finally:
        io.close()
    
    return df_final_enc1, df_final_enc2, df_final_enc3

# Fixation period functions
def get_fixation_periods(nwbfile) -> Tuple[List, List, List]:
    """Extracts fixation periods from events."""
    events = nwbfile.acquisition['events']
    timestamps = events.timestamps[:]
    data = events.data[:]
    
    fixation_starts = []
    fixation_ends = []
    trial_ids = []
    
    for i in range(len(data) - 1):
        if data[i] == 11 and data[i + 1] == 1:  # Fixation onset to picture shown
            trial_id = len(fixation_starts) + 1
            fixation_starts.append(timestamps[i])
            fixation_ends.append(timestamps[i + 1])
            trial_ids.append(trial_id)
    
    return fixation_starts, fixation_ends, trial_ids

def process_fixation_periods(nwbfile, subject_id: int) -> pd.DataFrame:
    """Process fixation periods for all neurons."""
    fixation_starts, fixation_ends, trial_ids = get_fixation_periods(nwbfile)
    
    if not fixation_starts:
        return pd.DataFrame()
    
    total_neurons = len(nwbfile.units['spike_times'])
    all_results = []
    
    for neuron_id in range(total_neurons):
        spike_times = nwbfile.units['spike_times'][neuron_id]
        spikes_list = []
        rates_list = []
        
        for start, end in zip(fixation_starts, fixation_ends):
            spikes, rate = calculate_spike_rate(spike_times, start, end)
            spikes_list.append(spikes)
            rates_list.append(rate)
        
        all_results.append(pd.DataFrame({
            'subject_id': subject_id,
            'Neuron_ID': neuron_id,
            'trial_id': trial_ids,
            'Fixation_Start': fixation_starts,
            'Fixation_End': fixation_ends,
            'Spikes_in_Fixation': spikes_list,
            'Spikes_rate_Fixation': rates_list,
        }))
    
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

# Delay period functions  
def get_delay_periods(nwbfile) -> Tuple[List, List, List]:
    """Extracts delay periods from events."""
    events = nwbfile.acquisition['events']
    timestamps = events.timestamps[:]
    data = events.data[:]
    
    delay_starts = []
    delay_ends = []
    trial_ids = []
    
    for i in range(len(data) - 1):
        if data[i] == 6 and data[i + 1] == 7:
            trial_id = len(delay_starts) + 1
            delay_starts.append(timestamps[i])
            delay_ends.append(timestamps[i + 1])
            trial_ids.append(trial_id)
    
    return delay_starts, delay_ends, trial_ids

def process_delay_periods(nwbfile, subject_id: int) -> pd.DataFrame:
    """Process delay periods for all neurons."""
    delay_starts, delay_ends, trial_ids = get_delay_periods(nwbfile)
    
    if not delay_starts:
        return pd.DataFrame()
    
    total_neurons = len(nwbfile.units['spike_times'])
    all_results = []
    
    for neuron_id in range(total_neurons):
        spike_times = nwbfile.units['spike_times'][neuron_id]
        spikes_list = []
        rates_list = []
        
        for start, end in zip(delay_starts, delay_ends):
            spikes, rate = calculate_spike_rate(spike_times, start, end)
            spikes_list.append(spikes)
            rates_list.append(rate)
        
        all_results.append(pd.DataFrame({
            'subject_id': subject_id,
            'Neuron_ID': neuron_id,
            'trial_id': trial_ids,
            'Delay_Start': delay_starts,
            'Delay_End': delay_ends,
            'Spikes_in_Delay': spikes_list,
            'Spikes_rate_Delay': rates_list,
        }))
    
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

# Probe period functions
def get_probe_periods_with_trials(nwbfile) -> Tuple[List, List, List, List, List, List]:
    """Extracts probe periods with trial information."""
    trials_table = nwbfile.intervals['trials']
    
    probe_starts = []
    probe_ends = []
    image_ids = []
    trial_nums = []
    in_out = []
    accuracy = []
    
    events = nwbfile.acquisition['events']
    event_times = events.timestamps[:]
    event_data = events.data[:]
    
    for trial_idx in range(len(trials_table)):
        trial_start = trials_table['start_time'][trial_idx]
        trial_end = trials_table['stop_time'][trial_idx]
        
        # Find probe events within this trial
        mask = (event_times > trial_start) & (event_times < trial_end)
        probe_indices = np.where((event_data[:-1] == 7) & (event_data[1:] == 8) & mask[:-1])[0]
        
        for idx in probe_indices:
            probe_starts.append(event_times[idx])
            probe_ends.append(event_times[idx + 1])
            image_ids.append(trials_table['loadsProbe_PicIDs'][trial_idx])
            trial_nums.append(trial_idx + 1)
            in_out.append(trials_table['probe_in_out'][trial_idx])
            accuracy.append(trials_table['response_accuracy'][trial_idx])
    
    return probe_starts, probe_ends, image_ids, trial_nums, in_out, accuracy

def process_probe_periods(nwbfile, subject_id: int) -> pd.DataFrame:
    """Process probe periods for all neurons."""
    probe_data = get_probe_periods_with_trials(nwbfile)
    probe_starts, probe_ends, image_ids, trial_nums, in_out, accuracy = probe_data
    
    if not probe_starts:
        return pd.DataFrame()
    
    total_neurons = len(nwbfile.units['spike_times'])
    all_results = []
    
    for neuron_id in range(total_neurons):
        spike_times = nwbfile.units['spike_times'][neuron_id]
        spikes_list = []
        rates_list = []
        
        for start, end in zip(probe_starts, probe_ends):
            spikes, rate = calculate_spike_rate(spike_times, start, end)
            spikes_list.append(spikes)
            rates_list.append(rate)
        
        all_results.append(pd.DataFrame({
            'subject_id': subject_id,
            'Neuron_ID': neuron_id,
            'trial_id': trial_nums,
            'Probe_Start_Time': probe_starts,
            'Probe_End_Time': probe_ends,
            'Probe_Image_ID': [img_id - 1 for img_id in image_ids],
            'Spikes_in_Probe': spikes_list,
            'Spikes_rate_Probe': rates_list,
            'probe_in_out': in_out,
            'response_accuracy': accuracy
        }))
    
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

# Main processing pipeline
def process_all_files(filepaths: List[str]) -> None:
    """Process all NWB files for all periods."""
    output_dir = "/./"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data containers
    all_enc1, all_enc2, all_enc3 = [], [], []
    all_fixation, all_delay, all_probe = [], [], []
    
    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        print(f"Processing: {filepath}")
        subject_id = extract_subject_id(filepath)
        
        try:
            nwbfile, io = read_nwb_file(filepath)
            
            # Process all periods for this file
            enc1, enc2, enc3 = process_encoding_file(filepath)
            all_enc1.append(enc1)
            all_enc2.append(enc2) 
            all_enc3.append(enc3)
            
            fixation_data = process_fixation_periods(nwbfile, subject_id)
            all_fixation.append(fixation_data)
            
            delay_data = process_delay_periods(nwbfile, subject_id)
            all_delay.append(delay_data)
            
            probe_data = process_probe_periods(nwbfile, subject_id)
            all_probe.append(probe_data)
            
            io.close()
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    # Concatenate and save all data
    def save_data(data_list, filename):
        if data_list:
            combined = pd.concat([d for d in data_list if not d.empty], ignore_index=True)
            combined.to_excel(os.path.join(output_dir, filename), index=False)
            print(f"Saved {filename} with {len(combined)} rows")
        else:
            print(f"No data for {filename}")
    
    save_data(all_enc1, "all_spike_rate_data_encoding1.xlsx")
    save_data(all_enc2, "all_spike_rate_data_encoding2.xlsx") 
    save_data(all_enc3, "all_spike_rate_data_encoding3.xlsx")
    save_data(all_fixation, "all_spike_rate_data_fixation.xlsx")
    save_data(all_delay, "all_spike_rate_data_delay.xlsx")
    save_data(all_probe, "all_spike_rate_data_probe.xlsx")
    
    print("All data processing complete!")

# Run the processing
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    filepaths = [f"/./000469/sub-{i+1}/sub-{i+1}_ses-2_ecephys+image.nwb" for i in range(21)]
    process_all_files(filepaths)
