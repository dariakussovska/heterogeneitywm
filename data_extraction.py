import pandas as pd
import numpy as np
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path

class NWBProcessor:
    def __init__(self, output_dir="/home/daria/PROJECT"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_subject_id(self, file_path):
        """Extract subject ID from file path."""
        match = re.search(r'sub-(\d+)', str(file_path))
        return int(match.group(1)) if match else None
    
    def get_image_order(self, nwbfile):
        """Extract and visualize stimulus images."""
        stim_templates = nwbfile.stimulus_template['StimulusTemplates']
        sorted_keys = sorted(stim_templates.images.keys())
        
        # Create visualization
        num_images = len(sorted_keys)
        num_cols = 6
        num_rows = (num_images + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
        axes = axes.flatten()
        
        image_order = {}
        for rank, key in enumerate(sorted_keys):
            image_data = stim_templates.images[key].data[:]
            axes[rank].imshow(image_data, cmap='gray')
            axes[rank].set_title(f"ID: {key}\nRank: {rank}")
            axes[rank].axis('off')
            image_order[key] = rank
        
        for ax in axes[num_images:]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return image_order
    
    def process_encoding_periods(self, nwbfile, image_order, subject_id):
        """Process encoding periods data."""
        stim_pres = nwbfile.stimulus['StimulusPresentation']
        trial_data = nwbfile.intervals['trials']
        
        results = {1: [], 2: [], 3: []}
        
        for i in range(len(trial_data)):
            idx_base = i * 4
            
            for enc_num in [1, 2, 3]:
                stim_idx = stim_pres.data[idx_base + enc_num - 1]
                image_id = list(image_order.keys())[stim_idx]
                
                start_time = trial_data[f'timestamps_Encoding{enc_num}'][i]
                stop_time = trial_data[f'timestamps_Encoding{enc_num}_end'][i]
                
                results[enc_num].append({
                    'subject_id': subject_id,
                    'trial_id': i + 1,
                    'start_time': start_time,
                    'stop_time': stop_time,
                    'image_id': image_id,
                    'stimulus_index': stim_idx,
                    'image_rank': image_order[image_id]
                })
        
        return (pd.DataFrame(results[1]), pd.DataFrame(results[2]), pd.DataFrame(results[3]))
    
    def get_period_times(self, nwbfile, start_code, end_code, period_name):
        """Extract period times based on event codes."""
        events = nwbfile.acquisition['events']
        timestamps = events.timestamps[:]
        data = events.data[:]
        
        period_data = []
        trial_id = 0
        
        for i in range(len(data) - 1):
            if data[i] == start_code and data[i + 1] == end_code:
                trial_id += 1
                period_data.append({
                    'start_time': timestamps[i],
                    'end_time': timestamps[i + 1],
                    'trial_id': trial_id
                })
        
        return period_data
    
    def calculate_spike_rates_for_period(self, nwbfile, period_data, subject_id, period_name):
        """Calculate spike rates for any period type."""
        if not period_data:
            return pd.DataFrame()
        
        all_results = []
        total_neurons = len(nwbfile.units['spike_times'])
        
        for neuron_id in range(total_neurons):
            spike_times = nwbfile.units['spike_times'][neuron_id]
            
            for period in period_data:
                start_time, end_time = period['start_time'], period['end_time']
                duration = end_time - start_time
                
                if duration > 0:
                    spikes = [spike for spike in spike_times if start_time < spike < end_time]
                    spike_rate = len(spikes) / duration
                else:
                    spikes = []
                    spike_rate = 0
                
                result = {
                    'subject_id': subject_id,
                    'Neuron_ID': neuron_id,
                    'trial_id': period['trial_id'],
                    f'{period_name}_Start': start_time,
                    f'{period_name}_End': end_time,
                    f'Spikes_in_{period_name}': [spikes],
                    f'Spikes_rate_{period_name}': spike_rate
                }
                
                all_results.append(result)
        
        return pd.DataFrame(all_results)
    
    def process_probe_periods(self, nwbfile, subject_id):
        """Process probe periods with additional trial data."""
        events = nwbfile.acquisition['events']
        timestamps = events.timestamps[:]
        data = events.data[:]
        
        trials_table = nwbfile.intervals['trials']
        probe_data = []
        
        for trial_idx, (start_time, stop_time) in enumerate(zip(
            trials_table['start_time'].data[:],
            trials_table['stop_time'].data[:]
        )):
            within_trial = (timestamps > start_time) & (timestamps < stop_time)
            probe_events = np.where((data[:-1] == 7) & (data[1:] == 8) & within_trial[:-1])[0]
            
            for idx in probe_events:
                probe_data.append({
                    'start_time': timestamps[idx],
                    'end_time': timestamps[idx + 1],
                    'trial_id': trial_idx + 1,
                    'image_id': trials_table['loadsProbe_PicIDs'].data[trial_idx],
                    'probe_in_out': trials_table['probe_in_out'].data[trial_idx],
                    'response_accuracy': trials_table['response_accuracy'].data[trial_idx]
                })
        
        return self.calculate_spike_rates_for_period(nwbfile, probe_data, subject_id, 'Probe')
    
    def process_single_file(self, filepath):
        """Process a single NWB file and extract all data."""
        print(f"Processing: {filepath}")
        
        with NWBHDF5IO(filepath, 'r', load_namespaces=True) as io:
            nwbfile = io.read()
            subject_id = self.extract_subject_id(filepath)
            
            # Get image order for encoding periods
            image_order = self.get_image_order(nwbfile)
            
            # Process encoding periods
            df_enc1, df_enc2, df_enc3 = self.process_encoding_periods(nwbfile, image_order, subject_id)
            
            # Add spike rates to encoding periods
            encoding_dfs = {}
            for enc_num, df_enc in zip([1, 2, 3], [df_enc1, df_enc2, df_enc3]):
                encoding_dfs[enc_num] = self.add_spike_rates_to_encoding(nwbfile, df_enc, f'Encoding{enc_num}')
            
            # Process other periods
            fixation_data = self.get_period_times(nwbfile, 11, 1, 'Fixation')
            df_fixation = self.calculate_spike_rates_for_period(nwbfile, fixation_data, subject_id, 'Fixation')
            
            delay_data = self.get_period_times(nwbfile, 6, 7, 'Delay')
            df_delay = self.calculate_spike_rates_for_period(nwbfile, delay_data, subject_id, 'Delay')
            
            df_probe = self.process_probe_periods(nwbfile, subject_id)
            
            return {
                'encoding1': encoding_dfs[1],
                'encoding2': encoding_dfs[2],
                'encoding3': encoding_dfs[3],
                'fixation': df_fixation,
                'delay': df_delay,
                'probe': df_probe
            }
    
    def add_spike_rates_to_encoding(self, nwbfile, df_encoding, period_name):
        """Add spike rates to encoding period dataframes."""
        total_neurons = len(nwbfile.units['spike_times'])
        all_results = []
        
        for neuron_id in range(total_neurons):
            spike_times = nwbfile.units['spike_times'][neuron_id]
            neuron_results = []
            
            for _, row in df_encoding.iterrows():
                start_time, stop_time = row['start_time'], row['stop_time']
                
                if pd.isna(start_time) or pd.isna(stop_time) or (stop_time - start_time) <= 0:
                    spikes = []
                    spike_rate = 0
                else:
                    spikes = [spike for spike in spike_times if start_time <= spike < stop_time]
                    spike_rate = len(spikes) / (stop_time - start_time)
                
                result = row.to_dict()
                result.update({
                    'Neuron_ID': neuron_id,
                    'Spikes': [spikes],
                    f'Spikes_rate_{period_name}': spike_rate
                })
                neuron_results.append(result)
            
            all_results.extend(neuron_results)
        
        return pd.DataFrame(all_results)
    
    def process_all_files(self, filepaths):
        """Process all NWB files and save results."""
        all_data = {
            'encoding1': [], 'encoding2': [], 'encoding3': [], 
            'fixation': [], 'delay': [], 'probe': []
        }
        
        for filepath in filepaths:
            filepath = Path(filepath)
            if filepath.exists():
                try:
                    results = self.process_single_file(filepath)
                    for key in all_data.keys():
                        if not results[key].empty:
                            all_data[key].append(results[key])
                    print(f"✓ Successfully processed {filepath.name}")
                except Exception as e:
                    print(f"✗ Error processing {filepath.name}: {e}")
            else:
                print(f"✗ File not found: {filepath}")
        
        # Combine and save all data
        for data_type, data_list in all_data.items():
            if data_list:
                combined_df = pd.concat(data_list, ignore_index=True)
                output_path = self.output_dir / f"all_spike_rate_data_{data_type}.xlsx"
                combined_df.to_excel(output_path, index=False)
                print(f"✓ Saved {data_type} data: {output_path}")

# Main execution
if __name__ == "__main__":
    # Create file paths
    filepaths = [f"/home/daria/PROJECT/000469/sub-{i+1}/sub-{i+1}_ses-2_ecephys+image.nwb" 
                for i in range(21)]
    
    # Initialize processor
    processor = NWBProcessor("/home/daria/PROJECT")
    
    # Process all files
    processor.process_all_files(filepaths)
    
    print("All data processing complete!")
    print("Generated files:")
    for period in ['encoding1', 'encoding2', 'encoding3', 'fixation', 'delay', 'probe']:
        print(f"  - all_spike_rate_data_{period}.xlsx")

