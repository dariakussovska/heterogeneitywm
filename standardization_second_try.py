import pandas as pd
import ast
import os

BASE_DIR = '/home/daria/PROJECT'

filtered_files = {
    'Encoding1': os.path.join(BASE_DIR, 'all_spike_rate_data_encoding1.xlsx'),
    'Encoding2': os.path.join(BASE_DIR, 'all_spike_rate_data_encoding2.xlsx'),
    'Encoding3': os.path.join(BASE_DIR, 'all_spike_rate_data_encoding3.xlsx'),
    'Delay': os.path.join(BASE_DIR, 'all_spike_rate_data_delay.xlsx'),
    'Probe': os.path.join(BASE_DIR, 'all_spike_rate_data_probe.xlsx')
}

def load_enc1_start_times(enc1_path):
    enc1_df = pd.read_excel(enc1_path)
    return enc1_df[['subject_id', 'trial_id', 'start_time']].drop_duplicates().rename(columns={'start_time': 'start_time_enc1'})

enc1_start_times = load_enc1_start_times(filtered_files['Encoding1'])

def standardize_spikes(file_path, spikes_column, period_name):
    df = pd.read_excel(file_path)
    
    df = pd.merge(
        df,
        enc1_start_times,
        on=['subject_id', 'trial_id'],
        how='left'
    )

    if 'start_time_enc1' in df.columns and spikes_column in df.columns:
        def standardize_row_spikes(row):
            spikes = row[spikes_column]
            
            if isinstance(spikes, str):
                try:
                    spikes = ast.literal_eval(spikes)
                except:
                    spikes = []
            
            def flatten_spikes(spike_data):
                flat_list = []
                for item in spike_data:
                    if isinstance(item, list):
                        flat_list.extend(flatten_spikes(item))
                    else:
                        flat_list.append(item)
                return flat_list
            
            if isinstance(spikes, list):
                spikes = flatten_spikes(spikes)
            
            if isinstance(spikes, list) and not pd.isna(row['start_time_enc1']):
                try:
                    return [float(spike) - float(row['start_time_enc1']) for spike in spikes if spike is not None]
                except (TypeError, ValueError) as e:
                    print(f"Error processing spikes: {spikes}, error: {e}")
                    return []
            return []

        df['Standardized_Spikes'] = df.apply(standardize_row_spikes, axis=1)
        return df
    else:
        print(f"Missing required columns in {file_path}. Skipping standardization.")
        return None

# Test with one file first
print("Testing with Encoding1 file...")
test_df = pd.read_excel(filtered_files['Encoding1'])
print("Sample spike data:")
print(test_df['Spikes'].head(3))

standardized_data = {}
for period_name, file_path in filtered_files.items():
    print(f"Processing {period_name}...")
    spikes_column = 'Spikes' 
    if period_name == 'Delay':
        spikes_column = 'Spikes_in_Delay'
    elif period_name == 'Probe':
        spikes_column = 'Spikes_in_Probe'

    standardized_df = standardize_spikes(
        file_path=file_path,
        spikes_column=spikes_column,
        period_name=period_name
    )
    if standardized_df is not None:
        standardized_data[period_name] = standardized_df

print("Standardization completed for all filtered files.")

# Load trial info files
trial_info_new = pd.read_excel(os.path.join(BASE_DIR, 'new_trial_info.xlsx'))
trial_info_final = pd.read_excel(os.path.join(BASE_DIR, 'new_trial_final.xlsx'))

# Standardize column types
trial_info_new['subject_id'] = trial_info_new['subject_id'].astype(str).str.strip()
trial_info_new['trial_id'] = trial_info_new['trial_id'].astype(int)
trial_info_final['subject_id'] = trial_info_final['subject_id'].astype(str).str.strip()
trial_info_final['trial_id'] = trial_info_final['trial_id'].astype(int)

# Load encoding data to get the actual stimulus indices for each period
print("Loading encoding data to get proper stimulus indices...")
enc1_data = pd.read_excel(filtered_files['Encoding1'])[['subject_id', 'trial_id', 'stimulus_index']].rename(columns={'stimulus_index': 'stimulus_index_enc1'})
enc2_data = pd.read_excel(filtered_files['Encoding2'])[['subject_id', 'trial_id', 'stimulus_index']].rename(columns={'stimulus_index': 'stimulus_index_enc2'})
enc3_data = pd.read_excel(filtered_files['Encoding3'])[['subject_id', 'trial_id', 'stimulus_index']].rename(columns={'stimulus_index': 'stimulus_index_enc3'})

# Merge all encoding stimulus indices into trial info
trial_info_with_stimuli = trial_info_new.copy()
trial_info_with_stimuli = pd.merge(trial_info_with_stimuli, enc1_data, on=['subject_id', 'trial_id'], how='left')
trial_info_with_stimuli = pd.merge(trial_info_with_stimuli, enc2_data, on=['subject_id', 'trial_id'], how='left')
trial_info_with_stimuli = pd.merge(trial_info_with_stimuli, enc3_data, on=['subject_id', 'trial_id'], how='left')

print("Stimulus indices merged into trial info:")
print(trial_info_with_stimuli[['subject_id', 'trial_id', 'num_images_presented', 'stimulus_index_enc1', 'stimulus_index_enc2', 'stimulus_index_enc3']].head())

# Function to determine the correct stimulus index for delay based on num_images_presented
def get_delay_stimulus_index(row):
    num_images = row['num_images_presented']
    if pd.isna(num_images):
        return None
    
    num_images = int(num_images)
    if num_images == 1:
        return row.get('stimulus_index_enc1')
    elif num_images == 2:
        return row.get('stimulus_index_enc2')
    elif num_images == 3:
        return row.get('stimulus_index_enc3')
    else:
        return None

# Process each period with correct stimulus indices
for period_name, df in standardized_data.items():
    df['subject_id'] = df['subject_id'].astype(str).str.strip()
    df['trial_id'] = df['trial_id'].astype(int)

    if period_name == 'Encoding1':
        # For Encoding1, use stimulus_index_enc1
        df = pd.merge(df, trial_info_with_stimuli[['subject_id', 'trial_id', 'num_images_presented', 'stimulus_index_enc1']], 
                     on=['subject_id', 'trial_id'], how='left')
        df.rename(columns={'stimulus_index_enc1': 'stimulus_index'}, inplace=True)
        
    elif period_name == 'Encoding2':
        # For Encoding2, use stimulus_index_enc2
        df = pd.merge(df, trial_info_with_stimuli[['subject_id', 'trial_id', 'num_images_presented', 'stimulus_index_enc2']], 
                     on=['subject_id', 'trial_id'], how='left')
        df.rename(columns={'stimulus_index_enc2': 'stimulus_index'}, inplace=True)
        
    elif period_name == 'Encoding3':
        # For Encoding3, use stimulus_index_enc3
        df = pd.merge(df, trial_info_with_stimuli[['subject_id', 'trial_id', 'num_images_presented', 'stimulus_index_enc3']], 
                     on=['subject_id', 'trial_id'], how='left')
        df.rename(columns={'stimulus_index_enc3': 'stimulus_index'}, inplace=True)
        
    elif period_name == 'Delay':
        # For Delay, use the appropriate stimulus index based on num_images_presented
        df = pd.merge(df, trial_info_with_stimuli[['subject_id', 'trial_id', 'num_images_presented', 
                                                  'stimulus_index_enc1', 'stimulus_index_enc2', 'stimulus_index_enc3']], 
                     on=['subject_id', 'trial_id'], how='left')
        # Apply the function to determine correct stimulus index
        df['stimulus_index'] = df.apply(get_delay_stimulus_index, axis=1)
        
    elif period_name == 'Probe':
        # For Probe, include all stimulus indices for categorization
        df = pd.merge(df, trial_info_with_stimuli[['subject_id', 'trial_id', 'num_images_presented', 
                                                  'stimulus_index_enc1', 'stimulus_index_enc2', 'stimulus_index_enc3']], 
                     on=['subject_id', 'trial_id'], how='left')

    # Add other trial info columns
    df = pd.merge(df, trial_info_final[['subject_id', 'trial_id', 'new_trial_id', 'trial_id_final']], 
                 on=['subject_id', 'trial_id'], how='left')

    standardized_data[period_name] = df

print("Stimulus indices properly assigned to each period:")
for period_name, df in standardized_data.items():
    print(f"{period_name}: stimulus_index column {'exists' if 'stimulus_index' in df.columns else 'missing'}")
    if 'stimulus_index' in df.columns:
        print(f"  Sample values: {df['stimulus_index'].dropna().unique()[:5]}")

cleaned_data_dir = os.path.join(BASE_DIR, 'clean_data')
graph_data_dir = os.path.join(BASE_DIR, 'graph_data')
os.makedirs(cleaned_data_dir, exist_ok=True)
os.makedirs(graph_data_dir, exist_ok=True)

columns_to_remove = [
    'mean_1st', 'cat_1st',
    'im_cat_2nd', 'mean_2nd', 'cat_2nd',
    'p_val', 'CI', 'start_time_enc1'
]

def clean_standardized_data():
    for period_name, df in standardized_data.items():
        df['Neuron_ID_3'] = (
            df['subject_id'].astype(str) + '0' + df['Neuron_ID'].astype(str)
        ).astype(int)

        df_cleaned = df.drop(columns=columns_to_remove, errors='ignore')

        output_file = os.path.join(cleaned_data_dir, f'cleaned_{period_name}.xlsx')
        df_cleaned.to_excel(output_file, index=False)
        print(f"Cleaned data for {period_name} saved to: {output_file}")

clean_standardized_data()
print("All standardized data has been cleaned and saved in the clean_data folder.")

# Process fixation data similarly
fixation_file_path = os.path.join(BASE_DIR, 'all_spike_rate_data_fixation.xlsx')

def standardize_fixation_period(file_path, spikes_column, start_time_column):
    df = pd.read_excel(file_path)

    if start_time_column in df.columns and spikes_column in df.columns:
        def standardize_row_spikes(row):
            spikes = row[spikes_column]
            if isinstance(spikes, str):
                try:
                    spikes = ast.literal_eval(spikes)
                except:
                    spikes = []
            
            def flatten_spikes(spike_data):
                flat_list = []
                for item in spike_data:
                    if isinstance(item, list):
                        flat_list.extend(flatten_spikes(item))
                    else:
                        flat_list.append(item)
                return flat_list
            
            if isinstance(spikes, list):
                spikes = flatten_spikes(spikes)
            
            if isinstance(spikes, list) and not pd.isna(row[start_time_column]):
                try:
                    return [float(spike) - float(row[start_time_column]) for spike in spikes if spike is not None]
                except (TypeError, ValueError) as e:
                    print(f"Error processing spikes: {spikes}, error: {e}")
                    return []
            return []

        df['Standardized_Spikes'] = df.apply(standardize_row_spikes, axis=1)
        print("Fixation data standardized successfully.")
        return df
    else:
        print(f"Missing required columns in {file_path}. Skipping standardization.")
        return None

standardized_fixation_data = standardize_fixation_period(
    file_path=fixation_file_path,
    spikes_column='Spikes_in_Fixation',
    start_time_column='Fixation_Start'
)

if standardized_fixation_data is not None:
    standardized_fixation_data['subject_id'] = standardized_fixation_data['subject_id'].astype(str).str.strip()
    standardized_fixation_data['trial_id'] = standardized_fixation_data['trial_id'].astype(int)
    
    # Add trial info to fixation data
    standardized_fixation_data = pd.merge(
        standardized_fixation_data,
        trial_info_with_stimuli[['subject_id', 'trial_id', 'num_images_presented']],
        on=['subject_id', 'trial_id'],
        how='left'
    )
    
    standardized_fixation_data = pd.merge(
        standardized_fixation_data,
        trial_info_final[['subject_id', 'trial_id', 'new_trial_id', 'trial_id_final']],
        on=['subject_id', 'trial_id'],
        how='left'
    )

    standardized_fixation_data['Neuron_ID_3'] = (
        standardized_fixation_data['subject_id'].astype(str) + '0' + 
        standardized_fixation_data['Neuron_ID'].astype(str)
    ).astype(int)

    output_file = os.path.join(cleaned_data_dir, 'cleaned_Fixation.xlsx')
    standardized_fixation_data.to_excel(output_file, index=False)
    print(f"Cleaned fixation data saved to: {output_file}")

print("Fixation period cleaning and saving completed.")

def create_graph_data():
    for period_name, df in standardized_data.items():
        df_graph = df.copy()
        
        output_file = os.path.join(graph_data_dir, f'graph_{period_name.lower()}.xlsx')
        df_graph.to_excel(output_file, index=False)
        print(f"Graph data for {period_name} saved to: {output_file}")

create_graph_data()
print("Graph data creation completed.")

fixation_graph_data = pd.read_excel(os.path.join(cleaned_data_dir, 'cleaned_Fixation.xlsx'))
fixation_graph_output = os.path.join(graph_data_dir, 'graph_fixation.xlsx')
fixation_graph_data.to_excel(fixation_graph_output, index=False)
print(f"Fixation graph data saved to: {fixation_graph_output}")

# Final verification
print("\n=== FINAL VERIFICATION ===")
for period_name in ['Encoding1', 'Encoding2', 'Encoding3', 'Delay', 'Probe']:
    if period_name in standardized_data:
        df = standardized_data[period_name]
        print(f"\n{period_name}:")
        print(f"  - num_images_presented: {'PRESENT' if 'num_images_presented' in df.columns else 'MISSING'}")
        print(f"  - stimulus_index: {'PRESENT' if 'stimulus_index' in df.columns else 'MISSING'}")
        if period_name == 'Probe':
            print(f"  - stimulus_index_enc1: {'PRESENT' if 'stimulus_index_enc1' in df.columns else 'MISSING'}")
            print(f"  - stimulus_index_enc2: {'PRESENT' if 'stimulus_index_enc2' in df.columns else 'MISSING'}")
            print(f"  - stimulus_index_enc3: {'PRESENT' if 'stimulus_index_enc3' in df.columns else 'MISSING'}")
