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
            
            # Debug: print the type and first few elements
            if isinstance(spikes, str):
                try:
                    spikes = ast.literal_eval(spikes)
                except:
                    spikes = []
            
            # Handle nested lists by flattening
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

# Test with one file first to see the actual data structure
print("Testing with Encoding1 file...")
test_df = pd.read_excel(filtered_files['Encoding1'])
print("Sample spike data:")
print(test_df['Spikes'].head(3))
print("Data types:")
print(test_df['Spikes'].apply(type).value_counts())

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

trial_info_new = pd.read_excel(os.path.join(BASE_DIR, 'new_trial_info.xlsx'))
trial_info_final = pd.read_excel(os.path.join(BASE_DIR, 'new_trial_final.xlsx'))

trial_info_new['subject_id'] = trial_info_new['subject_id'].astype(str).str.strip()
trial_info_new['trial_id'] = trial_info_new['trial_id'].astype(int)
trial_info_final['subject_id'] = trial_info_final['subject_id'].astype(str).str.strip()
trial_info_final['trial_id'] = trial_info_final['trial_id'].astype(int)

for period_name, df in standardized_data.items():
    df['subject_id'] = df['subject_id'].astype(str).str.strip()
    df['trial_id'] = df['trial_id'].astype(int)

    df = pd.merge(
        df,
        trial_info_new[['subject_id', 'trial_id', 'new_trial_id']],
        on=['subject_id', 'trial_id'],
        how='left'  
    )
    
    df = pd.merge(
        df,
        trial_info_final[['subject_id', 'trial_id', 'trial_id_final']],
        on=['subject_id', 'trial_id'],
        how='left'  
    )

    standardized_data[period_name] = df

print("new_trial_id and trial_id_final columns added to all standardized DataFrames.")

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
            
            # Handle nested lists by flattening
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
    
    standardized_fixation_data = pd.merge(
        standardized_fixation_data,
        trial_info_new[['subject_id', 'trial_id', 'new_trial_id']],
        on=['subject_id', 'trial_id'],
        how='left'
    )
    
    standardized_fixation_data = pd.merge(
        standardized_fixation_data,
        trial_info_final[['subject_id', 'trial_id', 'trial_id_final']],
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
    # First load and prepare trial info for graph data
    trial_info_new_graph = pd.read_excel(os.path.join(BASE_DIR, 'new_trial_info.xlsx'))
    trial_info_final_graph = pd.read_excel(os.path.join(BASE_DIR, 'new_trial_final.xlsx'))
    
    trial_info_new_graph['subject_id'] = trial_info_new_graph['subject_id'].astype(str).str.strip()
    trial_info_new_graph['trial_id'] = trial_info_new_graph['trial_id'].astype(int)
    trial_info_final_graph['subject_id'] = trial_info_final_graph['subject_id'].astype(str).str.strip()
    trial_info_final_graph['trial_id'] = trial_info_final_graph['trial_id'].astype(int)
    
    for period_name, file_path in filtered_files.items():
        # Load original data for this period
        df_graph = pd.read_excel(file_path)
        
        # Clean and prepare the data
        df_graph['subject_id'] = df_graph['subject_id'].astype(str).str.strip()
        df_graph['trial_id'] = df_graph['trial_id'].astype(int)
        
        # Merge trial info (same as cleaned data)
        df_graph = pd.merge(
            df_graph,
            trial_info_new_graph[['subject_id', 'trial_id', 'new_trial_id']],
            on=['subject_id', 'trial_id'],
            how='left'
        )
        
        df_graph = pd.merge(
            df_graph,
            trial_info_final_graph[['subject_id', 'trial_id', 'trial_id_final']],
            on=['subject_id', 'trial_id'],
            how='left'
        )
        
        # Add Neuron_ID_3 (same as cleaned data)
        df_graph['Neuron_ID_3'] = (
            df_graph['subject_id'].astype(str) + '0' + df_graph['Neuron_ID'].astype(str)
        ).astype(int)
        
        # Remove unnecessary columns (same as cleaned data)
        df_graph = df_graph.drop(columns=columns_to_remove, errors='ignore')
        
        # Determine the spikes column and start time column for this period
        if period_name == 'Encoding1':
            spikes_col = 'Spikes'
            start_time_col = 'start_time'
        elif period_name == 'Encoding2':
            spikes_col = 'Spikes'
            start_time_col = 'start_time'  # Adjust based on actual column name
        elif period_name == 'Encoding3':
            spikes_col = 'Spikes'
            start_time_col = 'start_time'  # Adjust based on actual column name
        elif period_name == 'Delay':
            spikes_col = 'Spikes_in_Delay'
            start_time_col = 'start_time'  # Adjust based on actual column name
        elif period_name == 'Probe':
            spikes_col = 'Spikes_in_Probe'
            start_time_col = 'start_time'  # Adjust based on actual column name
        
        # Standardize from this period's own start time (DIFFERENT from cleaned data)
        if start_time_col in df_graph.columns and spikes_col in df_graph.columns:
            def standardize_own_epoch_spikes(row):
                spikes = row[spikes_col]
                
                if isinstance(spikes, str):
                    try:
                        spikes = ast.literal_eval(spikes)
                    except:
                        spikes = []
                
                # Handle nested lists by flattening
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
                
                if isinstance(spikes, list) and not pd.isna(row[start_time_col]):
                    try:
                        return [float(spike) - float(row[start_time_col]) for spike in spikes if spike is not None]
                    except (TypeError, ValueError) as e:
                        print(f"Error processing spikes: {spikes}, error: {e}")
                        return []
                return []

            df_graph['Standardized_Spikes'] = df_graph.apply(standardize_own_epoch_spikes, axis=1)
            print(f"Graph data for {period_name} standardized from its own start time ({start_time_col})")
        else:
            print(f"Warning: Missing columns {start_time_col} or {spikes_col} for {period_name}")
            # If columns missing, use the cleaned data but this shouldn't happen
            df_graph = standardized_data[period_name].copy()
        
        output_file = os.path.join(graph_data_dir, f'graph_{period_name.lower()}.xlsx')
        df_graph.to_excel(output_file, index=False)
        print(f"Graph data for {period_name} saved to: {output_file}")

create_graph_data()
print("Graph data creation completed.")

# For fixation graph data (already uses its own start time)
fixation_graph_data = pd.read_excel(os.path.join(cleaned_data_dir, 'cleaned_Fixation.xlsx'))
fixation_graph_output = os.path.join(graph_data_dir, 'graph_fixation.xlsx')
fixation_graph_data.to_excel(fixation_graph_output, index=False)
print(f"Fixation graph data saved to: {fixation_graph_output}")

print("All processing completed!")
print("Cleaned data: Standardized from Encoding1 start time")
print("Graph data: Standardized from each period's own start time")
print("Both have trial info, Neuron_ID_3, and all necessary columns")
