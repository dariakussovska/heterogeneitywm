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
        def standardize_spikes(row):
            spikes = ast.literal_eval(row[spikes_column]) if isinstance(row[spikes_column], str) else row[spikes_column]
            if isinstance(spikes, list) and not pd.isna(row['start_time_enc1']):
                return [spike - row['start_time_enc1'] for spike in spikes]
            return []

        df['Standardized_Spikes'] = df.apply(standardize_spikes, axis=1)
        return df
    else:
        print(f"Missing required columns in {file_path}. Skipping standardization.")
        return None

standardized_data = {}
for period_name, file_path in filtered_files.items():
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

significant_neurons = pd.read_excel(os.path.join(BASE_DIR, 'Neuron_Check_Significant_All.xlsx'))
significant_neurons.rename(columns={'Signi': 'Significance'}, inplace=True)

significant_neurons['subject_id'] = significant_neurons['subject_id'].astype(str).str.strip()
significant_neurons['Neuron_ID'] = significant_neurons['Neuron_ID'].astype(int)

for period_name, df in standardized_data.items():
    df['subject_id'] = df['subject_id'].astype(str).str.strip()
    df['Neuron_ID'] = df['Neuron_ID'].astype(int)

    df = pd.merge(
        df,
        significant_neurons[['subject_id', 'Neuron_ID', 'Significance']],
        on=['subject_id', 'Neuron_ID'],
        how='left'  
    )

    df['Significance'] = df['Significance'].fillna('N')
    standardized_data[period_name] = df

print("Significance column added to all standardized DataFrames.")

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
trial_info_path = os.path.join(BASE_DIR, 'trial_info copy.xlsx')

def standardize_fixation_period(file_path, spikes_column, start_time_column):
    df = pd.read_excel(file_path)

    if start_time_column in df.columns and spikes_column in df.columns:
        def standardize_spikes(row):
            spikes = ast.literal_eval(row[spikes_column]) if isinstance(row[spikes_column], str) else row[spikes_column]
            if isinstance(spikes, list) and not pd.isna(row[start_time_column]):
                return [spike - row[start_time_column] for spike in spikes]
            return []

        df['Standardized_Spikes'] = df.apply(standardize_spikes, axis=1)
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

trial_info = pd.read_excel(trial_info_path)
trial_info['subject_id'] = trial_info['subject_id'].astype(str).str.strip()
trial_info['trial_id'] = trial_info['trial_id'].astype(int)

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

significant_neurons_extended = pd.read_excel(os.path.join(BASE_DIR, 'merged_significant_neurons_with_brain_regions.xlsx'))

def create_graph_data():
    for period_name, df in standardized_data.items():
        df_graph = df.copy()
        
        df_graph = pd.merge(
            df_graph,
            significant_neurons_extended[['subject_id', 'Neuron_ID', 'im_cat_1st', 'Location']],
            on=['subject_id', 'Neuron_ID'],
            how='left'
        )
        
        df_graph.rename(columns={'im_cat_1st': 'preferred_image_id'}, inplace=True)
        
        output_file = os.path.join(graph_data_dir, f'graph_{period_name.lower()}.xlsx')
        df_graph.to_excel(output_file, index=False)
        print(f"Graph data for {period_name} saved to: {output_file}")

create_graph_data()
print("Graph data creation completed.")

fixation_graph_data = pd.read_excel(os.path.join(cleaned_data_dir, 'cleaned_Fixation.xlsx'))
fixation_graph_data = pd.merge(
    fixation_graph_data,
    significant_neurons_extended[['subject_id', 'Neuron_ID', 'Location']],
    on=['subject_id', 'Neuron_ID'],
    how='inner'
)

fixation_graph_output = os.path.join(graph_data_dir, 'graph_fixation.xlsx')
fixation_graph_data.to_excel(fixation_graph_output, index=False)
print(f"Fixation graph data saved to: {fixation_graph_output}")
