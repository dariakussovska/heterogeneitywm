import numpy as np
import pandas as pd

# Load the data (replace these paths with the actual paths to your data files)
enc1_data = pd.read_excel('/home/daria/PROJECT/clean_data/cleaned_Encoding1.xlsx')
significant_neurons_df = pd.read_excel('/home/daria/PROJECT/Neuron_Check_Significant_All.xlsx')

# Filter significant neurons to only include Y or N (exclude NaN or other values)
significant_neurons_filtered = significant_neurons_df[significant_neurons_df['Signi'].isin(['Y', 'N'])]

# Display the standardized spike times for Enc1
print(enc1_data[['Neuron_ID_3', 'trial_id', 'start_time', 'Spikes', 'Standardized_Spikes']].head())

# Add a column for the preferred image ID to the Enc1 data
def add_preferred_image_id(enc1_df, significant_neurons_df):
    # Merge Enc1 data with significant neurons data on 'subject_id' and 'Neuron_ID'
    merged_df = pd.merge(enc1_df, significant_neurons_df[['subject_id', 'Neuron_ID', 'im_cat_1st', 'Signi']], on=['subject_id', 'Neuron_ID'], how='inner')
    # Rename the 'im_cat_1st' column to 'preferred_image_id'
    merged_df.rename(columns={'im_cat_1st': 'preferred_image_id'}, inplace=True)
    return merged_df

# Apply the function to the Enc1 data using filtered significant neurons
enc1_data_with_preferred = add_preferred_image_id(enc1_data, significant_neurons_filtered)

# Display the Enc1 data with the preferred image ID
print(enc1_data_with_preferred[['subject_id', 'Neuron_ID', 'trial_id', 'preferred_image_id', 'Standardized_Spikes', 'Signi']].head())

# Load the trial information data
trial_info = pd.read_excel('/home/daria/PROJECT/trial_info.xlsx')

# Filter Enc1 data by single image trials using subject_id and trial_id
enc1_data_filtered = pd.merge(enc1_data_with_preferred, trial_info, on=['subject_id', 'trial_id'], how='inner')

# Display the filtered Enc1 data
print(enc1_data_filtered[['subject_id', 'Neuron_ID', 'trial_id', 'preferred_image_id', 'num_images_presented', 'Standardized_Spikes', 'Signi']].head())

# Define the file path where the filtered Enc1 data will be saved
filtered_enc1_file_path = '/home/daria/PROJECT/graph_data/graph_Encoding1.xlsx'

# Save the filtered Enc1 data to an Excel file
enc1_data_filtered.to_excel(filtered_enc1_file_path, index=False)

# Confirm the data has been saved
print(f"Filtered Enc1 data saved to {filtered_enc1_file_path}")

# Function to add a column indicating preferred or non-preferred trials
def categorize_trials_by_preference(df):
    df['Category'] = df.apply(lambda row: 'Preferred' if row['preferred_image_id'] == row['stimulus_index'] else 'Non-Preferred', axis=1)
    return df

# Apply the function to the Enc1 data
enc1_data_categorized = categorize_trials_by_preference(enc1_data_filtered)

# Display the categorized Enc1 data
print(enc1_data_categorized[['subject_id', 'Neuron_ID', 'trial_id', 'preferred_image_id', 'stimulus_index', 'Category', 'Signi']].head())

enc1_data_categorized.to_excel(filtered_enc1_file_path, index=False)

# Confirm the data has been saved
print(f"Categorized Enc1 data saved to {filtered_enc1_file_path}")

categorized_enc1_file_path = '/home/daria/PROJECT/graph_data/graph_encoding1.xlsx'

enc1_data_categorized = pd.read_excel(categorized_enc1_file_path)

# Load the significant neurons data
significant_neurons = pd.read_excel('/home/daria/PROJECT/all_neuron_brain_regions_cleaned.xlsx')

# Function to add brain region information
def add_brain_region_info(enc1_df, significant_neurons_df):
    # Merge Enc1 data with significant neurons data on 'subject_id' and 'Neuron_ID'
    merged_df = pd.merge(enc1_df, significant_neurons_df[['subject_id', 'Neuron_ID', 'Location']], on=['subject_id', 'Neuron_ID'], how='inner')
    return merged_df

# Apply the function to the categorized Enc1 data
enc1_data_with_brain_region = add_brain_region_info(enc1_data_categorized, significant_neurons)

# Display the Enc1 data with the brain region information
print(enc1_data_with_brain_region[['subject_id', 'Neuron_ID', 'trial_id', 'preferred_image_id', 'stimulus_index', 'Category', 'Location', 'Signi']].head())

# Define the file path where the updated Enc1 data will be saved
updated_enc1_file_path = '/home/daria/PROJECT/graph_data/graph_encoding1.xlsx'

# Save the updated Enc1 data to an Excel file
enc1_data_with_brain_region.to_excel(updated_enc1_file_path, index=False)

# Confirm the data has been saved
print(f"Updated Enc1 data with brain region saved to {updated_enc1_file_path}")

import pandas as pd

# Load the Enc1 and delay data with brain region from the Excel file
enc1_data_with_brain_region = pd.read_excel('/home/daria/PROJECT/graph_data/graph_encoding1.xlsx')
delay_data = pd.read_excel('/home/daria/PROJECT/graph_data/graph_delay.xlsx')

# Standardize spike times by subtracting the start time of the delay period
def standardize_spike_times_delay(df):
    df['Standardized_Spikes'] = df.apply(
        lambda row: [spike - row['Delay_Start'] for spike in eval(row['Spikes_in_Delay'])] if pd.notna(row['Spikes_in_Delay']) else [],
        axis=1
    )
    return df

delay_data = standardize_spike_times_delay(delay_data)

# Select only the specified columns from enc1 data
columns_to_keep_from_enc1 = [
    'subject_id', 'Neuron_ID', 'trial_id', 'Significance', 'new_trial_id',
    'Neuron_ID_3', 'preferred_image_id', 'image_id_enc1', 'stimulus_index_enc1',
    'image_id_enc2', 'stimulus_index_enc2', 'image_id_enc3', 'stimulus_index_enc3',
    'num_images_presented', 'Category', 'Location', 'Signi'  # Added Signi column
]

enc1_data_filtered = enc1_data_with_brain_region[columns_to_keep_from_enc1]

# Merge delay data with the filtered Enc1 data
merged_data = pd.merge(
    delay_data,
    enc1_data_filtered,
    on=['subject_id', 'Neuron_ID', 'trial_id'],
    how='inner'
)

# Save the merged data
merged_data.to_excel('/home/daria/PROJECT/graph_data/graph_delay.xlsx', index=False)

print("Merged data saved")

import pandas as pd

probe_data = pd.read_excel('/home/daria/PROJECT/graph_data/all_spike_rate_data_probe.xlsx')

# Standardize spike times by subtracting the start time of the probe period
def standardize_spike_times_probe(df):
    df['Standardized_Spikes'] = df.apply(
        lambda row: [spike - row['Probe_Start_Time'] for spike in eval(row['Spikes_in_Probe'])] if pd.notna(row['Spikes_in_Probe']) else [],
        axis=1
    )
    return df

probe_data = standardize_spike_times_probe(probe_data)

# Select only the specified columns from Enc1 data
columns_to_keep_from_enc1 = [
    'subject_id', 'Neuron_ID', 'trial_id', 'Significance', 'new_trial_id',
    'Neuron_ID_3', 'preferred_image_id', 'image_id_enc1', 'stimulus_index_enc1',
    'image_id_enc2', 'stimulus_index_enc2', 'image_id_enc3', 'stimulus_index_enc3',
    'num_images_presented', 'Category', 'Location', 'Signi'  # Added Signi column
]

enc1_data_filtered = enc1_data_with_brain_region[columns_to_keep_from_enc1]

# Merge probe data with the filtered Enc1 data
merged_data_probe = pd.merge(
    probe_data,
    enc1_data_filtered,
    on=['subject_id', 'Neuron_ID', 'trial_id'],
    how='inner'
)

# Save the merged data
merged_data_probe.to_excel('/home/daria/PROJECT/graph_data/graph_probe.xlsx', index=False)

print("Merged data saved")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Add a new column for categorizing trials based on the preferred vs non-preferred images in Enc1 and Probe periods
def categorize_trial(row):
    if row['Category'] == 'Preferred' and row['Probe_Image_ID'] == row['preferred_image_id']:
        return 'Preferred Encoded'
    elif row['Category'] == 'Preferred' and row['Probe_Image_ID'] != row['preferred_image_id']:
        return 'Preferred Nonencoded'
    elif row['Category'] == 'Non-Preferred' and row['Probe_Image_ID'] == row['preferred_image_id']:
        return 'Nonpreferred Encoded'
    else:
        return 'Nonpreferred Nonencoded'

merged_data_probe['Trial_Type'] = merged_data_probe.apply(categorize_trial, axis=1)

# Save the updated data
merged_data_probe.to_excel('/home/daria/PROJECT/graph_data/graph_probe.xlsx', index=False)

import numpy as np
import pandas as pd

# Load the data (replace these paths with the actual paths to your data files)
fixation_data = pd.read_excel('/home/daria/PROJECT/graph_data/graph_fixation.xlsx')
significant_neurons = pd.read_excel('/home/daria/PROJECT/all_neuron_brain_regions_cleaned.xlsx')
trial_info = pd.read_excel('/home/daria/PROJECT/trial_info.xlsx')

# Function to standardize spike times by subtracting the start time of the period
def standardize_spike_times_fixation(df):
    df['Standardized_Spikes_in_Fixation'] = df.apply(lambda row: [spike - row['Fixation_Start'] for spike in eval(row['Spikes_in_Fixation'])] if pd.notna(row['Spikes_in_Fixation']) else [], axis=1)
    return df

# Apply the function to the fixation data
fixation_data = standardize_spike_times_fixation(fixation_data)

# Display the standardized spike times for fixation
print(fixation_data[['Neuron_ID', 'trial_id', 'Fixation_Start', 'Spikes_in_Fixation', 'Standardized_Spikes_in_Fixation']].head())

# Merge fixation data with significant neurons and trial information
filtered_fixation_data = pd.merge(fixation_data, significant_neurons[['subject_id', 'Neuron_ID', 'Location']], on=['subject_id', 'Neuron_ID'], how='inner')
filtered_fixation_data = pd.merge(filtered_fixation_data, trial_info, on=['subject_id', 'trial_id'], how='inner')

# Display the filtered fixation data
print(filtered_fixation_data.head())

# Save the filtered fixation data to a new Excel file
filtered_fixation_data.to_excel('/home/daria/PROJECT/graph_data/graph_fixation.xlsx', index=False)

import pandas as pd
import numpy as np

# Load the data
merged_data_probe = pd.read_excel('/home/daria/PROJECT/graph_data/graph_probe.xlsx')

def categorize_probe(row):
    preferred = row['preferred_image_id']
    probe = row['Probe_Image_ID']
    num_images = row['num_images_presented']
    
    enc1 = row.get('stimulus_index_enc1')
    enc2 = row.get('stimulus_index_enc2')
    enc3 = row.get('stimulus_index_enc3')

    # Build encoding image list, excluding '5' (means no image shown)
    encoded_images = [enc for enc in [enc1, enc2, enc3] if pd.notna(enc) and enc != 5]

    in_encoding = preferred in encoded_images
    in_probe = probe == preferred

    if num_images == 1:
        if enc1 == 5:
            return 'Unknown'
        if row['Category'] == 'Preferred' and probe == preferred and enc1 == preferred:
            return 'Preferred Encoded'
        elif row['Category'] == 'Preferred' and probe != preferred:
            return 'Preferred Nonencoded'
        elif row['Category'] == 'Non-Preferred' and probe == preferred:
            return 'Nonpreferred Encoded'
        else:
            return 'Nonpreferred Nonencoded'

    elif num_images in [2, 3]:
        if in_encoding and in_probe:
            return 'Preferred Encoded'
        elif in_encoding and not in_probe:
            return 'Preferred Nonencoded'
        elif not in_encoding and in_probe:
            return 'Nonpreferred Encoded'
        else:
            return 'Nonpreferred Nonencoded'
    
    return 'Unknown'

merged_data_probe['Category_Probe'] = merged_data_probe.apply(categorize_probe, axis=1)
merged_data_probe.to_excel('/home/daria/PROJECT/graph_data/graph_probe.xlsx', index=False)

print("All processing complete! Significant neurons (Signi = Y or N) have been included in all datasets.")
