import numpy as np
import pandas as pd

# Load the data
enc1_data = pd.read_excel('/home/daria/PROJECT/clean_data/cleaned_Encoding1.xlsx')
significant_neurons_df = pd.read_excel('/home/daria/PROJECT/Neuron_Check_Significant_All.xlsx')

# Filter significant neurons to only include Y or N
significant_neurons_filtered = significant_neurons_df[significant_neurons_df['Signi'].isin(['Y', 'N'])]

# SIMPLIFIED APPROACH: Add all necessary columns in one merge
def add_significance_data(df, sig_neurons_df):
    """Add Signi, preferred_image_id, and other columns in one merge"""
    
    # Check what key columns are available
    if 'Neuron_ID_3' in df.columns and 'Neuron_ID_3' in sig_neurons_df.columns:
        # Merge on Neuron_ID_3
        columns_to_merge = ['Neuron_ID_3', 'im_cat_1st', 'Signi']
        # Add subject_id if available in both
        if 'subject_id' in df.columns and 'subject_id' in sig_neurons_df.columns:
            columns_to_merge = ['subject_id', 'Neuron_ID_3', 'im_cat_1st', 'Signi']
            
        merged_df = pd.merge(df, sig_neurons_df[columns_to_merge], 
                            on=[col for col in columns_to_merge if col != 'im_cat_1st' and col != 'Signi'], 
                            how='inner')
    
    elif 'subject_id' in df.columns and 'Neuron_ID' in df.columns:
        # Merge on subject_id and Neuron_ID
        merged_df = pd.merge(df, sig_neurons_df[['subject_id', 'Neuron_ID', 'im_cat_1st', 'Signi']], 
                            on=['subject_id', 'Neuron_ID'], how='inner')
    else:
        print("Warning: Could not find appropriate key columns for merge")
        return df
    
    # Rename im_cat_1st to preferred_image_id
    if 'im_cat_1st' in merged_df.columns:
        merged_df.rename(columns={'im_cat_1st': 'preferred_image_id'}, inplace=True)
    
    return merged_df

# Apply the function to add all significance data at once
enc1_data_with_significance = add_significance_data(enc1_data, significant_neurons_filtered)

# Load the trial information data
trial_info = pd.read_excel('/home/daria/PROJECT/trial_info.xlsx')

# Add significance data to trial_info as well
trial_info_with_significance = add_significance_data(trial_info, significant_neurons_filtered)

# Merge with trial info
enc1_data_merged = pd.merge(enc1_data_with_significance, trial_info_with_significance[['subject_id', 'trial_id', 'num_images_presented']], 
                           on=['subject_id', 'trial_id'], how='inner')

# Function to add a column indicating preferred or non-preferred trials
def categorize_trials_by_preference(df):
    df['Category'] = df.apply(lambda row: 'Preferred' if row['preferred_image_id'] == row['stimulus_index'] else 'Non-Preferred', axis=1)
    return df

# Apply the function to the Enc1 data
enc1_data_categorized = categorize_trials_by_preference(enc1_data_merged)

# Load brain regions data and add significance
significant_neurons_brain = pd.read_excel('/home/daria/PROJECT/all_neuron_brain_regions_cleaned.xlsx')
significant_neurons_brain_with_signi = add_significance_data(significant_neurons_brain, significant_neurons_filtered)

# Function to add brain region information
def add_brain_region_info(enc1_df, brain_regions_df):
    # Merge with brain regions data
    if 'Neuron_ID_3' in enc1_df.columns and 'Neuron_ID_3' in brain_regions_df.columns:
        merged_df = pd.merge(enc1_df, brain_regions_df[['Neuron_ID_3', 'Location', 'Signi']], 
                            on='Neuron_ID_3', how='inner', suffixes=('', '_brain'))
    elif 'subject_id' in enc1_df.columns and 'Neuron_ID' in enc1_df.columns:
        merged_df = pd.merge(enc1_df, brain_regions_df[['subject_id', 'Neuron_ID', 'Location', 'Signi']], 
                            on=['subject_id', 'Neuron_ID'], how='inner', suffixes=('', '_brain'))
    else:
        print("Warning: Could not merge brain regions")
        return enc1_df
    
    return merged_df

# Add brain region information
enc1_data_with_brain_region = add_brain_region_info(enc1_data_categorized, significant_neurons_brain_with_signi)

# Save final Enc1 data
final_enc1_file_path = '/home/daria/PROJECT/graph_data/graph_encoding1.xlsx'
enc1_data_with_brain_region.to_excel(final_enc1_file_path, index=False)
print(f"Final Enc1 data saved to {final_enc1_file_path}")

# PROBE CATEGORIZATION FUNCTION - ADJUSTED FOR YOUR COLUMN NAMES
def categorize_probe(row):
    """Categorize probe trials based on your logic"""
    preferred = row['preferred_image_id']
    probe = row['Probe_Image_ID']
    num_images = row['num_images_presented']
    
    # Get stimulus indices for each encoding period - using the column names from your probe data
    enc1 = row.get('stimulus_index_enc1', np.nan)
    enc2 = row.get('stimulus_index_enc2', np.nan) 
    enc3 = row.get('stimulus_index_enc3', np.nan)

    # If we don't have the encoding period columns, use the general stimulus_index
    if pd.isna(enc1) and 'stimulus_index' in row:
        enc1 = row['stimulus_index']

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

# Now process probe data specifically with the correct column handling
def process_probe_data(probe_file_path, enc1_reference_data, significant_neurons_df, output_path):
    """Process probe data with proper column handling"""
    try:
        # Load probe data
        probe_data = pd.read_excel(probe_file_path)
        print(f"Processing probe data from {probe_file_path}")
        print(f"Probe data columns: {probe_data.columns.tolist()}")
        
        # Add significance data to probe data
        probe_data_with_signi = add_significance_data(probe_data, significant_neurons_df)
        
        # Standardize probe spikes if not already done
        if 'Standardized_Spikes' not in probe_data_with_signi.columns and 'Spikes_in_Probe' in probe_data_with_signi.columns:
            probe_data_with_signi['Standardized_Spikes'] = probe_data_with_signi.apply(
                lambda row: [spike - row['Probe_Start_Time'] for spike in eval(row['Spikes_in_Probe'])] if pd.notna(row['Spikes_in_Probe']) else [],
                axis=1
            )
        
        # Merge with enc1 reference data to get encoding period information
        merge_keys = ['subject_id', 'Neuron_ID', 'trial_id']
        common_keys = [key for key in merge_keys if key in probe_data_with_signi.columns and key in enc1_reference_data.columns]
        
        if common_keys:
            # Get the specific columns we need from enc1 data for probe categorization
            enc1_columns_needed = [
                'subject_id', 'Neuron_ID', 'trial_id', 'preferred_image_id', 
                'stimulus_index_enc1', 'stimulus_index_enc2', 'stimulus_index_enc3',
                'num_images_presented', 'Category', 'Location', 'Signi'
            ]
            
            # Only take columns that actually exist in enc1_reference_data
            enc1_columns_available = [col for col in enc1_columns_needed if col in enc1_reference_data.columns]
            
            probe_data_final = pd.merge(
                probe_data_with_signi, 
                enc1_reference_data[enc1_columns_available], 
                on=common_keys, 
                how='inner', 
                suffixes=('', '_enc1')
            )
            
            print("After merge, probe data columns:", probe_data_final.columns.tolist())
            
            # Add probe categorization
            print("Adding probe categorization...")
            probe_data_final['Probe_Category'] = probe_data_final.apply(categorize_probe, axis=1)
            print("Probe categories added:", probe_data_final['Probe_Category'].value_counts())
            
        else:
            probe_data_final = probe_data_with_signi
            print("Warning: Could not merge with enc1 data - probe categorization may not work properly")
        
        # Save the processed probe data
        probe_data_final.to_excel(output_path, index=False)
        print(f"Processed probe data saved to {output_path}")
        
        return probe_data_final
        
    except Exception as e:
        print(f"Error processing probe data: {e}")
        import traceback
        traceback.print_exc()
        return None

# Process other data types (delay, fixation) - simplified version
def process_other_data(file_path, enc1_reference_data, significant_neurons_df, output_suffix="_processed"):
    """Process delay, fixation data"""
    try:
        data = pd.read_excel(file_path)
        print(f"Processing {file_path}")
        
        # Add significance data
        data_with_signi = add_significance_data(data, significant_neurons_df)
        
        # Standardize spikes based on file type
        if 'Spikes_in_Delay' in data_with_signi.columns:
            data_with_signi['Standardized_Spikes'] = data_with_signi.apply(
                lambda row: [spike - row['Delay_Start'] for spike in eval(row['Spikes_in_Delay'])] if pd.notna(row['Spikes_in_Delay']) else [],
                axis=1
            )
        elif 'Spikes_in_Fixation' in data_with_signi.columns:
            data_with_signi['Standardized_Spikes'] = data_with_signi.apply(
                lambda row: [spike - row['Fixation_Start'] for spike in eval(row['Spikes_in_Fixation'])] if pd.notna(row['Spikes_in_Fixation']) else [],
                axis=1
            )
        
        # Merge with enc1 reference data
        merge_keys = ['subject_id', 'Neuron_ID', 'trial_id']
        common_keys = [key for key in merge_keys if key in data_with_signi.columns and key in enc1_reference_data.columns]
        
        if common_keys:
            data_final = pd.merge(data_with_signi, enc1_reference_data, on=common_keys, how='inner', suffixes=('', '_enc1'))
        else:
            data_final = data_with_signi
        
        # Save the processed data
        output_path = file_path.replace('.xlsx', f'{output_suffix}.xlsx')
        data_final.to_excel(output_path, index=False)
        print(f"Processed data saved to {output_path}")
        
        return data_final
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Process all files
enc1_reference = pd.read_excel(final_enc1_file_path)

# Process probe data with special handling
probe_data = process_probe_data(
    '/home/daria/PROJECT/all_spike_rate_data_probe.xlsx', 
    enc1_reference, 
    significant_neurons_filtered,
    '/home/daria/PROJECT/graph_data/graph_probe.xlsx'
)

# Process delay data
delay_data = process_other_data('/home/daria/PROJECT/graph_data/graph_delay.xlsx', enc1_reference, significant_neurons_filtered)

# Process fixation data
fixation_data = process_other_data('/home/daria/PROJECT/graph_data/graph_fixation.xlsx', enc1_reference, significant_neurons_filtered)

print("All data processing complete!")
