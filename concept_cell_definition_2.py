import numpy as np
import pandas as pd

# Load the data
enc1_data = pd.read_excel('/home/daria/PROJECT/clean_data/cleaned_Encoding1.xlsx')
significant_neurons_df = pd.read_excel('/home/daria/PROJECT/Neuron_Check_Significant_All.xlsx')

# Filter significant neurons to only include Y or N
significant_neurons_filtered = significant_neurons_df[significant_neurons_df['Signi'].isin(['Y', 'N'])]

print("Enc1 data columns:", enc1_data.columns.tolist())
print("Significant neurons columns:", significant_neurons_filtered.columns.tolist())

# STEP 1: Add Signi column to enc1_data based on subject_id and Neuron_ID
def add_signi_column(df, sig_neurons_df):
    """Add Signi column to dataframe"""
    # Check what key columns are available
    if 'subject_id' in df.columns and 'Neuron_ID' in df.columns:
        # Merge to add Signi column
        result = pd.merge(df, sig_neurons_df[['subject_id', 'Neuron_ID', 'Signi']], 
                         on=['subject_id', 'Neuron_ID'], how='inner')
        return result
    else:
        print("Cannot add Signi - missing subject_id or Neuron_ID")
        return df

# Add Signi column to enc1_data
enc1_data_with_signi = add_signi_column(enc1_data, significant_neurons_filtered)
print("After adding Signi:", enc1_data_with_signi.columns.tolist())

# STEP 2: Add preferred_image_id (im_cat_1st)
def add_preferred_image(df, sig_neurons_df):
    """Add preferred_image_id column"""
    if 'subject_id' in df.columns and 'Neuron_ID' in df.columns:
        result = pd.merge(df, sig_neurons_df[['subject_id', 'Neuron_ID', 'im_cat_1st']], 
                         on=['subject_id', 'Neuron_ID'], how='inner')
        result.rename(columns={'im_cat_1st': 'preferred_image_id'}, inplace=True)
        return result
    else:
        return df

# Add preferred_image_id
enc1_data_with_both = add_preferred_image(enc1_data_with_signi, significant_neurons_filtered)
print("After adding preferred_image_id:", enc1_data_with_both.columns.tolist())

# STEP 3: Add trial info (num_images_presented)
trial_info = pd.read_excel('/home/daria/PROJECT/trial_info.xlsx')
enc1_data_merged = pd.merge(enc1_data_with_both, trial_info[['subject_id', 'trial_id', 'num_images_presented']], 
                           on=['subject_id', 'trial_id'], how='inner')

# STEP 4: Add Category column
def categorize_trials_by_preference(df):
    df['Category'] = df.apply(lambda row: 'Preferred' if row['preferred_image_id'] == row['stimulus_index'] else 'Non-Preferred', axis=1)
    return df

enc1_data_categorized = categorize_trials_by_preference(enc1_data_merged)

# STEP 5: Add brain regions (without expecting Signi column there)
def add_brain_region(df, brain_regions_path):
    """Add brain region information"""
    brain_regions = pd.read_excel(brain_regions_path)
    
    # Check available keys
    if 'Neuron_ID_3' in df.columns and 'Neuron_ID_3' in brain_regions.columns:
        result = pd.merge(df, brain_regions[['Neuron_ID_3', 'Location']], 
                         on='Neuron_ID_3', how='inner')
    elif 'subject_id' in df.columns and 'Neuron_ID' in df.columns:
        result = pd.merge(df, brain_regions[['subject_id', 'Neuron_ID', 'Location']], 
                         on=['subject_id', 'Neuron_ID'], how='inner')
    else:
        print("Cannot add brain regions - no common keys")
        return df
    return result

# Add brain regions
enc1_data_final = add_brain_region(enc1_data_categorized, '/home/daria/PROJECT/all_neuron_brain_regions_cleaned.xlsx')

print("Final enc1 data columns:", enc1_data_final.columns.tolist())
print(enc1_data_final[['subject_id', 'Neuron_ID', 'trial_id', 'preferred_image_id', 'Category', 'Signi', 'Location']].head())

# Save the final enc1 data
enc1_data_final.to_excel('/home/daria/PROJECT/graph_data/graph_encoding1.xlsx', index=False)
print("Enc1 data saved!")

# STEP 6: Process probe data with categorization
def process_probe_data():
    """Process probe data with proper categorization"""
    # Load probe data
    probe_data = pd.read_excel('/home/daria/PROJECT/all_spike_rate_data_probe.xlsx')
    print("Probe data columns:", probe_data.columns.tolist())
    
    # STEP 1: Add Signi to probe data
    probe_with_signi = add_signi_column(probe_data, significant_neurons_filtered)
    
    # STEP 2: Add preferred_image_id to probe data
    probe_with_preferred = add_preferred_image(probe_with_signi, significant_neurons_filtered)
    
    # STEP 3: Merge with enc1 data to get encoding period info and Category
    probe_merged = pd.merge(probe_with_preferred, 
                           enc1_data_final[['subject_id', 'Neuron_ID', 'trial_id', 
                                          'stimulus_index_enc1', 'stimulus_index_enc2', 'stimulus_index_enc3',
                                          'num_images_presented', 'Category', 'Location']],
                           on=['subject_id', 'Neuron_ID', 'trial_id'], 
                           how='inner')
    
    # STEP 4: Add probe categorization
    def categorize_probe(row):
        preferred = row['preferred_image_id']
        probe = row['Probe_Image_ID']
        num_images = row['num_images_presented']
        
        enc1 = row.get('stimulus_index_enc1', np.nan)
        enc2 = row.get('stimulus_index_enc2', np.nan)
        enc3 = row.get('stimulus_index_enc3', np.nan)

        # Build encoding image list
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

    probe_merged['Probe_Category'] = probe_merged.apply(categorize_probe, axis=1)
    
    # Save probe data
    probe_merged.to_excel('/home/daria/PROJECT/graph_data/graph_probe.xlsx', index=False)
    print("Probe data saved with categorization!")
    
    return probe_merged

# Process probe data
probe_final = process_probe_data()

print("All processing complete!")
