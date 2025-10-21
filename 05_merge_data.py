import pandas as pd
import os

BASE_DIR = '/./'
CLEAN_DATA_DIR = os.path.join(BASE_DIR, 'clean_data')
GRAPH_DATA_DIR = os.path.join(BASE_DIR, 'graph_data')

# Load all source data
significant_neurons_df = pd.read_feather('/./Neuron_Check_Significant_All.feather')
trial_info_df = pd.read_feather('/./trial_info.feather')
brain_regions_df = pd.read_feather('/./all_neuron_brain_regions_cleaned.feather')

# Filter significant neurons to only include Y or N
significant_neurons_filtered = significant_neurons_df[significant_neurons_df['Signi'].isin(['Y', 'N'])]

# Add all required columns to a single file
def add_all_columns_to_file(file_path, sig_neurons_df, trial_info_df, brain_regions_df):
    """Add Signi, im_cat_1st, num_images_presented, and Location to a single file"""
    
    # Load the file
    df = pd.read_feather(file_path)
    
    # STEP 2A: Add Signi and im_cat_1st from significant neurons
    if 'subject_id' in df.columns and 'Neuron_ID' in df.columns:
        
        # Select only the columns we need from significant neurons
        sig_cols = ['subject_id', 'Neuron_ID', 'Signi', 'im_cat_1st']
        sig_subset = sig_neurons_df[sig_cols].copy()
        
        # Merge the data
        df = pd.merge(df, sig_subset, on=['subject_id', 'Neuron_ID'], how='left')
        
        # Check if merge was successful
        if 'Signi' in df.columns:
            signi_counts = df['Signi'].value_counts(dropna=False)
        
        if 'im_cat_1st' in df.columns:
            print(f"im_cat_1st sample values: {df['im_cat_1st'].dropna().unique()[:5]}")
    
    # Add num_images_presented from trial_info
    if 'subject_id' in df.columns and 'trial_id' in df.columns:
        
        # Select only the columns we need from trial_info
        trial_cols = ['subject_id', 'trial_id', 'num_images_presented']
        trial_subset = trial_info_df[trial_cols].copy()
        
        # Merge the data
        df = pd.merge(df, trial_subset, on=['subject_id', 'trial_id'], how='left')
        
        # Check if merge was successful
        if 'num_images_presented' in df.columns:
            num_images_counts = df['num_images_presented'].value_counts(dropna=False)
            
    # Add Location from brain regions
    if 'subject_id' in df.columns and 'Neuron_ID' in df.columns:
        
        # Select only the columns we need from brain regions
        brain_cols = ['subject_id', 'Neuron_ID', 'Location']
        brain_subset = brain_regions_df[brain_cols].copy()
        
        # Merge the data
        df = pd.merge(df, brain_subset, on=['subject_id', 'Neuron_ID'], how='left')
    
    # Alternative: Try merging by Neuron_ID_3 if available
    elif 'Neuron_ID_3' in df.columns and 'Neuron_ID_3' in brain_regions_df.columns:
        print("Adding Location using Neuron_ID_3...")
        
        # Select only the columns we need from brain regions
        brain_cols = ['Neuron_ID_3', 'Location']
        brain_subset = brain_regions_df[brain_cols].copy()
        
        # Merge the data
        df = pd.merge(df, brain_subset, on='Neuron_ID_3', how='left')
            
    print(f"Final columns: {df.columns.tolist()}")
    return df

# Process all clean_data files
print("PROCESSING CLEAN_DATA FILES")

clean_files = [f for f in os.listdir(CLEAN_DATA_DIR) if f.startswith('cleaned_') and f.endswith('.feather')]

for filename in clean_files:
    file_path = os.path.join(CLEAN_DATA_DIR, filename)
    
    # Add all columns
    df_updated = add_all_columns_to_file(file_path, significant_neurons_filtered, trial_info_df, brain_regions_df)
    
    # Save back to the same file
    df_updated.to_feather(file_path, index=False)
    print(f"Saved: {filename}\n")

# Process all graph_data files
print("PROCESSING GRAPH_DATA FILES")

graph_files = [f for f in os.listdir(GRAPH_DATA_DIR) if f.startswith('graph_') and f.endswith('.feather')]

for filename in graph_files:
    file_path = os.path.join(GRAPH_DATA_DIR, filename)
    
    # Add all columns
    df_updated = add_all_columns_to_file(file_path, significant_neurons_filtered, trial_info_df, brain_regions_df)
    
    # Save back to the same file
    df_updated.to_feather(file_path, index=False)
    print(f" Saved: {filename}\n")
