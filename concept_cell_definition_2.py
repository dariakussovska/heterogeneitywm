import pandas as pd
import os

BASE_DIR = '/home/daria/PROJECT'
CLEAN_DATA_DIR = os.path.join(BASE_DIR, 'clean_data')
GRAPH_DATA_DIR = os.path.join(BASE_DIR, 'graph_data')

# STEP 1: Load all source data
print("Loading source data...")
significant_neurons_df = pd.read_excel('/home/daria/PROJECT/Neuron_Check_Significant_All.xlsx')
trial_info_df = pd.read_excel('/home/daria/PROJECT/trial_info.xlsx')
brain_regions_df = pd.read_excel('/home/daria/PROJECT/all_neuron_brain_regions_cleaned.xlsx')

# Filter significant neurons to only include Y or N
significant_neurons_filtered = significant_neurons_df[significant_neurons_df['Signi'].isin(['Y', 'N'])]

print("Source data loaded:")
print(f"Significant neurons: {len(significant_neurons_filtered)} rows")
print(f"Trial info: {len(trial_info_df)} rows") 
print(f"Brain regions: {len(brain_regions_df)} rows")

# STEP 2: Function to add all required columns to a single file
def add_all_columns_to_file(file_path, sig_neurons_df, trial_info_df, brain_regions_df):
    """Add Signi, im_cat_1st, num_images_presented, and Location to a single file"""
    print(f"\nProcessing: {os.path.basename(file_path)}")
    
    # Load the file
    df = pd.read_excel(file_path)
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Original row count: {len(df)}")
    
    # STEP 2A: Add Signi and im_cat_1st from significant neurons
    if 'subject_id' in df.columns and 'Neuron_ID' in df.columns:
        print("Adding Signi and im_cat_1st...")
        
        # Select only the columns we need from significant neurons
        sig_cols = ['subject_id', 'Neuron_ID', 'Signi', 'im_cat_1st']
        sig_subset = sig_neurons_df[sig_cols].copy()
        
        # Merge the data
        df = pd.merge(df, sig_subset, on=['subject_id', 'Neuron_ID'], how='left')
        
        print(f"After Signi merge row count: {len(df)}")
        
        # Check if merge was successful
        if 'Signi' in df.columns:
            signi_counts = df['Signi'].value_counts(dropna=False)
            print(f"Signi distribution: {signi_counts.to_dict()}")
        
        if 'im_cat_1st' in df.columns:
            print(f"im_cat_1st sample values: {df['im_cat_1st'].dropna().unique()[:5]}")
    
    # STEP 2B: Add num_images_presented from trial_info
    if 'subject_id' in df.columns and 'trial_id' in df.columns:
        print("Adding num_images_presented...")
        
        # Select only the columns we need from trial_info
        trial_cols = ['subject_id', 'trial_id', 'num_images_presented']
        trial_subset = trial_info_df[trial_cols].copy()
        
        # Merge the data
        df = pd.merge(df, trial_subset, on=['subject_id', 'trial_id'], how='left')
        
        print(f"After trial info merge row count: {len(df)}")
        
        # Check if merge was successful
        if 'num_images_presented' in df.columns:
            num_images_counts = df['num_images_presented'].value_counts(dropna=False)
            print(f"num_images_presented distribution: {num_images_counts.to_dict()}")
    
    # STEP 2C: Add Location from brain regions
    if 'subject_id' in df.columns and 'Neuron_ID' in df.columns:
        print("Adding Location...")
        
        # Select only the columns we need from brain regions
        brain_cols = ['subject_id', 'Neuron_ID', 'Location']
        brain_subset = brain_regions_df[brain_cols].copy()
        
        # Merge the data
        df = pd.merge(df, brain_subset, on=['subject_id', 'Neuron_ID'], how='left')
        
        print(f"After brain regions merge row count: {len(df)}")
        
        # Check if merge was successful
        if 'Location' in df.columns:
            location_counts = df['Location'].value_counts(dropna=False)
            print(f"Location distribution: {location_counts.to_dict()}")
    
    # Alternative: Try merging by Neuron_ID_3 if available
    elif 'Neuron_ID_3' in df.columns and 'Neuron_ID_3' in brain_regions_df.columns:
        print("Adding Location using Neuron_ID_3...")
        
        # Select only the columns we need from brain regions
        brain_cols = ['Neuron_ID_3', 'Location']
        brain_subset = brain_regions_df[brain_cols].copy()
        
        # Merge the data
        df = pd.merge(df, brain_subset, on='Neuron_ID_3', how='left')
        
        print(f"After brain regions merge row count: {len(df)}")
        
        # Check if merge was successful
        if 'Location' in df.columns:
            location_counts = df['Location'].value_counts(dropna=False)
            print(f"Location distribution: {location_counts.to_dict()}")
    
    print(f"Final columns: {df.columns.tolist()}")
    return df

# STEP 3: Process all clean_data files
print("\n" + "="*50)
print("PROCESSING CLEAN_DATA FILES")
print("="*50)

clean_files = [f for f in os.listdir(CLEAN_DATA_DIR) if f.startswith('cleaned_') and f.endswith('.xlsx')]

for filename in clean_files:
    file_path = os.path.join(CLEAN_DATA_DIR, filename)
    
    # Add all columns
    df_updated = add_all_columns_to_file(file_path, significant_neurons_filtered, trial_info_df, brain_regions_df)
    
    # Save back to the same file
    df_updated.to_excel(file_path, index=False)
    print(f"Saved: {filename}\n")

# STEP 4: Process all graph_data files
print("\n" + "="*50)
print("PROCESSING GRAPH_DATA FILES")
print("="*50)

graph_files = [f for f in os.listdir(GRAPH_DATA_DIR) if f.startswith('graph_') and f.endswith('.xlsx')]

for filename in graph_files:
    file_path = os.path.join(GRAPH_DATA_DIR, filename)
    
    # Add all columns
    df_updated = add_all_columns_to_file(file_path, significant_neurons_filtered, trial_info_df, brain_regions_df)
    
    # Save back to the same file
    df_updated.to_excel(file_path, index=False)
    print(f" Saved: {filename}\n")

# Check one file from each directory to verify
def verify_file_columns(file_path, expected_columns):
    """Verify that a file has all expected columns"""
    df = pd.read_excel(file_path)
    missing_columns = [col for col in expected_columns if col not in df.columns]
    
    if missing_columns:
        print(f"{os.path.basename(file_path)}: Missing {missing_columns}")
    else:
        print(f"{os.path.basename(file_path)}: All columns present")

expected_columns = ['Signi', 'im_cat_1st', 'num_images_presented', 'Location']

# Check a sample from clean_data
if clean_files:
    sample_clean_file = os.path.join(CLEAN_DATA_DIR, clean_files[0])
    verify_file_columns(sample_clean_file, expected_columns)

# Check a sample from graph_data  
if graph_files:
    sample_graph_file = os.path.join(GRAPH_DATA_DIR, graph_files[0])
    verify_file_columns(sample_graph_file, expected_columns)
