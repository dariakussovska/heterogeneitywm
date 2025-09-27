import numpy as np
import pandas as pd
import os

BASE_DIR = '/home/daria/PROJECT'
CLEAN_DATA_DIR = os.path.join(BASE_DIR, 'clean_data')
GRAPH_DATA_DIR = os.path.join(BASE_DIR, 'graph_data')

# STEP 1: Load all source data
print("Loading source data...")
significant_neurons_df = pd.read_excel('/home/daria/PROJECT/Neuron_Check_Significant_All.xlsx')
trial_info = pd.read_excel('/home/daria/PROJECT/trial_info.xlsx')
brain_regions_df = pd.read_excel('/home/daria/PROJECT/all_neuron_brain_regions_cleaned.xlsx')

# Filter significant neurons to only include Y or N
significant_neurons_filtered = significant_neurons_df[significant_neurons_df['Signi'].isin(['Y', 'N'])]

print("Source data loaded:")
print(f"Significant neurons: {len(significant_neurons_filtered)} rows")
print(f"Trial info: {len(trial_info)} rows") 
print(f"Brain regions: {len(brain_regions_df)} rows")

# STEP 2: Load encoding data and add basic columns first
print("\nLoading encoding data...")
enc1_data = pd.read_excel(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding1.xlsx'))
enc2_data = pd.read_excel(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding2.xlsx')) 
enc3_data = pd.read_excel(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding3.xlsx'))

# Function to add basic columns (Signi, preferred_image_id, brain region, trial info)
def add_basic_columns(df, sig_neurons_df, brain_regions_df, trial_info_df):
    """Add Signi, preferred_image_id, Location, and trial info to dataframe"""
    
    # Add Signi and preferred_image_id from significant neurons
    if 'subject_id' in df.columns and 'Neuron_ID' in df.columns:
        sig_cols = ['subject_id', 'Neuron_ID', 'Signi', 'im_cat_1st']
        df = pd.merge(df, sig_neurons_df[sig_cols], on=['subject_id', 'Neuron_ID'], how='left')
        df.rename(columns={'im_cat_1st': 'preferred_image_id'}, inplace=True)
    
    # Add brain region
    if 'Neuron_ID_3' in df.columns and 'Neuron_ID_3' in brain_regions_df.columns:
        df = pd.merge(df, brain_regions_df[['Neuron_ID_3', 'Location']], on='Neuron_ID_3', how='left')
    elif 'subject_id' in df.columns and 'Neuron_ID' in df.columns:
        df = pd.merge(df, brain_regions_df[['subject_id', 'Neuron_ID', 'Location']], on=['subject_id', 'Neuron_ID'], how='left')
    
    # Add trial info (num_images_presented and stimulus indices)
    if 'subject_id' in df.columns and 'trial_id' in df.columns:
        trial_cols = [col for col in trial_info_df.columns if col not in ['subject_id', 'trial_id']]
        df = pd.merge(df, trial_info_df[['subject_id', 'trial_id'] + trial_cols], on=['subject_id', 'trial_id'], how='left')
    
    return df

# Add basic columns to encoding data
print("Adding basic columns to encoding data...")
enc1_data = add_basic_columns(enc1_data, significant_neurons_filtered, brain_regions_df, trial_info)
enc2_data = add_basic_columns(enc2_data, significant_neurons_filtered, brain_regions_df, trial_info)
enc3_data = add_basic_columns(enc3_data, significant_neurons_filtered, brain_regions_df, trial_info)

# STEP 3: Add Category to encoding data based on preferred_image_id vs stimulus_index
print("Adding Category to encoding data...")
def add_category_to_encoding(df):
    """Add Category column to encoding data: Preferred if preferred_image_id == stimulus_index"""
    if 'preferred_image_id' in df.columns and 'stimulus_index' in df.columns:
        df['Category'] = df.apply(
            lambda row: 'Preferred' if row['preferred_image_id'] == row['stimulus_index'] 
            else 'Non-Preferred' if pd.notna(row['stimulus_index']) 
            else 'Unknown', 
            axis=1
        )
    return df

enc1_data = add_category_to_encoding(enc1_data)
enc2_data = add_category_to_encoding(enc2_data)
enc3_data = add_category_to_encoding(enc3_data)

# Save the updated encoding data
enc1_data.to_excel(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding1.xlsx'), index=False)
enc2_data.to_excel(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding2.xlsx'), index=False)
enc3_data.to_excel(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding3.xlsx'), index=False)
print("Encoding data updated with Category")

# STEP 4: Function to determine Category for non-encoding files (delay, fixation, probe)
def get_category_from_encoding(row, enc1_df, enc2_df, enc3_df):
    """Get Category from the correct encoding period based on num_images_presented"""
    subject_id = row['subject_id']
    trial_id = row['trial_id']
    num_images = row.get('num_images_presented')
    
    if pd.isna(num_images):
        return 'Unknown'
    
    try:
        num_images = int(num_images)
    except (ValueError, TypeError):
        return 'Unknown'
    
    if num_images == 1:
        # Get Category from Encoding1
        match = enc1_df[(enc1_df['subject_id'] == subject_id) & (enc1_df['trial_id'] == trial_id)]
        if not match.empty and 'Category' in match.columns:
            return match.iloc[0]['Category']
    
    elif num_images == 2:
        # Get Category from Encoding2
        match = enc2_df[(enc2_df['subject_id'] == subject_id) & (enc2_df['trial_id'] == trial_id)]
        if not match.empty and 'Category' in match.columns:
            return match.iloc[0]['Category']
    
    elif num_images == 3:
        # Get Category from Encoding3
        match = enc3_df[(enc3_df['subject_id'] == subject_id) & (enc3_df['trial_id'] == trial_id)]
        if not match.empty and 'Category' in match.columns:
            return match.iloc[0]['Category']
    
    return 'Unknown'

# YOUR EXACT CATEGORIZE_PROBE FUNCTION
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

# STEP 5: Process delay, fixation, and probe files
def process_non_encoding_files():
    """Process delay, fixation, and probe files"""
    file_types = ['Delay', 'Fixation', 'Probe']
    
    for file_type in file_types:
        clean_file = os.path.join(CLEAN_DATA_DIR, f'cleaned_{file_type}.xlsx')
        graph_file = os.path.join(GRAPH_DATA_DIR, f'graph_{file_type.lower()}.xlsx')
        
        # Process clean data file
        if os.path.exists(clean_file):
            print(f"\nProcessing {file_type} clean data...")
            df = pd.read_excel(clean_file)
            print(f"Original columns: {df.columns.tolist()}")
            
            # Add basic columns
            df = add_basic_columns(df, significant_neurons_filtered, brain_regions_df, trial_info)
            
            # Add Category from encoding data
            df['Category'] = df.apply(
                lambda row: get_category_from_encoding(row, enc1_data, enc2_data, enc3_data), 
                axis=1
            )
            
            # For probe files, add Probe_Category
            if file_type == 'Probe':
                df['Probe_Category'] = df.apply(categorize_probe, axis=1)
            
            # Save clean data
            df.to_excel(clean_file, index=False)
            print(f"Updated {file_type} clean data")
            print(f"Category distribution: {df['Category'].value_counts().to_dict()}")
            if 'Probe_Category' in df.columns:
                print(f"Probe_Category distribution: {df['Probe_Category'].value_counts().to_dict()}")
        
        # Process graph data file
        if os.path.exists(graph_file):
            print(f"\nProcessing {file_type} graph data...")
            df = pd.read_excel(graph_file)
            print(f"Original columns: {df.columns.tolist()}")
            
            # Add basic columns
            df = add_basic_columns(df, significant_neurons_filtered, brain_regions_df, trial_info)
            
            # Add Category from encoding data
            df['Category'] = df.apply(
                lambda row: get_category_from_encoding(row, enc1_data, enc2_data, enc3_data), 
                axis=1
            )
            
            # For probe files, add Probe_Category
            if file_type == 'Probe':
                df['Probe_Category'] = df.apply(categorize_probe, axis=1)
            
            # Save graph data
            df.to_excel(graph_file, index=False)
            print(f"Updated {file_type} graph data")
            print(f"Category distribution: {df['Category'].value_counts().to_dict()}")
            if 'Probe_Category' in df.columns:
                print(f"Probe_Category distribution: {df['Probe_Category'].value_counts().to_dict()}")

# STEP 6: Execute the processing
print("\n=== PROCESSING NON-ENCODING FILES ===")
process_non_encoding_files()

print("\n=== PROCESSING COMPLETE ===")
print("All files have been updated in the correct order:")
print("1. Encoding files: Category based on preferred_image_id vs stimulus_index")
print("2. Delay/Fixation files: Category inherited from correct encoding period based on num_images_presented")
print("3. Probe files: Category inherited + Probe_Category using your exact logic")
