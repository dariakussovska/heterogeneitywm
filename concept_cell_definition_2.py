import numpy as np
import pandas as pd
import os

BASE_DIR = '/home/daria/PROJECT'
CLEAN_DATA_DIR = os.path.join(BASE_DIR, 'clean_data')
GRAPH_DATA_DIR = os.path.join(BASE_DIR, 'graph_data')

# Load the necessary data files
significant_neurons_df = pd.read_excel('/home/daria/PROJECT/Neuron_Check_Significant_All.xlsx')
trial_info = pd.read_excel('/home/daria/PROJECT/trial_info.xlsx')
brain_regions_df = pd.read_excel('/home/daria/PROJECT/all_neuron_brain_regions_cleaned.xlsx')

# Filter significant neurons to only include Y or N
significant_neurons_filtered = significant_neurons_df[significant_neurons_df['Signi'].isin(['Y', 'N'])]

# Load encoding data to get Category information for each trial
print("Loading encoding data for Category inheritance...")
enc1_data = pd.read_excel(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding1.xlsx'))
enc2_data = pd.read_excel(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding2.xlsx')) 
enc3_data = pd.read_excel(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding3.xlsx'))

# Add Category to encoding data
def add_category_to_encoding(df):
    """Add Category column to encoding data"""
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

# Create a mapping of trial Category based on num_images_presented
def create_trial_category_mapping(trial_info_df, enc1_df, enc2_df, enc3_df):
    """Create a mapping of (subject_id, trial_id) to Category based on num_images_presented"""
    trial_mapping = []
    
    for _, trial_row in trial_info_df.iterrows():
        subject_id = trial_row['subject_id']
        trial_id = trial_row['trial_id']
        num_images = trial_row['num_images_presented']
        
        if num_images == 1:
            # Get Category from Encoding1
            enc1_match = enc1_df[(enc1_df['subject_id'] == subject_id) & (enc1_df['trial_id'] == trial_id)]
            if not enc1_match.empty:
                category = enc1_match.iloc[0]['Category'] if 'Category' in enc1_match.columns else 'Unknown'
                trial_mapping.append({
                    'subject_id': subject_id,
                    'trial_id': trial_id,
                    'Category': category
                })
                
        elif num_images == 2:
            # Get Category from Encoding2
            enc2_match = enc2_df[(enc2_df['subject_id'] == subject_id) & (enc2_df['trial_id'] == trial_id)]
            if not enc2_match.empty:
                category = enc2_match.iloc[0]['Category'] if 'Category' in enc2_match.columns else 'Unknown'
                trial_mapping.append({
                    'subject_id': subject_id,
                    'trial_id': trial_id,
                    'Category': category
                })
                
        elif num_images == 3:
            # Get Category from Encoding3
            enc3_match = enc3_df[(enc3_df['subject_id'] == subject_id) & (enc3_df['trial_id'] == trial_id)]
            if not enc3_match.empty:
                category = enc3_match.iloc[0]['Category'] if 'Category' in enc3_match.columns else 'Unknown'
                trial_mapping.append({
                    'subject_id': subject_id,
                    'trial_id': trial_id,
                    'Category': category
                })
    
    return pd.DataFrame(trial_mapping)

# Create the trial category mapping
trial_category_map = create_trial_category_mapping(trial_info, enc1_data, enc2_data, enc3_data)

# Function to add Signi column
def add_signi_column(df, sig_neurons_df):
    """Add Signi column to dataframe"""
    if 'subject_id' in df.columns and 'Neuron_ID' in df.columns:
        result = pd.merge(df, sig_neurons_df[['subject_id', 'Neuron_ID', 'Signi']], 
                         on=['subject_id', 'Neuron_ID'], how='left')
        return result
    elif 'Neuron_ID_3' in df.columns and 'Neuron_ID_3' in sig_neurons_df.columns:
        result = pd.merge(df, sig_neurons_df[['Neuron_ID_3', 'Signi']], 
                         on='Neuron_ID_3', how='left')
        return result
    else:
        return df

# Function to add preferred_image_id
def add_preferred_image(df, sig_neurons_df):
    """Add preferred_image_id column"""
    if 'subject_id' in df.columns and 'Neuron_ID' in df.columns:
        result = pd.merge(df, sig_neurons_df[['subject_id', 'Neuron_ID', 'im_cat_1st']], 
                         on=['subject_id', 'Neuron_ID'], how='left')
        result.rename(columns={'im_cat_1st': 'preferred_image_id'}, inplace=True)
        return result
    elif 'Neuron_ID_3' in df.columns and 'Neuron_ID_3' in sig_neurons_df.columns:
        result = pd.merge(df, sig_neurons_df[['Neuron_ID_3', 'im_cat_1st']], 
                         on='Neuron_ID_3', how='left')
        result.rename(columns={'im_cat_1st': 'preferred_image_id'}, inplace=True)
        return result
    else:
        return df

# Function to add brain region
def add_brain_region(df, brain_regions_df):
    """Add brain region information"""
    if 'Neuron_ID_3' in df.columns and 'Neuron_ID_3' in brain_regions_df.columns:
        result = pd.merge(df, brain_regions_df[['Neuron_ID_3', 'Location']], 
                         on='Neuron_ID_3', how='left')
    elif 'subject_id' in df.columns and 'Neuron_ID' in df.columns:
        result = pd.merge(df, brain_regions_df[['subject_id', 'Neuron_ID', 'Location']], 
                         on=['subject_id', 'Neuron_ID'], how='left')
    else:
        return df
    return result

# Function to add trial info
def add_trial_info(df, trial_info_df):
    """Add trial information"""
    if 'subject_id' in df.columns and 'trial_id' in df.columns:
        available_columns = [col for col in trial_info_df.columns if col not in ['subject_id', 'trial_id']]
        result = pd.merge(df, trial_info_df[['subject_id', 'trial_id'] + available_columns], 
                         on=['subject_id', 'trial_id'], how='left')
        return result
    else:
        return df

# Function to add Category based on trial mapping
def add_category_from_mapping(df, category_map):
    """Add Category column based on trial mapping"""
    if 'subject_id' in df.columns and 'trial_id' in df.columns:
        result = pd.merge(df, category_map, on=['subject_id', 'trial_id'], how='left')
        # Fill missing Categories with 'Unknown'
        result['Category'] = result['Category'].fillna('Unknown')
        return result
    else:
        df['Category'] = 'Unknown'
        return df

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

# Process all files
def process_files(directory_path, is_clean_data=True):
    """Process all files in a directory"""
    for filename in os.listdir(directory_path):
        if ((is_clean_data and filename.startswith('cleaned_') and filename.endswith('.xlsx')) or
            (not is_clean_data and filename.startswith('graph_') and filename.endswith('.xlsx'))):
            
            file_path = os.path.join(directory_path, filename)
            print(f"Processing file: {filename}")
            
            # Load the file
            df = pd.read_excel(file_path)
            print(f"Original columns: {df.columns.tolist()}")
            
            # Add all required columns
            df = add_signi_column(df, significant_neurons_filtered)
            df = add_preferred_image(df, significant_neurons_filtered)
            df = add_trial_info(df, trial_info)
            df = add_brain_region(df, brain_regions_df)
            df = add_category_from_mapping(df, trial_category_map)  # Add Category from mapping
            
            # For probe files, add probe categorization
            if 'probe' in filename.lower():
                df['Probe_Category'] = df.apply(categorize_probe, axis=1)
            
            # Save back to the same file
            df.to_excel(file_path, index=False)
            print(f"Updated file: {filename}")
            print(f"Final columns: {df.columns.tolist()}\n")

# Process both directories
print("=== PROCESSING CLEAN_DATA FILES ===")
process_files(CLEAN_DATA_DIR, is_clean_data=True)

print("=== PROCESSING GRAPH_DATA FILES ===")
process_files(GRAPH_DATA_DIR, is_clean_data=False)

print("All files have been updated!")
print("For delay/fixation: Category is inherited from the corresponding encoding period based on num_images_presented")
print("For probe: Uses your exact categorization logic")
