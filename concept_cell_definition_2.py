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

print("Available columns in data sources:")
print("Significant neurons:", significant_neurons_filtered.columns.tolist())
print("Trial info:", trial_info.columns.tolist())
print("Brain regions:", brain_regions_df.columns.tolist())

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
        print("Cannot add Signi - missing key columns")
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
        print("Cannot add brain regions - no common keys")
        return df
    return result

# Function to add trial info (num_images_presented and stimulus indices)
def add_trial_info(df, trial_info_df):
    """Add trial information including num_images_presented and stimulus indices"""
    if 'subject_id' in df.columns and 'trial_id' in df.columns:
        # Get all available columns from trial_info
        available_columns = [col for col in trial_info_df.columns if col not in ['subject_id', 'trial_id']]
        result = pd.merge(df, trial_info_df[['subject_id', 'trial_id'] + available_columns], 
                         on=['subject_id', 'trial_id'], how='left')
        return result
    else:
        return df

# Function to add Category column
def add_category_column(df):
    """Add Category column based on preferred_image_id and stimulus_index"""
    if 'preferred_image_id' in df.columns and 'stimulus_index' in df.columns:
        df['Category'] = df.apply(
            lambda row: 'Preferred' if row['preferred_image_id'] == row['stimulus_index'] 
            else 'Non-Preferred' if pd.notna(row['stimulus_index']) 
            else 'Unknown', 
            axis=1
        )
    return df

# Function to add Probe_Category column
def add_probe_category(df):
    """Add Probe_Category column for probe data"""
    if 'Probe_Image_ID' in df.columns and 'preferred_image_id' in df.columns and 'num_images_presented' in df.columns:
        
        def categorize_probe(row):
            preferred = row['preferred_image_id']
            probe = row['Probe_Image_ID']
            num_images = row['num_images_presented']
            
            # Get stimulus indices for each encoding period
            enc1 = row.get('stimulus_index_enc1', np.nan)
            enc2 = row.get('stimulus_index_enc2', np.nan)
            enc3 = row.get('stimulus_index_enc3', np.nan)

            # Build encoding image list, excluding NaN and '5' (no image shown)
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

        df['Probe_Category'] = df.apply(categorize_probe, axis=1)
    return df

# Process all clean_data files
def process_clean_data_files():
    """Process all files in the clean_data directory"""
    for filename in os.listdir(CLEAN_DATA_DIR):
        if filename.startswith('cleaned_') and filename.endswith('.xlsx'):
            file_path = os.path.join(CLEAN_DATA_DIR, filename)
            print(f"Processing clean data file: {filename}")
            
            # Load the file
            df = pd.read_excel(file_path)
            print(f"Original columns: {df.columns.tolist()}")
            
            # Add all required columns
            df = add_signi_column(df, significant_neurons_filtered)
            df = add_preferred_image(df, significant_neurons_filtered)
            df = add_trial_info(df, trial_info)
            df = add_brain_region(df, brain_regions_df)
            df = add_category_column(df)
            
            # For probe files, add probe categorization
            if 'probe' in filename.lower():
                df = add_probe_category(df)
            
            # Save back to the same file
            df.to_excel(file_path, index=False)
            print(f"Updated clean data file: {filename}")
            print(f"Final columns: {df.columns.tolist()}\n")

# Process all graph_data files
def process_graph_data_files():
    """Process all files in the graph_data directory"""
    for filename in os.listdir(GRAPH_DATA_DIR):
        if filename.startswith('graph_') and filename.endswith('.xlsx'):
            file_path = os.path.join(GRAPH_DATA_DIR, filename)
            print(f"Processing graph data file: {filename}")
            
            # Load the file
            df = pd.read_excel(file_path)
            print(f"Original columns: {df.columns.tolist()}")
            
            # Add all required columns
            df = add_signi_column(df, significant_neurons_filtered)
            df = add_preferred_image(df, significant_neurons_filtered)
            df = add_trial_info(df, trial_info)
            df = add_brain_region(df, brain_regions_df)
            df = add_category_column(df)
            
            # For probe files, add probe categorization
            if 'probe' in filename.lower():
                df = add_probe_category(df)
            
            # Save back to the same file
            df.to_excel(file_path, index=False)
            print(f"Updated graph data file: {filename}")
            print(f"Final columns: {df.columns.tolist()}\n")

# Process both directories
print("=== PROCESSING CLEAN_DATA FILES ===")
process_clean_data_files()

print("=== PROCESSING GRAPH_DATA FILES ===")
process_graph_data_files()

print("All files have been updated with:")
print("- Signi column")
print("- Location column") 
print("- num_images_presented")
print("- preferred_image_id (im_cat_1st)")
print("- Category column")
print("- Probe_Category column (for probe files)")
