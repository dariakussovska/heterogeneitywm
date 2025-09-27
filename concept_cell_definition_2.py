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

# Load encoding data with Category information
print("Loading encoding data with Category information...")
enc1_data = pd.read_excel(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding1.xlsx'))
enc2_data = pd.read_excel(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding2.xlsx')) 
enc3_data = pd.read_excel(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding3.xlsx'))

# Add Category column to encoding data if not present
def add_category_to_encoding(df, enc_num):
    """Add Category column to encoding data based on preferred_image_id and stimulus_index"""
    if 'preferred_image_id' in df.columns and 'stimulus_index' in df.columns:
        df['Category'] = df.apply(
            lambda row: 'Preferred' if row['preferred_image_id'] == row['stimulus_index'] 
            else 'Non-Preferred' if pd.notna(row['stimulus_index']) 
            else 'Unknown', 
            axis=1
        )
        # Rename to indicate which encoding period this Category belongs to
        df.rename(columns={'Category': f'Category_enc{enc_num}'}, inplace=True)
    return df

enc1_data = add_category_to_encoding(enc1_data, 1)
enc2_data = add_category_to_encoding(enc2_data, 2)
enc3_data = add_category_to_encoding(enc3_data, 3)

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

# Function to add the correct Category based on num_images_presented
def add_category_from_encoding(df, enc1_data, enc2_data, enc3_data):
    """Add Category column based on the correct encoding period"""
    if 'subject_id' in df.columns and 'trial_id' in df.columns and 'num_images_presented' in df.columns:
        # Create a result dataframe
        result_dfs = []
        
        # Process each num_images_presented value separately
        for num_images in [1, 2, 3]:
            mask = df['num_images_presented'] == num_images
            subset = df[mask].copy()
            
            if len(subset) > 0:
                if num_images == 1:
                    # Use Category from Encoding1
                    category_data = enc1_data[['subject_id', 'trial_id', 'Category_enc1']].drop_duplicates()
                    subset = pd.merge(subset, category_data, on=['subject_id', 'trial_id'], how='left')
                    subset['Category'] = subset['Category_enc1']
                    subset.drop('Category_enc1', axis=1, inplace=True)
                    
                elif num_images == 2:
                    # Use Category from Encoding2
                    category_data = enc2_data[['subject_id', 'trial_id', 'Category_enc2']].drop_duplicates()
                    subset = pd.merge(subset, category_data, on=['subject_id', 'trial_id'], how='left')
                    subset['Category'] = subset['Category_enc2']
                    subset.drop('Category_enc2', axis=1, inplace=True)
                    
                elif num_images == 3:
                    # Use Category from Encoding3
                    category_data = enc3_data[['subject_id', 'trial_id', 'Category_enc3']].drop_duplicates()
                    subset = pd.merge(subset, category_data, on=['subject_id', 'trial_id'], how='left')
                    subset['Category'] = subset['Category_enc3']
                    subset.drop('Category_enc3', axis=1, inplace=True)
                
                result_dfs.append(subset)
        
        # Combine all subsets
        if result_dfs:
            result = pd.concat(result_dfs, ignore_index=True)
            # Add rows with unknown num_images_presented
            unknown_mask = ~df['num_images_presented'].isin([1, 2, 3]) | df['num_images_presented'].isna()
            if unknown_mask.any():
                unknown_subset = df[unknown_mask].copy()
                unknown_subset['Category'] = 'Unknown'
                result_dfs.append(unknown_subset)
            
            result = pd.concat(result_dfs, ignore_index=True)
            return result
        else:
            df['Category'] = 'Unknown'
            return df
    else:
        df['Category'] = 'Unknown'
        return df

# Function to add Probe_Category column
def add_probe_category(df):
    """Add Probe_Category column for probe data based on encoded vs non-encoded"""
    if 'Probe_Image_ID' in df.columns and 'preferred_image_id' in df.columns and 'num_images_presented' in df.columns and 'Category' in df.columns:
        
        def categorize_probe(row):
            preferred = row['preferred_image_id']
            probe = row['Probe_Image_ID']
            category = row['Category']
            
            # Check if the probe image is the preferred image
            is_preferred_probe = (probe == preferred)
            
            # Determine if encoded based on Category
            # If Category is 'Preferred', it means the preferred image was encoded in that trial
            # If Category is 'Non-Preferred', it means the preferred image was NOT encoded in that trial
            
            if category == 'Preferred' and is_preferred_probe:
                return 'Preferred Encoded'
            elif category == 'Preferred' and not is_preferred_probe:
                return 'Preferred Nonencoded'
            elif category == 'Non-Preferred' and is_preferred_probe:
                return 'Nonpreferred Encoded'
            elif category == 'Non-Preferred' and not is_preferred_probe:
                return 'Nonpreferred Nonencoded'
            else:
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
            
            # Add Category based on correct encoding period
            df = add_category_from_encoding(df, enc1_data, enc2_data, enc3_data)
            
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
            
            # Add Category based on correct encoding period
            df = add_category_from_encoding(df, enc1_data, enc2_data, enc3_data)
            
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
print("- Category column (from correct encoding period based on num_images_presented)")
print("- Probe_Category column (for probe files)")
