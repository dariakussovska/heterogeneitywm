import os 
import pandas as pd
import numpy as np 

# Load brain regions data
final_data = "/home/daria/PROJECT/Neuron_Check_Significant_All.xlsx"
df_final = pd.read_excel(final_data)
brain_regions_path = "/home/daria/PROJECT/all_neuron_brain_regions_cleaned.xlsx"
brain_regions_df = pd.read_excel(brain_regions_path)

# Load trial info for num_images_presented
trial_info_path = "/home/daria/PROJECT/trial_info.xlsx"
trial_info_df = pd.read_excel(trial_info_path)

def add_preferred_and_category(df, significant_neurons_df):
    df = df.copy()

    # Normalize dtypes for keys
    df['subject_id'] = df['subject_id'].astype(str).str.strip()
    df['Neuron_ID_3'] = pd.to_numeric(df['Neuron_ID_3'], errors='coerce').astype('Int64')

    # Get preferred image id (im_cat_1st) from significant neurons
    sig = significant_neurons_df[['subject_id','Neuron_ID_3','im_cat_1st','Signi']].copy()
    sig['subject_id'] = sig['subject_id'].astype(str).str.strip()
    sig['Neuron_ID_3'] = pd.to_numeric(sig['Neuron_ID_3'], errors='coerce').astype('Int64')

    # Remove stale cols
    cols_to_drop = ['Signi','Signi_x','Signi_y','im_cat_1st','im_cat_1st_x','im_cat_1st_y',
                    'preferred_image_id_from_sig','preferred_image_id_sig', 'Category']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Merge - im_cat_1st becomes our preferred_image_id
    df = pd.merge(df, sig, on=['subject_id','Neuron_ID_3'], how='left', suffixes=('', '_sig'))

    # Set preferred_image_id directly from im_cat_1st
    if 'im_cat_1st' in df.columns:
        df['preferred_image_id'] = df['im_cat_1st']
        df.drop(columns=['im_cat_1st'], inplace=True)
    else:
        df['preferred_image_id'] = np.nan

    # Standardize Signi
    if 'Signi_sig' in df.columns:
        if 'Signi' in df.columns:
            df.drop(columns=['Signi'], inplace=True)
        df.rename(columns={'Signi_sig': 'Signi'}, inplace=True)
    elif 'Signi' not in df.columns:
        df['Signi'] = np.nan

    # Normalize numeric types
    df['preferred_image_id'] = pd.to_numeric(df['preferred_image_id'], errors='coerce')
    df['stimulus_index'] = pd.to_numeric(df['stimulus_index'], errors='coerce')

    # Create single Category column
    def get_category(row):
        pref = row['preferred_image_id']
        stim = row['stimulus_index']
        if pd.isna(pref) or pd.isna(stim):
            return 'Unknown'
        if stim == 5:  # No image shown
            return 'No-Image'
        return 'Preferred' if pref == stim else 'Non-Preferred'
    
    df['Category'] = df.apply(get_category, axis=1)

    return df

def add_brain_region(df, brain_regions_df):
    df = df.copy()
    br = brain_regions_df.copy()

    # Normalize types
    df['subject_id'] = df['subject_id'].astype(str).str.strip()
    df['Neuron_ID_3'] = pd.to_numeric(df['Neuron_ID_3'], errors='coerce').astype('Int64')

    br['subject_id'] = br['subject_id'].astype(str).str.strip()
    br['Neuron_ID_3'] = pd.to_numeric(br['Neuron_ID_3'], errors='coerce').astype('Int64')

    # Find region column
    region_col = next((c for c in ['Location','Region','BrainRegion','brain_region'] if c in br.columns), None)
    if region_col is None:
        raise ValueError("brain_regions_df must contain a region column")

    # Merge
    br_slim = br[['subject_id', 'Neuron_ID_3', region_col]].drop_duplicates()
    merged = pd.merge(df, br_slim, on=['subject_id', 'Neuron_ID_3'], how='left')

    if region_col != 'Location':
        merged = merged.rename(columns={region_col: 'Location'})

    return merged

def add_num_images_presented(df, trial_info_df):
    """Add num_images_presented from trial_info.xlsx to all files"""
    
    # Prepare trial info data
    trial_info = trial_info_df.copy()
    trial_info['subject_id'] = trial_info['subject_id'].astype(str).str.strip()
    trial_info['Neuron_ID_3'] = pd.to_numeric(trial_info['Neuron_ID_3'], errors='coerce').astype('Int64')
    trial_info['trial_id'] = pd.to_numeric(trial_info['trial_id'], errors='coerce').astype('Int64')
    
    # Prepare main dataframe
    df_clean = df.copy()
    df_clean['subject_id'] = df_clean['subject_id'].astype(str).str.strip()
    df_clean['Neuron_ID_3'] = pd.to_numeric(df_clean['Neuron_ID_3'], errors='coerce').astype('Int64')
    df_clean['trial_id'] = pd.to_numeric(df_clean['trial_id'], errors='coerce').astype('Int64')
    
    # Merge num_images_presented
    df_clean = pd.merge(
        df_clean, 
        trial_info[['subject_id', 'Neuron_ID_3', 'trial_id', 'num_images_presented']],
        on=['subject_id', 'Neuron_ID_3', 'trial_id'], 
        how='left'
    )
    
    return df_clean

def categorize_probe(row):
    """Categorize probe trials based on your logic"""
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

# Process graph_data files only
graph_data_dir = "/home/daria/PROJECT/graph_data"
for filename in os.listdir(graph_data_dir):
    if filename.startswith('graph_') and filename.endswith('.xlsx'):
        file_path = os.path.join(graph_data_dir, filename)
        print(f"Processing: {filename}")
        
        try:
            df_graph = pd.read_excel(file_path)
            
            # Add trial info first to get num_images_presented (to ALL files)
            df_graph = add_num_images_presented(df_graph, trial_info_df)
            
            # Add preferred image (im_cat_1st), category, and brain region
            df_graph = add_preferred_and_category(df_graph, df_final)
            df_graph = add_brain_region(df_graph, brain_regions_df)
            
            # Special handling for probe data
            if 'probe' in filename.lower() and 'Probe_Image_ID' in df_graph.columns:
                # Add probe categorization using your exact function
                df_graph['Probe_Category'] = df_graph.apply(categorize_probe, axis=1)
            
            # Keep only essential columns
            essential_columns = [col for col in df_graph.columns if col not in ['p_val', 'CI', 'mean_1st', 'cat_1st', 'mean_2nd', 'cat_2nd']]
            df_graph = df_graph[essential_columns]
            
            # Save back
            df_graph.to_excel(file_path, index=False)
            print(f"✓ Added significance data to: {filename}")
            
        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")

print("Significance data added to all graph files successfully!")
