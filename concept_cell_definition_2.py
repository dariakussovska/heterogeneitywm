import os 
import pandas as pd
import numpy as np 

# Load brain regions data
final_data = "/home/daria/PROJECT/Neuron_Check_Significant_All.xlsx"
df_final = pd.read_excel(final_data)
brain_regions_path = "/home/daria/PROJECT/all_neuron_brain_regions_cleaned.xlsx"
brain_regions_df = pd.read_excel(brain_regions_path)

import numpy as np
import pandas as pd

def add_preferred_and_category(df, significant_neurons_df):
    df = df.copy()

    # normalize dtypes
    df['subject_id'] = df['subject_id'].astype(str).str.strip()
    df['Neuron_ID']  = df['Neuron_ID'].astype(int)

    sig = significant_neurons_df[['subject_id','Neuron_ID','im_cat_1st','Signi']].copy()
    sig['subject_id'] = sig['subject_id'].astype(str).str.strip()
    sig['Neuron_ID']  = sig['Neuron_ID'].astype(int)

    # 1) Merge into a TEMPORARY column to avoid duplicates
    sig = sig.rename(columns={'im_cat_1st': 'preferred_image_id_from_sig'})
    df = pd.merge(df, sig, on=['subject_id','Neuron_ID'], how='left')

    # 2) Coalesce: if df already had preferred_image_id, keep it; otherwise use from_sig
    if 'preferred_image_id' not in df.columns:
        df['preferred_image_id'] = np.nan
    df['preferred_image_id'] = df['preferred_image_id'].where(
        df['preferred_image_id'].notna(), df['preferred_image_id_from_sig']
    )

    # 3) Clean up the temp column
    df = df.drop(columns=['preferred_image_id_from_sig'])

    # 4) Pick the best stimulus column available in this file
    stim_cols = [
        'stimulus_index_enc1','stimulus_index_enc2','stimulus_index_enc3',
        'stimulus_index','image_id_enc1','Image_ID','image_id'
    ]
    present_stim_col = next((c for c in stim_cols if c in df.columns), None)

    # 5) Compute Category safely (no ambiguous Series)
    def get_category(row):
        pref = row['preferred_image_id']
        if pd.isna(pref):
            return 'Unknown'          # neuron not significant / no preferred image
        if present_stim_col is None or pd.isna(row[present_stim_col]):
            return 'No-Image'         # e.g., Delay/Fixation with no stimulus label
        return 'Preferred' if pref == row[present_stim_col] else 'Non-Preferred'

    df['Category'] = df.apply(get_category, axis=1)
    return df
   
def add_brain_region(df, brain_regions_df):
    df = df.copy()
    br = brain_regions_df.copy()

    # --- Normalize types in df ---
    df['subject_id'] = df['subject_id'].astype(str).str.strip()
    if 'Neuron_ID' in df.columns:
        df['Neuron_ID'] = pd.to_numeric(df['Neuron_ID'], errors='coerce').astype('Int64')
    if 'Neuron_ID_3' in df.columns:
        df['Neuron_ID_3'] = pd.to_numeric(df['Neuron_ID_3'], errors='coerce').astype('Int64')

    # Ensure df has Neuron_ID_3 (unique)
    if 'Neuron_ID_3' not in df.columns and {'subject_id','Neuron_ID'}.issubset(df.columns):
        df['Neuron_ID_3'] = (df['subject_id'].astype(str).str.strip() + '0' +
                             df['Neuron_ID'].astype('Int64').astype(str)).astype(int)

    # --- Normalize types / columns in brain regions ---
    # Common variants seen: Neuron_ID_3, Neuron_ID_2, Neuron_ID; sometimes subject_id is present
    br_cols = set(br.columns.str.strip())

    # Standardize subject_id if present
    if 'subject_id' in br_cols:
        br['subject_id'] = br['subject_id'].astype(str).str.strip()

    # Detect neuron key in brain regions
    if 'Neuron_ID_3' in br_cols:
        neuron_key_br = 'Neuron_ID_3'
        br[neuron_key_br] = pd.to_numeric(br[neuron_key_br], errors='coerce').astype('Int64')
    elif 'Neuron_ID_2' in br_cols:
        neuron_key_br = 'Neuron_ID_2'
        br[neuron_key_br] = pd.to_numeric(br[neuron_key_br], errors='coerce').astype('Int64')
    elif 'Neuron_ID' in br_cols:
        neuron_key_br = 'Neuron_ID'
        br[neuron_key_br] = pd.to_numeric(br[neuron_key_br], errors='coerce').astype('Int64')
    else:
        raise ValueError("brain_regions_df must contain one of: Neuron_ID_3, Neuron_ID_2, or Neuron_ID")

    # If brain regions is NOT already on Neuron_ID_3, try to construct it
    if neuron_key_br != 'Neuron_ID_3':
        # Need subject_id to build Neuron_ID_3; if missing, we will merge on (subject_id, Neuron_ID) fallback
        if 'subject_id' in br_cols and neuron_key_br in {'Neuron_ID','Neuron_ID_2'}:
            br['Neuron_ID_3'] = (br['subject_id'].astype(str).str.strip() + '0' +
                                 br[neuron_key_br].astype('Int64').astype(str)).astype(int)
        # else: keep as-is for fallback merge

    # Decide merge strategy (prefer unique key)
    if 'Neuron_ID_3' in br.columns and 'Neuron_ID_3' in df.columns:
        left_keys  = ['Neuron_ID_3']
        right_keys = ['Neuron_ID_3']
    elif {'subject_id','Neuron_ID'}.issubset(df.columns) and \
         'subject_id' in br.columns and neuron_key_br in br.columns and neuron_key_br == 'Neuron_ID':
        left_keys  = ['subject_id','Neuron_ID']
        right_keys = ['subject_id','Neuron_ID']
    else:
        raise ValueError(
            "Cannot align keys to merge brain regions.\n"
            f"df has: {df.columns.tolist()}\n"
            f"brain_regions_df has: {br.columns.tolist()}\n"
            "Provide either Neuron_ID_3 in both, or (subject_id, Neuron_ID) in both."
        )

    # Do the merge; keep Location (or whatever the region column is named)
    # Try common region column names
    region_col = next((c for c in ['Location','Region','BrainRegion','brain_region'] if c in br.columns), None)
    if region_col is None:
        raise ValueError("brain_regions_df must contain a region column (e.g., 'Location').")

    br_slim = br[left_keys + [region_col]].drop_duplicates()
    merged = pd.merge(df, br_slim, left_on=left_keys, right_on=left_keys, how='left')

    # Standardize region column name to 'Location'
    if region_col != 'Location':
        merged = merged.rename(columns={region_col: 'Location'})

    return merged

# Add significance results to all clean_data files with preferred image and category
clean_data_dir = "/home/daria/PROJECT/clean_data"
for filename in os.listdir(clean_data_dir):
    if filename.startswith('cleaned_') and filename.endswith('.xlsx'):
        file_path = os.path.join(clean_data_dir, filename)
        df_clean = pd.read_excel(file_path)
        
        # Add preferred image, category, and brain region
        df_clean = add_preferred_and_category(df_clean, df_final)
        df_clean = add_brain_region(df_clean, brain_regions_df)
        
        # Keep only essential columns (remove CI, p_val, etc.)
        essential_columns = [col for col in df_clean.columns if col not in ['p_val', 'CI', 'mean_1st', 'cat_1st', 'mean_2nd', 'cat_2nd']]
        df_clean = df_clean[essential_columns]
        
        # Save back
        df_clean.to_excel(file_path, index=False)
        print(f"Added significance data to: {filename}")

# Add significance results to all graph_data files with preferred image and category  
graph_data_dir = "/home/daria/PROJECT/graph_data"
for filename in os.listdir(graph_data_dir):
    if filename.startswith('graph_') and filename.endswith('.xlsx'):
        file_path = os.path.join(graph_data_dir, filename)
        df_graph = pd.read_excel(file_path)
        
        # For probe data, use different categorization
        if 'probe' in filename.lower():
            # Add preferred image and brain region
            df_graph = pd.merge(
                df_graph,
                df_final[['subject_id', 'Neuron_ID', 'im_cat_1st', 'Signi']],
                on=['subject_id', 'Neuron_ID'],
                how='left'
            )
            df_graph.rename(columns={'im_cat_1st': 'preferred_image_id'}, inplace=True)
            
            # Add brain region
            df_graph = add_preferred_and_category(df_graph, df_final)
            df_graph = add_brain_region(df_graph, brain_regions_df)
            
            # Special categorization for probe data
            if 'Probe_Image_ID' in df_graph.columns:
                def categorize_probe_trial(row):
                    if row['Category'] == 'Preferred' and row['Probe_Image_ID'] == row['preferred_image_id']:
                        return 'Preferred Encoded'
                    elif row['Category'] == 'Preferred' and row['Probe_Image_ID'] != row['preferred_image_id']:
                        return 'Preferred Nonencoded'
                    elif row['Category'] == 'Non-Preferred' and row['Probe_Image_ID'] == row['preferred_image_id']:
                        return 'Nonpreferred Encoded'
                    else:
                        return 'Nonpreferred Nonencoded'
                
                # First add basic Category column based on encoding stimulus
                df_graph['Category'] = df_graph.apply(
                    lambda row: 'Preferred' if row['preferred_image_id'] == row.get('stimulus_index', row.get('image_id_enc1', None)) else 'Non-Preferred', 
                    axis=1
                )
                
                # Then add detailed probe categorization
                df_graph['Probe_Category'] = df_graph.apply(categorize_probe_trial, axis=1)
        
        else:
            # For non-probe data, use standard categorization
            df_graph = add_preferred_and_category(df_graph, df_final)
            df_graph = add_brain_region(df_graph, brain_regions_df)
        
        # Keep only essential columns
        essential_columns = [col for col in df_graph.columns if col not in ['p_val', 'CI', 'mean_1st', 'cat_1st', 'mean_2nd', 'cat_2nd']]
        df_graph = df_graph[essential_columns]
        
        # Save back
        df_graph.to_excel(file_path, index=False)
        print(f"Added significance data to: {filename}")

print("Significance data added to all files successfully!")

