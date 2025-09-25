import os 
import pandas as pd
import numpy as np 

# Load brain regions data
brain_regions_path = "/home/daria/PROJECT/all_neuron_brain_regions_cleaned.xlsx"
brain_regions_df = pd.read_excel(brain_regions_path)

def add_preferred_and_category(df, significant_neurons_df):
    # normalize types for a clean merge
    df['subject_id'] = df['subject_id'].astype(str).str.strip()
    df['Neuron_ID'] = df['Neuron_ID'].astype(int)
    significant_neurons_df = significant_neurons_df.copy()
    significant_neurons_df['subject_id'] = significant_neurons_df['subject_id'].astype(str).str.strip()
    significant_neurons_df['Neuron_ID'] = significant_neurons_df['Neuron_ID'].astype(int)

    # merge preferred image + significance
    df = pd.merge(
        df,
        significant_neurons_df[['subject_id','Neuron_ID','im_cat_1st','Signi']],
        on=['subject_id','Neuron_ID'],
        how='left'
    ).rename(columns={'im_cat_1st':'preferred_image_id'})

    # find the best available stimulus column for this file
    stim_cols = [
        'stimulus_index_enc1', 'stimulus_index_enc2', 'stimulus_index_enc3',
        'stimulus_index', 'image_id_enc1', 'Image_ID', 'image_id'
    ]
    present_stim_col = next((c for c in stim_cols if c in df.columns), None)

    def get_category(row):
        if pd.isna(row.get('preferred_image_id', np.nan)):
            return 'Unknown'  # neuron not significant / no preferred image
        if present_stim_col is None or pd.isna(row.get(present_stim_col, np.nan)):
            return 'No-Image'  # e.g., Delay (no presented image)
        return 'Preferred' if row['preferred_image_id'] == row[present_stim_col] else 'Non-Preferred'

    df['Category'] = df.apply(get_category, axis=1)
    return df

# Function to add brain region information
def add_brain_region(df, brain_regions_df):
    df = pd.merge(
        df,
        brain_regions_df[['subject_id', 'Neuron_ID', 'Location']],
        on=['subject_id', 'Neuron_ID'],
        how='left'
    )
    return df

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

