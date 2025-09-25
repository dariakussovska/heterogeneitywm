# Load brain regions data
brain_regions_path = "/home/daria/PROJECT/all_neuron_brain_regions_cleaned.xlsx"
brain_regions_df = pd.read_excel(brain_regions_path)

# Function to add preferred image ID and categorize trials
def add_preferred_and_category(df, significant_neurons_df):
    # Merge with significant neurons to get preferred image
    df = pd.merge(
        df,
        significant_neurons_df[['subject_id', 'Neuron_ID', 'im_cat_1st', 'Signi']],
        on=['subject_id', 'Neuron_ID'],
        how='left'
    )
    
    # Rename columns
    df.rename(columns={'im_cat_1st': 'preferred_image_id'}, inplace=True)
    
    # Add category column (Preferred/Non-Preferred) - handle both column names
    def get_category(row):
        # Try both possible column names for the stimulus index
        if 'stimulus_index_enc1' in row and pd.notna(row['stimulus_index_enc1']):
            if row['preferred_image_id'] == row['stimulus_index_enc1']:
                return 'Preferred'
            else:
                return 'Non-Preferred'
        elif 'stimulus_index' in row and pd.notna(row['stimulus_index']):
            if row['preferred_image_id'] == row['stimulus_index']:
                return 'Preferred'
            else:
                return 'Non-Preferred'
        else:
            return 'Unknown'
    
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
    if filename.startswith('clean_') and filename.endswith('.xlsx'):
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

