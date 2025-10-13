import pandas as pd
import os

BASE_DIR = '/home/daria/PROJECT'
CLEAN_DATA_DIR = os.path.join(BASE_DIR, 'clean_data')
GRAPH_DATA_DIR = os.path.join(BASE_DIR, 'graph_data')

def get_preferred_stimulus_for_trial(enc1_ref, enc2_ref, enc3_ref, subject_id, trial_id, neuron_id):
    """Get the preferred stimulus for a trial by checking ALL encoding periods"""
    preferred_stimulus = None
    
    # Check Encoding 1
    match_enc1 = enc1_ref[
        (enc1_ref['subject_id'] == subject_id) & 
        (enc1_ref['trial_id'] == trial_id) & 
        (enc1_ref['Neuron_ID'] == neuron_id)
    ]
    if not match_enc1.empty and 'im_cat_1st' in match_enc1.columns:
        preferred_stimulus = match_enc1.iloc[0]['im_cat_1st']
    
    # If not found in Encoding 1, check Encoding 2
    if preferred_stimulus is None:
        match_enc2 = enc2_ref[
            (enc2_ref['subject_id'] == subject_id) & 
            (enc2_ref['trial_id'] == trial_id) & 
            (enc2_ref['Neuron_ID'] == neuron_id)
        ]
        if not match_enc2.empty and 'im_cat_1st' in match_enc2.columns:
            preferred_stimulus = match_enc2.iloc[0]['im_cat_1st']
    
    # If still not found, check Encoding 3
    if preferred_stimulus is None:
        match_enc3 = enc3_ref[
            (enc3_ref['subject_id'] == subject_id) & 
            (enc3_ref['trial_id'] == trial_id) & 
            (enc3_ref['Neuron_ID'] == neuron_id)
        ]
        if not match_enc3.empty and 'im_cat_1st' in match_enc3.columns:
            preferred_stimulus = match_enc3.iloc[0]['im_cat_1st']
    
    return preferred_stimulus

def add_category_to_encoding(df, encoding_name):
    """Add Category column to encoding files: Preferred if im_cat_1st == stimulus_index"""
    print(f"Processing {encoding_name}...")
    if 'im_cat_1st' in df.columns and 'stimulus_index' in df.columns:
        df['Category'] = df.apply(
            lambda row: 'Preferred' if row['im_cat_1st'] == row['stimulus_index'] else 'Non-Preferred', 
            axis=1
        )
        category_counts = df['Category'].value_counts()
        print(f"Category distribution for {encoding_name}: {category_counts.to_dict()}")
    else:
        print(f"Missing required columns in {encoding_name}")
    return df

def add_category_from_encoding(target_df, enc1_ref, enc2_ref, enc3_ref, file_type):
    """Add Category to non-encoding files using the preferred stimulus from ANY encoding period"""
    print(f"Adding Category to {file_type}...")
    
    target_df['Category'] = 'Unknown'
    target_df['im_cat_1st'] = None  # Add this column to store the preferred stimulus
    
    for idx, row in target_df.iterrows():
        subject_id = row['subject_id']
        trial_id = row['trial_id']
        neuron_id = row['Neuron_ID']
        current_stimulus = row.get('stimulus_index')
        
        # Get the preferred stimulus for this trial (from ANY encoding period)
        preferred_stimulus = get_preferred_stimulus_for_trial(
            enc1_ref, enc2_ref, enc3_ref, subject_id, trial_id, neuron_id
        )
        
        if preferred_stimulus is not None:
            # Store the preferred stimulus
            target_df.at[idx, 'im_cat_1st'] = preferred_stimulus
            
            # Categorize based on whether current stimulus matches preferred stimulus
            if current_stimulus == preferred_stimulus:
                target_df.at[idx, 'Category'] = 'Preferred'
            else:
                target_df.at[idx, 'Category'] = 'Non-Preferred'
    
    category_counts = target_df['Category'].value_counts()
    print(f"Category distribution for {file_type}: {category_counts.to_dict()}")
    return target_df

def add_probe_category(df):
    """Add Probe_Category column for probe files using consistent preferred stimulus"""
    print("Adding Probe_Category...")
    
    def categorize_probe(row):
        preferred = row['im_cat_1st']
        probe = row['Probe_Image_ID']
        category = row['Category']
        
        if pd.isna(preferred) or pd.isna(probe) or category == 'Unknown':
            return 'Unknown'
            
        if category == 'Preferred' and probe == preferred:
            return 'Preferred Encoded'
        elif category == 'Preferred' and probe != preferred:
            return 'Preferred Nonencoded'
        elif category == 'Non-Preferred' and probe == preferred:
            return 'Nonpreferred Encoded'
        elif category == 'Non-Preferred' and probe != preferred:
            return 'Nonpreferred Nonencoded'
        else:
            return 'Unknown'
    
    df['Probe_Category'] = df.apply(categorize_probe, axis=1)
    probe_category_counts = df['Probe_Category'].value_counts()
    print(f"Probe_Category distribution: {probe_category_counts.to_dict()}")
    return df

# Main execution
print("Loading encoding data for reference...")
enc1_data = pd.read_excel(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding1.xlsx'))
enc2_data = pd.read_excel(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding2.xlsx'))
enc3_data = pd.read_excel(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding3.xlsx'))

print("\n" + "="*50)
print("ADDING CATEGORY TO ENCODING FILES")
print("="*50)

# Add Category to CLEAN encoding files
enc1_data = add_category_to_encoding(enc1_data, "Encoding1 (clean)")
enc2_data = add_category_to_encoding(enc2_data, "Encoding2 (clean)")
enc3_data = add_category_to_encoding(enc3_data, "Encoding3 (clean)")

# Save the updated clean encoding files
enc1_data.to_excel(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding1.xlsx'), index=False)
enc2_data.to_excel(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding2.xlsx'), index=False)
enc3_data.to_excel(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding3.xlsx'), index=False)

# Add Category to GRAPH encoding files
print("\nProcessing graph encoding files...")
enc1_graph = pd.read_excel(os.path.join(GRAPH_DATA_DIR, 'graph_encoding1.xlsx'))
enc2_graph = pd.read_excel(os.path.join(GRAPH_DATA_DIR, 'graph_encoding2.xlsx'))
enc3_graph = pd.read_excel(os.path.join(GRAPH_DATA_DIR, 'graph_encoding3.xlsx'))

enc1_graph = add_category_to_encoding(enc1_graph, "Encoding1 (graph)")
enc2_graph = add_category_to_encoding(enc2_graph, "Encoding2 (graph)")
enc3_graph = add_category_to_encoding(enc3_graph, "Encoding3 (graph)")

# Save the updated graph encoding files
enc1_graph.to_excel(os.path.join(GRAPH_DATA_DIR, 'graph_encoding1.xlsx'), index=False)
enc2_graph.to_excel(os.path.join(GRAPH_DATA_DIR, 'graph_encoding2.xlsx'), index=False)
enc3_graph.to_excel(os.path.join(GRAPH_DATA_DIR, 'graph_encoding3.xlsx'), index=False)

print("\n" + "="*50)
print("PROCESSING NON-ENCODING FILES")
print("="*50)

# Process Delay files
delay_clean_path = os.path.join(CLEAN_DATA_DIR, 'cleaned_Delay.xlsx')
delay_graph_path = os.path.join(GRAPH_DATA_DIR, 'graph_delay.xlsx')

if os.path.exists(delay_clean_path):
    print("\nProcessing Delay files...")
    
    # Clean data
    delay_clean = pd.read_excel(delay_clean_path)
    delay_clean = add_category_from_encoding(delay_clean, enc1_data, enc2_data, enc3_data, "Delay (clean)")
    delay_clean.to_excel(delay_clean_path, index=False)
    
    # Graph data
    delay_graph = pd.read_excel(delay_graph_path)
    delay_graph = add_category_from_encoding(delay_graph, enc1_graph, enc2_graph, enc3_graph, "Delay (graph)")
    delay_graph.to_excel(delay_graph_path, index=False)
    
    print("Delay files updated")

# Process Probe files
probe_clean_path = os.path.join(CLEAN_DATA_DIR, 'cleaned_Probe.xlsx')
probe_graph_path = os.path.join(GRAPH_DATA_DIR, 'graph_probe.xlsx')

if os.path.exists(probe_clean_path):
    print("\nProcessing Probe files...")
    
    # Clean data
    probe_clean = pd.read_excel(probe_clean_path)
    probe_clean = add_category_from_encoding(probe_clean, enc1_data, enc2_data, enc3_data, "Probe (clean)")
    probe_clean = add_probe_category(probe_clean)
    probe_clean.to_excel(probe_clean_path, index=False)
    
    # Graph data
    probe_graph = pd.read_excel(probe_graph_path)
    probe_graph = add_category_from_encoding(probe_graph, enc1_graph, enc2_graph, enc3_graph, "Probe (graph)")
    probe_graph = add_probe_category(probe_graph)
    probe_graph.to_excel(probe_graph_path, index=False)
    
    print("Probe files updated")

# Process Fixation files (if they exist)
fixation_clean_path = os.path.join(CLEAN_DATA_DIR, 'cleaned_Fixation.xlsx')
fixation_graph_path = os.path.join(GRAPH_DATA_DIR, 'graph_fixation.xlsx')

if os.path.exists(fixation_clean_path):
    print("\nProcessing Fixation files...")
    
    # Clean data
    fixation_clean = pd.read_excel(fixation_clean_path)
    fixation_clean = add_category_from_encoding(fixation_clean, enc1_data, enc2_data, enc3_data, "Fixation (clean)")
    fixation_clean.to_excel(fixation_clean_path, index=False)
    
    # Graph data
    fixation_graph = pd.read_excel(fixation_graph_path)
    fixation_graph = add_category_from_encoding(fixation_graph, enc1_graph, enc2_graph, enc3_graph, "Fixation (graph)")
    fixation_graph.to_excel(fixation_graph_path, index=False)
    
    print("Fixation files updated")

print("\n" + "="*50)
print("PROCESSING COMPLETED!")
print("="*50)
