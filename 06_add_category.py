import pandas as pd
import os

BASE_DIR = './'
CLEAN_DATA_DIR = os.path.join(BASE_DIR, 'clean_data')
GRAPH_DATA_DIR = os.path.join(BASE_DIR, 'graph_data')

# Load the encoding data first to use as reference
print("Loading encoding data for reference...")
enc1_data = pd.read_feather(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding1.feather'))
enc2_data = pd.read_feather(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding2.feather'))
enc3_data = pd.read_feather(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding3.feather'))

# Add Category to encoding files (both clean and graph)
print("\n" + "="*50)
print("ADDING CATEGORY TO ENCODING FILES")
print("="*50)

def add_category_to_encoding(df, encoding_name):
    """Add Category column to encoding files: Preferred if im_cat_1st == stimulus_index"""
    print(f"Processing {encoding_name}...")
    if 'im_cat_1st' in df.columns and 'stimulus_index' in df.columns:
        # Create Category column based on your logic
        df['Category'] = df.apply(
            lambda row: 'Preferred' if row['im_cat_1st'] == row['stimulus_index'] else 'Non-Preferred', 
            axis=1
        )
        # Show results
        category_counts = df['Category'].value_counts()
        print(f"Category distribution for {encoding_name}: {category_counts.to_dict()}")
    else:
        print(f"Missing required columns in {encoding_name}")
    return df

# Add Category to CLEAN encoding files
enc1_data = add_category_to_encoding(enc1_data, "Encoding1 (clean)")
enc2_data = add_category_to_encoding(enc2_data, "Encoding2 (clean)")
enc3_data = add_category_to_encoding(enc3_data, "Encoding3 (clean)")

# Save the updated clean encoding files
enc1_data.to_feather(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding1.feather'))
enc2_data.to_feather(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding2.feather'))
enc3_data.to_feather(os.path.join(CLEAN_DATA_DIR, 'cleaned_Encoding3.feather'))

# Add Category to GRAPH encoding files
print("\nProcessing graph encoding files...")
enc1_graph = pd.read_feather(os.path.join(GRAPH_DATA_DIR, 'graph_encoding1.feather'))
enc2_graph = pd.read_feather(os.path.join(GRAPH_DATA_DIR, 'graph_encoding2.feather'))
enc3_graph = pd.read_feather(os.path.join(GRAPH_DATA_DIR, 'graph_encoding3.feather'))

enc1_graph = add_category_to_encoding(enc1_graph, "Encoding1 (graph)")
enc2_graph = add_category_to_encoding(enc2_graph, "Encoding2 (graph)")
enc3_graph = add_category_to_encoding(enc3_graph, "Encoding3 (graph)")

# Save the updated graph encoding files
enc1_graph.to_feather(os.path.join(GRAPH_DATA_DIR, 'graph_encoding1.feather'))
enc2_graph.to_feather(os.path.join(GRAPH_DATA_DIR, 'graph_encoding2.feather'))
enc3_graph.to_feather(os.path.join(GRAPH_DATA_DIR, 'graph_encoding3.feather'))

print("All encoding files updated with Category")

# Add Category to non-encoding files (Delay, Fixation, Probe)
def add_category_from_encoding(target_df, enc1_ref, enc2_ref, enc3_ref, file_type):
    """Add Category to non-encoding files by matching with the correct encoding period"""
    print(f"Adding Category to {file_type}...")
    
    # Create a new Category column
    target_df['Category'] = 'Unknown'  # Default value
    
    # Process each row based on num_images_presented
    for idx, row in target_df.iterrows():
        num_images = row.get('num_images_presented')
        if pd.isna(num_images):
            continue
            
        try:
            num_images = int(num_images)
        except (ValueError, TypeError):
            continue
            
        subject_id = row['subject_id']
        trial_id = row['trial_id']
        neuron_id = row['Neuron_ID']
        
        if num_images == 1:
            # Look in Encoding1
            match = enc1_ref[
                (enc1_ref['subject_id'] == subject_id) & 
                (enc1_ref['trial_id'] == trial_id) & 
                (enc1_ref['Neuron_ID'] == neuron_id)
            ]
            if not match.empty and 'Category' in match.columns:
                target_df.at[idx, 'Category'] = match.iloc[0]['Category']
                
        elif num_images == 2:
            # For load 2, check Encoding1 AND Encoding2 - if ANY have "Preferred", use "Preferred"
            is_preferred = False
            
            # Check Encoding1
            match_enc1 = enc1_ref[
                (enc1_ref['subject_id'] == subject_id) & 
                (enc1_ref['trial_id'] == trial_id) & 
                (enc1_ref['Neuron_ID'] == neuron_id)
            ]
            if not match_enc1.empty and 'Category' in match_enc1.columns:
                if match_enc1.iloc[0]['Category'] == 'Preferred':
                    is_preferred = True
            
            # Check Encoding2
            if not is_preferred:
                match_enc2 = enc2_ref[
                    (enc2_ref['subject_id'] == subject_id) & 
                    (enc2_ref['trial_id'] == trial_id) & 
                    (enc2_ref['Neuron_ID'] == neuron_id)
                ]
                if not match_enc2.empty and 'Category' in match_enc2.columns:
                    if match_enc2.iloc[0]['Category'] == 'Preferred':
                        is_preferred = True
            
            # Set category based on whether ANY encoding period has "Preferred"
            if is_preferred:
                target_df.at[idx, 'Category'] = 'Preferred'
            else:
                # If we found matches but none were "Preferred", set to "Non-Preferred"
                target_df.at[idx, 'Category'] = 'Non-Preferred'
                
        elif num_images == 3:
            # For load 3, check ALL encoding periods - if ANY have "Preferred", use "Preferred"
            is_preferred = False
            
            # Check Encoding1
            match_enc1 = enc1_ref[
                (enc1_ref['subject_id'] == subject_id) & 
                (enc1_ref['trial_id'] == trial_id) & 
                (enc1_ref['Neuron_ID'] == neuron_id)
            ]
            if not match_enc1.empty and 'Category' in match_enc1.columns:
                if match_enc1.iloc[0]['Category'] == 'Preferred':
                    is_preferred = True
            
            # Check Encoding2
            if not is_preferred:
                match_enc2 = enc2_ref[
                    (enc2_ref['subject_id'] == subject_id) & 
                    (enc2_ref['trial_id'] == trial_id) & 
                    (enc2_ref['Neuron_ID'] == neuron_id)
                ]
                if not match_enc2.empty and 'Category' in match_enc2.columns:
                    if match_enc2.iloc[0]['Category'] == 'Preferred':
                        is_preferred = True
            
            # Check Encoding3
            if not is_preferred:
                match_enc3 = enc3_ref[
                    (enc3_ref['subject_id'] == subject_id) & 
                    (enc3_ref['trial_id'] == trial_id) & 
                    (enc3_ref['Neuron_ID'] == neuron_id)
                ]
                if not match_enc3.empty and 'Category' in match_enc3.columns:
                    if match_enc3.iloc[0]['Category'] == 'Preferred':
                        is_preferred = True
            
            # Set category based on whether ANY encoding period has "Preferred"
            if is_preferred:
                target_df.at[idx, 'Category'] = 'Preferred'
            else:
                # If we found matches but none were "Preferred", set to "Non-Preferred"
                target_df.at[idx, 'Category'] = 'Non-Preferred'
    
    category_counts = target_df['Category'].value_counts()
    print(f"Category distribution for {file_type}: {category_counts.to_dict()}")
    return target_df

# Add Probe_Category for probe files
def add_probe_category(df):
    """Add Probe_Category column for probe files using your exact logic"""
    print("Adding Probe_Category...")
    
    def categorize_probe(row):
        preferred = row['im_cat_1st']
        probe = row['Probe_Image_ID']
        category = row['Category']
        
        # Your exact logic
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

# Process Delay and Probe files (both clean and graph)
print("\n" + "="*50)
print("PROCESSING NON-ENCODING FILES")
print("="*50)

# Process Delay files
delay_clean_path = os.path.join(CLEAN_DATA_DIR, 'cleaned_Delay.feather')
delay_graph_path = os.path.join(GRAPH_DATA_DIR, 'graph_delay.feather')

if os.path.exists(delay_clean_path):
    print("\nProcessing Delay files...")
    
    # Clean data
    delay_clean = pd.read_feather(delay_clean_path)
    delay_clean = add_category_from_encoding(delay_clean, enc1_data, enc2_data, enc3_data, "Delay (clean)")
    delay_clean.to_feather(delay_clean_path)
    
    # Graph data
    delay_graph = pd.read_feather(delay_graph_path)
    delay_graph = add_category_from_encoding(delay_graph, enc1_graph, enc2_graph, enc3_graph, "Delay (graph)")
    delay_graph.to_feather(delay_graph_path)
    
    print("Delay files updated")

# Process Probe files
probe_clean_path = os.path.join(CLEAN_DATA_DIR, 'cleaned_Probe.feather')
probe_graph_path = os.path.join(GRAPH_DATA_DIR, 'graph_probe.feather')

if os.path.exists(probe_clean_path):
    print("\nProcessing Probe files...")
    
    # Clean data
    probe_clean = pd.read_feather(probe_clean_path)
    probe_clean = add_category_from_encoding(probe_clean, enc1_data, enc2_data, enc3_data, "Probe (clean)")
    probe_clean = add_probe_category(probe_clean)
    probe_clean.to_feather(probe_clean_path)
    
    # Graph data
    probe_graph = pd.read_feather(probe_graph_path)
    probe_graph = add_category_from_encoding(probe_graph, enc1_graph, enc2_graph, enc3_graph, "Probe (graph)")
    probe_graph = add_probe_category(probe_graph)
    probe_graph.to_feather(probe_graph_path)
    
    print("Probe files updated")

# Process Fixation files (if they exist)
fixation_clean_path = os.path.join(CLEAN_DATA_DIR, 'cleaned_Fixation.feather')
fixation_graph_path = os.path.join(GRAPH_DATA_DIR, 'graph_fixation.feather')

if os.path.exists(fixation_clean_path):
    print("\nProcessing Fixation files...")
    
    # Clean data
    fixation_clean = pd.read_feather(fixation_clean_path)
    fixation_clean = add_category_from_encoding(fixation_clean, enc1_data, enc2_data, enc3_data, "Fixation (clean)")
    fixation_clean.to_feather(fixation_clean_path)
    
    # Graph data
    fixation_graph = pd.read_feather(fixation_graph_path)
    fixation_graph = add_category_from_encoding(fixation_graph, enc1_graph, enc2_graph, enc3_graph, "Fixation (graph)")
    fixation_graph.to_feather(fixation_graph_path)
    
    print("Fixation files updated")

print("\n" + "="*50)
print("PROCESSING COMPLETED!")
print("="*50)
