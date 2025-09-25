import pandas as pd
import numpy as np
from ast import literal_eval
import os 
import matplotlib.pyplot as plt

def compute_firing_rate(df, start_time=0.2, end_time=1.0):
    """
    Computes firing rate for each (subject_id, trial_id, Neuron_ID) using Standardized_Spikes.
    Only spikes between 0.2s and 1.0s are considered.
    """
    spike_rates = []

    for _, row in df.iterrows():
        spikes = row['Standardized_Spikes']
        spike_times = literal_eval(str(spikes)) if isinstance(spikes, str) else spikes
        spike_times = spike_times if isinstance(spike_times, list) else []

        spikes_in_window = [s for s in spike_times if start_time <= s <= end_time]
        firing_rate = len(spikes_in_window) / (end_time - start_time)
        spike_rates.append(firing_rate)

    df['Spike_Rate_new'] = spike_rates
    return df

# Load input Excel file
input_path = "/home/daria/PROJECT/clean_data/cleaned_Encoding1.xlsx"
df = pd.read_excel(input_path)

# Compute firing rates
df = compute_firing_rate(df)

def extract_top_two_categories(df):
    def get_top_two(subject_id, neuron_id, df):
        cat_1st, cat_2nd = [], []
        mean_1st = mean_2nd = 0
        im_cat_1st = im_cat_2nd = ''

        for image in df['stimulus_index'].dropna().unique():
            image_str = str(image)
            mask = (
                (df['subject_id'] == subject_id) &
                (df['Neuron_ID'] == neuron_id) &
                (df['stimulus_index'].astype(str).str.contains(image_str))
            )
            rates = df.loc[mask, 'Spike_Rate_new']
            mean_rate = rates.mean()

            if mean_rate > mean_1st:
                mean_2nd, cat_2nd, im_cat_2nd = mean_1st, cat_1st, im_cat_1st
                mean_1st, cat_1st, im_cat_1st = mean_rate, rates.tolist(), image
            elif mean_1st >= mean_rate > mean_2nd:
                mean_2nd, cat_2nd, im_cat_2nd = mean_rate, rates.tolist(), image

        return im_cat_1st, mean_1st, cat_1st, im_cat_2nd, mean_2nd, cat_2nd

    results = []
    for subj in df['subject_id'].unique():
        df_subj = df[df['subject_id'] == subj]
        for neuron in df_subj['Neuron_ID'].unique():
            res = get_top_two(subj, neuron, df)
            results.append({
                'subject_id': subj,
                'Neuron_ID': neuron,
                'im_cat_1st': res[0],
                'mean_1st': res[1],
                'cat_1st': res[2],
                'im_cat_2nd': res[3],
                'mean_2nd': res[4],
                'cat_2nd': res[5]
            })
    return pd.DataFrame(results)

df_top_2 = extract_top_two_categories(df)

def resampling(arr1, arr2, iteration=1000):
    min_len = min(len(arr1), len(arr2))
    diffs = []

    for _ in range(iteration):
        sample1 = np.random.choice(arr1, min_len, replace=True)
        sample2 = np.random.choice(arr2, min_len, replace=True)
        diffs.append(np.mean(sample1) - np.mean(sample2))

    CI = np.percentile(diffs, [2.5, 97.5])
    m1 = 2 * sum(i > 0 for i in diffs) / iteration
    m2 = 1 - sum(i < 0 for i in diffs) / iteration
    m3 = 2 * sum(i < 0 for i in diffs) / iteration
    m4 = 1 - sum(i > 0 for i in diffs) / iteration
    p_val = min(m1, m2, m3, m4)
    return p_val, CI

def run_significance_test(df, iteration=1000):
    def str_to_float_list(x): return [float(i) for i in x]

    p_vals, CIs, signs = [], [], []

    for _, row in df.iterrows():
        arr1 = str_to_float_list(row['cat_1st'])
        arr2 = str_to_float_list(row['cat_2nd'])
        p, ci = resampling(arr1, arr2, iteration)

        p_vals.append(p)
        CIs.append(ci)
        signs.append('Y' if p < 0.05 and len(arr1) > 1 and len(arr2) > 1 else 'N')

    df['p_val'] = p_vals
    df['CI'] = CIs
    df['Signi'] = signs
    return df

df_final = run_significance_test(df_top_2)
df_final.to_excel('/home/daria/PROJECT/Neuron_Check_Significant_All.xlsx', index=False)
print("Final file saved at: Neuron_Check_Significant_All.xlsx")

significant_neurons = df_final[df_final['Signi'] == 'Y']
print("Significant Neurons:")
print(significant_neurons)

# Load brain regions data
brain_regions_path = "/Users/darikussovska/Desktop/PROJECT/merged_significant_neurons_with_brain_regions.xlsx"
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
    
    # Add category column (Preferred/Non-Preferred)
    df['Category'] = df.apply(
        lambda row: 'Preferred' if row['preferred_image_id'] == row['stimulus_index'] else 'Non-Preferred', 
        axis=1
    )
    
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
clean_data_dir = "/Users/darikussovska/Desktop/PROJECT/clean_data"
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

def plot_example_concept_cells(df_original, df_sig, save_dir, max_plots=2):
    """
    Plot and save bar plots of concept cells based on their firing rates across categories.
    Only 'Signi' == 'Y' neurons will be plotted, and up to max_plots will be saved.
    """

    # Filter significant neurons only
    significant_neurons = df_sig[df_sig['Signi'] == 'Y']
    saved = 0

    os.makedirs(save_dir, exist_ok=True)

    for idx, row in significant_neurons.iterrows():
        subject_id = row['subject_id']
        neuron_id = row['Neuron_ID']
        max_image_id = row['im_cat_1st']

        # Filter for the neuron's data
        neuron_data = df_original[
            (df_original['subject_id'] == subject_id) &
            (df_original['Neuron_ID'] == neuron_id)
        ]

        # Group by image ID and compute stats
        grouped = neuron_data.groupby('stimulus_index')['Spike_Rate_new']
        mean_rates = grouped.mean()
        sem_rates = grouped.sem()

        # Plot
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.bar(mean_rates.index, mean_rates.values, yerr=sem_rates.values, capsize=5, color='blue', alpha=0.7)

        if max_image_id in mean_rates.index:
            ax.bar([max_image_id], [mean_rates[max_image_id]], yerr=[sem_rates[max_image_id]],
                   capsize=5, color='red', alpha=0.9)

        ax.set_title(f'Subject {subject_id} - Neuron {neuron_id} - Category Selectivity')
        ax.set_xlabel('Image ID')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.grid(True)
        plt.xticks(rotation=90)

        # Save only a few .eps examples
        if saved < max_plots:
            filename = os.path.join(save_dir, f'concept_cell_{saved+1}.eps')
            plt.savefig(filename, format='eps')
            print(f"Saved: {filename}")
            saved += 1

        plt.show()

        if saved >= max_plots:
            break

plot_example_concept_cells(
    df_original=df,
    df_sig=df_final,
    save_dir='/home/daria/PROJECT',
    max_plots=2
)
