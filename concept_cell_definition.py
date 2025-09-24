import pandas as pd
import numpy as np
from ast import literal_eval

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
input_path = "/Users/darikussovska/Desktop/PROJECT/clean_data/cleaned_Encoding1.xlsx"
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
df_final.to_excel('/Users/darikussovska/Desktop/PROJECT/Neuron_Check_Significant_All.xlsx', index=False)
print("Final file saved at: Neuron_Check_Significant_All.xlsx")

significant_neurons = df_final[df_final['Signi'] == 'Y']
print("Significant Neurons:")
print(significant_neurons)

# Add significance results to all clean_data files
clean_data_dir = "/Users/darikussovska/Desktop/PROJECT/clean_data"
for filename in os.listdir(clean_data_dir):
    if filename.startswith('cleaned_') and filename.endswith('.xlsx'):
        file_path = os.path.join(clean_data_dir, filename)
        df_clean = pd.read_excel(file_path)
        
        # Merge significance data
        df_clean = pd.merge(
            df_clean,
            df_final[['subject_id', 'Neuron_ID', 'Signi']],
            on=['subject_id', 'Neuron_ID'],
            how='left'
        )
        
        # Save back
        df_clean.to_excel(file_path, index=False)
        print(f"Added significance data to: {filename}")

# Add significance results to all graph_data files  
graph_data_dir = "/Users/darikussovska/Desktop/PROJECT/graph_data"
for filename in os.listdir(graph_data_dir):
    if filename.startswith('graph_') and filename.endswith('.xlsx'):
        file_path = os.path.join(graph_data_dir, filename)
        df_graph = pd.read_excel(file_path)
        
        # Merge significance data
        df_graph = pd.merge(
            df_graph,
            df_final[['subject_id', 'Neuron_ID', 'Signi']],
            on=['subject_id', 'Neuron_ID'],
            how='left'
        )
        
        # Save back
        df_graph.to_excel(file_path, index=False)
        print(f"Added significance data to: {filename}")

print("Significance data added to all files successfully!")
