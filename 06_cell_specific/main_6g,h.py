import numpy as np
import ast
import matplotlib.pyplot as plt

def generate_filtered_spike_trains(df_encoding, df_delay, cell_type_map,
                                   encoding_col='Standardized_Spikes',
                                   delay_col='Standardized_Spikes',
                                   trial_col='trial_id',
                                   encoding_duration=1,
                                   delay_duration=2.5,
                                   preferred_col='Category',
                                   preferred_label='Preferred'):
    """
    Returns:
    spike_dict[neuron_uid] = {
        'subject_id': subject_id,
        'cell_type': 'PY' or 'IN',
        'encoding': np.array([...]),
        'delay':    np.array([...])
    }
    """
    spike_dict = {}

    for (subject_id, neuron_id) in cell_type_map.keys():
        neuron_uid = f"{subject_id}_{neuron_id}"
        cell_type = cell_type_map[(subject_id, neuron_id)]

        spikes_encoding = []
        spikes_delay = []

        # ---------- Encoding ----------
        df_enc = df_encoding[
            (df_encoding['subject_id'] == subject_id) &
            (df_encoding['Neuron_ID_3'] == neuron_id) &
            (df_encoding[preferred_col] == preferred_label)  # <-- Filter Preferred
        ]
        trials_enc = sorted(df_enc[trial_col].unique())
        trial_index_map_enc = {tid: i for i, tid in enumerate(trials_enc)}

        for trial_id in trials_enc:
            i = trial_index_map_enc[trial_id]
            df_trial = df_enc[df_enc[trial_col] == trial_id]
            for spikes_str in df_trial[encoding_col].dropna():
                if spikes_str != '[]':
                    spikes = np.array(ast.literal_eval(spikes_str), dtype=float)
                    spikes = spikes[(spikes >= 0) & (spikes <= encoding_duration)]
                    if spikes.size:
                        spikes += i * encoding_duration
                        spikes_encoding.extend(spikes)

        # ---------- Delay ----------
        df_del = df_delay[
            (df_delay['subject_id'] == subject_id) &
            (df_delay['Neuron_ID_3'] == neuron_id) &
            (df_delay[preferred_col] == preferred_label)  # <-- Filter Preferred
        ]
        trials_del = sorted(df_del[trial_col].unique())
        trial_index_map_del = {tid: i for i, tid in enumerate(trials_del)}

        for trial_id in trials_del:
            i = trial_index_map_del[trial_id]
            df_trial = df_del[df_del[trial_col] == trial_id]
            for spikes_str in df_trial[delay_col].dropna():
                if spikes_str != '[]':
                    spikes = np.array(ast.literal_eval(spikes_str), dtype=float)
                    spikes = spikes[(spikes >= 0) & (spikes <= delay_duration)]
                    if spikes.size:
                        spikes += i * delay_duration
                        spikes_delay.extend(spikes)

        spike_dict[neuron_uid] = {
            'subject_id': subject_id,
            'cell_type': cell_type,
            'encoding': np.sort(np.array(spikes_encoding)) if spikes_encoding else np.array([]),
            'delay':    np.sort(np.array(spikes_delay))    if spikes_delay    else np.array([])
        }

    return spike_dict

import pandas as pd
import numpy as np 

# Load your metadata and spikes
df_meta = pd.read_excel('../Clustering_3D.xlsx')
df_encoding = pd.read_excel('../clean_data/graph_encoding1.xlsx')
df_delay    = pd.read_excel('../clean_data/graph_dela.xlsx')

# Filter metadata to significant neurons
df_meta = df_meta[df_meta['Signi'] == 'Y']

# Create (subject_id, neuron_id) → Cell Type mapping
cell_type_map = {
    (row['subject_id'], row['Neuron_ID_3']): row['Cell_Type_New']
    for _, row in df_meta.iterrows()
}

# Filter spike files to only significant neurons
valid_ids = set(cell_type_map.keys())
df_encoding = df_encoding[df_encoding.apply(lambda row: (row['subject_id'], row['Neuron_ID_3']) in valid_ids, axis=1)]
df_delay    = df_delay   [df_delay   .apply(lambda row: (row['subject_id'], row['Neuron_ID_3']) in valid_ids, axis=1)]

# Run the function
spike_dict = generate_filtered_spike_trains(df_encoding, df_delay, cell_type_map)

# Inspect
print(f"\nExtracted {len(spike_dict)} neurons.")
for neuron_uid, entry in spike_dict.items():
    print(f"\nNeuron {neuron_uid} — Subject {entry['subject_id']} ({entry['cell_type']})")
    print("Encoding (first 5 spikes):", entry['encoding'][:50])
    print("Delay (first 5 spikes):", entry['delay'][:50])
    break

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

def compute_cv2(spike_times):
    if len(spike_times) < 3:
        return np.nan
    isi = np.diff(spike_times)
    cv2 = 2 * np.abs(np.diff(isi)) / (isi[:-1] + isi[1:])
    return np.mean(cv2)

# Collect CV2s for neurons that have values for both Encoding and Delay
cv2_data = {'PY': {'encoding': [], 'delay': []},
            'IN': {'encoding': [], 'delay': []}}

for neuron_uid, entry in spike_dict.items():
    cell_type = entry['cell_type']
    cv2_enc = compute_cv2(entry['encoding'])
    cv2_del = compute_cv2(entry['delay'])

    if not np.isnan(cv2_enc) and not np.isnan(cv2_del):
        cv2_data[cell_type]['encoding'].append(cv2_enc)
        cv2_data[cell_type]['delay'].append(cv2_del)

# Perform Wilcoxon signed-rank test
p_py = wilcoxon(cv2_data['PY']['encoding'], cv2_data['PY']['delay']).pvalue if len(cv2_data['PY']['encoding']) > 0 else np.nan
p_in = wilcoxon(cv2_data['IN']['encoding'], cv2_data['IN']['delay']).pvalue if len(cv2_data['IN']['encoding']) > 0 else np.nan

# Prepare means and stds
labels = ['Encoding', 'Delay']
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(6, 5))

py_means = [np.mean(cv2_data['PY']['encoding']), np.mean(cv2_data['PY']['delay'])]
py_stds  = [np.std(cv2_data['PY']['encoding']),  np.std(cv2_data['PY']['delay'])]

in_means = [np.mean(cv2_data['IN']['encoding']), np.mean(cv2_data['IN']['delay'])]
in_stds  = [np.std(cv2_data['IN']['encoding']),  np.std(cv2_data['IN']['delay'])]

# Plot bars
ax.bar(x - width/2, py_means, width, yerr=py_stds, label='PY', color='skyblue', capsize=5)
ax.bar(x + width/2, in_means, width, yerr=in_stds, label='IN', color='salmon', capsize=5)

# Add significance stars
def add_sig_star(p, xpos1, xpos2, y, height=0.01):
    if p < 0.001:
        label = '***'
    elif p < 0.01:
        label = '**'
    elif p < 0.05:
        label = '*'
    else:
        return
    ax.plot([xpos1, xpos1, xpos2, xpos2], [y, y+height, y+height, y], lw=1.2, c='black')
    ax.text((xpos1 + xpos2)/2, y + height + 0.005, label, ha='center', va='bottom', fontsize=12)

# Add stars
max_py = max(py_means) + max(py_stds)
max_in = max(in_means) + max(in_stds)

add_sig_star(p_py, x[0] - width/2, x[1] - width/2, max_py)
add_sig_star(p_in, x[0] + width/2, x[1] + width/2, max_in)

ax.set_ylabel('CV2')
ax.set_title('CV2 in Encoding vs Delay')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
save_path = "./main_6gh.eps"
plt.savefig(save_path, format='eps', dpi=300)
plt.show()

# Print p-values
print(f"Wilcoxon test PY: p = {p_py:.4f}")
print(f"Wilcoxon test IN: p = {p_in:.4f}")
