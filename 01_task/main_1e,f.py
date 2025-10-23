import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import PoissonRegressor
from tqdm import tqdm
import ast

trial_info = pd.read_excel('../data/new_trial_info.xlsx')
subject_trials = trial_info[trial_info['subject_id'] == 1][['new_trial_id', 'num_images_presented', 'stimulus_index', 'RT', 'response_accuracy']]
y_matrix = subject_trials

# 1) Load Your Main Data

df_delay_filtered = pd.read_excel('../graph_data/graph_delay.xlsx')
y_matrix = y_matrix.reset_index(drop=True)

# 2) Load Neuron Locations Data (for location-based filtering)

df_neuron_locations = pd.read_excel('../data/all_neuron_brain_regions_cleaned.xlsx')

# Example list of desired locations for filtering
desired_locations = ['amygdala_left', 'amygdala_right', 'hippocampus_left', 'hippocampus_right', 'pre_supplementary_motor_area_right', 'pre_supplementary_motor_area_left',
                    'ventral_medial_prefrontal_cortex_right', 'ventral_medial_prefrontal_cortex_left', 'dorsal_anterior_cingulate_cortex_right', 'dorsal_anterior_cingulate_cortex_left']


# We'll compute the set of neurons that are in these locations:

neurons_in_desired_locations = df_neuron_locations[
    df_neuron_locations['Location'].isin(desired_locations)
]['Neuron_ID_3'].unique()

# Design matrix
final_neurons = df_delay_filtered['Neuron_ID_3'].unique()
final_neurons = sorted(final_neurons)

trial_count = y_matrix.shape[0]
neuron_count = len(final_neurons)
design_matrix = np.empty((trial_count, neuron_count), dtype=object)

def parse_spike_times(spike_str):
    try:
        return ast.literal_eval(spike_str)
    except (ValueError, SyntaxError):
        return []

for trial_index, trial_row in y_matrix.iterrows():
    trial_id = trial_row['new_trial_id']

    for neuron_index, neuron_id in enumerate(final_neurons):
        spikes_for_this_trial = []

        df_subset = df_delay_filtered[
            (df_delay_filtered['new_trial_id'] == trial_id) &
            (df_delay_filtered['Neuron_ID_3'] == neuron_id)
        ]
        if not df_subset.empty:
            spike_times_str = df_subset['Standardized_Spikes'].values[0]
            spikes_for_this_trial.extend(parse_spike_times(spike_times_str))

        design_matrix[trial_index, neuron_index] = np.array(spikes_for_this_trial)

df_design_matrix = pd.DataFrame(
    design_matrix,
    columns=[f'Neuron_{int(nid)}' for nid in final_neurons]
)

delay_start = 0.0
delay_end   = 2.5   # set to your exact window

def count_spikes_in_window(spike_times, start, end):
    if spike_times is None or len(spike_times) == 0:
        return 0
    st = np.asarray(spike_times, dtype=float)
    st = st[np.isfinite(st)]
    return np.count_nonzero((st >= start) & (st < end))

counts_matrix = np.zeros((trial_count, neuron_count), dtype=int)

for trial_index in range(trial_count):
    for neuron_index, neuron_id in enumerate(final_neurons):
        neuron_spikes = design_matrix[trial_index, neuron_index]  
        counts_matrix[trial_index, neuron_index] = count_spikes_in_window(
            neuron_spikes, delay_start, delay_end
        )

# Combine hemispheric labels into broader region labels
df_neuron_locations['Region_Label'] = df_neuron_locations['Location'].replace({
   # 'PY': 'PY', 
   # 'IN': 'IN'
     'amygdala_left': 'amygdala',
     'amygdala_right': 'amygdala',
     'hippocampus_left': 'hippocampus',
     'hippocampus_right': 'hippocampus',
     'pre_supplementary_motor_area_right': 'preSMA',
     'pre_supplementary_motor_area_left': 'preSMA',
     'ventral_medial_prefrontal_cortex_right': 'vmPFC',
     'ventral_medial_prefrontal_cortex_left': 'vmPFC',
     'dorsal_anterior_cingulate_cortex_right': 'dACC',
     'dorsal_anterior_cingulate_cortex_left': 'dACC'
})

stripped_ids = [str(nid).replace('Neuron_', '') for nid in df_design_matrix.columns]
neuron_to_region = dict(zip(df_neuron_locations['Neuron_ID_3'].astype(str), df_neuron_locations['Region_Label']))
region_labels = [neuron_to_region.get(nid, 'Unknown') for nid in stripped_ids]

SEED = 17
rng = np.random.default_rng(SEED)

n_iterations = 1000      # bootstrap null iterations
alpha_neuron = 0.05      # per-neuron significance threshold (via null quantile)

neural_data = counts_matrix
X_raw = subject_trials['num_images_presented'].to_numpy().astype(float)
cell_type_labels = np.asarray(region_labels)  

if neural_data.ndim == 3:
    y_counts = neural_data.sum(axis=2)
else:
    y_counts = neural_data

n_trials, n_neurons = y_counts.shape

# Optional: block ids for within-subject permutation
blocks = subject_trials['subject_id'].to_numpy() if 'subject_id' in subject_trials.columns else None

# z-score predictor for stability
X = (X_raw - X_raw.mean()) / (X_raw.std(ddof=0) + 1e-12)
X = X.reshape(-1, 1)

# =====================
# Helper: block-wise permutation of predictor
# =====================
def permute_within_blocks(values, blocks, rng):
    if blocks is None:
        return rng.permutation(values)
    out = np.empty_like(values)
    for b in np.unique(blocks):
        idx = np.where(blocks == b)[0]
        out[idx] = rng.permutation(values[idx])
    return out

# =====================
# 1) Fit real Poisson GLM slope per neuron
# =====================
real_slopes = np.empty(n_neurons, dtype=float)
for j in range(n_neurons):
    model = PoissonRegressor(alpha=0.0, max_iter=1000).fit(X, y_counts[:, j])
    real_slopes[j] = model.coef_[0]

# =====================
# 2) Build per-neuron null distribution of slopes (permuted predictor)
# =====================
scrambled_slopes = np.empty((n_neurons, n_iterations), dtype=float)
for it in tqdm(range(n_iterations), desc="Building per-neuron null (permuted predictor)"):
    X_shuf = permute_within_blocks(X.squeeze(), blocks, rng).reshape(-1, 1)
    for j in range(n_neurons):
        model = PoissonRegressor(alpha=0.0, max_iter=500).fit(X_shuf, y_counts[:, j])
        scrambled_slopes[j, it] = model.coef_[0]

# =====================
# 3) Per-neuron thresholds from its own null (|slope| quantile)
# =====================
abs_null = np.abs(scrambled_slopes)  # [n_neurons x n_iterations]
q = 1.0 - alpha_neuron
thr_per_neuron = np.quantile(abs_null, q, axis=1, method='higher')  # [n_neurons]

# Observed "significant" mask by neuron
sig_obs_neuron = (np.abs(real_slopes) >= thr_per_neuron)

# =====================
# 4) Null distribution of "number of significant neurons" per region
#     p-value = (# null iterations >= observed) / (# iterations)
# =====================
types = ['amygdala', 'hippocampus', 'preSMA', 'vmPFC', 'dACC']
summary_rows = []
null_counts_by_type = {}

for t in types:
    idx = np.where(cell_type_labels == t)[0]
    if idx.size == 0:
        print(f"[WARN] No neurons labeled '{t}'. Skipping.")
        continue

    # Observed count for this region
    observed_count = int(np.sum(sig_obs_neuron[idx]))

    # For each iteration, count how many neurons in this region exceed their per-neuron threshold
    abs_null_type = abs_null[idx, :]            # [n_type_neurons x n_iterations]
    thr_type = thr_per_neuron[idx][:, None]     # [n_type_neurons x 1]
    sig_in_iter = (abs_null_type >= thr_type)   # boolean
    null_counts = sig_in_iter.sum(axis=0)       # [n_iterations]

    # One-sided bootstrap p (plain definition; no +1)
    p_boot = np.sum(null_counts >= observed_count) / float(n_iterations)

    null_counts_by_type[t] = null_counts
    summary_rows.append({
        'cell_type': t,
        'n_neurons': int(idx.size),
        'observed_significant': observed_count,
        'null_mean': float(np.mean(null_counts)),
        'null_std': float(np.std(null_counts, ddof=1)),
        'null_p95': float(np.percentile(null_counts, 95)),
        'p_boot_one_sided': p_boot
    })

summary_df = pd.DataFrame(summary_rows)
print("\n=== Bootstrap test of number of significant neurons per region ===")
print(summary_df.to_string(index=False))

# =====================
# 5) Plot % significant per region with bootstrap p-value stars
# =====================
def p_to_stars(p):
    if p < 0.005: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''

if len(summary_df) > 0:
    sdf = summary_df.copy()
    sdf['observed_percent'] = 100.0 * sdf['observed_significant'] / sdf['n_neurons'].clip(lower=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    xt = np.arange(len(sdf))

    bars = ax.bar(xt, sdf['observed_percent'].values, width=0.6, color='black')

    ymax = float(sdf['observed_percent'].max()) if len(sdf) else 0.0
    headroom = max(5.0, 0.10 * max(1.0, ymax))
    for i, row in sdf.reset_index(drop=True).iterrows():
        stars = p_to_stars(row['p_boot_one_sided'])
        if stars:
            ax.text(i, row['observed_percent'] + 0.03 * max(1.0, ymax),
                    stars, ha='center', va='bottom', fontsize=14, fontweight='bold', color='red')
        ax.text(i, -2.0, f"(n={int(row['n_neurons'])})", ha='center', va='top', fontsize=9)

    ax.set_xticks(xt)
    ax.set_xticklabels(sdf['cell_type'].values, rotation=0)
    ax.set_ylabel("% significant neurons")
    ax.set_ylim(-5, ymax + headroom)
    ax.set_title("Observed % significant neurons (Brain Regions)")
    plt.tight_layout()
    plt.savefig("./poisson_load.eps", format='eps', dpi=300)
    plt.show()
