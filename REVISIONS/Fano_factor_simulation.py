import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import ast
import os
trial_info = pd.read_excel('../data/new_trial_final.xlsx')
subject_trials = trial_info[trial_info['subject_id'] == 14][[
    'trial_id_final', 'num_images_presented',
    'stimulus_index_enc1', 'stimulus_index_enc2', 'stimulus_index_enc3',
    'response_accuracy'
]]
print(subject_trials)
y_matrix = subject_trials

df_delay_filtered = pd.read_excel('../clean_data/cleaned_Delay.xlsx')
df_fixation       = pd.read_excel('../clean_data/cleaned_Fixation.xlsx')
df_clustering     = pd.read_excel('all_neuron_brain_regions_merged.xlsx')

y_matrix = y_matrix.reset_index(drop=True)

# ── Settings ────────────────────────────────────────────────────────────────
bin_sizes     = [0.25, 0.5, 0.75, 1.0]
step_size     = 0.025
durations     = {1: 2.5, 2: 2.5, 3: 2.5}
n_splits      = 3
n_repeats     = 10
n_sims        = 100
random_state  = 42

folder_path = './'
os.makedirs(folder_path, exist_ok=True)

def parse_spike_times(spike_str):
    if pd.isna(spike_str):
        return []
    try:
        parsed = ast.literal_eval(spike_str)
        return list(parsed) if isinstance(parsed, (list, tuple, np.ndarray)) else []
    except Exception:
        return []

# Filter significant neurons present in delay + fixation 
candidate_neurons = df_clustering[df_clustering['Signi'] == 'Y']['Neuron_ID_3'].dropna().unique()
delay_neurons     = df_delay_filtered['Neuron_ID_3'].dropna().unique()
fix_neurons       = df_fixation['Neuron_ID_3'].dropna().unique()
final_neurons     = np.array(sorted(
    np.intersect1d(candidate_neurons, np.intersect1d(delay_neurons, fix_neurons))
))
print(f"Neurons included: {len(final_neurons)}")

trial_count   = y_matrix.shape[0]
neuron_count  = len(final_neurons)
neuron_to_col = {nid: idx for idx, nid in enumerate(final_neurons)}

# Build design matrix (spike times per trial x neuron)
design_matrix = np.empty((trial_count, neuron_count), dtype=object)

delay_lookup = {}
for _, row in df_delay_filtered.iterrows():
    key = (row['trial_id_final'], row['Neuron_ID_3'])
    delay_lookup[key] = np.array(parse_spike_times(row['Standardized_Spikes']), dtype=float)

for trial_idx, row in y_matrix.iterrows():
    trial_id = row['trial_id_final']
    for neuron_id in final_neurons:
        col_idx = neuron_to_col[neuron_id]
        design_matrix[trial_idx, col_idx] = delay_lookup.get(
            (trial_id, neuron_id), np.array([], dtype=float)
        )

# Fixation normalization
df_fix_filt  = df_fixation[df_fixation['Neuron_ID_3'].isin(final_neurons)].copy()
fixation_means = df_fix_filt.groupby('Neuron_ID_3')['Spikes_rate_Fixation'].mean().to_dict()
fixation_stds  = df_fix_filt.groupby('Neuron_ID_3')['Spikes_rate_Fixation'].std().to_dict()

# Trial labels
y_labels = []
for _, row in y_matrix.iterrows():
    load = row['num_images_presented']
    if   load == 1: y_labels.append(row['stimulus_index_enc1'])
    elif load == 2: y_labels.append(row['stimulus_index_enc2'])
    elif load == 3: y_labels.append(row['stimulus_index_enc3'])
    else:           y_labels.append(np.nan)
y_labels = np.array(y_labels)

valid = ~pd.isna(y_labels)
y_matrix      = y_matrix.loc[valid].reset_index(drop=True)
y_labels      = y_labels[valid]
design_matrix = design_matrix[valid]

load_trials = {
    load: np.where(y_matrix['num_images_presented'].values == load)[0]
    for load in [1, 2, 3]
}
print({f"Load{k}": len(v) for k, v in load_trials.items()}, "trials")

from collections import defaultdict

def create_time_bins(start_time, end_time, bin_size, step_size):
    return np.arange(start_time, end_time - bin_size + step_size, step_size)

def count_spikes_in_bins(spike_times, time_bins, bin_size):
    spike_times = np.asarray(spike_times, dtype=float)
    return np.array(
        [np.sum((spike_times >= t) & (spike_times < t + bin_size)) for t in time_bins],
        dtype=float
    )

def svc_cv_accuracy(X, y, n_splits=3, n_repeats=10, seed=42):
    """Repeated stratified K-fold SVC; returns mean accuracy (%)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or np.min(counts) < n_splits:
        return np.nan
    fold_accs = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for tr, te in skf.split(X, y):
            if len(np.unique(y[tr])) < 2:
                continue
            clf = SVC(kernel='linear', random_state=seed + rep)
            clf.fit(X[tr], y[tr])
            fold_accs.append(np.mean(clf.predict(X[te]) == y[te]) * 100)
    return np.nanmean(fold_accs) if fold_accs else np.nan

def _build_X(spikes_matrix, neuron_list, bin_size, end_time, step_size,
              fixation_means, fixation_stds):
    """Bin spike times and z-score by fixation baseline. Returns (n_trials, n_neurons*n_bins)."""
    time_bins = create_time_bins(0.0, end_time, bin_size, step_size)
    n_t, n_n = spikes_matrix.shape[0], len(neuron_list)
    X = np.zeros((n_t, n_n, len(time_bins)), dtype=float)
    for i in range(n_t):
        for j, nid in enumerate(neuron_list):
            spikes  = spikes_matrix[i, j]
            binned  = count_spikes_in_bins(spikes, time_bins, bin_size)
            mu = fixation_means.get(nid, 0.0)
            sd = fixation_stds.get(nid, 1.0)
            X[i, j, :] = (binned - mu) / sd if (sd and sd > 0 and not np.isnan(sd)) else binned
    return X.reshape(n_t, -1)

def estimate_gamma_poisson_params(design_matrix, trial_idxs, y_labels, final_neurons, T, min_trials=5):
    rbar   = {nid: {} for nid in final_neurons}
    kappa  = {nid: {} for nid in final_neurons}
    counts = {nid: defaultdict(list) for nid in final_neurons}
    for trial_idx in np.asarray(trial_idxs, dtype=int):
        c = y_labels[trial_idx]
        for j, nid in enumerate(final_neurons):
            spikes = np.asarray(design_matrix[trial_idx, j], dtype=float)
            counts[nid][c].append(int(np.sum((spikes >= 0) & (spikes < T))))
    for nid in final_neurons:
        for c, arr in counts[nid].items():
            arr = np.asarray(arr, dtype=float)
            mu  = arr.mean() if len(arr) > 0 else 0.0
            rbar[nid][c] = mu / T if T > 0 else 0.0
            if len(arr) < min_trials or mu == 0:
                kappa[nid][c] = np.inf
            else:
                var = arr.var(ddof=1)
                kappa[nid][c] = np.inf if var <= mu else (mu ** 2) / (var - mu)
    return rbar, kappa

def simulate_persistent_spikes_fano(trial_idxs, y_labels, final_neurons, rbar, kappa, T, rng):
    trial_idxs = np.asarray(trial_idxs, dtype=int)
    sim = np.empty((len(trial_idxs), len(final_neurons)), dtype=object)
    for i, trial_idx in enumerate(trial_idxs):
        c = y_labels[trial_idx]
        for j, nid in enumerate(final_neurons):
            rb = rbar[nid].get(c, 0.0)
            k  = kappa[nid].get(c, np.inf)
            if rb <= 0:
                sim[i, j] = np.array([], dtype=float)
                continue
            r_trial = rb if np.isinf(k) else rng.gamma(shape=k, scale=rb / k)
            nspk    = rng.poisson(r_trial * T)
            sim[i, j] = np.sort(rng.uniform(0, T, size=nspk)) if nspk > 0 else np.array([], dtype=float)
    return sim

def compute_real_curve(bin_sizes, load_trials, y_labels, durations, final_neurons,
                       design_matrix, fixation_means, fixation_stds,
                       step_size=0.025, n_splits=3, n_repeats=10, random_state=42):
    """Decode real data across bin sizes; returns dict[bin_size][LoadN] -> mean accuracy %."""
    real = {b: {} for b in bin_sizes}
    for load in [1, 2, 3]:
        trial_idxs = np.asarray(load_trials[load], dtype=int)
        T = durations[load]
        y_local = y_labels[trial_idxs]
        for b in bin_sizes:
            X = _build_X(design_matrix[trial_idxs], final_neurons, b, T,
                         step_size, fixation_means, fixation_stds)
            real[b][f'Load{load}'] = svc_cv_accuracy(X, y_local, n_splits, n_repeats, random_state)
            print(f"  Real Load{load} bin={b}: {real[b][f'Load{load}']:.2f}%")
    return real

def run_fano_matched_null(bin_sizes, load_trials, y_labels, durations, final_neurons,
                          design_matrix, fixation_means, fixation_stds,
                          step_size=0.025, n_sims=1000, seed=42,
                          n_splits=3, n_repeats=10, random_state=42):
    """Fano-matched Gamma-Poisson null; returns dict[bin_size][LoadN] -> array of n_sims accuracies %."""
    rng = np.random.default_rng(seed)
    out = {b: {f'Load{l}': [] for l in [1,2,3]} for b in bin_sizes}

    for load in [1, 2, 3]:
        trial_idxs = np.asarray(load_trials[load], dtype=int)
        T          = durations[load]
        y_local    = y_labels[trial_idxs]

        rbar, kappa = estimate_gamma_poisson_params(
            design_matrix, trial_idxs, y_labels, final_neurons, T
        )

        for sim_idx in range(n_sims):
            sim_spikes = simulate_persistent_spikes_fano(
                trial_idxs, y_labels, final_neurons, rbar, kappa, T, rng
            )
            cv_seed = int(rng.integers(0, 1_000_000_000))
            for b in bin_sizes:
                X_sim = _build_X(sim_spikes, final_neurons, b, T, step_size,
                                 fixation_means, fixation_stds)
                out[b][f'Load{load}'].append(
                    svc_cv_accuracy(X_sim, y_local, n_splits, n_repeats, cv_seed)
                )
            if (sim_idx + 1) % 100 == 0:
                print(f"  Load{load} sim {sim_idx+1}/{n_sims}")

    for b in bin_sizes:
        for k in out[b]:
            out[b][k] = np.asarray(out[b][k], dtype=float)
    return out

print("Functions defined.")

print("Computing real decoding curve...")
real_curve = compute_real_curve(
    bin_sizes=bin_sizes,
    load_trials=load_trials,
    y_labels=y_labels,
    durations=durations,
    final_neurons=final_neurons,
    design_matrix=design_matrix,
    fixation_means=fixation_means,
    fixation_stds=fixation_stds,
    step_size=step_size,
    n_splits=n_splits,
    n_repeats=n_repeats,
    random_state=random_state
)

print("\nComputing fano-matched null distribution...")
fano_null = run_fano_matched_null(
    bin_sizes=bin_sizes,
    load_trials=load_trials,
    y_labels=y_labels,
    durations=durations,
    final_neurons=final_neurons,
    design_matrix=design_matrix,
    fixation_means=fixation_means,
    fixation_stds=fixation_stds,
    step_size=step_size,
    n_sims=n_sims,
    seed=random_state,
    n_splits=n_splits,
    n_repeats=n_repeats,
    random_state=random_state
)
print("\nDone.")

bin_vals = np.array(bin_sizes)

# Compute slopes 
real_slopes = {}
for load in ['Load1', 'Load2', 'Load3']:
    y = np.array([real_curve[b][load] for b in bin_sizes])
    real_slopes[load] = np.polyfit(bin_vals, y, 1)[0]

null_slopes = {}
for load in ['Load1', 'Load2', 'Load3']:
    n = len(fano_null[bin_sizes[0]][load])
    slopes = [
        np.polyfit(bin_vals, [fano_null[b][load][i] for b in bin_sizes], 1)[0]
        for i in range(n)
    ]
    null_slopes[load] = np.array(slopes)

# Permutation statistics 
p_values_slope = {}
z_scores       = {}
print("=" * 60)
print(f"{'Load':<8} {'Real slope':>12} {'Null mean':>12} {'Null SD':>10} {'p (perm)':>12} {'z':>8}")
print("-" * 60)
for load in ['Load1', 'Load2', 'Load3']:
    rs   = real_slopes[load]
    nd   = null_slopes[load]
    nd   = nd[~np.isnan(nd)]
    p    = (np.sum(nd >= rs) + 1) / (len(nd) + 1)
    z    = (rs - nd.mean()) / nd.std() if nd.std() > 0 else np.nan
    p_values_slope[load] = p
    z_scores[load]       = z
    print(f"{load:<8} {rs:>12.4f} {nd.mean():>12.4f} {nd.std():>10.4f} {p:>12.5f} {z:>8.2f}")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

load_colors = {'Load1': '#2c7bb6', 'Load2': '#5aae61', 'Load3': '#d7191c'}

for ax, load in zip(axes, ['Load1', 'Load2', 'Load3']):
    nd   = null_slopes[load]
    nd   = nd[~np.isnan(nd)]
    rs   = real_slopes[load]
    p    = p_values_slope[load]
    z    = z_scores[load]
    stars = ('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns')

    # Null distribution
    ax.hist(nd, bins=40, color='lightgray', edgecolor='gray', alpha=0.85, label='Fano-matched null')

    # Real slope
    ax.axvline(rs, color=load_colors[load], linewidth=2.5, label=f'Real slope')

    # Null mean
    ax.axvline(nd.mean(), color='black', linewidth=1.2, linestyle='--', label='Null mean')

    # Annotations
    ymax = ax.get_ylim()[1]
    ax.text(0.97, 0.95, f"p = {p:.4f} {stars}\nz = {z:.2f}",
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

    ax.set_title(f'{load}', fontsize=13)
    ax.set_xlabel('Slope (% accuracy / s)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.suptitle(
    'Real vs Fano-matched Null: Decoding Slope Across Bin Sizes\n'
    f'(SVC, {n_repeats}×{n_splits}-fold CV, {n_sims} simulations)',
    fontsize=13
)
plt.tight_layout()

save_path = os.path.join(folder_path, 'fano_slope_comparison.eps')
plt.savefig(save_path, format='eps', dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved to {save_path}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

load_colors = {
    'Load1': '#2c7bb6',
    'Load2': '#5aae61',
    'Load3': '#d7191c'
}

for ax, load in zip(axes, ['Load1', 'Load2', 'Load3']):

    # Real curve
    y_real = np.array([
        real_curve[b][load]
        for b in bin_sizes
    ])

    # Shuffle mean and SD
    y_null_mean = np.array([
        np.mean(fano_null[b][load])
        for b in bin_sizes
    ])

    y_null_std = np.array([
        np.std(fano_null[b][load])
        for b in bin_sizes
    ])

    # Real line
    ax.plot(
        bin_vals,
        y_real,
        '-o',
        color=load_colors[load],
        linewidth=2.5,
        markersize=7,
        label='Real'
    )

    # Shuffle line
    ax.plot(
        bin_vals,
        y_null_mean,
        '-o',
        color='black',
        linewidth=2,
        markersize=6,
        label='Fano-matched shuffle'
    )

    # SD band
    ax.fill_between(
        bin_vals,
        y_null_mean - y_null_std,
        y_null_mean + y_null_std,
        color='gray',
        alpha=0.25,
        label='Shuffle ±1 SD'
    )

    p = p_values_slope[load]
    z = z_scores[load]

    stars = (
        '***' if p < 0.001 else
        '**'  if p < 0.01 else
        '*'   if p < 0.05 else
        'ns'
    )

    ax.text(
        0.97, 0.95,
        f"Slope={real_slopes[load]:.4f}\np={p:.4f} {stars}\nz={z:.2f}",
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3',
                  fc='white',
                  ec='gray')
    )

    ax.set_title(load)
    ax.set_xlabel('Bin size (s)')
    ax.set_ylabel('Decoding accuracy (%)')
    ax.grid(alpha=0.3)
    ax.legend()

plt.suptitle(
    'Real vs Fano-Matched Shuffle Decoding Across Bin Sizes',
    fontsize=14
)

plt.tight_layout()
plt.show()
