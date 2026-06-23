import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu, ttest_rel
from scipy.io import loadmat
import h5py
import os
import glob
import random
import warnings
warnings.filterwarnings('ignore')

def create_time_bins(duration, bin_size, step_size):
    return np.arange(0, duration - bin_size + step_size, step_size)

def count_spikes_in_bins(spike_times, time_bins, bin_size):
    return np.array([np.sum((spike_times >= t) & (spike_times < t + bin_size))
                     for t in time_bins], dtype=float)

def build_binned_counts_matrix(design_matrix, trial_indices, total_duration, bin_size, step_size):
    bins = create_time_bins(total_duration, bin_size, step_size)
    n_trials = len(trial_indices)
    n_neurons = design_matrix.shape[1]
    X = np.zeros((n_trials, n_neurons, len(bins)), dtype=float)

    for ii, tr in enumerate(trial_indices):
        for j in range(n_neurons):
            spikes = design_matrix[tr, j]
            X[ii, j, :] = count_spikes_in_bins(spikes, bins, bin_size)

    return X, bins

def fit_poisson_nb(X_train, y_train, eps=1e-6):
    classes = np.unique(y_train)
    K = len(classes)
    n_neurons = X_train.shape[1]

    lambda_ = np.zeros((K, n_neurons), dtype=float)
    log_prior = np.zeros(K, dtype=float)

    for ki, c in enumerate(classes):
        Xc = X_train[y_train == c]
        lam = Xc.mean(axis=0)
        lam = np.clip(lam, eps, None)
        lambda_[ki] = lam
        log_prior[ki] = np.log(len(Xc) / len(X_train))

    log_lambda = np.log(lambda_)
    return classes, log_lambda, lambda_, log_prior

def posterior_poisson_nb(x, log_lambda, lambda_, log_prior):
    ll = (x[None, :] * log_lambda - lambda_).sum(axis=1)
    log_post = ll + log_prior
    log_post -= logsumexp(log_post)
    return np.exp(log_post)

def pred_and_confidence(posterior):
    """Get prediction and confidence from posterior probabilities"""
    pred_idx = np.argmax(posterior, axis=1)
    confidence = np.max(posterior, axis=1)
    return pred_idx, confidence

def encoding_trained_delay_scores_loocv(
    design_matrix,
    trial_indices,
    y_labels,
    enc_duration=0.25,
    delay_duration=0.75,
    bin_size=0.025,
    step_size=0.01,
    eps=1e-6
):
    """
    Leave-one-trial-out CV with encoding and delay periods
    """
    # Timing parameters (20kHz sampling)
    dt = 1.0 / 20000
    fixation_duration = 1.0  # 20000 points
    encoding_start_time = fixation_duration
    encoding_end_time = fixation_duration + enc_duration
    delay_start_time = encoding_end_time
    delay_end_time = delay_start_time + delay_duration
    total_duration = delay_end_time
    
    # Build counts for entire trial
    X, bins = build_binned_counts_matrix(
        design_matrix=design_matrix,
        trial_indices=trial_indices,
        total_duration=total_duration,
        bin_size=bin_size,
        step_size=step_size
    )

    # Find bins for encoding period
    enc_bins_start = np.searchsorted(bins, encoding_start_time)
    enc_bins_end = np.searchsorted(bins, encoding_end_time)
    enc_idx = np.arange(enc_bins_start, enc_bins_end)
    
    # Find bins for delay period
    delay_bins_start = np.searchsorted(bins, delay_start_time)
    delay_bins_end = np.searchsorted(bins, delay_end_time)
    delay_idx = np.arange(delay_bins_start, delay_bins_end)
    
    # Get delay bin times
    bins_delay = bins[delay_idx] - delay_start_time

    n_trials = X.shape[0]
    n_delay = len(delay_idx)

    # Fixed class order across folds
    classes_all = np.unique(y_labels)
    K = len(classes_all)
    post_delay = np.zeros((n_trials, n_delay, K), dtype=float)

    for test_i in range(n_trials):
        train_mask = np.ones(n_trials, dtype=bool)
        train_mask[test_i] = False

        # Training: encoding bins from training trials
        X_enc_train = X[train_mask, :, :][:, :, enc_idx]
        X_enc_train = np.transpose(X_enc_train, (0, 2, 1)).reshape(-1, X.shape[1])
        y_enc_train = np.repeat(y_labels[train_mask], len(enc_idx))

        classes_fold, log_lambda, lambda_, log_prior = fit_poisson_nb(X_enc_train, y_enc_train, eps=eps)

        # Map fold class indices to global indices
        fold_to_global = {}
        for fold_i, c in enumerate(classes_fold):
            if c in classes_all:
                fold_to_global[fold_i] = int(np.where(classes_all == c)[0][0])

        # Decode delay for held-out trial
        for t_i, t in enumerate(delay_idx):
            x = X[test_i, :, t]
            p_fold = posterior_poisson_nb(x, log_lambda, lambda_, log_prior)

            p_global = np.zeros(K, dtype=float)
            for fold_i, glob_i in fold_to_global.items():
                p_global[glob_i] = p_fold[fold_i]

            s = p_global.sum()
            if s > 0:
                p_global /= s
            post_delay[test_i, t_i, :] = p_global

    return bins_delay, post_delay, classes_all

def duration_per_item_presented_nonpresented_rnn(
    score_delay, classes, y_labels,
    bins_delay, step_size, conf_thr,
    min_bins=2, require_pos_slope=False
):
    """
    Calculate confidence durations for RNN data
    """
    K = len(classes)
    n_trials = score_delay.shape[0]
    
    pres_out = []
    non_out = []
    
    # Create mapping from class index to label
    idx_to_label = {i: classes[i] for i in range(K)}
    
    for ti in range(n_trials):
        # Determine presented stimulus for this trial
        true_label = y_labels[ti]
        presented = {true_label}
        
        pred_idx, conf = pred_and_confidence(score_delay[ti])
        slope = np.diff(conf, prepend=conf[0]) / step_size
        
        above = conf >= conf_thr
        i = 0
        pres_dur = 0.0
        non_dur = 0.0
        
        while i < len(conf):
            if above[i]:
                start = i
                while i < len(conf) and above[i]:
                    i += 1
                end = i - 1
                
                if (end - start + 1) < min_bins:
                    continue
                if require_pos_slope and slope[start] <= 0:
                    continue
                
                # event label = majority predicted class
                seg_pred = pred_idx[start:end+1]
                ev_class_idx = int(np.bincount(seg_pred).argmax())
                ev_label = idx_to_label[ev_class_idx]
                
                dur = (end - start + 1) * step_size
                
                if ev_label in presented:
                    pres_dur += dur
                else:
                    non_dur += dur
            else:
                i += 1
        
        pres_out.append(pres_dur)
        non_out.append(non_dur)
    
    return np.asarray(pres_out), np.asarray(non_out)

def find_threshold_via_null_for_model(
    design_matrix, y_labels,
    enc_duration=0.25,
    delay_duration=0.75,
    bin_size=0.025,
    step_size=0.01,
    n_shuffles=100,
    percentile=95,
    verbose=False
):
    """
    Find threshold for a single model using shuffled labels
    """
    n_trials = len(y_labels)
    
    # Run decoder with TRUE labels
    if verbose:
        print(f"  Running decoder with true labels...")
    
    bins_delay, score_delay, classes = encoding_trained_delay_scores_loocv(
        design_matrix=design_matrix,
        trial_indices=np.arange(n_trials),
        y_labels=y_labels,
        enc_duration=enc_duration,
        delay_duration=delay_duration,
        bin_size=bin_size,
        step_size=step_size
    )
    
    # Collect all confidence values from real data
    real_confidences = []
    for ti in range(n_trials):
        _, conf = pred_and_confidence(score_delay[ti])
        real_confidences.extend(conf)
    
    real_confidences = np.array(real_confidences)
    
    # Run decoder with SHUFFLED labels
    null_confidences = []
    
    for shuffle_idx in range(n_shuffles):
        if verbose and shuffle_idx % 20 == 0:
            print(f"  Shuffle {shuffle_idx}/{n_shuffles}")
        
        # Shuffle labels
        y_shuffled = np.random.permutation(y_labels)
        
        try:
            _, score_delay_shuf, _ = encoding_trained_delay_scores_loocv(
                design_matrix=design_matrix,
                trial_indices=np.arange(n_trials),
                y_labels=y_shuffled,
                enc_duration=enc_duration,
                delay_duration=delay_duration,
                bin_size=bin_size,
                step_size=step_size
            )
            
            # Sample trials to keep size manageable
            for ti in range(min(n_trials, 10)):
                _, conf = pred_and_confidence(score_delay_shuf[ti])
                null_confidences.extend(conf)
                
        except Exception as e:
            continue
    
    null_confidences = np.array(null_confidences)
    
    # Determine threshold
    threshold = np.percentile(null_confidences, percentile)
    
    if verbose:
        print(f"  Threshold ({percentile}th %ile of null): {threshold:.3f}")
        print(f"  Real conf mean: {real_confidences.mean():.3f}, null conf mean: {null_confidences.mean():.3f}")
    
    return threshold, real_confidences, null_confidences, bins_delay, score_delay, classes

def load_model_v73(model_path):
    """Load MATLAB v7.3 file using h5py"""
    data = {}
    with h5py.File(model_path, 'r') as f:
        for key in f.keys():
            if key.startswith('#'):
                continue
            dataset = f[key]
            if isinstance(dataset, h5py.Dataset):
                arr = dataset[()]
                if len(arr.shape) >= 2:
                    arr = arr.T
                data[key] = arr
    return data

def load_and_prepare_model(model_path, use_inhibitory=True):
    """Load model and prepare design matrix"""
    print(f"  Loading {os.path.basename(model_path)}...")
    
    try:
        # Try v7.3 first
        mat_data = load_model_v73(model_path)
    except:
        # Fall back to scipy
        mat_data = loadmat(model_path)
    
    # Find spike data
    spike_data = None
    for key in ['outs', 'stable_outs', 'all_spks', 'spks', 'r']:
        if key in mat_data:
            spike_data = mat_data[key]
            print(f"    Using '{key}' with shape {spike_data.shape}")
            break
    
    if spike_data is None:
        raise ValueError(f"No spike data found in {model_path}")
    
    # Find inhibitory units
    if use_inhibitory and 'inh' in mat_data:
        inh = mat_data['inh'].flatten()
        if len(inh) == spike_data.shape[1]:
            inh_indices = np.where(inh == 1)[0]
            if len(inh_indices) > 0:
                spike_data = spike_data[:, inh_indices, :] if len(spike_data.shape) == 3 else spike_data[:, inh_indices]
                print(f"    Using {len(inh_indices)} inhibitory units")
    
    # Ensure 3D shape
    if len(spike_data.shape) == 2:
        spike_data = spike_data.reshape(spike_data.shape[0], 1, spike_data.shape[1])
    
    # Convert to spike times (20kHz)
    dt = 1.0 / 20000
    n_trials, n_neurons, n_timepoints = spike_data.shape
    time_axis = np.arange(n_timepoints) * dt
    
    # Create design matrix
    design_matrix = np.empty((n_trials, n_neurons), dtype=object)
    
    for trial in range(n_trials):
        for neuron in range(n_neurons):
            spike_inds = np.where(spike_data[trial, neuron, :] > 0)[0]
            design_matrix[trial, neuron] = time_axis[spike_inds]
    
    # Create labels (first half -1, second half +1)
    n_trials_per_cond = n_trials // 2
    y_labels = np.array([-1] * n_trials_per_cond + [1] * (n_trials - n_trials_per_cond))
    
    return design_matrix, y_labels, n_trials, n_neurons

# =============================================
# MAIN ANALYSIS
# =============================================

# Set up paths
good_models_dir = "./RNN_models"
model_files = sorted(glob.glob(os.path.join(good_models_dir, "*.mat")))

print("="*80)
print("RNN MODEL ANALYSIS WITH PER-MODEL NULL THRESHOLDS")
print("="*80)
print(f"Found {len(model_files)} model files")

# Parameters
enc_duration = 0.25  # 0.25s encoding
delay_duration = 0.75  # 0.75s delay
bin_size = 0.025
step_size = 0.01
n_shuffles = 50  # Number of shuffles per model (can increase for more accuracy)
percentile = 95  # Use 95th percentile of null as threshold

# Store results for all models
all_model_results = []

# Process each model
for model_idx, model_path in enumerate(model_files):
    print(f"\n{'='*60}")
    print(f"Model {model_idx+1}/{len(model_files)}: {os.path.basename(model_path)}")
    print(f"{'='*60}")
    
    try:
        # Load model
        design_matrix, y_labels, n_trials, n_neurons = load_and_prepare_model(model_path)
        print(f"    Trials: {n_trials}, Neurons: {n_neurons}")
        
        # Find threshold for this model using null shuffles
        threshold, real_conf, null_conf, bins_delay, score_delay, classes = find_threshold_via_null_for_model(
            design_matrix=design_matrix,
            y_labels=y_labels,
            enc_duration=enc_duration,
            delay_duration=delay_duration,
            bin_size=bin_size,
            step_size=step_size,
            n_shuffles=n_shuffles,
            percentile=percentile,
            verbose=True
        )
        
        # Calculate durations with this threshold
        pres_dur, non_dur = duration_per_item_presented_nonpresented_rnn(
            score_delay=score_delay,
            classes=classes,
            y_labels=y_labels,
            bins_delay=bins_delay,
            step_size=step_size,
            conf_thr=threshold,
            min_bins=2,
            require_pos_slope=False
        )
        
        diff_dur = pres_dur - non_dur
        
        # Store results
        model_result = {
            'model_name': os.path.basename(model_path),
            'threshold': threshold,
            'pres_dur': pres_dur,
            'non_dur': non_dur,
            'diff_dur': diff_dur,
            'mean_pres': np.mean(pres_dur),
            'std_pres': np.std(pres_dur),
            'mean_non': np.mean(non_dur),
            'std_non': np.std(non_dur),
            'mean_diff': np.mean(diff_dur),
            'std_diff': np.std(diff_dur),
            'real_conf': real_conf,
            'null_conf': null_conf,
            'bins_delay': bins_delay,
            'score_delay': score_delay,
            'n_trials': n_trials
        }
        
        all_model_results.append(model_result)
        
        print(f"\n  Results for {os.path.basename(model_path)}:")
        print(f"    Threshold: {threshold:.3f}")
        print(f"    Presented duration: {model_result['mean_pres']:.4f} ± {model_result['std_pres']:.4f} s")
        print(f"    Non-presented duration: {model_result['mean_non']:.4f} ± {model_result['std_non']:.4f} s")
        print(f"    Difference: {model_result['mean_diff']:.4f} ± {model_result['std_diff']:.4f} s")
        
    except Exception as e:
        print(f"  Error processing {os.path.basename(model_path)}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*80}")
print(f"Successfully processed {len(all_model_results)}/{len(model_files)} models")
print(f"{'='*80}")

# =============================================
# ACROSS-MODEL ANALYSIS
# =============================================

if len(all_model_results) > 0:
    
    # Collect across-model statistics
    thresholds = np.array([r['threshold'] for r in all_model_results])
    mean_pres = np.array([r['mean_pres'] for r in all_model_results])
    mean_non = np.array([r['mean_non'] for r in all_model_results])
    mean_diff = np.array([r['mean_diff'] for r in all_model_results])
    
    # All durations concatenated across models
    all_pres_durs = np.concatenate([r['pres_dur'] for r in all_model_results])
    all_non_durs = np.concatenate([r['non_dur'] for r in all_model_results])
    all_diff_durs = np.concatenate([r['diff_dur'] for r in all_model_results])
    
    print("\n" + "="*80)
    print("ACROSS-MODEL RESULTS")
    print("="*80)
    
    print(f"\nNull thresholds across models:")
    print(f"  Mean: {np.mean(thresholds):.3f} ± {np.std(thresholds):.3f}")
    print(f"  Range: [{np.min(thresholds):.3f}, {np.max(thresholds):.3f}]")
    
    print(f"\nPresented item durations:")
    print(f"  Per-model mean: {np.mean(mean_pres):.4f} ± {np.std(mean_pres):.4f} s")
    print(f"  Across all trials: {np.mean(all_pres_durs):.4f} ± {np.std(all_pres_durs):.4f} s")
    
    print(f"\nNon-presented item durations:")
    print(f"  Per-model mean: {np.mean(mean_non):.4f} ± {np.std(mean_non):.4f} s")
    print(f"  Across all trials: {np.mean(all_non_durs):.4f} ± {np.std(all_non_durs):.4f} s")
    
    print(f"\nDifference (presented - non-presented):")
    print(f"  Per-model mean: {np.mean(mean_diff):.4f} ± {np.std(mean_diff):.4f} s")
    print(f"  Across all trials: {np.mean(all_diff_durs):.4f} ± {np.std(all_diff_durs):.4f} s")
    
    # Statistical tests
    print(f"\nSTATISTICAL TESTS:")
    
    # Paired t-test across models (comparing presented vs non-presented per model)
    t_stat, p_val_ttest = ttest_rel(mean_pres, mean_non)
    print(f"  Paired t-test (per-model means): t={t_stat:.3f}, p={p_val_ttest:.4f}")
    
    # Wilcoxon signed-rank test across models
    if not np.all(mean_pres == mean_non):
        stat_wilcox, p_val_wilcox = wilcoxon(mean_pres, mean_non)
        print(f"  Wilcoxon signed-rank test: statistic={stat_wilcox:.1f}, p={p_val_wilcox:.4f}")
    
    # Wilcoxon on concatenated trials
    if not np.all(all_pres_durs == all_non_durs):
        stat_wilcox_all, p_val_wilcox_all = wilcoxon(all_pres_durs, all_non_durs)
        print(f"  Wilcoxon (all trials): statistic={stat_wilcox_all:.1f}, p={p_val_wilcox_all:.4e}")
    
    # Intermediate diagnostic plots removed. The final publication-style real-vs-RNN plot is kept below.

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE (Per Model)")
    print("="*80)
    print(f"{'Model':<40} {'Threshold':<10} {'Pres Dur (s)':<15} {'Non Dur (s)':<15} {'Diff (s)':<12}")
    print("-"*92)
    
    for r in all_model_results:
        short_name = r['model_name'][:38]
        print(f"{short_name:<40} {r['threshold']:<10.3f} {r['mean_pres']:<15.4f} {r['mean_non']:<15.4f} {r['mean_diff']:<12.4f}")
    
    print("="*80)
    
    # Save results
    results_dict = {
        'thresholds': thresholds,
        'mean_pres': mean_pres,
        'mean_non': mean_non,
        'mean_diff': mean_diff,
        'all_pres_durs': all_pres_durs,
        'all_non_durs': all_non_durs,
        'all_diff_durs': all_diff_durs,
        'model_results': all_model_results
    }
    
    # Save to file
    import pickle
    with open('rnn_analysis_results.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    
    print("\n Results saved to 'rnn_analysis_results.pkl'")
    
else:
    print("\n No models were successfully processed!")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import pickle
import pandas as pd

# Set seaborn style for publication-ready plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

# =============================================
# Load your real data results
# =============================================

with open('./Bayesian_decoding/real_data_results_load1.pkl', 'rb') as f:
    real_results = pickle.load(f)

print("="*60)
print("LOADED REAL DATA RESULTS")
print("="*60)

# Extract trial-level data (45 trials)
real_pres = np.array(real_results['pres']).flatten()
real_non = np.array(real_results['non']).flatten()
real_diff = np.array(real_results['diff']).flatten()

print(f"Number of real trials: {len(real_pres)}")

# Calculate real data statistics
real_mean_pres = np.mean(real_pres)
real_mean_non = np.mean(real_non)
real_mean_diff = np.mean(real_diff)
real_std_pres = np.std(real_pres)
real_std_non = np.std(real_non)
real_std_diff = np.std(real_diff)

print(f"\nReal data summary (n={len(real_pres)} trials):")
print(f"  Presented duration: {real_mean_pres:.4f} ± {real_std_pres:.4f}s")
print(f"  Non-presented duration: {real_mean_non:.4f} ± {real_std_non:.4f}s")
print(f"  Difference: {real_mean_diff:.4f} ± {real_std_diff:.4f}s")

# =============================================
# RNN results
# =============================================

with open('./rnn_analysis_results.pkl', 'rb') as f:
    rnn_results = pickle.load(f)

print("\n" + "="*60)
print("LOADED RNN RESULTS")
print("="*60)
print(f"Number of RNN models: {len(rnn_results['model_results'])}")

# Extract per-model means (42 models)
rnn_model_means_pres = np.array([r['mean_pres'] for r in rnn_results['model_results']])
rnn_model_means_non = np.array([r['mean_non'] for r in rnn_results['model_results']])
rnn_model_means_diff = np.array([r['mean_diff'] for r in rnn_results['model_results']])

print(f"  RNN models: {len(rnn_model_means_pres)}")

# =============================================
# Time rescaling
# =============================================

rnn_delay_duration = 0.75   # seconds
real_delay_duration = 2.5   # seconds
time_scale = real_delay_duration / rnn_delay_duration  # 3.333...

print("\n" + "="*60)
print("TIME RESCALING")
print("="*60)
print(f"RNN delay duration: {rnn_delay_duration}s")
print(f"Real delay duration: {real_delay_duration}s")
print(f"Time scale factor: {time_scale:.3f}")

# Rescale RNN model means
rnn_model_means_pres_rescaled = rnn_model_means_pres * time_scale
rnn_model_means_non_rescaled = rnn_model_means_non * time_scale
rnn_model_means_diff_rescaled = rnn_model_means_diff * time_scale

print(f"\nAfter rescaling (RNN model-level means):")
print(f"  RNN presented duration: {np.mean(rnn_model_means_pres_rescaled):.4f} ± {np.std(rnn_model_means_pres_rescaled):.4f}s")
print(f"  RNN non-presented duration: {np.mean(rnn_model_means_non_rescaled):.4f} ± {np.std(rnn_model_means_non_rescaled):.4f}s")
print(f"  RNN difference: {np.mean(rnn_model_means_diff_rescaled):.4f} ± {np.std(rnn_model_means_diff_rescaled):.4f}s")

# Mann-Whitney U tests
stat_pres, p_pres = mannwhitneyu(real_pres, rnn_model_means_pres_rescaled, alternative='two-sided')
stat_non, p_non = mannwhitneyu(real_non, rnn_model_means_non_rescaled, alternative='two-sided')
stat_diff, p_diff = mannwhitneyu(real_diff, rnn_model_means_diff_rescaled, alternative='two-sided')

print(f"\nPresented duration:")
print(f"  Mann-Whitney U = {stat_pres:.1f}, p = {p_pres:.6f}")

print(f"\nNon-presented duration:")
print(f"  Mann-Whitney U = {stat_non:.1f}, p = {p_non:.6f}")

print(f"\nDifference (presented - non-presented):")
print(f"  Mann-Whitney U = {stat_diff:.1f}, p = {p_diff:.6f}")

# Effect size
def cohen_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / pooled_std

d_pres = cohen_d(real_pres, rnn_model_means_pres_rescaled)
d_non = cohen_d(real_non, rnn_model_means_non_rescaled)
d_diff = cohen_d(real_diff, rnn_model_means_diff_rescaled)

print(f"\nEffect sizes (Cohen's d):")
print(f"  Presented duration: d = {d_pres:.3f}")
print(f"  Non-presented duration: d = {d_non:.3f}")
print(f"  Difference: d = {d_diff:.3f}")

data_pres = pd.DataFrame({
    'Duration': np.concatenate([real_pres, rnn_model_means_pres_rescaled]),
    'Group': ['Real Data'] * len(real_pres) + ['RNN Models'] * len(rnn_model_means_pres_rescaled),
    'Metric': 'Presented Duration'
})

data_non = pd.DataFrame({
    'Duration': np.concatenate([real_non, rnn_model_means_non_rescaled]),
    'Group': ['Real Data'] * len(real_non) + ['RNN Models'] * len(rnn_model_means_non_rescaled),
    'Metric': 'Non-presented Duration'
})

data_diff = pd.DataFrame({
    'Duration': np.concatenate([real_diff, rnn_model_means_diff_rescaled]),
    'Group': ['Real Data'] * len(real_diff) + ['RNN Models'] * len(rnn_model_means_diff_rescaled),
    'Metric': 'Difference (Presented - Non-presented)'
})

all_data = pd.concat([data_pres, data_non, data_diff], ignore_index=True)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = {'Real Data': '#3498db', 'RNN Models': '#e74c3c'}

# Plot each metric
for idx, metric in enumerate(['Presented Duration', 'Non-presented Duration', 'Difference (Presented - Non-presented)']):
    ax = axes[idx]
    
    # Subset data
    subset = all_data[all_data['Metric'] == metric]
    
    # Create boxplot
    sns.boxplot(data=subset, x='Group', y='Duration', 
                palette=colors, width=0.6, 
                fliersize=0, linewidth=2, ax=ax)
    
    # Add swarmplot for individual points
    sns.swarmplot(data=subset, x='Group', y='Duration', 
                  color='black', size=4, alpha=0.6, ax=ax)
    
    # Add mean markers
    means = subset.groupby('Group')['Duration'].mean()
    for i, group in enumerate(['Real Data', 'RNN Models']):
        ax.scatter(i, means[group], color='black', s=100, marker='D', zorder=5, 
                  edgecolors='white', linewidth=1.5)
    
    # Customize appearance
    ax.set_xlabel('')
    ax.set_ylim(-0.5, 2.5)
    ax.set_ylabel('Duration (s)', fontsize=12)
    ax.set_title(metric, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add significance annotation
    p_val = p_pres if metric == 'Presented Duration' else (p_non if metric == 'Non-presented Duration' else p_diff)
    
    if p_val < 0.05:
        y_max = subset['Duration'].max()
        y_min = subset['Duration'].min()
        y_range = y_max - y_min
        
        # Add bracket
        ax.plot([0, 0, 1, 1], [y_max + y_range*0.02, y_max + y_range*0.08, 
                                 y_max + y_range*0.08, y_max + y_range*0.02], 
                color='black', linewidth=1.5)
        
        # Add p-value text
        if p_val < 0.005:
            p_text = '***'
        elif p_val < 0.01:
            p_text = '**'
        else:
            p_text = '*'
        
        ax.text(0.5, y_max + y_range*0.12, p_text, ha='center', fontsize=16, 
                fontweight='bold')

plt.ylim(-0.5, 2.5)
plt.suptitle(f'Model-Level Comparison: Real Data vs RNN Models\n(Rescaled from {rnn_delay_duration}s to {real_delay_duration}s delay)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Save as EPS 
plt.savefig('/home/daria/Desktop/real_vs_rnn_model_level.eps', format='eps', dpi=300, bbox_inches='tight')
print("\n Plots saved as:")
print("   - real_vs_rnn_model_level.eps")

plt.show()

comparison_results = {
    'real_data': {
        'pres_durations': real_pres.tolist(),
        'non_durations': real_non.tolist(),
        'diff_durations': real_diff.tolist(),
        'mean_pres': float(real_mean_pres),
        'mean_non': float(real_mean_non),
        'mean_diff': float(real_mean_diff),
        'std_pres': float(real_std_pres),
        'std_non': float(real_std_non),
        'std_diff': float(real_std_diff),
        'n_trials': len(real_pres)
    },
    'rnn_models': {
        'pres_means': rnn_model_means_pres_rescaled.tolist(),
        'non_means': rnn_model_means_non_rescaled.tolist(),
        'diff_means': rnn_model_means_diff_rescaled.tolist(),
        'mean_pres': float(np.mean(rnn_model_means_pres_rescaled)),
        'mean_non': float(np.mean(rnn_model_means_non_rescaled)),
        'mean_diff': float(np.mean(rnn_model_means_diff_rescaled)),
        'std_pres': float(np.std(rnn_model_means_pres_rescaled)),
        'std_non': float(np.std(rnn_model_means_non_rescaled)),
        'std_diff': float(np.std(rnn_model_means_diff_rescaled)),
        'n_models': len(rnn_model_means_pres_rescaled)
    },
    'statistical_tests': {
        'presented': {'p_value': float(p_pres), 'cohens_d': float(d_pres)},
        'non_presented': {'p_value': float(p_non), 'cohens_d': float(d_non)},
        'difference': {'p_value': float(p_diff), 'cohens_d': float(d_diff)}
    },
    'time_scaling': {
        'rnn_delay': rnn_delay_duration,
        'real_delay': real_delay_duration,
        'scale_factor': time_scale
    }
}

with open('real_vs_rnn_comparison.pkl', 'wb') as f:
    pickle.dump(comparison_results, f)

print(f"\nTime scaling: {rnn_delay_duration}s → {real_delay_duration}s (factor {time_scale:.2f})")

print(f"\nPresented duration:")
print(f"  Real: {real_mean_pres:.3f} ± {real_std_pres:.3f}s")
print(f"  RNN: {np.mean(rnn_model_means_pres_rescaled):.3f} ± {np.std(rnn_model_means_pres_rescaled):.3f}s")
print(f"  p = {p_pres:.4f}, d = {abs(d_pres):.2f}")

print(f"\nNon-presented duration:")
print(f"  Real: {real_mean_non:.3f} ± {real_std_non:.3f}s")
print(f"  RNN: {np.mean(rnn_model_means_non_rescaled):.3f} ± {np.std(rnn_model_means_non_rescaled):.3f}s")
print(f"  p = {p_non:.4f}, d = {abs(d_non):.2f}")

print(f"\nDifference:")
print(f"  Real: {real_mean_diff:.3f} ± {real_std_diff:.3f}s")
print(f"  RNN: {np.mean(rnn_model_means_diff_rescaled):.3f} ± {np.std(rnn_model_means_diff_rescaled):.3f}s")
print(f"  p = {p_diff:.4f}, d = {abs(d_diff):.2f}")
