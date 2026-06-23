import os
import ast
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import ttest_1samp, wilcoxon

# =========================
# PATHS
# =========================
PROJECT_DIR = "../"
CLEAN_DIR = os.path.join(PROJECT_DIR, "clean_data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "Bayesian_decoder_outputs_minimal")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRIAL_INFO_PATH = os.path.join(PROJECT_DIR, "trial_info.xlsx")
ENC1_PATH = os.path.join(CLEAN_DIR, "cleaned_Encoding1.xlsx")
ENC2_PATH = os.path.join(CLEAN_DIR, "cleaned_Encoding2.xlsx")
ENC3_PATH = os.path.join(CLEAN_DIR, "cleaned_Encoding3.xlsx")
DELAY_PATH = os.path.join(CLEAN_DIR, "cleaned_Delay.xlsx")
PROBE_PATH = os.path.join(CLEAN_DIR, "cleaned_Probe.xlsx")
CLUSTERING_PATH = os.path.join("./all_neuron_brain_regions_merged.xlsx")

# =========================
# SETTINGS
# =========================
BIN_SIZE = 0.25
STEP_SIZE = 0.10
DELAY_DUR = 2.5
N_LOAD1_TRIALS_TO_PLOT = 5
EXCLUDE_SUBJECTS = [6, 7, 8, 9, 16, 19]

TRIAL_COL = "trial_id"
NEURON_COL = "Neuron_ID_3"
SPIKE_COL = "Standardized_Spikes"

# Thresholds are derived from shuffled-label null decoders
# The threshold for each load is the THRESHOLD_PERCENTILE of null confidence values.
N_THRESHOLD_SHUFFLES = 100
THRESHOLD_PERCENTILE = 95
THRESHOLD_RANDOM_SEED = 0

print("Output directory:", OUTPUT_DIR)

trial_info = pd.read_excel(TRIAL_INFO_PATH)
df_enc1 = pd.read_excel(ENC1_PATH)
df_enc2 = pd.read_excel(ENC2_PATH)
df_enc3 = pd.read_excel(ENC3_PATH)
df_delay = pd.read_excel(DELAY_PATH)
df_probe = pd.read_excel(PROBE_PATH)
df_clustering = pd.read_excel(CLUSTERING_PATH)

y_matrix = trial_info[
    trial_info["subject_id"] == 14
][[
    "trial_id",
    "num_images_presented",
    "stimulus_index_enc1",
    "stimulus_index_enc2",
    "stimulus_index_enc3",
    "response_accuracy",
]].reset_index(drop=True)

# Pooled concept cells, excluding subjects with less than 135 trials 
neuron_ids = df_clustering[
    (df_clustering["Signi"] == "Y") &
    (~df_clustering["subject_id"].isin(EXCLUDE_SUBJECTS))
][NEURON_COL].dropna().unique()

# Keep original order returned by pandas unique(), to avoid changing column order.
neuron_ids = np.asarray(neuron_ids)

print("y_matrix shape:", y_matrix.shape)
print("n pooled concept cells:", len(neuron_ids))
print("first 10 neuron IDs:", neuron_ids[:10])

check_subjects = df_clustering[df_clustering[NEURON_COL].isin(neuron_ids)]["subject_id"].dropna().unique()
print("subjects represented in selected neurons:", sorted(check_subjects))

def parse_spike_times(spike_str):
    try:
        return ast.literal_eval(spike_str)
    except Exception:
        return []

trial_count = y_matrix.shape[0]
neuron_count = len(neuron_ids)
design_matrix = np.empty((trial_count, neuron_count), dtype=object)
print("design_matrix shape:", design_matrix.shape)

for trial_idx, row in y_matrix.iterrows():
    trial_id = row[TRIAL_COL]
    num_images = row["num_images_presented"]

    for neuron_idx, neuron in enumerate(neuron_ids):
        all_spikes = []

        # This intentionally matches the original notebook: one pooled object matrix
        # with spikes gathered from the available task epochs.
        for df in [df_enc1, df_enc2, df_enc3, df_delay, df_probe]:
            if (df is df_enc2 and num_images < 2) or (df is df_enc3 and num_images < 3):
                continue

            match = df[
                (df[TRIAL_COL] == trial_id) &
                (df[NEURON_COL] == neuron)
            ]

            if not match.empty:
                all_spikes.extend(parse_spike_times(match[SPIKE_COL].values[0]))

        design_matrix[trial_idx, neuron_idx] = np.asarray(all_spikes, dtype=float)

# =========================
# BINNING + POISSON BAYESIAN DECODER
# =========================
def create_time_bins(duration, bin_size=BIN_SIZE, step_size=STEP_SIZE):
    # IMPORTANT: this matches your original notebook exactly.
    return np.arange(0, duration - bin_size + step_size, step_size)


def count_spikes_in_bins(spike_times, time_bins, bin_size=BIN_SIZE):
    return np.array([
        np.sum((spike_times >= t) & (spike_times < t + bin_size))
        for t in time_bins
    ], dtype=float)


def build_binned_counts_matrix(design_matrix, trial_indices, total_duration, bin_size=BIN_SIZE, step_size=STEP_SIZE):
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
    lambda_ = np.zeros((len(classes), X_train.shape[1]), dtype=float)
    log_prior = np.zeros(len(classes), dtype=float)

    for ki, c in enumerate(classes):
        Xc = X_train[y_train == c]
        lam = Xc.mean(axis=0)
        lam = np.clip(lam, eps, None)
        lambda_[ki] = lam
        log_prior[ki] = np.log(len(Xc) / len(X_train))

    log_lambda = np.log(lambda_)
    return classes, log_lambda, lambda_, log_prior


def scores_poisson_nb(x, log_lambda, lambda_, log_prior):
    # log(x!) is omitted because it is identical across classes for the same x.
    ll = (x[None, :] * log_lambda - lambda_).sum(axis=1)
    return ll + log_prior


def posterior_from_scores(scores):
    log_post = scores - logsumexp(scores)
    return np.exp(log_post)


def get_epoch_windows(load):
    return {
        "enc1": (0.0, 1.0),
        "enc2": (1.0, 2.0),
        "enc3": (2.0, 3.0),
        "delay_start": load + 0.2,
    }


def bins_in_window(bin_starts, start, end):
    return np.where((bin_starts >= start) & (bin_starts < end))[0]


def encoding_epoch_trained_delay_decode_loocv(
    design_matrix,
    trial_indices,
    y_labels,
    load,
    train_epoch,
    delay_duration=DELAY_DUR,
    bin_size=BIN_SIZE,
    step_size=STEP_SIZE,
    eps=1e-6,
):
    """
    Original-matched logic:
      - build binned counts from the pooled design_matrix
      - train only on enc1/enc2/enc3 window depending on load
      - test across delay bins
      - LOOCV across trials
    """
    total_duration = load + 0.2 + delay_duration
    X, bins = build_binned_counts_matrix(
        design_matrix=design_matrix,
        trial_indices=trial_indices,
        total_duration=total_duration,
        bin_size=bin_size,
        step_size=step_size,
    )

    windows = get_epoch_windows(load)
    train_start, train_end = windows[train_epoch]
    delay_start = windows["delay_start"]

    train_idx = bins_in_window(bins, train_start, train_end)
    delay_idx = np.where(bins >= delay_start)[0]
    bins_delay = bins[delay_idx] - delay_start

    if len(train_idx) == 0:
        raise ValueError(f"No training bins found for {train_epoch}.")
    if len(delay_idx) == 0:
        raise ValueError("No delay bins found.")

    classes_all = np.unique(y_labels)
    K = len(classes_all)
    n_trials = X.shape[0]
    n_delay = len(delay_idx)

    post_delay = np.zeros((n_trials, n_delay, K), dtype=float)
    score_delay = np.full((n_trials, n_delay, K), -np.inf, dtype=float)

    for test_i in range(n_trials):
        train_mask = np.ones(n_trials, dtype=bool)
        train_mask[test_i] = False

        X_train_epoch = X[train_mask, :, :][:, :, train_idx]
        X_train_epoch = np.transpose(X_train_epoch, (0, 2, 1)).reshape(-1, X.shape[1])
        y_train_epoch = np.repeat(y_labels[train_mask], len(train_idx))

        classes_fold, log_lambda, lambda_, log_prior = fit_poisson_nb(
            X_train_epoch,
            y_train_epoch,
            eps=eps,
        )

        fold_to_global = {
            fold_i: int(np.where(classes_all == c)[0][0])
            for fold_i, c in enumerate(classes_fold)
            if c in classes_all
        }

        for t_i, t in enumerate(delay_idx):
            x = X[test_i, :, t]
            s_fold = scores_poisson_nb(x, log_lambda, lambda_, log_prior)

            s_global = np.full(K, -np.inf, dtype=float)
            for fold_i, glob_i in fold_to_global.items():
                s_global[glob_i] = s_fold[fold_i]

            score_delay[test_i, t_i, :] = s_global
            post_delay[test_i, t_i, :] = posterior_from_scores(s_global)

    return bins_delay, post_delay, score_delay, classes_all

# =========================
# RUN DECODER ACROSS LOADS
# =========================
def get_load_trials_labels_epoch(load):
    trials = y_matrix[y_matrix["num_images_presented"] == load].index.values

    if load == 1:
        label_col = "stimulus_index_enc1"
        train_epoch = "enc1"
    elif load == 2:
        label_col = "stimulus_index_enc2"   # last encoded item for load 2
        train_epoch = "enc2"
    elif load == 3:
        label_col = "stimulus_index_enc3"   # last encoded item for load 3
        train_epoch = "enc3"
    else:
        raise ValueError("load must be 1, 2, or 3")

    y = y_matrix.loc[trials, label_col].values
    return trials, y, train_epoch, label_col


results = {}

for load in [1, 2, 3]:
    trials, y, train_epoch, label_col = get_load_trials_labels_epoch(load)

    print(f"\n================ LOAD {load} ================")
    print(f"Training window: {train_epoch}")
    print(f"Decoded label: {label_col}")
    print(f"n trials: {len(trials)}")
    print(f"classes: {np.unique(y)}")

    bins_delay, post_delay, score_delay, classes = encoding_epoch_trained_delay_decode_loocv(
        design_matrix=design_matrix,
        trial_indices=trials,
        y_labels=y,
        load=load,
        train_epoch=train_epoch,
        delay_duration=DELAY_DUR,
        bin_size=BIN_SIZE,
        step_size=STEP_SIZE,
    )

    pred_idx = np.argmax(post_delay, axis=2)
    pred_labels = classes[pred_idx]
    correct_matrix = (pred_labels == y[:, None]).astype(float)
    acc_by_time = correct_matrix.mean(axis=0)

    chance = 1 / len(classes)
    p_vals = np.array([
        ttest_1samp(correct_matrix[:, t], chance).pvalue
        for t in range(correct_matrix.shape[1])
    ])
    significant = p_vals < 0.05

    print(f"Mean accuracy across delay: {acc_by_time.mean():.3f}")
    print(f"Max accuracy across delay:  {acc_by_time.max():.3f} at {bins_delay[np.argmax(acc_by_time)]:.2f} s")
    print(f"Chance level:               {chance:.3f}")
    print(f"Significant time bins p<.05: {significant.sum()} / {len(significant)}")

    results[load] = {
        "load": load,
        "train_epoch": train_epoch,
        "label_col": label_col,
        "trials": trials,
        "trial_ids": y_matrix.loc[trials, TRIAL_COL].values,
        "y": y,
        "classes": classes,
        "bins_delay": bins_delay,
        "post_delay": post_delay,
        "score_delay": score_delay,
        "pred_labels": pred_labels,
        "correct_matrix": correct_matrix,
        "acc_by_time": acc_by_time,
        "p_vals": p_vals,
        "significant": significant,
        "mean_accuracy": float(acc_by_time.mean()),
        "max_accuracy": float(acc_by_time.max()),
        "best_time": float(bins_delay[np.argmax(acc_by_time)]),
        "chance": float(chance),
    }

# =========================
# DECODING ACCURACY SUMMARY
# =========================
accuracy_summary = pd.DataFrame([
    {
        "load": load,
        "train_epoch": results[load]["train_epoch"],
        "decoded_label": results[load]["label_col"],
        "n_trials": len(results[load]["trials"]),
        "n_classes": len(results[load]["classes"]),
        "chance": results[load]["chance"],
        "mean_accuracy": results[load]["mean_accuracy"],
        "max_accuracy": results[load]["max_accuracy"],
        "best_time": results[load]["best_time"],
        "n_sig_bins_p05": int(results[load]["significant"].sum()),
    }
    for load in [1, 2, 3]
])

print(accuracy_summary)

accuracy_csv = os.path.join(OUTPUT_DIR, "bayesian_accuracy_summary.csv")
accuracy_summary.to_csv(accuracy_csv, index=False)
print("Saved accuracy summary:", accuracy_csv)

# =========================
# POSTERIOR PLOTS FOR FIRST LOAD 1 TRIALS
# =========================
def plot_first_load1_posteriors(results, n_trials=N_LOAD1_TRIALS_TO_PLOT, save_dir=OUTPUT_DIR):
    data = results[1]
    bins = data["bins_delay"]
    post = data["post_delay"]
    classes = data["classes"]
    trial_ids = data["trial_ids"]
    y = data["y"]

    chosen_local = np.arange(min(n_trials, post.shape[0]))

    for i in chosen_local:
        trial_id = trial_ids[i]
        true_stim = y[i]

        plt.figure(figsize=(8, 3))
        for class_i, stim_id in enumerate(classes):
            lw = 3 if stim_id == true_stim else 1.5
            plt.plot(
                bins,
                post[i, :, class_i],
                linewidth=lw,
                label=f"Stim {stim_id}",
            )

        plt.xlabel("Delay time (s)")
        plt.ylabel("Posterior probability")
        plt.ylim(0, 1)
        plt.title(f"Load 1 pooled Bayesian posterior | trial {trial_id}\ntrue stimulus = {true_stim}")
        plt.legend(title="Stimulus identity", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        fig_path = os.path.join(save_dir, f"load1_trial_{trial_id}_posterior.png")
        plt.savefig(fig_path, dpi=300)
        plt.show()
        print("Saved posterior plot:", fig_path)


plot_first_load1_posteriors(results, n_trials=N_LOAD1_TRIALS_TO_PLOT, save_dir=OUTPUT_DIR)

# =========================
# PRESENTED-ITEM DURATION FROM SCORE MARGINS
# =========================
def get_presented_items(y_matrix, global_trial_idx, load):
    if load == 1:
        return [y_matrix.loc[global_trial_idx, "stimulus_index_enc1"]]
    elif load == 2:
        return [
            y_matrix.loc[global_trial_idx, "stimulus_index_enc1"],
            y_matrix.loc[global_trial_idx, "stimulus_index_enc2"],
        ]
    elif load == 3:
        return [
            y_matrix.loc[global_trial_idx, "stimulus_index_enc1"],
            y_matrix.loc[global_trial_idx, "stimulus_index_enc2"],
            y_matrix.loc[global_trial_idx, "stimulus_index_enc3"],
        ]
    else:
        raise ValueError("load must be 1, 2, or 3")


def pred_and_confidence(score_delay_trial):
    pred_idx = np.argmax(score_delay_trial, axis=1)
    s_sorted = np.sort(score_delay_trial, axis=1)
    conf = s_sorted[:, -1] - s_sorted[:, -2]
    return pred_idx, conf


def duration_per_item_presented_nonpresented(
    score_delay,
    classes,
    trials,
    y_matrix,
    load,
    step_size=STEP_SIZE,
    conf_thr=0.0,
    min_bins=2,
):
    pres_out = []
    non_out = []

    for ti in range(score_delay.shape[0]):
        global_trial_idx = trials[ti]
        presented = set(get_presented_items(y_matrix, global_trial_idx, load))
        n_pres_items = len(presented)
        n_non_items = len(classes) - n_pres_items

        pred_idx, conf = pred_and_confidence(score_delay[ti])
        above = conf >= conf_thr

        pres_dur = 0.0
        non_dur = 0.0
        i = 0

        while i < len(conf):
            if above[i]:
                start = i
                while i < len(conf) and above[i]:
                    i += 1
                end = i - 1

                if (end - start + 1) < min_bins:
                    continue

                seg_pred = pred_idx[start:end + 1]
                ev_class_idx = int(np.bincount(seg_pred).argmax())
                ev_label = classes[ev_class_idx]
                dur = (end - start + 1) * step_size

                if ev_label in presented:
                    pres_dur += dur
                else:
                    non_dur += dur
            else:
                i += 1

        pres_out.append(pres_dur / n_pres_items if n_pres_items > 0 else 0.0)
        non_out.append(non_dur / n_non_items if n_non_items > 0 else 0.0)

    return np.asarray(pres_out), np.asarray(non_out), np.asarray(pres_out) - np.asarray(non_out)

# =========================
# DERIVE CONFIDENCE THRESHOLDS FROM SHUFFLED-LABEL NULLS
# =========================
def collect_confidences_from_score_delay(score_delay):
    """Confidence metric used for real-data duration extraction: top log-score minus second log-score."""
    conf_all = []
    for ti in range(score_delay.shape[0]):
        _, conf = pred_and_confidence(score_delay[ti])
        conf_all.extend(conf)
    return np.asarray(conf_all, dtype=float)


def derive_threshold_for_load(
    load,
    results,
    design_matrix,
    n_shuffles=N_THRESHOLD_SHUFFLES,
    percentile=THRESHOLD_PERCENTILE,
    seed=THRESHOLD_RANDOM_SEED,
):
    """
    Derive the duration/event threshold for one load using shuffled labels.

    Rerun the same LOOCV Bayesian decoder
    with shuffled labels, collect the decoder confidence during the delay, and
    use a high percentile of that null distribution as the threshold.

    Here confidence is top log-score minus second-highest log-score.
    """
    data = results[load]
    rng = np.random.default_rng(seed + load)

    null_confidences = []

    for shuf_i in range(n_shuffles):
        y_shuf = rng.permutation(data["y"])

        _, _, score_delay_shuf, _ = encoding_epoch_trained_delay_decode_loocv(
            design_matrix=design_matrix,
            trial_indices=data["trials"],
            y_labels=y_shuf,
            load=load,
            train_epoch=data["train_epoch"],
            delay_duration=DELAY_DUR,
            bin_size=BIN_SIZE,
            step_size=STEP_SIZE,
        )

        null_confidences.extend(collect_confidences_from_score_delay(score_delay_shuf))

        if (shuf_i + 1) % 20 == 0 or (shuf_i + 1) == n_shuffles:
            print(f"Load {load}: completed {shuf_i + 1}/{n_shuffles} threshold shuffles")

    null_confidences = np.asarray(null_confidences, dtype=float)
    threshold = float(np.percentile(null_confidences, percentile))

    real_confidences = collect_confidences_from_score_delay(data["score_delay"])

    return {
        "load": load,
        "threshold": threshold,
        "percentile": percentile,
        "n_shuffles": n_shuffles,
        "null_confidences": null_confidences,
        "real_confidences": real_confidences,
        "null_mean": float(np.mean(null_confidences)),
        "null_std": float(np.std(null_confidences)),
        "real_mean": float(np.mean(real_confidences)),
        "real_std": float(np.std(real_confidences)),
    }


threshold_results = {}
CONF_THR_BY_LOAD = {}

for load in [1, 2, 3]:
    threshold_results[load] = derive_threshold_for_load(
        load=load,
        results=results,
        design_matrix=design_matrix,
        n_shuffles=N_THRESHOLD_SHUFFLES,
        percentile=THRESHOLD_PERCENTILE,
        seed=THRESHOLD_RANDOM_SEED,
    )
    CONF_THR_BY_LOAD[load] = threshold_results[load]["threshold"]

print("\nDerived confidence thresholds:")
for load, thr in CONF_THR_BY_LOAD.items():
    print(f"  Load {load}: {thr:.4f}")

threshold_pkl_path = os.path.join(OUTPUT_DIR, "bayesian_thresholds_from_shuffled_nulls.pkl")
with open(threshold_pkl_path, "wb") as f:
    pickle.dump(threshold_results, f)

# =========================
# DERIVE CONFIDENCE THRESHOLDS FROM SHUFFLED-LABEL NULLS
# =========================
def collect_confidences_from_score_delay(score_delay):
    """Confidence metric used for real-data duration extraction: top log-score minus second log-score."""
    conf_all = []
    for ti in range(score_delay.shape[0]):
        _, conf = pred_and_confidence(score_delay[ti])
        conf_all.extend(conf)
    return np.asarray(conf_all, dtype=float)


def derive_threshold_for_load(
    load,
    results,
    design_matrix,
    n_shuffles=N_THRESHOLD_SHUFFLES,
    percentile=THRESHOLD_PERCENTILE,
    seed=THRESHOLD_RANDOM_SEED,
):
    """
    Derive the duration/event threshold for one load using shuffled labels.

    Rerun the same LOOCV Bayesian decoder
    with shuffled labels, collect the decoder confidence during the delay, and
    use a high percentile of that null distribution as the threshold.

    Here confidence is top log-score minus second-highest log-score.
    """
    data = results[load]
    rng = np.random.default_rng(seed + load)

    null_confidences = []

    for shuf_i in range(n_shuffles):
        y_shuf = rng.permutation(data["y"])

        _, _, score_delay_shuf, _ = encoding_epoch_trained_delay_decode_loocv(
            design_matrix=design_matrix,
            trial_indices=data["trials"],
            y_labels=y_shuf,
            load=load,
            train_epoch=data["train_epoch"],
            delay_duration=DELAY_DUR,
            bin_size=BIN_SIZE,
            step_size=STEP_SIZE,
        )

        null_confidences.extend(collect_confidences_from_score_delay(score_delay_shuf))

        if (shuf_i + 1) % 20 == 0 or (shuf_i + 1) == n_shuffles:
            print(f"Load {load}: completed {shuf_i + 1}/{n_shuffles} threshold shuffles")

    null_confidences = np.asarray(null_confidences, dtype=float)
    threshold = float(np.percentile(null_confidences, percentile))

    real_confidences = collect_confidences_from_score_delay(data["score_delay"])

    return {
        "load": load,
        "threshold": threshold,
        "percentile": percentile,
        "n_shuffles": n_shuffles,
        "null_confidences": null_confidences,
        "real_confidences": real_confidences,
        "null_mean": float(np.mean(null_confidences)),
        "null_std": float(np.std(null_confidences)),
        "real_mean": float(np.mean(real_confidences)),
        "real_std": float(np.std(real_confidences)),
    }


threshold_results = {}
CONF_THR_BY_LOAD = {}

for load in [1, 2, 3]:
    threshold_results[load] = derive_threshold_for_load(
        load=load,
        results=results,
        design_matrix=design_matrix,
        n_shuffles=N_THRESHOLD_SHUFFLES,
        percentile=THRESHOLD_PERCENTILE,
        seed=THRESHOLD_RANDOM_SEED,
    )
    CONF_THR_BY_LOAD[load] = threshold_results[load]["threshold"]

print("\nDerived confidence thresholds:")
for load, thr in CONF_THR_BY_LOAD.items():
    print(f"  Load {load}: {thr:.4f}")

threshold_pkl_path = os.path.join(OUTPUT_DIR, "bayesian_thresholds_from_shuffled_nulls.pkl")
with open(threshold_pkl_path, "wb") as f:
    pickle.dump(threshold_results, f)

# =========================
# SAVE PICKLE + CSV OUTPUTS
# =========================
duration_results = {}

for load in [1, 2, 3]:
    data = results[load]
    conf_thr = CONF_THR_BY_LOAD[load]

    pres, non, diff = duration_per_item_presented_nonpresented(
        score_delay=data["score_delay"],
        classes=data["classes"],
        trials=data["trials"],
        y_matrix=y_matrix,
        load=load,
        step_size=STEP_SIZE,
        conf_thr=conf_thr,
        min_bins=2,
    )

    # Wilcoxon can fail if all differences are zero; handle cleanly.
    try:
        stat, p = wilcoxon(pres, non)
    except ValueError:
        stat, p = np.nan, np.nan

    out = {
        "load": load,
        "train_epoch": data["train_epoch"],
        "decoded_label": data["label_col"],
        "conf_thr": conf_thr,
        "trials": data["trials"],
        "trial_ids": data["trial_ids"],
        "y": data["y"],
        "classes": data["classes"],
        "bins_delay": data["bins_delay"],
        "presented_duration_per_item": pres,
        "nonpresented_duration_per_item": non,
        "difference_pres_minus_non": diff,
        "mean_presented_duration_per_item": float(np.mean(pres)),
        "mean_nonpresented_duration_per_item": float(np.mean(non)),
        "mean_difference_pres_minus_non": float(np.mean(diff)),
        "median_difference_pres_minus_non": float(np.median(diff)),
        "wilcoxon_stat": float(stat) if np.isfinite(stat) else np.nan,
        "wilcoxon_p": float(p) if np.isfinite(p) else np.nan,
        "acc_by_time": data["acc_by_time"],
        "mean_accuracy": data["mean_accuracy"],
        "max_accuracy": data["max_accuracy"],
        "best_time": data["best_time"],
    }

    duration_results[load] = out

    pkl_path = os.path.join(OUTPUT_DIR, f"bayesian_presented_item_duration_load{load}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(out, f)

    print(f"\nLoad {load}")
    print(f"mean presented duration/item:    {pres.mean():.4f} s")
    print(f"mean nonpresented duration/item: {non.mean():.4f} s")
    print(f"mean difference:                 {diff.mean():.4f} s")
    print(f"Wilcoxon p:                      {p}")

