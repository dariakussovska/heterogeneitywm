import pandas as pd
import numpy as np
import ast
from scipy.special import logsumexp

# =========================
# USER PATHS
# =========================

TRIAL_INFO_PATH = "/Users/darikussovska/Desktop/PROJECT/trial_info.xlsx"
CONCEPT_CELL_PATH = "/Users/darikussovska/Desktop/PROJECT/merged_significant_neurons_with_brain_regions.xlsx"

ENC1_PATH = "/Users/darikussovska/Desktop/PROJECT/encoding1.xlsx"
ENC2_PATH = "/Users/darikussovska/Desktop/PROJECT/encoding2.xlsx"
ENC3_PATH = "/Users/darikussovska/Desktop/PROJECT/encoding3.xlsx"
DELAY_PATH = "/Users/darikussovska/Desktop/PROJECT/maintenance.xlsx"

OUTPUT_PATH = "/Users/darikussovska/Desktop/PROJECT/trial_level_bayesian_decoder_margins.xlsx"

# =========================
# CONFIG
# =========================

BIN_SIZE = 0.25
STEP_SIZE = 0.10
DELAY_DUR = 2.5
MIN_CONCEPT_CELLS = 5

SUBJECT_COL = "subject_id"
TRIAL_COL = "trial_id"
NEURON_COL = "Neuron_ID"
SIGNI_COL = "Signi"

# Change these if your files use different spike columns
ENC_SPIKE_COL = "Standardized_Spikes_New"
DELAY_SPIKE_COL = "Standardized_Spikes_in_Delay"


# =========================
# BASIC UTILITIES
# =========================

def parse_spikes(x):
    if isinstance(x, (list, np.ndarray)):
        return np.asarray(x, dtype=float)

    if pd.isna(x):
        return np.array([], dtype=float)

    if isinstance(x, str):
        x = x.strip()
        if x in ["", "[]", "nan", "NaN"]:
            return np.array([], dtype=float)
        try:
            return np.asarray(ast.literal_eval(x), dtype=float)
        except Exception:
            return np.fromstring(x.replace("[", "").replace("]", ""), sep=" ")

    return np.array([], dtype=float)


def create_time_bins(duration, bin_size, step_size):
    return np.arange(0, duration - bin_size + 1e-9, step_size)


def count_spikes_in_bins(spike_times, time_bins, bin_size):
    spike_times = np.asarray(spike_times, dtype=float)
    return np.array([
        np.sum((spike_times >= t) & (spike_times < t + bin_size))
        for t in time_bins
    ], dtype=float)


# =========================
# POISSON NAIVE BAYES
# =========================

def fit_poisson_nb(X_train, y_train, eps=1e-6):
    classes = np.unique(y_train)
    n_neurons = X_train.shape[1]

    lambdas = np.zeros((len(classes), n_neurons))
    log_prior = np.zeros(len(classes))

    for i, c in enumerate(classes):
        Xc = X_train[y_train == c]
        lambdas[i] = np.clip(Xc.mean(axis=0), eps, None)
        log_prior[i] = np.log(len(Xc) / len(X_train))

    return classes, np.log(lambdas), lambdas, log_prior


def posterior_poisson_nb(x, log_lambda, lambdas, log_prior):
    ll = (x[None, :] * log_lambda - lambdas).sum(axis=1)
    log_post = ll + log_prior
    log_post -= logsumexp(log_post)
    return np.exp(log_post)


# =========================
# BUILD SUBJECT DESIGN MATRIX
# =========================

def build_subject_design_matrix(
    subject_id,
    trial_info,
    concept_neurons,
    enc1_df,
    enc2_df,
    enc3_df,
    delay_df,
):
    subject_trials = trial_info[trial_info[SUBJECT_COL] == subject_id].copy()
    subject_trials = subject_trials.sort_values(TRIAL_COL).reset_index(drop=True)

    neurons = sorted(concept_neurons)
    n_trials = len(subject_trials)
    n_neurons = len(neurons)

    design_matrix_by_load = {}

    for load in [1, 2, 3]:
        load_trials = subject_trials[subject_trials["num_images_presented"] == load].copy()
        load_trials = load_trials.reset_index(drop=True)

        design_matrix = np.empty((len(load_trials), n_neurons), dtype=object)

        for i, row in load_trials.iterrows():
            trial_id = row[TRIAL_COL]

            for j, neuron_id in enumerate(neurons):
                all_spikes = []

                # Encoding 1
                s1 = get_spikes(enc1_df, subject_id, trial_id, neuron_id, ENC_SPIKE_COL)
                all_spikes.extend(s1)

                # Encoding 2 shifted by +1.0 s
                if load >= 2:
                    s2 = get_spikes(enc2_df, subject_id, trial_id, neuron_id, ENC_SPIKE_COL)
                    all_spikes.extend(s2 + 1.0)

                # Encoding 3 shifted by +2.0 s
                if load >= 3:
                    s3 = get_spikes(enc3_df, subject_id, trial_id, neuron_id, ENC_SPIKE_COL)
                    all_spikes.extend(s3 + 2.0)

                # Delay shifted after encoding duration
                enc_dur = load + 0.2
                sd = get_spikes(delay_df, subject_id, trial_id, neuron_id, DELAY_SPIKE_COL)
                all_spikes.extend(sd + enc_dur)

                design_matrix[i, j] = np.asarray(all_spikes, dtype=float)

        design_matrix_by_load[load] = {
            "design_matrix": design_matrix,
            "trial_table": load_trials,
            "neurons": neurons,
        }

    return design_matrix_by_load


def get_spikes(df, subject_id, trial_id, neuron_id, spike_col):
    sub = df[
        (df[SUBJECT_COL] == subject_id) &
        (df[TRIAL_COL] == trial_id) &
        (df[NEURON_COL] == neuron_id)
    ]

    if len(sub) == 0:
        return np.array([], dtype=float)

    spikes = []
    for val in sub[spike_col].values:
        spikes.extend(parse_spikes(val))

    return np.asarray(spikes, dtype=float)


# =========================
# DECODER
# =========================

def build_binned_counts_matrix(design_matrix, total_duration, bin_size, step_size):
    bins = create_time_bins(total_duration, bin_size, step_size)

    n_trials, n_neurons = design_matrix.shape
    X = np.zeros((n_trials, n_neurons, len(bins)))

    for i in range(n_trials):
        for j in range(n_neurons):
            X[i, j, :] = count_spikes_in_bins(design_matrix[i, j], bins, bin_size)

    return X, bins


def encoding_trained_delay_posteriors_loocv(
    design_matrix,
    y_labels,
    enc_duration,
    delay_duration,
    bin_size=0.25,
    step_size=0.1,
    eps=1e-6,
):
    total_duration = enc_duration + delay_duration

    X, bins = build_binned_counts_matrix(
        design_matrix,
        total_duration,
        bin_size,
        step_size
    )

    enc_bins = len(create_time_bins(enc_duration, bin_size, step_size))
    enc_idx = np.arange(enc_bins)
    delay_idx = np.arange(enc_bins, X.shape[2])

    bins_delay = bins[delay_idx] - enc_duration

    classes_all = np.unique(y_labels)
    n_trials = X.shape[0]
    n_delay = len(delay_idx)
    K = len(classes_all)

    post_delay = np.zeros((n_trials, n_delay, K))

    for test_i in range(n_trials):
        train_mask = np.ones(n_trials, dtype=bool)
        train_mask[test_i] = False

        y_train_trial = y_labels[train_mask]

        # Skip fold if training set loses too many classes
        if len(np.unique(y_train_trial)) < 2:
            post_delay[test_i, :, :] = np.nan
            continue

        X_enc_train = X[train_mask][:, :, enc_idx]
        X_enc_train = np.transpose(X_enc_train, (0, 2, 1)).reshape(-1, X.shape[1])
        y_enc_train = np.repeat(y_train_trial, len(enc_idx))

        classes_fold, log_lambda, lambdas, log_prior = fit_poisson_nb(
            X_enc_train,
            y_enc_train,
            eps=eps
        )

        for t_i, t in enumerate(delay_idx):
            x = X[test_i, :, t]
            p_fold = posterior_poisson_nb(x, log_lambda, lambdas, log_prior)

            p_global = np.zeros(K)
            for fold_i, c in enumerate(classes_fold):
                global_i = np.where(classes_all == c)[0][0]
                p_global[global_i] = p_fold[fold_i]

            if p_global.sum() > 0:
                p_global /= p_global.sum()

            post_delay[test_i, t_i, :] = p_global

    return bins_delay, post_delay, classes_all


# =========================
# TRIAL-LEVEL MARGIN
# =========================

def compute_trial_margin(post_trial, classes, presented_items):
    presented_items = [x for x in presented_items if not pd.isna(x)]
    presented_items = np.asarray(presented_items)

    presented_mask = np.isin(classes, presented_items)
    nonpresented_mask = ~presented_mask

    if presented_mask.sum() == 0 or nonpresented_mask.sum() == 0:
        return np.nan, np.nan, np.nan

    presented_prob = np.nanmean(post_trial[:, presented_mask].sum(axis=1))
    nonpresented_max = np.nanmean(post_trial[:, nonpresented_mask].max(axis=1))

    margin = presented_prob - nonpresented_max

    return margin, presented_prob, nonpresented_max


# =========================
# MAIN SCRIPT
# =========================

trial_info = pd.read_excel(TRIAL_INFO_PATH)
concept_df = pd.read_excel(CONCEPT_CELL_PATH)

enc1_df = pd.read_excel(ENC1_PATH)
enc2_df = pd.read_excel(ENC2_PATH)
enc3_df = pd.read_excel(ENC3_PATH)
delay_df = pd.read_excel(DELAY_PATH)

concept_df = concept_df[concept_df[SIGNI_COL] == "Y"].copy()

all_rows = []

valid_subjects = []

for subject_id in sorted(trial_info[SUBJECT_COL].dropna().unique()):

    subject_concepts = concept_df[concept_df[SUBJECT_COL] == subject_id]
    concept_neurons = subject_concepts[NEURON_COL].dropna().unique()

    if len(concept_neurons) < MIN_CONCEPT_CELLS:
        continue

    valid_subjects.append(subject_id)

    print(f"Subject {subject_id}: {len(concept_neurons)} concept cells")

    design_by_load = build_subject_design_matrix(
        subject_id=subject_id,
        trial_info=trial_info,
        concept_neurons=concept_neurons,
        enc1_df=enc1_df,
        enc2_df=enc2_df,
        enc3_df=enc3_df,
        delay_df=delay_df,
    )

    for load in [1, 2, 3]:

        trial_table = design_by_load[load]["trial_table"]
        design_matrix = design_by_load[load]["design_matrix"]

        if len(trial_table) < 5:
            continue

        if load == 1:
            y = trial_table["stimulus_index_enc1"].values
        elif load == 2:
            y = trial_table["stimulus_index_enc2"].values
        else:
            y = trial_table["stimulus_index_enc3"].values

        valid = ~pd.isna(y)
        trial_table = trial_table[valid].reset_index(drop=True)
        design_matrix = design_matrix[valid]
        y = y[valid]

        if len(np.unique(y)) < 2:
            continue

        enc_dur = load + 0.2

        bins_delay, post_delay, classes = encoding_trained_delay_posteriors_loocv(
            design_matrix=design_matrix,
            y_labels=y,
            enc_duration=enc_dur,
            delay_duration=DELAY_DUR,
            bin_size=BIN_SIZE,
            step_size=STEP_SIZE,
        )

        for i, row in trial_table.iterrows():

            if load == 1:
                presented = [row["stimulus_index_enc1"]]
            elif load == 2:
                presented = [
                    row["stimulus_index_enc1"],
                    row["stimulus_index_enc2"],
                ]
            else:
                presented = [
                    row["stimulus_index_enc1"],
                    row["stimulus_index_enc2"],
                    row["stimulus_index_enc3"],
                ]

            margin, presented_prob, nonpresented_max = compute_trial_margin(
                post_trial=post_delay[i],
                classes=classes,
                presented_items=presented,
            )

            mean_posteriors = np.nanmean(post_delay[i], axis=0)
            predicted_class = classes[np.nanargmax(mean_posteriors)]

            all_rows.append({
                "subject_id": subject_id,
                "trial_id": row[TRIAL_COL],
                "num_images_presented": load,
                "n_concept_cells": len(concept_neurons),
                "stimulus_index_enc1": row.get("stimulus_index_enc1", np.nan),
                "stimulus_index_enc2": row.get("stimulus_index_enc2", np.nan),
                "stimulus_index_enc3": row.get("stimulus_index_enc3", np.nan),
                "response_accuracy": row.get("response_accuracy", np.nan),
                "decoder_classes": list(classes),
                "predicted_class_mean_delay": predicted_class,
                "mean_presented_posterior": presented_prob,
                "mean_max_nonpresented_posterior": nonpresented_max,
                "mean_bayesian_decoder_margin": margin,
            })

results_df = pd.DataFrame(all_rows)
results_df.to_excel(OUTPUT_PATH, index=False)

print(f"\nSaved trial-level decoder margins to:\n{OUTPUT_PATH}")
print(f"Valid subjects used: {valid_subjects}")
