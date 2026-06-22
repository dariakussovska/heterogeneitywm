# =========================
# IMPORTS + CONFIG
# =========================

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logsumexp

# ---- Edit these paths if needed ----
TRIAL_INFO_PATH = "../trial_info.xlsx"
CONCEPT_CELL_PATH = "../data/all_neuron_brain_regions_cleaned.xlsx"

ENC1_PATH = "../graph_data/graph_encoding1.xlsx"
ENC2_PATH = "../graph_data/graph_encoding2.xlsx"
ENC3_PATH = "../graph_data/graph_encoding3.xlsx"
DELAY_PATH = "../graph_data/graph_delay.xlsx"

# ---- Outputs ----
OUTPUT_DIR = "./bayesian_decoder_outputs"
POSTERIOR_PLOT_DIR = os.path.join(OUTPUT_DIR, "posterior_load1_examples")
os.makedirs(POSTERIOR_PLOT_DIR, exist_ok=True)

ACCURACY_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "bayesian_decoder_accuracy_by_subject_load.xlsx")
TRIAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "bayesian_decoder_trial_predictions.xlsx")

# ---- Decoder settings ----
BIN_SIZE = 0.25
STEP_SIZE = 0.10
DELAY_DUR = 2.5
MIN_CONCEPT_CELLS = 5
N_LOAD1_POSTERIOR_TRIALS = 6
RANDOM_STATE = 0

# ---- Column names ----
SUBJECT_COL = "subject_id"
NEURON_COL = "Neuron_ID"
SIGNI_COL = "Signi"
ENC_SPIKE_COL = "Standardized_Spikes"
DELAY_SPIKE_COL = "Standardized_Spikes"

# The original analysis decodes the last encoded item at each load:
# load 1 -> stimulus_index_enc1
# load 2 -> stimulus_index_enc2
# load 3 -> stimulus_index_enc3
LABEL_MODE = "last_encoded_item"

# =========================
# BASIC HELPERS
# =========================

def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

def infer_trial_col(df):
    if "trial_id" in df.columns:
        return "trial_id"
    if "trial_id_final" in df.columns:
        return "trial_id_final"
    raise KeyError("Could not find trial ID column. Expected 'trial_id' or 'trial_id_final'.")

def parse_spikes(x):
    if isinstance(x, (list, np.ndarray)):
        return np.asarray(x, dtype=float)

    if pd.isna(x):
        return np.array([], dtype=float)

    if isinstance(x, str):
        x = x.strip()
        if x in ["", "[]", "nan", "NaN", "None"]:
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

def get_label_col(load):
    if LABEL_MODE != "last_encoded_item":
        raise ValueError("Only LABEL_MODE='last_encoded_item' is implemented here.")
    return f"stimulus_index_enc{load}"

def presented_items_for_row(row, load):
    cols = [f"stimulus_index_enc{k}" for k in range(1, load + 1)]
    return [row[c] for c in cols if c in row.index and not pd.isna(row[c])]

# =========================
# LOAD DATA
# =========================

for p in [TRIAL_INFO_PATH, CONCEPT_CELL_PATH, ENC1_PATH, ENC2_PATH, ENC3_PATH, DELAY_PATH]:
    require_file(p)

trial_info = pd.read_excel(TRIAL_INFO_PATH)
concept_df = pd.read_excel(CONCEPT_CELL_PATH)
enc1_df = pd.read_excel(ENC1_PATH)
enc2_df = pd.read_excel(ENC2_PATH)
enc3_df = pd.read_excel(ENC3_PATH)
delay_df = pd.read_excel(DELAY_PATH)

TRIAL_COL = infer_trial_col(trial_info)

# If spike tables use trial_id_final instead of trial_id, normalize them to TRIAL_COL where possible.
for name, df in [("enc1", enc1_df), ("enc2", enc2_df), ("enc3", enc3_df), ("delay", delay_df)]:
    spike_trial_col = infer_trial_col(df)
    if spike_trial_col != TRIAL_COL:
        df.rename(columns={spike_trial_col: TRIAL_COL}, inplace=True)

concept_df = concept_df[concept_df[SIGNI_COL].astype(str).str.upper() == "Y"].copy()

print(f"Loaded trial table: {trial_info.shape}")
print(f"Loaded concept-cell table after Signi == 'Y': {concept_df.shape}")
print(f"Trial column used: {TRIAL_COL}")

# =========================
# DESIGN MATRIX CONSTRUCTION
# =========================

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

def build_subject_design_matrix(subject_id, concept_neurons):
    subject_trials = trial_info[trial_info[SUBJECT_COL] == subject_id].copy()
    subject_trials = subject_trials.sort_values(TRIAL_COL).reset_index(drop=True)

    neurons = sorted(concept_neurons)
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

                # Encoding 1: unshifted
                s1 = get_spikes(enc1_df, subject_id, trial_id, neuron_id, ENC_SPIKE_COL)
                all_spikes.extend(s1)

                # Encoding 2: shifted by +1.0 s
                if load >= 2:
                    s2 = get_spikes(enc2_df, subject_id, trial_id, neuron_id, ENC_SPIKE_COL)
                    all_spikes.extend(s2 + 1.0)

                # Encoding 3: shifted by +2.0 s
                if load >= 3:
                    s3 = get_spikes(enc3_df, subject_id, trial_id, neuron_id, ENC_SPIKE_COL)
                    all_spikes.extend(s3 + 2.0)

                # Delay: shifted to occur after encoding + transition
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

def build_binned_counts_matrix(design_matrix, total_duration, bin_size, step_size):
    bins = create_time_bins(total_duration, bin_size, step_size)
    n_trials, n_neurons = design_matrix.shape
    X = np.zeros((n_trials, n_neurons, len(bins)))

    for i in range(n_trials):
        for j in range(n_neurons):
            X[i, j, :] = count_spikes_in_bins(design_matrix[i, j], bins, bin_size)

    return X, bins

# =========================
# POISSON NAIVE BAYES DECODER
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
    # Constant log-factorial term is omitted because it is identical across classes for fixed x.
    ll = (x[None, :] * log_lambda - lambdas).sum(axis=1)
    log_post = ll + log_prior
    log_post -= logsumexp(log_post)
    return np.exp(log_post)

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
        step_size,
    )

    enc_bins = len(create_time_bins(enc_duration, bin_size, step_size))
    enc_idx = np.arange(enc_bins)
    delay_idx = np.arange(enc_bins, X.shape[2])
    bins_delay = bins[delay_idx] - enc_duration

    classes_all = np.unique(y_labels)
    n_trials = X.shape[0]
    n_delay = len(delay_idx)
    K = len(classes_all)

    post_delay = np.full((n_trials, n_delay, K), np.nan)

    for test_i in range(n_trials):
        train_mask = np.ones(n_trials, dtype=bool)
        train_mask[test_i] = False

        y_train_trial = y_labels[train_mask]

        if len(np.unique(y_train_trial)) < 2:
            continue

        # Treat each encoding time bin from each training trial as one training observation.
        X_enc_train = X[train_mask][:, :, enc_idx]
        X_enc_train = np.transpose(X_enc_train, (0, 2, 1)).reshape(-1, X.shape[1])
        y_enc_train = np.repeat(y_train_trial, len(enc_idx))

        classes_fold, log_lambda, lambdas, log_prior = fit_poisson_nb(
            X_enc_train,
            y_enc_train,
            eps=eps,
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

def predict_from_delay_posteriors(post_delay, classes):
    mean_post = np.nanmean(post_delay, axis=1)

    valid = ~np.isnan(mean_post).all(axis=1)
    pred = np.full(post_delay.shape[0], np.nan)

    if valid.any():
        pred[valid] = classes[np.nanargmax(mean_post[valid], axis=1)]

    return pred, mean_post, valid

# =========================
# RUN DECODER ACROSS LOADS
# =========================

accuracy_rows = []
trial_prediction_rows = []
posterior_store = {}
valid_subjects = []

for subject_id in sorted(trial_info[SUBJECT_COL].dropna().unique()):
    subject_concepts = concept_df[concept_df[SUBJECT_COL] == subject_id]
    concept_neurons = subject_concepts[NEURON_COL].dropna().unique()

    if len(concept_neurons) < MIN_CONCEPT_CELLS:
        continue

    valid_subjects.append(subject_id)
    print(f"\nSubject {subject_id}: {len(concept_neurons)} concept cells")

    design_by_load = build_subject_design_matrix(subject_id, concept_neurons)

    for load in [1, 2, 3]:
        trial_table = design_by_load[load]["trial_table"].copy()
        design_matrix = design_by_load[load]["design_matrix"]

        if len(trial_table) < 5:
            print(f"  Load {load}: skipped; too few trials ({len(trial_table)})")
            continue

        label_col = get_label_col(load)
        if label_col not in trial_table.columns:
            print(f"  Load {load}: skipped; missing label column {label_col}")
            continue

        y = trial_table[label_col].values.astype(float)
        valid = ~pd.isna(y)
        trial_table = trial_table[valid].reset_index(drop=True)
        design_matrix = design_matrix[valid]
        y = y[valid]

        if len(np.unique(y)) < 2:
            print(f"  Load {load}: skipped; fewer than 2 stimulus classes")
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

        pred, mean_post, valid_pred = predict_from_delay_posteriors(post_delay, classes)
        accuracy = np.mean(pred[valid_pred] == y[valid_pred]) if valid_pred.any() else np.nan
        chance = 1.0 / len(classes)

        print(f"  Load {load}: accuracy = {accuracy:.3f} | chance = {chance:.3f} | n trials = {valid_pred.sum()} | classes = {list(classes)}")

        accuracy_rows.append({
            "subject_id": subject_id,
            "load": load,
            "n_concept_cells": len(concept_neurons),
            "n_trials": int(valid_pred.sum()),
            "n_classes": len(classes),
            "classes": list(classes),
            "chance": chance,
            "accuracy": accuracy,
        })

        for i, row in trial_table.iterrows():
            trial_prediction_rows.append({
                "subject_id": subject_id,
                "trial_id": row[TRIAL_COL],
                "load": load,
                "true_label_decoded": y[i],
                "predicted_label_mean_delay": pred[i],
                "correct": bool(pred[i] == y[i]) if not pd.isna(pred[i]) else np.nan,
                "presented_items": presented_items_for_row(row, load),
                "response_accuracy": row.get("response_accuracy", np.nan),
            })

        # Store only what is needed for load-1 posterior plotting.
        if load == 1:
            posterior_store[subject_id] = {
                "bins_delay": bins_delay,
                "post_delay": post_delay,
                "classes": classes,
                "trial_table": trial_table.copy(),
                "y": y.copy(),
                "pred": pred.copy(),
            }

accuracy_df = pd.DataFrame(accuracy_rows)
trial_predictions_df = pd.DataFrame(trial_prediction_rows)

print("\n=========================")
print("Accuracy summary by load")
print("=========================")
if len(accuracy_df):
    print(accuracy_df.groupby("load")["accuracy"].agg(["mean", "std", "count"]))
else:
    print("No valid decoding results were produced. Check paths, columns, and MIN_CONCEPT_CELLS.")

accuracy_df.to_excel(ACCURACY_OUTPUT_PATH, index=False)
trial_predictions_df.to_excel(TRIAL_OUTPUT_PATH, index=False)

print(f"\nSaved accuracy table to: {ACCURACY_OUTPUT_PATH}")
print(f"Saved trial predictions to: {TRIAL_OUTPUT_PATH}")
print(f"Valid subjects used: {valid_subjects}")

# =========================
# PLOT LOAD-1 POSTERIORS FOR EXAMPLE TRIALS
# =========================

def choose_load1_trials_for_plot(trial_table, y, pred, n_trials=6, random_state=0):
    """
    Choose a mix of correct and incorrect load-1 trials when possible.
    This avoids cherry-picking only easy-looking trials.
    """
    rng = np.random.default_rng(random_state)
    idx_all = np.arange(len(trial_table))

    correct_idx = idx_all[pred == y]
    incorrect_idx = idx_all[pred != y]

    chosen = []
    n_correct = min(len(correct_idx), n_trials // 2)
    n_incorrect = min(len(incorrect_idx), n_trials - n_correct)

    if n_correct > 0:
        chosen.extend(rng.choice(correct_idx, size=n_correct, replace=False).tolist())
    if n_incorrect > 0:
        chosen.extend(rng.choice(incorrect_idx, size=n_incorrect, replace=False).tolist())

    remaining_needed = n_trials - len(chosen)
    if remaining_needed > 0:
        remaining = np.setdiff1d(idx_all, np.asarray(chosen, dtype=int))
        if len(remaining) > 0:
            chosen.extend(rng.choice(remaining, size=min(remaining_needed, len(remaining)), replace=False).tolist())

    return chosen

def plot_load1_posteriors_for_subject(subject_id, store, n_trials=6, save_dir=POSTERIOR_PLOT_DIR):
    bins_delay = store["bins_delay"]
    post_delay = store["post_delay"]
    classes = store["classes"]
    trial_table = store["trial_table"]
    y = store["y"]
    pred = store["pred"]

    chosen = choose_load1_trials_for_plot(
        trial_table=trial_table,
        y=y,
        pred=pred,
        n_trials=n_trials,
        random_state=RANDOM_STATE,
    )

    for i in chosen:
        row = trial_table.iloc[i]
        trial_id = row[TRIAL_COL]
        true_label = y[i]
        predicted_label = pred[i]

        plt.figure(figsize=(8, 5))

        for c_i, stim_id in enumerate(classes):
            label = f"Stim {int(stim_id) if float(stim_id).is_integer() else stim_id}"
            lw = 3 if stim_id == true_label else 1.8
            plt.plot(
                bins_delay,
                post_delay[i, :, c_i],
                marker="o",
                linewidth=lw,
                label=label,
            )

        title_status = "correct" if predicted_label == true_label else "incorrect"
        plt.title(
            f"Subject {subject_id} | Load 1 | Trial {trial_id} | {title_status}\n"
            f"True stim: {true_label} | Predicted: {predicted_label}"
        )
        plt.xlabel("Delay time (s)")
        plt.ylabel("Posterior probability")
        plt.ylim(-0.02, 1.02)
        plt.legend(title="Stimulus identity", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        outpath = os.path.join(
            save_dir,
            f"subject_{subject_id}_load1_trial_{trial_id}_posterior.png",
        )
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Saved: {outpath}")

# Plot for every subject that had valid load-1 decoding.
for subject_id, store in posterior_store.items():
    print(f"\nPlotting load-1 posterior examples for subject {subject_id}")
    plot_load1_posteriors_for_subject(
        subject_id=subject_id,
        store=store,
        n_trials=N_LOAD1_POSTERIOR_TRIALS,
        save_dir=POSTERIOR_PLOT_DIR,
    )

