import os
import ast
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import poisson as poisson_dist, wilcoxon, chisquare, fisher_exact, chi2_contingency
from scipy.stats import chi2 as chi2_dist
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

BASE = "../"
OUT = BASE + ../
DESKTOP = "../"
os.makedirs(OUT, exist_ok=True)

BIN_WIDTH = 0.03            # 30 ms, matches burst-detection bin size used elsewhere in this project
ENC_DURATION = 1.0          # s, natural encoding window
MAINT_DURATION = 2.8        # s, full recorded delay period
TEST_FRACTION = 0.30        # single held-out split, used only for the "final parameter estimate" fits
SEED = 0

K_FOLDS = 5                 # cross-validated model-comparison fold count
N_RESAMPLES = 5             # number of resampled 1.0s maintenance windows
WIN_STARTS_RANGE = (0.0, 1.5)  # maintenance window start times sampled from this range
N_BINS_WIN = 33             # bins in a resampled 1.0s window (1.0/0.03 -> 33)
VALID_STARTS = np.round(np.arange(WIN_STARTS_RANGE[0], WIN_STARTS_RANGE[1] + 1e-9, BIN_WIDTH), 4)

NEAR_FLOOR = 1e-3           # Hz; lambda values at/below this are treated as "at the numerical floor"

BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_LIGHT = "#9ec5f4"
GRAY_LIGHT, GRAY_MED, GRAY_DARK = "#c3c2b7", "#898781", "#52514e"
INK, GRID, SURFACE = "#0b0b0b", "#e1e0d9", "#fcfcfb"
PY_RED, IN_BLUE = "#c0392b", "#2a78d6"
CLASS_COLORS = {"persistent": "#c7c4b4", "hybrid": "#8f8c7e", "intermittent": "#35332c"}

print("Setup OK")

def poisson_logpmf(y, lam, bin_width):
    mu = np.clip(lam * bin_width, 1e-12, None)
    return poisson_dist.logpmf(y, mu)


def fit_2state_poisson_hmm(Y, bin_width, n_restarts=1, max_iter=150, tol=1e-4, seed=0):
    """Y: (n_trials, T) integer spike-count matrix. Returns dict with lam (2,), A (2,2), pi (2,),
    train_ll, n_iter for the best-of-n_restarts EM fit, states sorted low-to-high rate."""
    n_trials, T = Y.shape
    rng = np.random.default_rng(seed)
    trial_mean_rate = Y.mean(axis=1) / bin_width
    best = None
    for r in range(n_restarts):
        if r == 0:
            lam = np.array([np.percentile(trial_mean_rate, 25), np.percentile(trial_mean_rate, 75)])
            A = np.array([[0.90, 0.10], [0.10, 0.90]])
        elif r == 1:
            lam = np.array([np.percentile(trial_mean_rate, 10), np.percentile(trial_mean_rate, 90)])
            A = np.array([[0.97, 0.03], [0.03, 0.97]])
        else:
            base = max(trial_mean_rate.mean(), 1e-3)
            lam = np.sort(base * rng.uniform(0.3, 2.5, size=2))
            A = rng.dirichlet([8, 1], size=2)
        lam = np.clip(lam, 1e-3, None)
        if lam[0] == lam[1]:
            lam[1] += 1e-3
        pi = np.array([0.5, 0.5])
        prev_ll = -np.inf
        it = 0
        for it in range(max_iter):
            logB = np.stack([poisson_logpmf(Y, lam[0], bin_width), poisson_logpmf(Y, lam[1], bin_width)], axis=-1)
            B = np.exp(logB - logB.max(axis=-1, keepdims=True))
            Bmax = logB.max(axis=-1)
            alpha_hat = np.zeros((n_trials, T, 2))
            c = np.zeros((n_trials, T))
            a0 = pi[None, :] * B[:, 0, :]
            c[:, 0] = np.clip(a0.sum(axis=1), 1e-300, None)
            alpha_hat[:, 0, :] = a0 / c[:, 0, None]
            for t in range(1, T):
                a_t = (alpha_hat[:, t - 1, :] @ A) * B[:, t, :]
                c[:, t] = np.clip(a_t.sum(axis=1), 1e-300, None)
                alpha_hat[:, t, :] = a_t / c[:, t, None]
            beta_hat = np.zeros((n_trials, T, 2))
            beta_hat[:, T - 1, :] = 1.0
            for t in range(T - 2, -1, -1):
                b_t = (B[:, t + 1, :] * beta_hat[:, t + 1, :]) @ A.T
                beta_hat[:, t, :] = b_t / c[:, t + 1, None]
            gamma = alpha_hat * beta_hat
            gamma /= gamma.sum(axis=-1, keepdims=True)
            xi_sum = np.zeros((2, 2))
            for t in range(T - 1):
                num = (alpha_hat[:, t, :, None] * A[None, :, :] * B[:, t + 1, None, :] * beta_hat[:, t + 1, None, :])
                num = num / c[:, t + 1, None, None]
                xi_sum += num.sum(axis=0)
            train_ll = float((np.log(c) + Bmax).sum())
            pi_new = gamma[:, 0, :].mean(axis=0)
            gamma_sum_notlast = gamma[:, :-1, :].sum(axis=(0, 1))
            A_new = xi_sum / np.clip(gamma_sum_notlast[:, None], 1e-300, None)
            A_new /= A_new.sum(axis=1, keepdims=True)
            gamma_sum_all = gamma.sum(axis=(0, 1))
            lam_new = (gamma * Y[:, :, None]).sum(axis=(0, 1)) / np.clip(gamma_sum_all, 1e-300, None) / bin_width
            lam_new = np.clip(lam_new, 1e-4, None)
            pi, A, lam = pi_new, A_new, lam_new
            if train_ll - prev_ll < tol and it > 3:
                prev_ll = train_ll
                break
            prev_ll = train_ll
        if best is None or prev_ll > best["train_ll"]:
            order = np.argsort(lam)
            best = dict(lam=lam[order].copy(), A=A[np.ix_(order, order)].copy(),
                        pi=pi[order].copy(), train_ll=prev_ll, n_iter=it + 1)
    return best


def fit_1state_poisson(Y, bin_width):
    lam = max(Y.mean() / bin_width, 1e-4)
    ll = float(poisson_logpmf(Y, lam, bin_width).sum())
    return dict(lam=lam, train_ll=ll)


def held_out_loglik_2state(Y_test, bin_width, lam, A, pi):
    n_trials, T = Y_test.shape
    logB = np.stack([poisson_logpmf(Y_test, lam[0], bin_width), poisson_logpmf(Y_test, lam[1], bin_width)], axis=-1)
    B = np.exp(logB - logB.max(axis=-1, keepdims=True))
    Bmax = logB.max(axis=-1)
    alpha_hat = np.zeros((n_trials, T, 2))
    c = np.zeros((n_trials, T))
    a0 = pi[None, :] * B[:, 0, :]
    c[:, 0] = np.clip(a0.sum(axis=1), 1e-300, None)
    alpha_hat[:, 0, :] = a0 / c[:, 0, None]
    for t in range(1, T):
        a_t = (alpha_hat[:, t - 1, :] @ A) * B[:, t, :]
        c[:, t] = np.clip(a_t.sum(axis=1), 1e-300, None)
        alpha_hat[:, t, :] = a_t / c[:, t, None]
    return float((np.log(c) + Bmax).sum())


def held_out_loglik_1state(Y_test, bin_width, lam):
    return float(poisson_logpmf(Y_test, lam, bin_width).sum())

def simulate_2state(lam_low, lam_high, A, n_trials, T, bin_width, rng):
    Y = np.zeros((n_trials, T), dtype=int)
    for n in range(n_trials):
        z = rng.choice(2, p=[0.5, 0.5])
        for t in range(T):
            lam = lam_low if z == 0 else lam_high
            Y[n, t] = rng.poisson(lam * bin_width)
            z = rng.choice(2, p=A[z])
    return Y

T_SIM = int(round(MAINT_DURATION / BIN_WIDTH))
rng = np.random.default_rng(42)

print("--- positive control: true 2-state bursty process ---")
true_lam = np.array([1.0, 15.0])
true_A = np.array([[0.92, 0.08], [0.05, 0.95]])
Y_sim = simulate_2state(true_lam[0], true_lam[1], true_A, 130, T_SIM, BIN_WIDTH, rng)
idx = rng.permutation(130)
n_tr = int(130 * (1 - TEST_FRACTION))
tr_idx, te_idx = idx[:n_tr], idx[n_tr:]
f2 = fit_2state_poisson_hmm(Y_sim[tr_idx], BIN_WIDTH, n_restarts=3, seed=1)
f1 = fit_1state_poisson(Y_sim[tr_idx], BIN_WIDTH)
ll2 = held_out_loglik_2state(Y_sim[te_idx], BIN_WIDTH, f2["lam"], f2["A"], f2["pi"])
ll1 = held_out_loglik_1state(Y_sim[te_idx], BIN_WIDTH, f1["lam"])
n_bins_test = Y_sim[te_idx].size
print(f"true lam={true_lam}, true dwell-diag={np.diag(true_A)}")
print(f"fit  lam={f2['lam'].round(2)}, fit dwell-diag={np.diag(f2['A']).round(3)}")
print(f"held-out delta-LL/bin (2state - 1state) = {(ll2-ll1)/n_bins_test:.4f}  (expect > 0)")
assert ll2 > ll1, "positive control FAILED: 2-state should beat 1-state on truly bursty data"

print()
print("--- negative control: true single-rate process ---")
Y0 = rng.poisson(5.0 * BIN_WIDTH, size=(130, T_SIM))
f2b = fit_2state_poisson_hmm(Y0[tr_idx], BIN_WIDTH, n_restarts=3, seed=2)
f1b = fit_1state_poisson(Y0[tr_idx], BIN_WIDTH)
ll2b = held_out_loglik_2state(Y0[te_idx], BIN_WIDTH, f2b["lam"], f2b["A"], f2b["pi"])
ll1b = held_out_loglik_1state(Y0[te_idx], BIN_WIDTH, f1b["lam"])
delta_null = (ll2b - ll1b) / n_bins_test
print(f"fit lam={f2b['lam'].round(2)} (true=5.0 for both -- should be close together)")
print(f"held-out delta-LL/bin (2state - 1state) = {delta_null:.5f}  (expect ~0, small in magnitude)")
assert abs(delta_null) < 0.01, "negative control FAILED: 2-state spuriously beating 1-state by a lot on pure noise"

enc_df = pd.read_excel(BASE + "graph_encoding1.xlsx")
enc_df["subject_id"] = enc_df["subject_id"].astype(int)
enc_df["trial_id"] = enc_df["trial_id"].astype(int)

delay_df = pd.read_excel(BASE + "graph_delay.xlsx")
delay_df["subject_id"] = delay_df["subject_id"].astype(int)
delay_df["trial_id"] = delay_df["trial_id"].astype(int)

sig_df = pd.read_excel(BASE + "merged_significant_neurons_with_brain_regions.xlsx")
concept_ids = set(sig_df.loc[sig_df["Signi"] == "Y", "Neuron_ID_3"])
print(f"{len(concept_ids)} concept cells (Signi == 'Y')")

r2_labels = pd.read_csv(DESKTOP + "cell_clustering_no_waveform_labels.csv", sep="\t")
r2_labels["Cell_Type_New"] = r2_labels["Cell_Type_New"].fillna("").astype(str).str.strip()
labelled_IN_ids = set(r2_labels.loc[r2_labels.Cell_Type_New == "IN", "Neuron_ID_3"].astype(int))
labelled_PY_ids = set(r2_labels.loc[r2_labels.Cell_Type_New == "PY", "Neuron_ID_3"].astype(int))
print(f"{len(labelled_IN_ids)} Labelled_IN, {len(labelled_PY_ids)} Labelled_PY (R2 > 0.3, ACG-classified)")


def parse_spikes(x, trial_dur):
    if pd.isna(x):
        return np.array([], dtype=float)
    if isinstance(x, str):
        s = x.strip()
        if s in ("", "[]"):
            return np.array([], dtype=float)
        try:
            arr = np.array(ast.literal_eval(s), dtype=float)
        except Exception:
            return np.array([], dtype=float)
    elif isinstance(x, (list, tuple, np.ndarray)):
        arr = np.array(x, dtype=float)
    else:
        return np.array([], dtype=float)
    if arr.ndim == 0:
        arr = np.array([float(arr)])
    return arr[(arr >= 0.0) & (arr <= trial_dur)]


def parse_spikes_window(x, lo, hi):
    if pd.isna(x):
        return np.array([], dtype=float)
    if isinstance(x, str):
        s = x.strip()
        if s in ("", "[]"):
            return np.array([], dtype=float)
        try:
            arr = np.array(ast.literal_eval(s), dtype=float)
        except Exception:
            return np.array([], dtype=float)
    elif isinstance(x, (list, tuple, np.ndarray)):
        arr = np.array(x, dtype=float)
    else:
        return np.array([], dtype=float)
    if arr.ndim == 0:
        arr = np.array([float(arr)])
    return arr[(arr >= lo) & (arr < hi)]


def build_binned_matrix(neuron_df, duration, edges):
    trial_ids, rows = [], []
    for _, row in neuron_df.sort_values("trial_id").iterrows():
        spikes = parse_spikes(row["Standardized_Spikes"], duration)
        counts, _ = np.histogram(spikes, bins=edges)
        rows.append(counts)
        trial_ids.append(int(row["trial_id"]))
    return np.array(rows, dtype=int), np.array(trial_ids)


def build_binned_matrix_window(neuron_df, lo, hi, edges):
    trial_ids, rows = [], []
    for _, row in neuron_df.sort_values("trial_id").iterrows():
        spikes = parse_spikes_window(row["Standardized_Spikes"], lo, hi)
        counts, _ = np.histogram(spikes, bins=edges)
        rows.append(counts)
        trial_ids.append(int(row["trial_id"]))
    return np.array(rows, dtype=int), np.array(trial_ids)


T_BINS_ENC = int(round(ENC_DURATION / BIN_WIDTH))
time_edges_enc = np.arange(0.0, ENC_DURATION + BIN_WIDTH, BIN_WIDTH)[: T_BINS_ENC + 1]
T_BINS_MAINT = int(round(MAINT_DURATION / BIN_WIDTH))
time_edges_maint = np.arange(0.0, MAINT_DURATION + BIN_WIDTH, BIN_WIDTH)[: T_BINS_MAINT + 1]

WIN_LO, WIN_HI = 1.0, 2.0
edges_mid = time_edges_enc + WIN_LO  # identical 33-bin/30ms grid as encoding, shifted to [1.0, 2.0)s

print("Data and helpers loaded")

def fit_all_neurons_base(df_epoch, duration, edges, cache_path, label="epoch"):
    results = []
    neuron_ids_all = sorted(df_epoch["Neuron_ID_3"].unique())
    print(f"[{label}] {len(neuron_ids_all)} neurons total", flush=True)
    t0 = time.time()
    for i, nid in enumerate(neuron_ids_all):
        ndf = df_epoch[df_epoch["Neuron_ID_3"] == nid]
        if len(ndf) < 20:
            continue
        Y, trial_ids = build_binned_matrix(ndf, duration, edges)
        n_trials = Y.shape[0]

        seed_n = (int(nid) * 2654435761) % (2**32 - 1)
        rng_n = np.random.default_rng(seed_n)
        idx = rng_n.permutation(n_trials)
        n_test = max(int(round(n_trials * TEST_FRACTION)), 5)
        te_idx, tr_idx = idx[:n_test], idx[n_test:]

        f1_tr = fit_1state_poisson(Y[tr_idx], BIN_WIDTH)
        f2_tr = fit_2state_poisson_hmm(Y[tr_idx], BIN_WIDTH, n_restarts=2, seed=seed_n)
        ll1 = held_out_loglik_1state(Y[te_idx], BIN_WIDTH, f1_tr["lam"])
        ll2 = held_out_loglik_2state(Y[te_idx], BIN_WIDTH, f2_tr["lam"], f2_tr["A"], f2_tr["pi"])
        n_test_bins = Y[te_idx].size

        f2_all = fit_2state_poisson_hmm(Y, BIN_WIDTH, n_restarts=2, seed=seed_n + 1)
        dwell_low = 1.0 / max(1 - f2_all["A"][0, 0], 1e-6) * BIN_WIDTH
        dwell_high = 1.0 / max(1 - f2_all["A"][1, 1], 1e-6) * BIN_WIDTH

        subject_id = int(ndf["subject_id"].iloc[0])
        n_trials_total = int((df_epoch["Neuron_ID_3"] == nid).sum())
        total_spikes = int(Y.sum())
        results.append(dict(
            neuron_id=int(nid), subject_id=subject_id, is_concept_cell=int(nid) in concept_ids,
            n_trials=n_trials, n_test_trials=len(te_idx), total_spikes=total_spikes,
            ll1_per_bin=ll1 / n_test_bins, ll2_per_bin=ll2 / n_test_bins,
            delta_ll_per_bin=(ll2 - ll1) / n_test_bins,
            lam_low_hz=f2_all["lam"][0], lam_high_hz=f2_all["lam"][1],
            rate_ratio=f2_all["lam"][1] / max(f2_all["lam"][0], 1e-6),
            dwell_low_s=dwell_low, dwell_high_s=dwell_high,
            A_00=f2_all["A"][0, 0], A_11=f2_all["A"][1, 1],
            overall_rate_hz=Y.mean() / BIN_WIDTH,
        ))
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] {i+1}/{len(neuron_ids_all)}, {elapsed:.0f}s elapsed, "
                  f"ETA {elapsed/(i+1)*(len(neuron_ids_all)-i-1)/60:.1f} min", flush=True)
    rdf = pd.DataFrame(results)
    rdf["is_labelled_IN"] = rdf["neuron_id"].isin(labelled_IN_ids)
    rdf["is_labelled_PY"] = rdf["neuron_id"].isin(labelled_PY_ids)
    rdf.to_csv(cache_path, index=False)
    print(f"[{label}] {len(rdf)} neurons fit, saved to {cache_path}")
    return rdf


def fit_all_neurons_midwindow(df_epoch, cache_path, label="maint_midwindow"):
    results = []
    neuron_ids_all = sorted(df_epoch["Neuron_ID_3"].unique())
    print(f"[{label}] {len(neuron_ids_all)} neurons total", flush=True)
    t0 = time.time()
    for i, nid in enumerate(neuron_ids_all):
        ndf = df_epoch[df_epoch["Neuron_ID_3"] == nid]
        if len(ndf) < 20:
            continue
        Y, trial_ids = build_binned_matrix_window(ndf, WIN_LO, WIN_HI, edges_mid)
        n_trials = Y.shape[0]

        seed_n = (int(nid) * 2654435761) % (2**32 - 1)
        rng_n = np.random.default_rng(seed_n)
        idx = rng_n.permutation(n_trials)
        n_test = max(int(round(n_trials * TEST_FRACTION)), 5)
        te_idx, tr_idx = idx[:n_test], idx[n_test:]

        f1_tr = fit_1state_poisson(Y[tr_idx], BIN_WIDTH)
        f2_tr = fit_2state_poisson_hmm(Y[tr_idx], BIN_WIDTH, n_restarts=2, seed=seed_n)
        ll1 = held_out_loglik_1state(Y[te_idx], BIN_WIDTH, f1_tr["lam"])
        ll2 = held_out_loglik_2state(Y[te_idx], BIN_WIDTH, f2_tr["lam"], f2_tr["A"], f2_tr["pi"])
        n_test_bins = Y[te_idx].size

        f2_all = fit_2state_poisson_hmm(Y, BIN_WIDTH, n_restarts=2, seed=seed_n + 1)
        dwell_low = 1.0 / max(1 - f2_all["A"][0, 0], 1e-6) * BIN_WIDTH
        dwell_high = 1.0 / max(1 - f2_all["A"][1, 1], 1e-6) * BIN_WIDTH

        subject_id = int(ndf["subject_id"].iloc[0])
        total_spikes = int(Y.sum())
        results.append(dict(
            neuron_id=int(nid), subject_id=subject_id, is_concept_cell=int(nid) in concept_ids,
            n_trials=n_trials, n_test_trials=len(te_idx), total_spikes=total_spikes,
            ll1_per_bin=ll1 / n_test_bins, ll2_per_bin=ll2 / n_test_bins,
            delta_ll_per_bin=(ll2 - ll1) / n_test_bins,
            lam_low_hz=f2_all["lam"][0], lam_high_hz=f2_all["lam"][1],
            rate_ratio=f2_all["lam"][1] / max(f2_all["lam"][0], 1e-6),
            dwell_low_s=dwell_low, dwell_high_s=dwell_high,
            A_00=f2_all["A"][0, 0], A_11=f2_all["A"][1, 1],
            overall_rate_hz=Y.mean() / BIN_WIDTH,
        ))
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] {i+1}/{len(neuron_ids_all)}, {elapsed:.0f}s elapsed, "
                  f"ETA {elapsed/(i+1)*(len(neuron_ids_all)-i-1)/60:.1f} min", flush=True)
    rdf = pd.DataFrame(results)
    rdf["is_labelled_IN"] = rdf["neuron_id"].isin(labelled_IN_ids)
    rdf["is_labelled_PY"] = rdf["neuron_id"].isin(labelled_PY_ids)
    rdf.to_csv(cache_path, index=False)
    print(f"[{label}] {len(rdf)} neurons fit, saved to {cache_path}")
    return rdf


# --- run all three base fits ---
enc_base = fit_all_neurons_base(enc_df, ENC_DURATION, time_edges_enc,
                                 OUT + "hmm_2state_vs_1state_results_ENCODING.csv", label="encoding")
maint_full_base = fit_all_neurons_base(delay_df, MAINT_DURATION, time_edges_maint,
                                        OUT + "hmm_2state_vs_1state_results.csv", label="maintenance_full")
maint_mid_base = fit_all_neurons_midwindow(delay_df,
                                            OUT + "hmm_2state_vs_1state_results_MAINT_MIDWINDOW_1to2s.csv",
                                            label="maintenance_midwindow")
def kfold_eval(Y, bin_width, k_folds, seed):
    """Every trial held out exactly once. Returns (total_ll1, total_ll2, total_bins)."""
    n_trials = Y.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_trials)
    folds = np.array_split(idx, k_folds)
    total_ll1, total_ll2, total_bins = 0.0, 0.0, 0
    for fi, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(idx, test_idx, assume_unique=True)
        if len(test_idx) == 0 or len(train_idx) < 5:
            continue
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        f1 = fit_1state_poisson(Y_tr, bin_width)
        f2 = fit_2state_poisson_hmm(Y_tr, bin_width, seed=seed + fi + 1)
        ll1 = held_out_loglik_1state(Y_te, bin_width, f1["lam"])
        ll2 = held_out_loglik_2state(Y_te, bin_width, f2["lam"], f2["A"], f2["pi"])
        total_ll1 += ll1
        total_ll2 += ll2
        total_bins += Y_te.size
    return total_ll1, total_ll2, total_bins


def kfold_natural_fit_one_neuron(ndf, duration, edges, seed, k_folds=K_FOLDS):
    Y, _ = build_binned_matrix(ndf, duration, edges)
    if Y.shape[0] < 20:
        return dict(delta_ll_per_bin=np.nan)
    ll1, ll2, bins_ = kfold_eval(Y, BIN_WIDTH, k_folds, seed)
    if bins_ == 0:
        return dict(delta_ll_per_bin=np.nan)
    return dict(delta_ll_per_bin=(ll2 - ll1) / bins_)


def kfold_resampled_fit_one_neuron(ndf, seed, n_resamples=N_RESAMPLES, k_folds=K_FOLDS):
    rng = np.random.default_rng(seed)
    starts = rng.choice(VALID_STARTS, size=n_resamples,
                         replace=False if len(VALID_STARTS) >= n_resamples else True)
    total_ll1, total_ll2, total_bins = 0.0, 0.0, 0
    for ri, start in enumerate(starts):
        edges_w = np.linspace(float(start), float(start) + 1.0, N_BINS_WIN + 1)
        Y, _ = build_binned_matrix_window(ndf, float(start), float(start) + 1.0, edges_w)
        if Y.shape[0] < 20:
            continue
        ll1, ll2, bins_ = kfold_eval(Y, BIN_WIDTH, k_folds, seed + ri * 1000)
        total_ll1 += ll1
        total_ll2 += ll2
        total_bins += bins_
    if total_bins == 0:
        return dict(delta_ll_per_bin=np.nan)
    return dict(delta_ll_per_bin=(total_ll2 - total_ll1) / total_bins)


# --- encoding: k-fold on the natural window, all neurons ---
enc_kfold_rows = []
t0 = time.time()
enc_neuron_ids = sorted(enc_df["Neuron_ID_3"].unique())
for i, nid in enumerate(enc_neuron_ids):
    ndf = enc_df[enc_df.Neuron_ID_3 == nid]
    res = kfold_natural_fit_one_neuron(ndf, ENC_DURATION, time_edges_enc,
                                        seed=(int(nid) * 2654435761) % (2**32 - 1))
    enc_kfold_rows.append(dict(neuron_id=int(nid), **res))
    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"  [encoding k-fold] {i+1}/{len(enc_neuron_ids)}, {elapsed:.0f}s elapsed, "
              f"ETA {elapsed/(i+1)*(len(enc_neuron_ids)-i-1)/60:.1f} min", flush=True)
enc_kfold_v3 = pd.DataFrame(enc_kfold_rows)
enc_kfold_v3.to_csv(OUT + "hmm_kfold5_encoding_v3.csv", index=False)
print(f"encoding k-fold done: {len(enc_kfold_v3)} neurons")

# --- maintenance: k-fold within 5 resampled windows, all neurons ---
maint_kfold_rows = []
t0 = time.time()
maint_neuron_ids = sorted(delay_df["Neuron_ID_3"].unique())
for i, nid in enumerate(maint_neuron_ids):
    ndf = delay_df[delay_df.Neuron_ID_3 == nid]
    res = kfold_resampled_fit_one_neuron(ndf, seed=(int(nid) * 2654435761) % (2**32 - 1))
    maint_kfold_rows.append(dict(neuron_id=int(nid), **res))
    if (i + 1) % 25 == 0:
        elapsed = time.time() - t0
        print(f"  [maintenance k-fold] {i+1}/{len(maint_neuron_ids)}, {elapsed:.0f}s elapsed, "
              f"ETA {elapsed/(i+1)*(len(maint_neuron_ids)-i-1)/60:.1f} min", flush=True)
maint_kfold_v3 = pd.DataFrame(maint_kfold_rows)
maint_kfold_v3.to_csv(OUT + "hmm_kfold5_maintenance_v3_fullrange.csv", index=False)
print(f"maintenance k-fold done: {len(maint_kfold_v3)} neurons")

def derive_v3(d):
    d = d.copy()
    d["A00_sat"] = d.A_00 >= 0.999
    d["A11_sat"] = d.A_11 >= 0.999
    d["lam_low_floor"] = d.lam_low_hz <= NEAR_FLOOR
    d["lam_high_floor"] = d.lam_high_hz <= NEAR_FLOOR
    both_sat = d.A00_sat & d.A11_sat
    a11_only = d.A11_sat & ~d.A00_sat
    a00_only = d.A00_sat & ~d.A11_sat
    excluded = both_sat | (a11_only & d.lam_low_floor) | (a00_only & d.lam_high_floor)
    d["excluded_v3"] = excluded
    d["force_persistent_v3"] = a11_only & ~d.lam_low_floor & ~excluded
    d["active"] = d.overall_rate_hz >= 1.0
    d["clean_active"] = ~d.excluded_v3 & d.active
    return d


def classify(row, duration):
    if pd.isna(row["delta_ll_per_bin"]) or pd.isna(row["rate_ratio"]):
        return np.nan
    if row.get("force_persistent_v3", False):
        return "persistent"
    if row["delta_ll_per_bin"] <= 0 or row["rate_ratio"] < 1.5:
        return "persistent"
    if row["dwell_high_s"] < 0.3 * duration:
        return "intermittent"
    return "hybrid"


# --- assemble encoding dataframe ---
enc = enc_base.drop(columns=["delta_ll_per_bin"]).merge(
    enc_kfold_v3[["neuron_id", "delta_ll_per_bin"]], on="neuron_id", how="left")
enc["is_concept_cell"] = enc.neuron_id.isin(concept_ids)
enc = derive_v3(enc)
enc["favors_2state"] = enc["delta_ll_per_bin"] > 0
enc["model_class"] = enc.apply(lambda r: classify(r, ENC_DURATION), axis=1)

# --- assemble maintenance dataframe (mid-window fit + full-window overall_rate_hz + resampled k-fold LL) ---
maint = maint_mid_base.drop(columns=["overall_rate_hz", "delta_ll_per_bin"]).merge(
    maint_full_base[["neuron_id", "overall_rate_hz"]], on="neuron_id", how="left").merge(
    maint_kfold_v3[["neuron_id", "delta_ll_per_bin"]], on="neuron_id", how="left")
maint["is_concept_cell"] = maint.neuron_id.isin(concept_ids)
maint = derive_v3(maint)
maint["favors_2state"] = maint["delta_ll_per_bin"] > 0
maint["model_class"] = maint.apply(lambda r: classify(r, ENC_DURATION), axis=1)

print(f"encoding: clean&active {enc.clean_active.sum()}/{len(enc)}  "
      f"(reclassified persistent: {int(enc.force_persistent_v3.sum())})")
print(f"maintenance: clean&active {maint.clean_active.sum()}/{len(maint)}  "
      f"(reclassified persistent: {int(maint.force_persistent_v3.sum())})")

def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)


def p_to_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"


panel_specs = [
    ("All cells", "Encoding", enc[enc.clean_active]),
    ("Concept cells", "Encoding", enc[enc.clean_active & enc.is_concept_cell]),
    ("All cells", "Maintenance", maint[maint.clean_active]),
    ("Concept cells", "Maintenance", maint[maint.clean_active & maint.is_concept_cell]),
]
fig, axes = plt.subplots(2, 2, figsize=(9, 9.5))
fig.patch.set_facecolor(SURFACE)
axes = axes.ravel()
TWO_STATE_COLOR = {"Encoding": BLUE_LIGHT, "Maintenance": BLUE}
classic_rows = []
for ax, (pop, epoch, df) in zip(axes, panel_specs):
    n = len(df)
    k2 = int(df.favors_2state.sum())
    k1 = n - k2
    frac1, frac2 = k1 / n, k2 / n
    wt = wilcoxon(df.delta_ll_per_bin)
    p = wt.pvalue
    classic_rows.append(dict(population=pop, epoch=epoch, n=n, n_one_state=k1, n_two_state=k2,
                              frac_two_state=frac2, wilcoxon_W=wt.statistic, wilcoxon_p=p))
    style_ax(ax)
    ax.bar([0, 1], [frac1, frac2], width=0.6, color=[GRAY_MED, TWO_STATE_COLOR[epoch]], edgecolor="none")
    ax.axhline(0.5, color=INK, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["one-state", "two-state"], fontsize=10)
    ax.set_ylim(0, 1.15); ax.set_ylabel("fraction of neurons", fontsize=9)
    ax.set_title(f"{pop}, {epoch} (n={n})", fontsize=11)
    y_bracket = max(frac1, frac2) + 0.08
    ax.plot([0, 0, 1, 1], [y_bracket - 0.02, y_bracket, y_bracket, y_bracket - 0.02], color=INK, linewidth=1)
    ax.text(0.5, y_bracket + 0.03, f"{p_to_stars(p)}\np={p:.2g}", ha="center", fontsize=8.5)
fig.suptitle("One-state vs. two-state model preference", fontsize=12, y=1.0)
fig.tight_layout()
fig.savefig(OUT + "01_onestate_vs_twostate.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
fig.savefig(OUT + "01_onestate_vs_twostate.eps", bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)
classic_df = pd.DataFrame(classic_rows)
classic_df.to_csv(OUT + "01_onestate_vs_twostate_stats.csv", index=False)
print(classic_df.to_string(index=False))

class_order = ["persistent", "hybrid", "intermittent"]
fig, axes = plt.subplots(2, 2, figsize=(9, 9.5))
fig.patch.set_facecolor(SURFACE)
axes = axes.ravel()
modelclass_rows = []
for ax, (pop, epoch, df) in zip(axes, panel_specs):
    n = len(df)
    counts = [int((df.model_class == c).sum()) for c in class_order]
    fracs = [c / n for c in counts]
    chi2, p = chisquare(counts)
    modelclass_rows.append(dict(population=pop, epoch=epoch, n=n,
                                 persistent=counts[0], hybrid=counts[1], intermittent=counts[2],
                                 chi2=chi2, p=p))
    style_ax(ax)
    ax.bar(range(3), fracs, width=0.6, color=[CLASS_COLORS[c] for c in class_order], edgecolor="none")
    ax.axhline(1/3, color=INK, linewidth=1, linestyle="--", alpha=0.6, label="uniform (1/3)")
    ax.set_xticks(range(3)); ax.set_xticklabels(class_order, fontsize=10)
    ax.set_ylim(0, 0.75); ax.set_ylabel("fraction of neurons", fontsize=9)
    ax.set_title(f"{pop}, {epoch} (n={n})", fontsize=11)
    y_bracket = max(fracs) + 0.10
    ax.plot([0, 0, 2, 2], [y_bracket - 0.015, y_bracket, y_bracket, y_bracket - 0.015], color=INK, linewidth=1)
    ax.text(1, y_bracket + 0.025, f"{p_to_stars(p)}  P={p:.2g}\n(chi-square vs. uniform)", ha="center", fontsize=8)
axes[0].legend(fontsize=8, loc="upper right", framealpha=0.9)
fig.suptitle("Persistent / hybrid / intermittent breakdown", fontsize=12, y=1.0)
fig.tight_layout()
fig.savefig(OUT + "02_modelclass_all_vs_concept.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
fig.savefig(OUT + "02_modelclass_all_vs_concept.eps", bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)
modelclass_df = pd.DataFrame(modelclass_rows)
modelclass_df.to_csv(OUT + "02_modelclass_all_vs_concept_stats.csv", index=False)
print(modelclass_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 6.5))
fig.patch.set_facecolor(SURFACE)
py_in_rows = []
for ax, (epoch, df) in zip(axes, [("Encoding", enc), ("Maintenance", maint)]):
    py = df[df.clean_active & df.is_labelled_PY]
    inn = df[df.clean_active & df.is_labelled_IN]
    n_py, n_in = len(py), len(inn)
    style_ax(ax)
    ax.axhline(0, color=INK, linewidth=1)
    for j, c in enumerate(class_order):
        py_c = int((py.model_class == c).sum())
        in_c = int((inn.model_class == c).sum())
        table = [[py_c, n_py - py_c], [in_c, n_in - in_c]]
        _, p = fisher_exact(table)
        frac_py, frac_in = py_c / n_py, in_c / n_in
        py_in_rows.append(dict(epoch=epoch, cls=c, n_py=n_py, n_in=n_in,
                                py_count=py_c, in_count=in_c, frac_py=frac_py, frac_in=frac_in, p=p))
        ax.bar(j, frac_py, width=0.6, bottom=0, color=PY_RED, edgecolor="none")
        ax.bar(j, -frac_in, width=0.6, bottom=0, color=IN_BLUE, edgecolor="none")
        ax.text(j, frac_py + 0.02, f"{py_c}/{n_py}", ha="center", va="bottom", fontsize=8)
        ax.text(j, -frac_in - 0.02, f"{in_c}/{n_in}", ha="center", va="top", fontsize=8)
        y_top = max(frac_py, frac_in) + 0.09
        ax.text(j, y_top, f"{p_to_stars(p)}\nP={p:.2g}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(class_order, fontsize=10)
    ax.set_ylim(-0.75, 0.75)
    ax.set_ylabel(f"fraction of neurons\n(PY, top, n={n_py}   |   IN, bottom, n={n_in})", fontsize=9)
    ax.set_title(epoch, fontsize=12)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=PY_RED, label="Labelled_PY"), Patch(color=IN_BLUE, label="Labelled_IN")],
           loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.02), frameon=False)
fig.suptitle("Model-class distribution: PY vs. IN", fontsize=12, y=1.10)
fig.tight_layout()
fig.savefig(OUT + "03_modelclass_PY_vs_IN.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
fig.savefig(OUT + "03_modelclass_PY_vs_IN.eps", bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)
py_in_df = pd.DataFrame(py_in_rows)
py_in_df.to_csv(OUT + "03_modelclass_PY_vs_IN_stats.csv", index=False)
print(py_in_df.to_string(index=False))

status_color = {
    "favors_2state": BLUE, "favors_2state_excluded": BLUE,
    "favors_1state": GRAY_LIGHT, "favors_1state_excluded": GRAY_LIGHT,
    "excluded_fit": ORANGE, "inactive": GRAY_LIGHT, "no_fit": GRAY_LIGHT,
}
status_alpha = {
    "favors_2state": 0.72, "favors_1state": 0.72,
    "favors_2state_excluded": 0.30, "favors_1state_excluded": 0.30,
    "excluded_fit": 0.75, "inactive": 0.25, "no_fit": 0.25,
}

import matplotlib.colors as mcolors
def blend_with_surface(hexcolor, alpha, surface=SURFACE):
    fg = np.array(mcolors.to_rgb(hexcolor)); bg = np.array(mcolors.to_rgb(surface))
    return tuple(alpha * fg + (1 - alpha) * bg)
status_color_blended = {k: blend_with_surface(v, status_alpha[k]) for k, v in status_color.items()}


def scatter_rows(df):
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r.delta_ll_per_bin):
            status = "excluded_fit" if r.excluded_v3 else ("inactive" if not r.active else "no_fit")
            yv = -0.008
        else:
            status = "favors_2state" if r.favors_2state else "favors_1state"
            if not r.clean_active:
                status = status + "_excluded"
            yv = r.delta_ll_per_bin
        rows.append(dict(rate_hz=r.overall_rate_hz, yv=yv, status=status))
    return pd.DataFrame(rows)


rate_min, rate_max = 0.02, 40
y_cap, y_min = 0.08, -0.01
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
fig.patch.set_facecolor(SURFACE)
for ax, (epoch, df) in zip(axes, [("Encoding", enc), ("Maintenance", maint)]):
    style_ax(ax)
    sdf = scatter_rows(df)
    x = sdf.rate_hz.clip(lower=rate_min)
    clipped = sdf.yv > y_cap
    y = sdf.yv.clip(upper=y_cap)
    colors = sdf.status.map(status_color_blended)
    for status in sdf.status.unique():
        m = sdf.status == status
        ax.scatter(x[m], y[m], s=10, c=[colors[i] for i in colors[m].index],
                   edgecolors=[INK if c else "none" for c in clipped[m]], linewidths=0.5)
    ax.axvline(1.0, color=GRAY_DARK, linewidth=1, linestyle=":")
    ax.axhline(0, color=GRAY_DARK, linewidth=1, linestyle=":")
    ax.set_xscale("log"); ax.set_xlim(rate_min, rate_max); ax.set_ylim(y_min, y_cap)
    ax.set_xlabel("overall firing rate (Hz, log scale)", fontsize=10)
    ax.set_title(f"{epoch} (n={len(df)})", fontsize=12)
axes[0].set_ylabel("$\Delta$LL / bin (held-out)", fontsize=10)
fig.suptitle("Firing rate vs. held-out $\Delta$LL/bin, all 902 neurons", fontsize=11, y=1.05)
fig.tight_layout()
fig.savefig(OUT + "04_rate_vs_deltaLL_scatter.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
fig.savefig(OUT + "04_rate_vs_deltaLL_scatter.eps", bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)
print("saved 04_rate_vs_deltaLL_scatter")

thresholds = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200]

enc_clean = enc[~enc.excluded_v3].copy()
maint_clean = maint[~maint.excluded_v3].copy()
# total_spikes: encoding fit already stores it; maintenance mid-window fit also stores it directly
enc_clean["favors_2state_single"] = enc_clean["delta_ll_per_bin"].notna() & (enc_clean["delta_ll_per_bin"] > 0)
maint_clean["favors_2state_single"] = maint_clean["delta_ll_per_bin"].notna() & (maint_clean["delta_ll_per_bin"] > 0)


def sweep(df, epoch_label):
    rows = []
    for pop_label, mask in [("All cells", df.neuron_id.notna()), ("Concept cells", df.is_concept_cell)]:
        sub_pop = df[mask]
        for t in thresholds:
            sub = sub_pop[sub_pop.total_spikes >= t]
            n = len(sub)
            frac = float(sub.favors_2state_single.mean()) if n > 0 else None
            rows.append(dict(epoch=epoch_label, population=pop_label, threshold=t, n=n, frac_favor_2state=frac))
    return rows

sens_rows = sweep(enc_clean, "Encoding") + sweep(maint_clean, "Maintenance")
sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(OUT + "05_spikecount_sensitivity_stats.csv", index=False)

pop_colors = {"All cells": GRAY_MED, "Concept cells": "#1baf7a"}
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
fig.patch.set_facecolor(SURFACE)
metrics = [("n", "n neurons remaining", (0, 920)), ("frac_favor_2state", "fraction favoring 2-state", (0.45, 0.95))]
for row, (mkey, mlabel, ylim) in enumerate(metrics):
    for col, epoch in enumerate(["Encoding", "Maintenance"]):
        ax = axes[row, col]
        style_ax(ax)
        for pop in ["All cells", "Concept cells"]:
            sub = sens_df[(sens_df.epoch == epoch) & (sens_df.population == pop)].sort_values("threshold")
            ax.plot(sub.threshold, sub[mkey], marker="o", markersize=4, linewidth=2, color=pop_colors[pop], label=pop)
        if mkey == "frac_favor_2state":
            ax.axhline(0.5, color=INK, linewidth=1, linestyle="--", alpha=0.5)
        ax.set_ylim(*ylim)
        if row == 0: ax.set_title(epoch, fontsize=12)
        if col == 0: ax.set_ylabel(mlabel, fontsize=9.5)
        if row == 1: ax.set_xlabel("min total-spike-count threshold", fontsize=9.5)
axes[1, 0].legend(fontsize=9, loc="lower left", framealpha=0.9)
fig.suptitle("Sensitivity to minimum spike-count threshold (clean fits only)", fontsize=11.5, y=1.0)
fig.tight_layout()
fig.savefig(OUT + "05_spikecount_sensitivity.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
fig.savefig(OUT + "05_spikecount_sensitivity.eps", bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)
print(sens_df.to_string(index=False))

# --- paired McNemar: binary favors_2state, encoding vs maintenance ---
def paired_mcnemar(name, mask_fn):
    e_ids = set(enc[enc.clean_active & mask_fn(enc)].neuron_id)
    m_ids = set(maint[maint.clean_active & mask_fn(maint)].neuron_id)
    common = sorted(e_ids & m_ids)
    e_sub = enc[enc.neuron_id.isin(common)][["neuron_id", "favors_2state"]].rename(columns={"favors_2state": "enc"})
    m_sub = maint[maint.neuron_id.isin(common)][["neuron_id", "favors_2state"]].rename(columns={"favors_2state": "maint"})
    paired = e_sub.merge(m_sub, on="neuron_id")
    table = pd.crosstab(paired.enc, paired.maint)
    exact = len(paired) < 25 or table.values.min() < 5
    res = mcnemar(table, exact=exact, correction=not exact)
    print(f"{name} (n={len(paired)}): enc={paired.enc.mean()*100:.1f}%  maint={paired.maint.mean()*100:.1f}%  "
          f"{'exact' if exact else 'chi2'} McNemar stat={res.statistic:.3f} p={res.pvalue:.4g}")
    return table, res

print("=== Paired McNemar (binary favors_2state) ===")
paired_mcnemar("All cells", lambda d: pd.Series(True, index=d.index))
paired_mcnemar("Concept cells", lambda d: d.is_concept_cell)
print()

# --- Stuart-Maxwell: 3-way class distribution, encoding vs maintenance ---
def stuart_maxwell(paired):
    k = len(class_order)
    idx = {c: i for i, c in enumerate(class_order)}
    table = np.zeros((k, k))
    for _, row in paired.iterrows():
        table[idx[row.class_enc], idx[row.class_maint]] += 1
    r, c = table.sum(axis=1), table.sum(axis=0)
    d = (r - c)[:-1]
    V = np.zeros((k - 1, k - 1))
    for i in range(k - 1):
        V[i, i] = r[i] + c[i] - 2 * table[i, i]
        for j in range(k - 1):
            if i != j:
                V[i, j] = -(table[i, j] + table[j, i])
    try:
        stat = float(d @ np.linalg.inv(V) @ d)
        p = float(1 - chi2_dist.cdf(stat, k - 1))
    except np.linalg.LinAlgError:
        stat, p = np.nan, np.nan
    return stat, p

print("=== Stuart-Maxwell (marginal homogeneity, enc vs maint) ===")
for name, mask_fn in [("All cells", lambda d: pd.Series(True, index=d.index)),
                       ("Concept cells", lambda d: d.is_concept_cell),
                       ("Labelled_IN", lambda d: d.is_labelled_IN),
                       ("Labelled_PY", lambda d: d.is_labelled_PY)]:
    e_ids = set(enc[enc.clean_active & mask_fn(enc)].neuron_id)
    m_ids = set(maint[maint.clean_active & mask_fn(maint)].neuron_id)
    common = sorted(e_ids & m_ids)
    e_sub = enc[enc.neuron_id.isin(common)][["neuron_id", "model_class"]].rename(columns={"model_class": "class_enc"})
    m_sub = maint[maint.neuron_id.isin(common)][["neuron_id", "model_class"]].rename(columns={"model_class": "class_maint"})
    paired = e_sub.merge(m_sub, on="neuron_id")
    stat, p = stuart_maxwell(paired)
    print(f"{name} (n paired={len(paired)}): stat={stat:.3f}  p={p:.4g}")
print()

# --- IN vs PY within epoch (chi-square) ---
print("=== IN vs PY, model-class distribution, within epoch ===")
for label, df in [("Encoding", enc), ("Maintenance", maint)]:
    in_df = df[df.clean_active & df.is_labelled_IN]
    py_df = df[df.clean_active & df.is_labelled_PY]
    table = np.array([[(in_df.model_class == c).sum() for c in class_order],
                       [(py_df.model_class == c).sum() for c in class_order]])
    chi2, p, dof, expected = chi2_contingency(table)
    print(f"{label}: chi2({dof})={chi2:.3f}  p={p:.4g}")
print()

# --- persistent-class composition breakdown ---
def reason(row):
    if pd.isna(row["delta_ll_per_bin"]) or pd.isna(row["rate_ratio"]):
        return np.nan
    if row["force_persistent_v3"]:
        return "boundary_reclass"
    if row["delta_ll_per_bin"] <= 0:
        return "one_state_won"
    if row["rate_ratio"] < 1.5:
        return "low_rate_ratio"
    return "not_persistent"

enc["reason"] = enc.apply(reason, axis=1)
maint["reason"] = maint.apply(reason, axis=1)
print("=== Persistent-class composition ===")
for epoch_label, df in [("Encoding", enc), ("Maintenance", maint)]:
    for pop_label, mask_fn in [("All cells", lambda d: pd.Series(True, index=d.index)),
                                ("Concept cells", lambda d: d.is_concept_cell),
                                ("Labelled_IN", lambda d: d.is_labelled_IN),
                                ("Labelled_PY", lambda d: d.is_labelled_PY)]:
        sub = df[df.clean_active & mask_fn(df)]
        n = len(sub)
        n_persistent = int((sub.reason != "not_persistent").sum())
        boundary = int((sub.reason == "boundary_reclass").sum())
        one_state = int((sub.reason == "one_state_won").sum())
        low_ratio = int((sub.reason == "low_rate_ratio").sum())
        print(f"  [{epoch_label}] {pop_label:16s} n={n:4d}  persistent={n_persistent:4d} "
              f"[boundary-reclass={boundary:3d}, one-state-won={one_state:3d}, low-rate-ratio={low_ratio:3d}]")

print()
print("ALL DONE")




