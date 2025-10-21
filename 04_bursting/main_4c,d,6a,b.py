import os
import ast
import hashlib
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal.windows import gaussian

# =========================
# Parameters
# =========================
bin_size = 0.07                     # seconds
TRIAL_WIN = 1                       # seconds per trial window; change to 2.8 for Delay
GLOBAL_SEED = 5
sigma_sec = 0.04                    # Gaussian kernel SD in seconds
prominence_threshold_percentile = 90 # percentile on smoothed rate
min_inter_burst_interval = 0.14      # seconds (distance between peaks)
N_SURROGATES = 100                   # Poisson repetitions per subject
PRINT_DEBUG = True

# Derived
min_bin_separation = int(np.ceil(min_inter_burst_interval / bin_size))
# =========================

# =========================
# Paths (edit as needed)
# =========================
path_decay_acg  = '../Clustering_3D.feather'
path_all_meta   = '../data/all_neuron_brain_regions_cleaned.feather'
path_trials     = '../graph_data/graph_encoding1.feather'

# =========================
# Load data
# =========================
df_metadata_decay_acg = pd.read_feather(path_decay_acg)
df_metadata_all       = pd.read_feather(path_all_meta)
df_enc1_filtered      = pd.read_feather(path_trials)

# Apply R² filter only to decay/ACG categories
df_metadata_decay_acg = df_metadata_decay_acg[df_metadata_decay_acg["R2"] > 0.3]

# Use full metadata (no filtering) for Concept, All cells
df_metadata_all_cells = df_metadata_all.copy()
df_metadata_concept   = df_metadata_all.copy()

# For cell-type split, original code used the decay/acg dataframe; keep consistent
df_metadata_in_py     = df_metadata_decay_acg.copy()

# =========================
# New knobs for per-subject low/high splits
# =========================
PER_SUBJ = 10   # minimum number of eligible neurons per subject to include for ACG/Decay categories
OVERLAP  = 1    

categories = [
    "All_cells",
    "Lowest_ACG", "Highest_ACG",
    "Lowest_decay", "Highest_decay",
    "Concept_cells",
    "Pyramidal", "Interneurons"
]

def parse_spike_entry(val):
    """Safely parse a spike string/list and keep spikes within [0, TRIAL_WIN]."""
    if val is None:
        return np.array([], dtype=float)
    if isinstance(val, str):
        if val.strip() in ("", "[]"):
            return np.array([], dtype=float)
        try:
            arr = np.array(ast.literal_eval(val), dtype=float)
        except Exception:
            return np.array([], dtype=float)
    elif isinstance(val, (list, tuple, np.ndarray)):
        arr = np.array(val, dtype=float)
    else:
        return np.array([], dtype=float)
    if arr.ndim == 0:
        arr = np.array([float(arr)])
    return arr[(arr >= 0.0) & (arr <= TRIAL_WIN)]

def stable_seed(*parts):
    """Stable 64-bit seed from arbitrary parts (strings/ints)."""
    s = "|".join(map(str, parts)).encode("utf-8")
    h = hashlib.sha256(s).digest()
    return (int.from_bytes(h[:8], "little") ^ GLOBAL_SEED) % (2**32 - 1)

def first_nonnull_series(series):
    """Return at most one spike-list cell to avoid double-counting duplicates."""
    for v in series:
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return [v]
    return []

def _split_low_high_for_subject(df_subject_metadata, feature, per_subj=PER_SUBJ, overlap=OVERLAP):
    """
    Deterministically split one subject's neurons into low/high sets on `feature`.
    - Requires at least `per_subj` rows (after dropping NaNs in feature).
    - Uses stable sort with Neuron_ID_3 as a tiebreaker for reproducibility.
    - Returns (low_ids, high_ids); each has size floor(per_subj/2)+overlap.
    """
    if df_subject_metadata.empty:
        return [], []
    if "Neuron_ID_3" not in df_subject_metadata.columns or feature not in df_subject_metadata.columns:
        return [], []

    df = df_subject_metadata[["Neuron_ID_3", feature]].dropna(subset=[feature]).copy()
    if len(df) < per_subj:
        return [], []  # not enough neurons for this subject

    df_sorted = df.sort_values(
        by=[feature, "Neuron_ID_3"],
        ascending=[True, True],
        kind="mergesort"  # stable
    )

    m = (per_subj // 2) + overlap
    low_ids  = df_sorted.head(m)["Neuron_ID_3"].tolist()
    high_ids = df_sorted.tail(m)["Neuron_ID_3"].tolist()
    return low_ids, high_ids

def subject_metadata_for_category(category, subject_id):
    """Pick the right metadata frame for a subject depending on category."""
    if category == "Concept_cells":
        return df_metadata_concept[df_metadata_concept['subject_id'] == subject_id]
    elif category in ["Pyramidal", "Interneurons"]:
        return df_metadata_in_py[df_metadata_in_py['subject_id'] == subject_id]
    elif category == "All_cells":
        return df_metadata_all_cells[df_metadata_all_cells['subject_id'] == subject_id]
    else:
        # ACG/Decay categories use the R²-filtered decay/acg dataframe
        return df_metadata_decay_acg[df_metadata_decay_acg['subject_id'] == subject_id]

def choose_neurons(category, df_subject_metadata):
    """Return Neuron_ID_3 list for this subject & category, using the new per-subject split for ACG/Decay."""
    if category in ("Lowest_ACG", "Highest_ACG"):
        low_ids, high_ids = _split_low_high_for_subject(df_subject_metadata, feature="ACG_Norm")
        return low_ids if category == "Lowest_ACG" else high_ids

    elif category in ("Lowest_decay", "Highest_decay"):
        low_ids, high_ids = _split_low_high_for_subject(df_subject_metadata, feature="Decay")
        return low_ids if category == "Lowest_decay" else high_ids

    elif category == "All_cells":
        return df_subject_metadata["Neuron_ID_3"].dropna().tolist()

    elif category == "Concept_cells":
        return df_subject_metadata[df_subject_metadata["Signi"] == "Y"]["Neuron_ID_3"].dropna().tolist()

    elif category == "Pyramidal":
        return df_subject_metadata[df_subject_metadata["Cell_Type_New"] == "PY"]["Neuron_ID_3"].dropna().tolist()

    elif category == "Interneurons":
        return df_subject_metadata[df_subject_metadata["Cell_Type_New"] == "IN"]["Neuron_ID_3"].dropna().tolist()

    return []

def subjects_for_category(category):
    """
    Subjects eligible per category.
    - For ACG/Decay categories: require >= PER_SUBJ neurons with a valid feature value (after R² filter applied upstream).
    - Other categories keep your previous logic (thresholds unchanged unless noted).
    """
    if category in ("Lowest_ACG", "Highest_ACG"):
        eligible = df_metadata_decay_acg.dropna(subset=["ACG_Norm"])
        counts = eligible.groupby("subject_id")["Neuron_ID_3"].count()
        return counts[counts >= PER_SUBJ].index.tolist()

    if category in ("Lowest_decay", "Highest_decay"):
        eligible = df_metadata_decay_acg.dropna(subset=["Decay"])
        counts = eligible.groupby("subject_id")["Neuron_ID_3"].count()
        return counts[counts >= PER_SUBJ].index.tolist()

    if category == "Concept_cells":
        counts = df_metadata_concept[df_metadata_concept["Signi"] == "Y"]["subject_id"].value_counts()
        return counts[counts >= 3].index.tolist()

    if category in ("Pyramidal", "Interneurons"):
        label = "PY" if category == "Pyramidal" else "IN"
        counts = df_metadata_in_py[df_metadata_in_py["Cell_Type_New"] == label]["subject_id"].value_counts()
        return counts[counts >= 3].index.tolist()

    if category == "All_cells":
        counts = df_metadata_all_cells["subject_id"].value_counts()
        return counts[counts >= 10].index.tolist()

    return []

results_delay = {
    cat: {
        "real_bursts": [], "poisson_bursts": [],
        "real_by_subj": {}, "poisson_by_subj": {}
    } for cat in categories
}

for category in categories:
    subj_ids = subjects_for_category(category)
    if PRINT_DEBUG:
        print(f"[{category}] candidate subjects: {len(subj_ids)}")

    for subject_id in sorted(subj_ids):
        # Trials for this subject (here: only memory load == 1)
        df_subject_trials = df_enc1_filtered[
            (df_enc1_filtered['subject_id'] == subject_id) &
            (df_enc1_filtered['num_images_presented'] == 1)
        ]
        if df_subject_trials.empty:
            continue

        # Metadata slice & neuron selection (now per-subject split when ACG/Decay)
        df_sm = subject_metadata_for_category(category, subject_id)
        neuron_ids = choose_neurons(category, df_sm)
        if len(neuron_ids) == 0:
            continue

        # Restrict trials to those neurons
        df_cat = df_subject_trials[df_subject_trials['Neuron_ID_3'].isin(neuron_ids)]
        trial_ids = sorted(df_cat['trial_id'].unique())
        n_trials = len(trial_ids)
        if n_trials == 0:
            continue

        # REAL BURSTS 
        all_spikes = []
        for trial_idx, trial_id in enumerate(trial_ids):
            df_trial = df_cat[df_cat['trial_id'] == trial_id]
            for neuron_id in sorted(df_trial['Neuron_ID_3'].unique()):
                spikes_series = df_trial.loc[df_trial['Neuron_ID_3'] == neuron_id, 'Standardized_Spikes'].dropna()
                for s in first_nonnull_series(spikes_series):
                    arr = parse_spike_entry(s)
                    if arr.size:
                        all_spikes.extend(arr + trial_idx * TRIAL_WIN)

        if len(all_spikes) == 0:
            results_delay[category]["real_bursts"].append(0.0)
            results_delay[category]["poisson_bursts"].append(0.0)
            results_delay[category]["real_by_subj"][str(subject_id)] = 0.0
            results_delay[category]["poisson_by_subj"][str(subject_id)] = 0.0
            continue

        total_duration = n_trials * TRIAL_WIN
        time_bins = np.arange(0.0, total_duration + bin_size, bin_size)

        spike_counts, _ = np.histogram(all_spikes, bins=time_bins)

        # Smoothing kernel
        kernel = gaussian(len(spike_counts), std=sigma_sec / bin_size)
        kernel = kernel / np.sum(kernel)

        smoothed = np.convolve(spike_counts, kernel, mode='same')
        threshold = np.percentile(smoothed, prominence_threshold_percentile)

        real_peaks, _ = signal.find_peaks(
            smoothed, prominence=threshold, distance=min_bin_separation
        )
        real_burst_count = int(len(real_peaks))

        # Save real
        results_delay[category]["real_bursts"].append(float(real_burst_count))
        results_delay[category]["real_by_subj"][str(subject_id)] = float(real_burst_count)

        # ---------- POISSON SURROGATES ----------
        rates_by_trial_idx = {ti: [] for ti in range(n_trials)}
        for trial_idx, trial_id in enumerate(trial_ids):
            df_trial = df_cat[df_cat['trial_id'] == trial_id]
            for neuron_id in sorted(df_trial['Neuron_ID_3'].unique()):
                spikes_series = df_trial.loc[df_trial['Neuron_ID_3'] == neuron_id, 'Standardized_Spikes'].dropna()
                cnt = 0
                for s in first_nonnull_series(spikes_series):
                    arr = parse_spike_entry(s)
                    cnt += arr.size
                rate = cnt / TRIAL_WIN
                if rate >= 0:
                    rates_by_trial_idx[trial_idx].append(rate)

        rng = np.random.default_rng(stable_seed(category, subject_id, "poisson"))

        poisson_bursts = []
        for _ in range(N_SURROGATES):
            poisson_spikes = []
            for trial_idx in range(n_trials):
                trial_rates = rates_by_trial_idx[trial_idx]
                if not trial_rates:
                    continue
                for rate in trial_rates:
                    expected = rng.poisson(rate * TRIAL_WIN)
                    if expected <= 0:
                        continue
                    synthetic = rng.uniform(0.0, TRIAL_WIN, expected) + trial_idx * TRIAL_WIN
                    poisson_spikes.extend(synthetic)

            if not poisson_spikes:
                poisson_bursts.append(0)
                continue

            spike_counts_pois, _ = np.histogram(poisson_spikes, bins=time_bins)
            smoothed_pois = np.convolve(spike_counts_pois, kernel, mode='same')
            peaks, _ = signal.find_peaks(smoothed_pois, prominence=threshold, distance=min_bin_separation)
            poisson_bursts.append(int(len(peaks)))

        poisson_mean_bursts = float(np.mean(poisson_bursts)) if len(poisson_bursts) else 0.0
        results_delay[category]["poisson_bursts"].append(poisson_mean_bursts)
        results_delay[category]["poisson_by_subj"][str(subject_id)] = poisson_mean_bursts

    if PRINT_DEBUG:
        rm = results_delay[category]["real_by_subj"]
        pm = results_delay[category]["poisson_by_subj"]
        common = sorted(set(rm) & set(pm))
        print(f"  -> saved subjects: real={len(rm)} poisson={len(pm)} paired={len(common)}")

if PRINT_DEBUG:
    rows = []
    for cat in categories:
        rm = results_delay[cat]["real_by_subj"]; pm = results_delay[cat]["poisson_by_subj"]
        rows.append({
            "Category": cat,
            "n_real_subj": len(rm),
            "n_pois_subj": len(pm),
            "n_paired": len(set(rm) & set(pm)),
            "real_mean": np.mean(results_delay[cat]["real_bursts"]) if results_delay[cat]["real_bursts"] else np.nan,
            "pois_mean": np.mean(results_delay[cat]["poisson_bursts"]) if results_delay[cat]["poisson_bursts"] else np.nan,
        })
    print("\nSanity summary:")
    print(pd.DataFrame(rows))

import os
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu, friedmanchisquare
from statsmodels.stats.multitest import multipletests

out_dir = "./"
os.makedirs(out_dir, exist_ok=True)

def symbol_from_p(p):
    """Hashes for within-category (Real vs Poisson) brackets."""
    if p is None or np.isnan(p): return ""
    if p < 0.005:   return "***"
    if p < 0.01:    return "**"
    if p< 0.05:     return "*"
    return ""

def star_from_p(p):
    """Asterisks for cross-category (Real vs Real) comparisons."""
    if p is None or np.isnan(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

def draw_bracket(ax, x1, x2, y_base, h=2.0, txt="", txt_pad=0.4, lw=1.5, color="k"):
    """Draw a bracket between x1 and x2 starting at y_base with height h and place txt above."""
    ax.plot([x1, x1, x2, x2], [y_base, y_base+h, y_base+h, y_base], lw=lw, c=color, clip_on=False)
    ax.text((x1+x2)/2, y_base + h + txt_pad, txt, ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=color)

def sanitize_yerr(yerr_list):
    """Return yerr usable by Matplotlib.
    - If all values are non-finite, return None (no error bars).
    - Otherwise, replace NaN/inf with 0 so bars still draw.
    """
    arr = np.asarray(yerr_list, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return None
    arr[~np.isfinite(arr)] = 0.0
    return arr

def get_map(cd, key):
    m = cd.get(key, None)
    return m if isinstance(m, dict) else None

def get_list(cd, key):
    arr = cd.get(key, None)
    if arr is None: return None
    arr = np.asarray(arr, dtype=float)
    return arr[~np.isnan(arr)]

# =========================
# Descriptives for plotting (mean ± SEM)
# =========================
category_means = {}

for cat in categories:
    cd = results_delay[cat]
    # Prefer subject maps (paired-friendly) for descriptives
    real_map = get_map(cd, "real_by_subj")
    pois_map = get_map(cd, "poisson_by_subj")

    if real_map is not None:
        real_vals = np.asarray(list(real_map.values()), float)
        real_vals = real_vals[~np.isnan(real_vals)]
    else:
        real_vals = get_list(cd, "real_bursts")

    if pois_map is not None:
        pois_vals = np.asarray(list(pois_map.values()), float)
        pois_vals = pois_vals[~np.isnan(pois_vals)]
    else:
        pois_vals = get_list(cd, "poisson_bursts")

    def mean_sem(x):
        if x is None or x.size == 0: return np.nan, np.nan, 0
        mu = float(np.mean(x))
        se = float(np.std(x, ddof=1) / np.sqrt(len(x))) if x.size > 1 else np.nan
        return mu, se, int(x.size)

    r_mean, r_sem, n_r = mean_sem(real_vals)
    p_mean, p_sem, n_p = mean_sem(pois_vals)

    category_means[cat] = {
        "real_mean": r_mean, "real_sem": r_sem, "n_real": n_r,
        "poisson_mean": p_mean, "poisson_sem": p_sem, "n_pois": n_p
    }

real_means    = [category_means[c]["real_mean"]    for c in categories]
poisson_means = [category_means[c]["poisson_mean"] for c in categories]
real_sems     = [category_means[c]["real_sem"]     for c in categories]
poisson_sems  = [category_means[c]["poisson_sem"]  for c in categories]

# WITHIN-CATEGORY: Real vs Poisson (paired by subject) + FDR-BH (Family A)

stats_within = {cat: {} for cat in categories}
raw_pvals_within, cats_with_within = [], []
all_diffs_for_condition_omnibus = []  # y - x (Poisson - Real) to test Condition effect

for cat in categories:
    cd = results_delay[cat]
    real_map = get_map(cd, "real_by_subj")
    pois_map = get_map(cd, "poisson_by_subj")

    pval = np.nan
    stat = np.nan
    n_pairs = 0
    test_name = "NA"

    if real_map is not None and pois_map is not None:
        common = sorted(set(real_map) & set(pois_map))
        if len(common) >= 2:
            x = np.array([real_map[s] for s in common], float)
            y = np.array([pois_map[s] for s in common], float)
            d = y - x
            all_diffs_for_condition_omnibus.extend(d[~np.isnan(d)])
            try:
                res = wilcoxon(x, y, zero_method='pratt', alternative='two-sided', method="asymptotic")
                stat, pval, test_name = float(res.statistic), float(res.pvalue), "Wilcoxon (paired)"
            except Exception:
                pval, stat, test_name = 1.0, np.nan, "Wilcoxon (ties -> p=1)"
            n_pairs = len(common)

    if not np.isnan(pval):
        raw_pvals_within.append(pval)
        cats_with_within.append(cat)

    stats_within[cat] = {
        "test": test_name, "stat": stat, "p_raw": pval, "n_pairs": n_pairs
    }

# FDR-BH within family
corrected_within = {}
if len(raw_pvals_within) > 0:
    reject, p_corr, _, _ = multipletests(raw_pvals_within, alpha=0.05, method='fdr_bh')
    corrected_within = {cat: (bool(r), float(pc)) for cat, r, pc in zip(cats_with_within, reject, p_corr)}

# Condition main effect (Real vs Poisson) across categories
omnibus_condition = {"test": "Wilcoxon on (Poisson-Real) paired diffs across all categories",
                     "stat": np.nan, "p": np.nan, "n_pairs": 0}
ad = np.asarray(all_diffs_for_condition_omnibus, float)
ad = ad[~np.isnan(ad)]
if ad.size >= 2 and not np.allclose(ad, 0):
    try:
        res = wilcoxon(ad, np.zeros_like(ad), zero_method='wilcox', alternative='two-sided', mode='auto')
        omnibus_condition.update({"stat": float(res.statistic), "p": float(res.pvalue), "n_pairs": int(ad.size)})
    except Exception:
        omnibus_condition.update({"p": 1.0, "n_pairs": int(ad.size)})

# =========================
# REAL vs REAL pairwise across categories (prefer paired), FDR-BH (Family B)
# =========================
pairwise_rows, raw_pvals_pairs = [], []
real_maps = {c: get_map(results_delay[c], "real_by_subj") for c in categories}

for a, b in combinations(categories, 2):
    Amap, Bmap = real_maps.get(a), real_maps.get(b)
    test_name, stat, pval = "NA", np.nan, np.nan
    paired_used, n_common = False, 0

    # Paired path (preferred)
    if Amap is not None and Bmap is not None:
        common = sorted(set(Amap) & set(Bmap))
        n_common = len(common)
        if n_common >= 2:
            Ax = np.array([Amap[s] for s in common], float)
            Bx = np.array([Bmap[s] for s in common], float)
            try:
                res = wilcoxon(Bx, Ax, zero_method='wilcox', alternative='two-sided', mode='auto')
                stat, pval, test_name = float(res.statistic), float(res.pvalue), "Wilcoxon (paired)"
            except Exception:
                pval, stat, test_name = 1.0, np.nan, "Wilcoxon (paired; ties -> p=1)"
            paired_used = True

    # Unpaired fallback (only if paired not possible)
    if (not paired_used) and (np.isnan(pval) or pval is np.nan):
        Ax_all = get_list(results_delay[a], "real_bursts")
        Bx_all = get_list(results_delay[b], "real_bursts")
        if Ax_all is not None and Ax_all.size > 0 and Bx_all is not None and Bx_all.size > 0:
            mw = mannwhitneyu(Bx_all, Ax_all, alternative="two-sided")
            stat, pval, test_name = float(mw.statistic), float(mw.pvalue), "Mann–Whitney U (unpaired)"
            n_common = 0

    pairwise_rows.append({
        "A": a, "B": b, "test": test_name, "stat": stat, "p_raw": pval,
        "paired": paired_used, "n_common": n_common
    })
    raw_pvals_pairs.append(pval)

df_pairs = pd.DataFrame(pairwise_rows)
if not df_pairs.empty and df_pairs["p_raw"].notna().any():
    rej_pairs, p_corr_pairs, _, _ = multipletests(df_pairs["p_raw"].fillna(1.0), alpha=0.05, method='fdr_bh')
    df_pairs["p_FDR_BH"] = p_corr_pairs
    df_pairs["Significant (FDR<0.05)"] = rej_pairs.astype(bool)
    df_pairs["Stars"] = [star_from_p(p) for p in df_pairs["p_FDR_BH"]]
    df_pairs = df_pairs.sort_values(["p_FDR_BH", "p_raw"], na_position="last")
else:
    df_pairs["p_FDR_BH"] = np.nan
    df_pairs["Significant (FDR<0.05)"] = False
    df_pairs["Stars"] = ""

df_pairs.to_csv(f"{out_dir}/real_vs_real_pairwise_stats.csv", index=False)

# OMNIBUS #2: Across-category effect on REAL via Friedman (within-subject)
omnibus_categories = {"test": "Friedman (REAL across categories)", "chi2": np.nan, "p": np.nan, "n_subjects": 0}

# Subjects that exist in ALL categories for REAL
real_common_subjects = None
for c in categories:
    rmap = real_maps.get(c)
    if rmap is None:
        real_common_subjects = set()  # cannot run
        break
    subj_set = set(k for k, v in rmap.items() if v is not None and not np.isnan(v))
    real_common_subjects = subj_set if real_common_subjects is None else (real_common_subjects & subj_set)

if real_common_subjects and len(real_common_subjects) >= 3 and len(categories) >= 3:
    S = sorted(real_common_subjects)
    mat = np.column_stack([np.asarray([real_maps[c][s] for s in S], float) for c in categories])
    ok = ~np.any(np.isnan(mat), axis=1)
    mat_ok = mat[ok]
    if mat_ok.shape[0] >= 3:
        args = [mat_ok[:, j] for j in range(mat_ok.shape[1])]
        chi2, pf = friedmanchisquare(*args)
        omnibus_categories.update({"chi2": float(chi2), "p": float(pf), "n_subjects": int(mat_ok.shape[0])})
        
plt.figure(figsize=(15, 8))
x = np.arange(len(categories))
width = 0.35

yerr_real = sanitize_yerr(real_sems)
yerr_pois = sanitize_yerr(poisson_sems)

bars_real = plt.bar(x - width/2, real_means,    width,
                    yerr=yerr_real,    capsize=5, label="Real Data",     color='black')
bars_pois = plt.bar(x + width/2, poisson_means, width,
                    yerr=yerr_pois, capsize=5, label="Poisson Surrogate", color='gray')

plt.xticks(x, categories, rotation=45, ha='right')
plt.ylabel("Burst Count")
plt.ylim(0, 90)
plt.title("Mean Burst Count: Real Data vs. Poisson Surrogate (Across Subjects and Categories)")
plt.legend()
plt.tight_layout()

ax = plt.gca()
ymin, ymax = ax.get_ylim()
ymax_needed = ymax

# Within-category Real vs Poisson brackets (FDR-corrected, Family A)
for idx, cat in enumerate(categories):
    tup = corrected_within.get(cat, None)
    if not tup:
        continue
    rej, p_corr = tup
    sym = symbol_from_p(p_corr)  # '' | '#' | '##'
    if not sym:
        continue
    # tallest bar top + SEM (if any)
    top_real = (real_means[idx] if np.isfinite(real_means[idx]) else 0.0) + (real_sems[idx] if np.isfinite(real_sems[idx]) else 0.0)
    top_pois = (poisson_means[idx] if np.isfinite(poisson_means[idx]) else 0.0) + (poisson_sems[idx] if np.isfinite(poisson_sems[idx]) else 0.0)
    top_pair = max(top_real, top_pois)
    pad, h = 3.0, 2.0
    base_y = top_pair + pad
    ymax_needed = max(ymax_needed, base_y + h + 1.0)
    x1 = x[idx] - width/2
    x2 = x[idx] + width/2
    draw_bracket(ax, x1, x2, base_y, h=h, txt=sym, lw=1.5, color='k')

if ymax_needed > ymax:
    ax.set_ylim(ymin, ymax_needed)

# Cross-category REAL vs REAL brackets for a baseline (to avoid clutter), Family B
idx_map = {c: i for i, c in enumerate(categories)}
baseline_cat = "All_cells" if "All_cells" in categories else categories[0]
pairs_to_annotate = [(baseline_cat, c) for c in categories if c != baseline_cat]

cur_ymin, cur_ymax = ax.get_ylim()
y_base = cur_ymax + 3.0
y_step = 3.5
extra2 = 0
k = 0

for a, b in pairs_to_annotate:
    if a not in idx_map or b not in idx_map:
        continue
    row = df_pairs[((df_pairs["A"]==a) & (df_pairs["B"]==b)) |
                   ((df_pairs["A"]==b) & (df_pairs["B"]==a))]
    if row.empty:
        continue
    stars = row.iloc[0]["Stars"]
    if not isinstance(stars, str) or len(stars) == 0:
        continue
    i, j = idx_map[a], idx_map[b]
    x1 = i - width/2
    x2 = j - width/2
    draw_bracket(ax, x1, x2, y_base + k*y_step, h=1.6, txt=stars, lw=1.5, color='k')
    extra2 = max(extra2, (y_base + k*y_step + 1.6 + 1.0) - cur_ymax)
    k += 1

if extra2 > 0:
    ax.set_ylim(cur_ymin, cur_ymax + extra2)

# Save & show
plt.savefig(f"{out_dir}/04_bursting/main_4cd_6ab.eps",
            format='eps', dpi=300, bbox_inches='tight')
plt.show()

within_rows = []
for cat in categories:
    row = {
        "Category": cat,
        "Within test": stats_within[cat].get("test", "NA"),
        "Within stat": stats_within[cat].get("stat", np.nan),
        "Within p_raw": stats_within[cat].get("p_raw", np.nan),
        "Within n_pairs": stats_within[cat].get("n_pairs", 0),
    }
    if cat in corrected_within:
        rej, pc = corrected_within[cat]
        row.update({"Within p_FDR_BH": pc, "Hash": symbol_from_p(pc), "Sig(FDR<0.05)": rej})
    within_rows.append(row)

if within_rows:
    print("\nWithin-category (Real vs Poisson) — paired by subject, FDR-BH (Family A):")
    print(pd.DataFrame(within_rows))

print("\nOMNIBUS #1 — Condition main effect (Real vs Poisson) across categories (Wilcoxon on pooled diffs):")
print(omnibus_condition)

print("\nREAL vs REAL pairwise comparisons (prefer paired when overlap exists) — FDR-BH (Family B):")
print(df_pairs)

print("\nOMNIBUS #2 — Across-category effect on REAL (Friedman within-subject):")
print(omnibus_categories)


# Significance matrix 

if not df_pairs.empty and df_pairs["p_FDR_BH"].notna().any():
    cats_for_matrix = sorted(set(df_pairs["A"]).union(set(df_pairs["B"])))
    mat = np.full((len(cats_for_matrix), len(cats_for_matrix)), np.nan, dtype=float)
    stars_mat = [["" for _ in cats_for_matrix] for __ in cats_for_matrix]

    for _, r in df_pairs.iterrows():
        a, b = r["A"], r["B"]
        i, j = cats_for_matrix.index(a), cats_for_matrix.index(b)
        pF = r.get("p_FDR_BH", np.nan)
        mat[i, j] = pF; mat[j, i] = pF
        s = r.get("Stars", "")
        stars_mat[i][j] = s; stars_mat[j][i] = s

    plt.figure(figsize=(9, 7))
    with np.errstate(divide='ignore'):
        im = plt.imshow(-np.log10(mat), interpolation="nearest")
    plt.xticks(range(len(cats_for_matrix)), cats_for_matrix, rotation=45, ha="right")
    plt.yticks(range(len(cats_for_matrix)), cats_for_matrix)
    cbar = plt.colorbar(im)
    cbar.set_label("-log10(p_FDR)")
    for i in range(len(cats_for_matrix)):
        for j in range(len(cats_for_matrix)):
            if i == j: 
                continue
            if stars_mat[i][j]:
                plt.text(j, i, stars_mat[i][j], ha="center", va="center", fontweight="bold")
    plt.title("REAL vs REAL: FDR-corrected pairwise significance")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/significance_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()
else:
    print("\n[Note] Skipping significance matrix: no valid FDR-corrected pairwise p-values.")
