import os
import ast
import time
import warnings
from itertools import combinations
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal.windows import gaussian
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)

BASE = "../"
DESKTOP = "../"
OUT = BASE + "CFI_Figures"
os.makedirs(OUT, exist_ok=True)

BIN_SIZE = 0.03
SIGMA = 0.05
PROM_PCT = 50
DURATION = {"encoding": 1.0, "maintenance": 2.8}

COINCIDENCE_WINDOW = 0.030
N_SHIFTS_FULL_COHORT = 50
EPS = 0.5

N_PERM_MIXED = 300
N_PERM_GEE = 5000
SEED = 0

EPOCHS = ["encoding", "maintenance"]
EPOCH_CFI_COL = {"encoding": "cfi_enc_z", "maintenance": "cfi_delay_z"}

print("Setup OK")

enc_df = pd.read_excel(BASE + "graph_encoding1.xlsx")
delay_df = pd.read_excel(BASE + "graph_delay.xlsx")
for _df in (enc_df, delay_df):
    _df["subject_id"] = _df["subject_id"].astype(int)
    _df["trial_id"] = _df["trial_id"].astype(int)

r2_labels = pd.read_csv(DESKTOP + "cell_clustering_no_waveform_labels.csv", sep="\t")
full_labels = pd.read_csv(BASE + "cell_types.csv")


def neurons_for(category):
    if category == "Labelled_IN":
        return set(r2_labels.loc[r2_labels.Cell_Type_New == "IN", "Neuron_ID_3"])
    if category == "Labelled_PY":
        return set(r2_labels.loc[r2_labels.Cell_Type_New == "PY", "Neuron_ID_3"])
    if category == "All_cells":
        return set(full_labels["Neuron_ID_3"])
    raise ValueError(category)


ALL_CATEGORIES = ["All_cells", "Labelled_IN", "Labelled_PY"]
MIN_NEURONS = {"All_cells": 10}  # Methods §3: >=10 for All_cells, >=3 for R2-filtered categories
print(f"All_cells: {len(neurons_for('All_cells'))} neurons; "
      f"Labelled_IN: {len(neurons_for('Labelled_IN'))}; Labelled_PY: {len(neurons_for('Labelled_PY'))}")

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


_kernel_width = int(5 * SIGMA / BIN_SIZE)
_kernel = gaussian(_kernel_width, std=SIGMA / BIN_SIZE)


def compute_burst_counts():
    rows = []
    for epoch, df_epoch in [("encoding", enc_df), ("maintenance", delay_df)]:
        trial_dur = DURATION[epoch]
        time_bins = np.arange(0.0, trial_dur + BIN_SIZE, BIN_SIZE)
        for category in ALL_CATEGORIES:
            neuron_set = neurons_for(category)
            cat_df = df_epoch[df_epoch.Neuron_ID_3.isin(neuron_set)]
            min_n = MIN_NEURONS.get(category, 3)
            n_per_subj = cat_df.groupby("subject_id")["Neuron_ID_3"].nunique()
            elig_subj = n_per_subj[n_per_subj >= min_n].index

            for subject_id in elig_subj:
                sub_df = cat_df[cat_df.subject_id == subject_id]
                n_neurons = sub_df.Neuron_ID_3.nunique()
                for trial_id, trial_df in sub_df.groupby("trial_id"):
                    spikes = []
                    for s in trial_df.Standardized_Spikes:
                        spikes.extend(parse_spikes(s, trial_dur))
                    if not spikes:
                        rows.append(dict(category=category, epoch=epoch, subject_id=subject_id,
                                          trial_id=trial_id, n_neurons=n_neurons, burst_count=0))
                        continue
                    counts, _ = np.histogram(spikes, bins=time_bins)
                    smoothed = np.convolve(counts, _kernel, mode="same")
                    thresh = np.percentile(smoothed, PROM_PCT)
                    peaks, _ = signal.find_peaks(smoothed, prominence=thresh)
                    rows.append(dict(category=category, epoch=epoch, subject_id=subject_id,
                                      trial_id=trial_id, n_neurons=n_neurons, burst_count=len(peaks)))
    return pd.DataFrame(rows)


t0 = time.time()
burst_full = compute_burst_counts()
print(f"{len(burst_full)} burst rows computed in {time.time()-t0:.1f}s")
display(burst_full.groupby(["category", "epoch"])["burst_count"].agg(["mean", "median", "count"]))

def compute_firing_rate(categories, min_neurons):
    rows = []
    for epoch, df_epoch in [("encoding", enc_df), ("maintenance", delay_df)]:
        dur = DURATION[epoch]
        for category in categories:
            neuron_set = neurons_for(category)
            cat_df = df_epoch[df_epoch.Neuron_ID_3.isin(neuron_set)]
            n_per_subj = cat_df.groupby("subject_id")["Neuron_ID_3"].nunique()
            elig_subj = n_per_subj[n_per_subj >= min_neurons.get(category, 3)].index
            for subject_id in elig_subj:
                sub_df = cat_df[cat_df.subject_id == subject_id]
                n_neurons = sub_df.Neuron_ID_3.nunique()
                for trial_id, trial_df in sub_df.groupby("trial_id"):
                    n_spikes = sum(len(parse_spikes(s, dur)) for s in trial_df.Standardized_Spikes)
                    fr = n_spikes / (n_neurons * dur)
                    rows.append(dict(category=category, epoch=epoch, subject_id=subject_id,
                                      trial_id=trial_id, n_neurons=n_neurons, n_spikes=n_spikes,
                                      firing_rate_hz=fr))
    return pd.DataFrame(rows)


firing_full = compute_firing_rate(ALL_CATEGORIES, MIN_NEURONS)
print(f"{len(firing_full)} firing-rate rows computed")


def get_firing_rate(cat, epoch):
    return firing_full[(firing_full["category"] == cat) & (firing_full["epoch"] == epoch)][
        ["subject_id", "trial_id", "firing_rate_hz"]
    ]

def parse_spike_entry_raw(val):
    if pd.isna(val):
        return np.array([], dtype=float)
    text = str(val).strip()
    if text in {"", "nan", "None", "[]"}:
        return np.array([], dtype=float)
    try:
        return np.asarray(ast.literal_eval(text), dtype=float)
    except (ValueError, SyntaxError):
        cleaned = text.replace("[", "").replace("]", "")
        tokens = [t.strip() for t in cleaned.split(",") if t.strip()]
        return np.asarray([float(t) for t in tokens], dtype=float)


def build_spike_cache(df, neuron_ids):
    df = df[df["Neuron_ID_3"].isin(neuron_ids)]
    cache = {}
    for neuron_id, ndf in df.groupby("Neuron_ID_3"):
        trial_map = {}
        for trial_id, g in ndf.groupby("trial_id"):
            spikes = parse_spike_entry_raw(g["Spikes"].iloc[0])
            start, stop = float(g["start_time"].iloc[0]), float(g["stop_time"].iloc[0])
            trial_map[int(trial_id)] = spikes[(spikes >= start) & (spikes <= stop)] - start
        cache[int(neuron_id)] = trial_map
    return cache


def count_coincidences(spikes_a, spikes_b, window=COINCIDENCE_WINDOW):
    if len(spikes_a) == 0 or len(spikes_b) == 0:
        return 0
    diffs = np.abs(spikes_a[:, None] - spikes_b[None, :])
    return int(np.sum(diffs <= window))


def per_trial_cfi_series(a_map, b_map, shared_trials, n_shifts, eps=EPS):
    n = len(shared_trials)
    cfis = np.empty(n)
    for i, trial in enumerate(shared_trials):
        observed = count_coincidences(a_map[trial], b_map[trial])
        shift_total = 0.0
        for s in range(1, n_shifts + 1):
            other_trial = shared_trials[(i + s) % n]
            shift_total += count_coincidences(a_map[trial], b_map[other_trial])
        expected = shift_total / n_shifts
        cfis[i] = np.log2((observed + eps) / (expected + eps))
    return cfis


neuron_subject_map = {}
for _df in (enc_df, delay_df):
    for sid, nid in _df[["subject_id", "Neuron_ID_3"]].drop_duplicates().itertuples(index=False):
        neuron_subject_map.setdefault(sid, set()).add(nid)


def choose_neurons(category, subject_id):
    return sorted(nid for nid in neurons_for(category)
                  if nid in neuron_subject_map.get(subject_id, set()))


def subjects_for_category(category):
    min_n = MIN_NEURONS.get(category, 3)
    eligible = []
    for subject_id in sorted(neuron_subject_map):
        if len(choose_neurons(category, subject_id)) >= min_n:
            eligible.append(int(subject_id))
    return eligible


for _cat in ALL_CATEGORIES:
    print(f"{_cat}: {len(subjects_for_category(_cat))} eligible subjects for CFI")

trial_info_rt_acc = pd.read_excel(BASE + "trial_info.xlsx")[
    ["subject_id", "trial_id", "RT", "response_accuracy"]].drop_duplicates()
trial_info_rt_acc["subject_id"] = trial_info_rt_acc["subject_id"].astype(int)
trial_info_rt_acc["trial_id"] = trial_info_rt_acc["trial_id"].astype(int)


def compute_cfi_dataset(pairs_by_subject, n_shifts, label):
    all_ids = sorted(set(nid for ids in pairs_by_subject.values() for nid in ids))
    enc_cache = build_spike_cache(enc_df, all_ids)
    delay_cache = build_spike_cache(delay_df, all_ids)
    rows = []
    n_pairs_total = sum(len(ids) * (len(ids) - 1) // 2 for ids in pairs_by_subject.values())
    print(f"  {label}: {n_pairs_total} pairs across {len(pairs_by_subject)} subjects")
    for subject_id, ids in pairs_by_subject.items():
        for a, b in combinations(sorted(ids), 2):
            a_enc, b_enc = enc_cache.get(a, {}), enc_cache.get(b, {})
            a_delay, b_delay = delay_cache.get(a, {}), delay_cache.get(b, {})
            shared = sorted(set(a_enc) & set(b_enc) & set(a_delay) & set(b_delay))
            if len(shared) < 2:
                continue
            cfi_enc_series = per_trial_cfi_series(a_enc, b_enc, shared, n_shifts)
            cfi_delay_series = per_trial_cfi_series(a_delay, b_delay, shared, n_shifts)
            for trial_id, ce, cd in zip(shared, cfi_enc_series, cfi_delay_series):
                rows.append((subject_id, trial_id, ce, cd))
    pair_trial_df = pd.DataFrame(rows, columns=["subject_id", "trial_id", "cfi_enc", "cfi_delay"])
    ts = pair_trial_df.groupby(["subject_id", "trial_id"]).agg(
        cfi_enc=("cfi_enc", "mean"), cfi_delay=("cfi_delay", "mean"),
        n_pairs=("cfi_enc", "size")).reset_index()
    ts["cfi_delta"] = ts["cfi_delay"] - ts["cfi_enc"]
    for col in ["cfi_enc", "cfi_delay", "cfi_delta"]:
        ts[f"{col}_z"] = ts.groupby("subject_id")[col].transform(lambda s: (s - s.mean()) / s.std(ddof=1))
    ts = ts.merge(trial_info_rt_acc, on=["subject_id", "trial_id"], how="left")
    print(f"  {label}: {len(ts)} trials, {ts['subject_id'].nunique()} subjects")
    return ts


cfi_datasets = {}
t0 = time.time()
for _category, _label in [("Labelled_IN", "IN (R2)"), ("Labelled_PY", "PY (R2)"), ("All_cells", "All cells (full-label)")]:
    _pairs_by_subject = {sid: choose_neurons(_category, sid) for sid in subjects_for_category(_category)}
    cfi_datasets[_category] = compute_cfi_dataset(_pairs_by_subject, N_SHIFTS_FULL_COHORT, _label)
print(f"CFI computed for all 3 categories in {time.time()-t0:.1f}s")


def load_cfi_file(cat):
    return cfi_datasets[cat]


trial_info = pd.read_excel(BASE + "trial_info.xlsx")[["subject_id", "trial_id", "num_images_presented"]]


def zwithin(df, col, out):
    df[out] = df.groupby("subject_id")[col].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0.0
    )
    return df

def fit_mixedlm_rs(data, outcome, focal, covariates):
    covs = " + ".join(covariates)
    formula = f"{outcome} ~ {focal}" + (f" + {covs}" if covs else "")
    md_ = smf.mixedlm(formula, data=data, groups=data["subject_id"], re_formula=f"~{focal}")
    return md_.fit(reml=True, method="lbfgs")


def perm_mixedlm(data, outcome, focal, covariates, obs_coef, n_perm=N_PERM_MIXED, seed=SEED):
    rng = np.random.default_rng(seed)
    d = data.copy()
    coefs = np.full(n_perm, np.nan)
    for i in range(n_perm):
        d[focal] = d.groupby("subject_id")[focal].transform(lambda x: rng.permutation(x.values))
        try:
            res = fit_mixedlm_rs(d, outcome, focal, covariates)
            coefs[i] = res.params[focal]
        except Exception:
            pass
    valid = ~np.isnan(coefs)
    n_valid = int(valid.sum())
    b = int(np.sum(np.abs(coefs[valid]) >= abs(obs_coef))) if n_valid > 0 else np.nan
    p = (b + 1) / (n_valid + 1) if n_valid > 0 else np.nan  # Phipson & Smyth (2010) bias-corrected permutation p
    return p, n_valid


def perm_gee_binomial(data, outcome, focal, covariates, obs_coef, n_perm=N_PERM_GEE, seed=SEED):
    rng = np.random.default_rng(seed)
    d = data.copy()
    covs = " + ".join(covariates)
    formula = f"{outcome} ~ {focal}" + (f" + {covs}" if covs else "")
    coefs = np.full(n_perm, np.nan)
    for i in range(n_perm):
        d[focal] = d.groupby("subject_id")[focal].transform(lambda x: rng.permutation(x.values))
        try:
            gee = smf.gee(formula, groups="subject_id", data=d, family=sm.families.Binomial()).fit()
            coefs[i] = gee.params[focal]
        except Exception:
            pass
    valid = ~np.isnan(coefs)
    n_valid = int(valid.sum())
    b = int(np.sum(np.abs(coefs[valid]) >= abs(obs_coef))) if n_valid > 0 else np.nan
    p = (b + 1) / (n_valid + 1) if n_valid > 0 else np.nan  # Phipson & Smyth (2010) bias-corrected permutation p
    return p, n_valid


print("Model helpers defined")

BURST_CFI_CONDITIONS = [
    ("Labelled_IN", "Labelled_IN", "IN bursts vs IN CFI"),
    ("Labelled_PY", "Labelled_PY", "PY bursts vs PY CFI"),
    ("All_cells", "All_cells", "All-cell bursts vs All-cell CFI"),
    ("All_cells", "Labelled_IN", "All-cell bursts vs IN CFI"),
    ("All_cells", "Labelled_PY", "All-cell bursts vs PY CFI"),
]

partA_rows = []
for burst_cat, cfi_cat, desc in BURST_CFI_CONDITIONS:
    for epoch in EPOCHS:
        cfi_col = EPOCH_CFI_COL[epoch]
        bdf = burst_full[(burst_full["category"] == burst_cat) & (burst_full["epoch"] == epoch)][
            ["subject_id", "trial_id", "burst_count"]
        ]
        fr = get_firing_rate(burst_cat, epoch)
        cfi = load_cfi_file(cfi_cat)[["subject_id", "trial_id", cfi_col]]

        merged = bdf.merge(fr, on=["subject_id", "trial_id"], how="inner").merge(
            cfi, on=["subject_id", "trial_id"], how="inner"
        )
        merged = zwithin(merged, "burst_count", "burst_count_z")
        merged = zwithin(merged, "firing_rate_hz", "firing_rate_z")
        merged = merged.dropna(subset=["burst_count_z", "firing_rate_z", cfi_col])

        res = fit_mixedlm_rs(merged, cfi_col, "burst_count_z", ["firing_rate_z"])
        obs_coef = res.params["burst_count_z"]
        wald_p = res.pvalues["burst_count_z"]
        perm_p, n_valid = perm_mixedlm(merged, cfi_col, "burst_count_z", ["firing_rate_z"], obs_coef, n_perm=5000)

        partA_rows.append(dict(
            condition=desc, burst_category=burst_cat, cfi_category=cfi_cat, epoch=epoch,
            n_trials=len(merged), n_subjects=merged["subject_id"].nunique(),
            coef=obs_coef, wald_p=wald_p, perm_p=perm_p, n_valid_perm=n_valid,
            firing_rate_coef=res.params["firing_rate_z"], firing_rate_wald_p=res.pvalues["firing_rate_z"],
        ))
        print(f"[A] {desc} | {epoch} | coef={obs_coef:.4f} perm_p={perm_p:.4f}", flush=True)

partA_df = pd.DataFrame(partA_rows).sort_values("perm_p").reset_index(drop=True)
partA_df.to_csv(OUT + "burst_vs_cfi_fulllabel_stats.csv", index=False)
display(partA_df)

CATS_B = ["Labelled_IN", "Labelled_PY", "All_cells"]

partB_rows = []
for cat in CATS_B:
    for epoch in EPOCHS:
        cfi_col = EPOCH_CFI_COL[epoch]
        cfi = load_cfi_file(cat)[["subject_id", "trial_id", cfi_col, "RT", "response_accuracy"]]
        merged = cfi.merge(trial_info, on=["subject_id", "trial_id"], how="inner")
        merged = merged[(merged["response_accuracy"] == 1) & (merged["RT"] > 0)].copy()
        merged["log_rt"] = np.log(merged["RT"])
        merged = zwithin(merged, "num_images_presented", "load_z")
        merged = merged.rename(columns={cfi_col: "cfi_z"})
        merged = merged.dropna(subset=["cfi_z", "load_z", "log_rt"])

        res = fit_mixedlm_rs(merged, "log_rt", "cfi_z", ["load_z"])
        obs_coef = res.params["cfi_z"]
        wald_p = res.pvalues["cfi_z"]
        perm_p, n_valid = perm_mixedlm(merged, "log_rt", "cfi_z", ["load_z"], obs_coef, n_perm=5000)

        partB_rows.append(dict(
            category=cat, epoch=epoch, n_trials=len(merged), n_subjects=merged["subject_id"].nunique(),
            coef=obs_coef, wald_p=wald_p, perm_p=perm_p, n_valid_perm=n_valid,
        ))
        print(f"[B-RT] {cat} | {epoch} | coef={obs_coef:.4f} perm_p={perm_p:.4f}", flush=True)

partB_df = pd.DataFrame(partB_rows).sort_values("perm_p").reset_index(drop=True)
partB_df.to_csv(OUT + "cfi_vs_rt_fulllabel_stats.csv", index=False)
display(partB_df)

partC_rows = []
for cat in CATS_B:
    for epoch in EPOCHS:
        cfi_col = EPOCH_CFI_COL[epoch]
        cfi = load_cfi_file(cat)[["subject_id", "trial_id", cfi_col, "response_accuracy"]]
        merged = cfi.merge(trial_info, on=["subject_id", "trial_id"], how="inner")
        merged = zwithin(merged, "num_images_presented", "load_z")
        merged = merged.rename(columns={cfi_col: "cfi_z"})
        merged = merged.dropna(subset=["cfi_z", "load_z", "response_accuracy"])

        gee = smf.gee(
            "response_accuracy ~ cfi_z + load_z", groups="subject_id",
            data=merged, family=sm.families.Binomial()
        ).fit()
        obs_coef = gee.params["cfi_z"]
        gee_p = gee.pvalues["cfi_z"]
        perm_p, n_valid = perm_gee_binomial(merged, "response_accuracy", "cfi_z", ["load_z"], obs_coef)

        partC_rows.append(dict(
            category=cat, epoch=epoch, n_trials=len(merged), n_subjects=merged["subject_id"].nunique(),
            coef=obs_coef, gee_p=gee_p, perm_p=perm_p, n_valid_perm=n_valid,
        ))
        print(f"[C-Acc] {cat} | {epoch} | coef={obs_coef:.4f} perm_p={perm_p:.4f}", flush=True)

partC_df = pd.DataFrame(partC_rows).sort_values("perm_p").reset_index(drop=True)
partC_df.to_csv(OUT + "cfi_vs_accuracy_fulllabel_stats.csv", index=False)
display(partC_df)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY_NS = "#c3c2b7"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.edgecolor": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": MUTED,
})


def stars(p):
    if pd.isna(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def plot_summary(df, conditions, title, ylabel, out_name, coef_col="coef", p_col="perm_p", figsize=(11, 5),
                  method_note="random-slope mixed model + within-subject permutation",
                  fr_note="firing rate partialled out"):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, epoch in zip(axes, ["encoding", "maintenance"]):
        ax.set_facecolor(SURFACE)
        sub = df[df["epoch"] == epoch]
        xs, heights, colors, labels, ps = [], [], [], [], []
        for i, (filt, label, color) in enumerate(conditions):
            row = sub
            for k, v in filt.items():
                row = row[row[k] == v]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            xs.append(i)
            heights.append(r[coef_col])
            p = r[p_col]
            colors.append(color if p < 0.05 else GRAY_NS)
            labels.append(label)
            ps.append(p)

        ax.bar(xs, heights, color=colors, width=0.62, edgecolor="none", zorder=3)
        ax.axhline(0, color=MUTED, linewidth=1, zorder=2)
        ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(epoch.capitalize(), fontsize=11, color=INK, pad=10)

        ymax, ymin = max(heights + [0]), min(heights + [0])
        pad = max(abs(ymax), abs(ymin)) * 0.35 + 0.005
        for x, h, p in zip(xs, heights, ps):
            offset = pad * 0.35 if h >= 0 else -pad * 0.35
            va = "bottom" if h >= 0 else "top"
            ax.text(x, h + offset, stars(p), ha="center", va=va, fontsize=10, color=INK, zorder=4)

    axes[0].set_ylabel(ylabel, fontsize=10, color=INK)
    all_heights = df[coef_col].tolist() + [0]
    ymax, ymin = max(all_heights), min(all_heights)
    pad = max(abs(ymax), abs(ymin)) * 0.5 + 0.01
    for ax in axes:
        ax.set_ylim(ymin - pad, ymax + pad)

    fig.suptitle(title, fontsize=13, color=INK, y=1.02)
    fig.text(0.5, -0.02,
              f"IN/PY: R2-filtered; All cells: full/unfiltered cohort; {fr_note}; {method_note} "
              f"(bars: * p<0.05, ** p<0.01, *** p<0.001, gray = ns)",
              ha="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT + out_name, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    eps_name = out_name.rsplit(".", 1)[0] + ".eps"
    fig.savefig(OUT + eps_name, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {OUT + out_name} and {OUT + eps_name}")
    return fig


condsA = [
    ({"condition": "IN bursts vs IN CFI"}, "IN bursts\nvs IN CFI", BLUE),
    ({"condition": "PY bursts vs PY CFI"}, "PY bursts\nvs PY CFI", ORANGE),
    ({"condition": "All-cell bursts vs All-cell CFI"}, "All bursts\nvs All CFI", AQUA),
    ({"condition": "All-cell bursts vs IN CFI"}, "All bursts\nvs IN CFI", YELLOW),
    ({"condition": "All-cell bursts vs PY CFI"}, "All bursts\nvs PY CFI", MAGENTA),
]
fig_a = plot_summary(partA_df, condsA,
             "Burst count vs. CFI (IN/PY: R2-filtered; All cells: full-label; firing-rate controlled)",
             "Mixed-model coefficient (burst_count_z on CFI_z)",
             "burst_vs_cfi_fulllabel_summary.png", figsize=(12, 5))
plt.show()

condsBCD = [
    ({"category": "Labelled_IN"}, "IN CFI\n(R2)", BLUE),
    ({"category": "Labelled_PY"}, "PY CFI\n(R2)", ORANGE),
    ({"category": "All_cells"}, "All-cell CFI\n(full-label)", AQUA),
]
fig_b = plot_summary(partB_df, condsBCD,
             "CFI vs. RT (log); IN/PY: R2-filtered, All cells: full-label; all subjects",
             "Mixed-model coefficient (CFI_z on log RT)",
             "cfi_vs_rt_fulllabel_summary.png",
             fr_note="firing rate NOT controlled (CFI is already shift-corrected; see text)")
plt.show()

fig_c = plot_summary(partC_df, condsBCD,
             "CFI vs. accuracy; IN/PY: R2-filtered, All cells: full-label; all subjects",
             "GEE coefficient (CFI_z on accuracy, logit)",
             "cfi_vs_accuracy_fulllabel_summary.png", p_col="perm_p",
             method_note="GEE (Independence, cluster=subject) + within-subject permutation",
             fr_note="firing rate NOT controlled (CFI is already shift-corrected; see text)")
plt.show()
