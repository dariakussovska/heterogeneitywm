import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import ast
import warnings

warnings.filterwarnings("ignore")
bin_size = 0.03
sigma = 0.05
prominence_threshold_percentile = 90

duration = {
    "Fixation": 1,
    "Encoding 1": 1,
    "Encoding 2": 1,
    "Encoding 3": 1,
    "Delay": 3,
   # "Probe": 1
}

# Per-subject low/high split knobs for ACG/Decay
PER_SUBJ = 10
OVERLAP = 2

# Output files
stats_excel_path = "/Users/darikussovska/Desktop/burst_statistics_results.xlsx"
plot_prefix = "/Users/darikussovska/Desktop"

# =========================
# Load data
# =========================
df_metadata = pd.read_excel("/Users/darikussovska/Desktop/revision_clustering_no_waveform_labels.xlsx")
df_metadata2 = pd.read_excel("/Users/darikussovska/Desktop/PROJECT/Neuron_Check_ANOVA_All.xlsx")

df_fixation = pd.read_excel("/Users/darikussovska/Desktop/PROJECT/new_clean_data/cleaned_Fixation.xlsx")
df_enc1 = pd.read_excel("/Users/darikussovska/Desktop/PROJECT/graph_encoding1.xlsx")
df_enc2 = pd.read_excel("/Users/darikussovska/Desktop/PROJECT/graph_encoding2.xlsx")
df_enc3 = pd.read_excel("/Users/darikussovska/Desktop/PROJECT/graph_encoding3.xlsx")
df_delay = pd.read_excel("/Users/darikussovska/Desktop/PROJECT/graph_delay.xlsx")
#df_probe = pd.read_excel("/Users/darikussovska/Desktop/PROJECT/graph_probe.xlsx")

encoding_map = {
    1: ("Encoding 1", df_enc1),
    2: ("Encoding 2", df_enc2),
    3: ("Encoding 3", df_enc3)
}

period_data = {
    "Fixation": df_fixation,
    "Delay": df_delay,
  #  "Probe": df_probe
}

categories = [
    "All_cells",
    "5_lowest_ACG",
    "5_highest_ACG",
    "5_lowest_decay",
    "5_highest_decay",
    "Concept_cells",
    "Interneurons",
    "Pyramidal",
]

periods = ["Fixation", "Encoding", "Delay_part1", "Delay_part2", "Delay_part3"]

# Store subject_id together with value
burst_counts = {
    cat: {
        period: {load: [] for load in [1, 2, 3]}
        for period in periods
    }
    for cat in categories
}

# =========================
# Helper functions
# =========================
def parse_spike_list(x, trial_dur):
    """Safely parse a stringified list of spikes; clip to [0, trial_dur]."""
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
        arr = np.array([float(arr)], dtype=float)

    return arr[(arr >= 0.0) & (arr <= float(trial_dur))]


def split_low_high(df_subj, feature, per_subj=PER_SUBJ, overlap=OVERLAP):
    """
    Deterministically split one subject's neurons into low/high sets by `feature`.
    Requires at least `per_subj` rows after dropping NaNs in feature.
    """
    if df_subj.empty or feature not in df_subj.columns:
        return [], []

    df = df_subj[["Neuron_ID_3", feature]].dropna(subset=[feature]).copy()
    if len(df) < per_subj:
        return [], []

    df_sorted = df.sort_values(
        [feature, "Neuron_ID_3"],
        ascending=[True, True],
        kind="mergesort"
    )

    m = (per_subj // 2) + overlap
    low_ids = df_sorted.head(m)["Neuron_ID_3"].tolist()
    high_ids = df_sorted.tail(m)["Neuron_ID_3"].tolist()

    return low_ids, high_ids


def symbol_from_p(p_corr):
    if pd.isna(p_corr):
        return ""
    if p_corr < 0.005:
        return "***"
    if p_corr < 0.01:
        return "**"
    if p_corr < 0.05:
        return "*"
    return ""


def add_hash(ax, text, x1, x2, base_y, height=0.6, text_pad=0.1, lw=1.5, color="k"):
    ax.plot(
        [x1, x1, x2, x2],
        [base_y, base_y + height, base_y + height, base_y],
        lw=lw,
        c=color,
        clip_on=False
    )
    ax.text(
        (x1 + x2) / 2,
        base_y + height + text_pad,
        text,
        ha="center",
        va="bottom",
        fontsize=12,
        color=color
    )


# =========================
# Determine eligible subjects per category
# =========================
subject_ids_per_category = {}

for category in categories:
    if category == "Concept_cells":
        subject_counts = df_metadata2[df_metadata2["Signi"] == "Y"]["subject_id"].value_counts()
        threshold = 3

    elif category in ["Pyramidal", "Interneurons"]:
        label = "PY" if category == "Pyramidal" else "IN"
        subject_counts = df_metadata[df_metadata["Cell_Type_New"] == label]["subject_id"].value_counts()
        threshold = 5

    elif category == "All_cells":
        subject_counts = df_metadata["subject_id"].value_counts()
        threshold = 10

    elif category in ["5_lowest_ACG", "5_highest_ACG"]:
        eligible = df_metadata[df_metadata["R2"] > 0.3].dropna(subset=["Mean_ACG"])
        subject_counts = eligible["subject_id"].value_counts()
        threshold = PER_SUBJ

    else:  # 5_lowest_decay / 5_highest_decay
        eligible = df_metadata[df_metadata["R2"] > 0.3].dropna(subset=["Decay"])
        subject_counts = eligible["subject_id"].value_counts()
        threshold = PER_SUBJ

    subject_ids_per_category[category] = subject_counts[subject_counts >= threshold].index.tolist()

# =========================
# Compute burst counts
# =========================
for group in categories:
    for subject_id in subject_ids_per_category[group]:
        df_meta = df_metadata[df_metadata["subject_id"] == subject_id].copy()
        df_meta_r2 = df_meta[df_meta["R2"] > 0.3].copy()

        low_acg, high_acg = split_low_high(df_meta_r2, "Mean_ACG", per_subj=PER_SUBJ, overlap=OVERLAP)
        low_dec, high_dec = split_low_high(df_meta_r2, "Decay", per_subj=PER_SUBJ, overlap=OVERLAP)

        concept_ids = df_metadata2[
            (df_metadata2["subject_id"] == subject_id) &
            (df_metadata2["Signi"] == "Y")
        ]["Neuron_ID_3"].tolist()

        neuron_groups = {
            "5_lowest_ACG": low_acg,
            "5_highest_ACG": high_acg,
            "5_lowest_decay": low_dec,
            "5_highest_decay": high_dec,
            "Interneurons": df_meta[df_meta["Cell_Type_New"] == "IN"]["Neuron_ID_3"].tolist(),
            "Pyramidal": df_meta[df_meta["Cell_Type_New"] == "PY"]["Neuron_ID_3"].tolist(),
            "All_cells": df_meta["Neuron_ID_3"].tolist(),
            "Concept_cells": concept_ids
        }

        neurons = neuron_groups.get(group, [])
        if len(neurons) == 0:
            continue

        for load in [1, 2, 3]:
            enc_label, df_enc = encoding_map[load]
            df_enc_sub = df_enc[
                (df_enc["subject_id"] == subject_id) &
                (df_enc["num_images_presented"] == load)
            ]

            for period_label in ["Fixation", "Encoding", "Delay"]:
                if period_label == "Encoding":
                    df_sub = df_enc_sub
                    trial_duration = duration[enc_label]
                else:
                    dfP = period_data[period_label]

                    if period_label == "Fixation":
                        if "num_images_presented" in dfP.columns:
                            df_sub = dfP[
                                (dfP["subject_id"] == subject_id) &
                                (dfP["num_images_presented"] == load)
                            ]
                        else:
                            df_sub = dfP[dfP["subject_id"] == subject_id]
                    else:
                        df_sub = dfP[
                            (dfP["subject_id"] == subject_id) &
                            (dfP["num_images_presented"] == load)
                        ]

                    trial_duration = duration[period_label]

                if df_sub.empty:
                    continue

                df_group = df_sub[df_sub["Neuron_ID_3"].isin(neurons)]
                if df_group.empty:
                    continue

                trial_ids = sorted(df_group["trial_id"].unique())
                if len(trial_ids) == 0:
                    continue

                all_spikes = []
                for trial_idx, trial_id in enumerate(trial_ids):
                    df_trial = df_group[df_group["trial_id"] == trial_id]

                    col = "Standardized_Spikes"
                    if col not in df_trial.columns:
                        raise KeyError(
                            f"Column '{col}' not found in {period_label}. "
                            f"Available columns: {df_trial.columns.tolist()}"
                        )

                    for neuron_id in sorted(df_trial["Neuron_ID_3"].unique()):
                        series = df_trial[df_trial["Neuron_ID_3"] == neuron_id][col].dropna()
                        for s in series:
                            arr = parse_spike_list(s, trial_duration)
                            if arr.size:
                                all_spikes.extend(arr + trial_idx * trial_duration)

                if not all_spikes:
                    continue

                total_dur = len(trial_ids) * trial_duration
                time_bins = np.arange(0.0, total_dur + bin_size, bin_size)
                counts, _ = np.histogram(all_spikes, bins=time_bins)

                std_bins = sigma / bin_size
                std_bins = max(1.0, std_bins)

                kernel_width = int(np.ceil(5 * std_bins))
                kernel_width = max(kernel_width, 3)
                if kernel_width % 2 == 0:
                    kernel_width += 1

                kernel = gaussian(kernel_width, std=std_bins)
                kernel /= kernel.sum()

                smooth_rate = np.convolve(counts, kernel, mode="same")

                threshold = np.percentile(smooth_rate, prominence_threshold_percentile)
                peaks, _ = signal.find_peaks(smooth_rate, prominence=threshold)

                if period_label == "Delay":
                    thirds = np.array_split(np.arange(len(smooth_rate)), 3)
                    for i, part in enumerate(thirds):
                        count = int(np.sum(np.isin(peaks, part)))
                        burst_counts[group][f"Delay_part{i+1}"][load].append({
                            "subject_id": subject_id,
                            "burst_count": count
                        })
                else:
                    burst_counts[group][period_label][load].append({
                        "subject_id": subject_id,
                        "burst_count": int(len(peaks))
                    })

plot_data = []
for load in [1, 2, 3]:
    for group in categories:
        for period in periods:
            for item in burst_counts[group][period][load]:
                plot_data.append({
                    "Load": load,
                    "Cell_Type": group,
                    "Period": period,
                    "subject_id": item["subject_id"],
                    "Burst_Count": item["burst_count"]
                })

df_plot_all = pd.DataFrame(plot_data)

# =========================
# Summary table
# =========================
summary_rows = []
for load in [1, 2, 3]:
    for group in categories:
        for period in periods:
            subset = df_plot_all[
                (df_plot_all["Load"] == load) &
                (df_plot_all["Cell_Type"] == group) &
                (df_plot_all["Period"] == period)
            ].copy()

            if subset.empty:
                summary_rows.append({
                    "Load": load,
                    "Cell_Type": group,
                    "Period": period,
                    "n_subjects": 0,
                    "mean_burst_count": np.nan,
                    "median_burst_count": np.nan,
                    "std_burst_count": np.nan,
                    "min_burst_count": np.nan,
                    "max_burst_count": np.nan
                })
            else:
                summary_rows.append({
                    "Load": load,
                    "Cell_Type": group,
                    "Period": period,
                    "n_subjects": subset["subject_id"].nunique(),
                    "mean_burst_count": subset["Burst_Count"].mean(),
                    "median_burst_count": subset["Burst_Count"].median(),
                    "std_burst_count": subset["Burst_Count"].std(),
                    "min_burst_count": subset["Burst_Count"].min(),
                    "max_burst_count": subset["Burst_Count"].max()
                })

df_summary = pd.DataFrame(summary_rows)

# =========================
# Statistical comparisons
# =========================
stats_results = []

for load in [1, 2, 3]:
    df_plot = df_plot_all[df_plot_all["Load"] == load].copy()

    plt.figure(figsize=(20, 10))

    for i, group in enumerate(categories, 1):
        ax = plt.subplot(2, 4, i)
        data = df_plot[df_plot["Cell_Type"] == group].copy()

        sns.boxplot(
            x="Period",
            y="Burst_Count",
            data=data,
            palette="Set2",
            showfliers=False,
            ax=ax
        )
        sns.stripplot(
            x="Period",
            y="Burst_Count",
            data=data,
            color="black",
            size=4,
            jitter=True,
            dodge=True,
            alpha=0.6,
            ax=ax
        )

        ax.set_title(f"{group} (Load {load})", fontsize=12)
        ax.set_ylim(0, 50)
        ax.set_xlabel("")
        ax.set_ylabel("")

        # collect all valid pairwise tests first
        group_test_rows = []

        for j in range(len(periods)):
            for k in range(j + 1, len(periods)):
                p1 = periods[j]
                p2 = periods[k]

                pair_df = data[data["Period"].isin([p1, p2])].copy()
                if pair_df.empty:
                    continue

                pivot = pair_df.pivot_table(
                    index="subject_id",
                    columns="Period",
                    values="Burst_Count",
                    aggfunc="first"
                )

                if p1 not in pivot.columns or p2 not in pivot.columns:
                    continue

                paired = pivot[[p1, p2]].dropna()
                n_pairs = len(paired)

                if n_pairs < 2:
                    continue

                x = paired[p1].values
                y = paired[p2].values

                # store some descriptive info even if test fails
                row = {
                    "Load": load,
                    "Cell_Type": group,
                    "Period_1": p1,
                    "Period_2": p2,
                    "n_pairs": n_pairs,
                    "mean_period1": np.mean(x),
                    "mean_period2": np.mean(y),
                    "median_period1": np.median(x),
                    "median_period2": np.median(y),
                    "wilcoxon_stat": np.nan,
                    "raw_p": np.nan,
                    "p_corrected_fdr_bh": np.nan,
                    "significance_symbol": "",
                    "test_used": "wilcoxon",
                    "correction_method": "fdr_bh"
                }

                try:
                    stat = wilcoxon(
                        x,
                        y,
                        alternative="two-sided",
                        zero_method="wilcox"
                    )
                    row["wilcoxon_stat"] = stat.statistic
                    row["raw_p"] = stat.pvalue
                except Exception:
                    pass

                group_test_rows.append(row)

        # FDR correction within each group x load block
        valid_idx = [idx for idx, row in enumerate(group_test_rows) if pd.notna(row["raw_p"])]
        if len(valid_idx) > 0:
            raw_ps = [group_test_rows[idx]["raw_p"] for idx in valid_idx]
            _, corr_ps, _, _ = multipletests(raw_ps, alpha=0.05, method="fdr_bh")

            for idx, p_corr in zip(valid_idx, corr_ps):
                group_test_rows[idx]["p_corrected_fdr_bh"] = p_corr
                group_test_rows[idx]["significance_symbol"] = symbol_from_p(p_corr)

        # append all rows to master stats table
        stats_results.extend(group_test_rows)

        # plot only significant corrected comparisons
        sig_rows = [row for row in group_test_rows if row["significance_symbol"] != ""]

        if len(sig_rows) > 0 and not data.empty:
            ymax = data["Burst_Count"].max()
            base = ymax + 1.0
            step = 0.8
            height = 0.45

            for idx, row in enumerate(sig_rows):
                j = periods.index(row["Period_1"])
                k = periods.index(row["Period_2"])
                y = base + idx * step
                add_hash(ax, row["significance_symbol"], j, k, y, height=height)

    plt.suptitle(
        f"Burst Count Across Periods (Delay in 3 Parts) - Load {load}",
        fontsize=20,
        y=1.02
    )
    plt.tight_layout()
    plt.savefig("/Users/darikussovska/Desktop/burst_plot_load_{}.eps".format(load),
            format="eps", dpi=300)
    plt.show()

df_stats = pd.DataFrame(stats_results)

# =========================
# Save all results to Excel
# =========================
with pd.ExcelWriter(stats_excel_path, engine="openpyxl") as writer:
    df_plot_all.to_excel(writer, sheet_name="burst_counts_long", index=False)
    df_summary.to_excel(writer, sheet_name="summary_by_group_period", index=False)
    df_stats.to_excel(writer, sheet_name="pairwise_stats", index=False)

print(f"Saved statistics Excel file to: {stats_excel_path}")
