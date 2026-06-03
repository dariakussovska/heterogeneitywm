import pandas as pd
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------
# Parameters
# ---------------------------------------------------

input_path = "/Users/darikussovska/Desktop/PROJECT/new_clean_data/cleaned_Encoding1.xlsx"
region_path = "/Users/darikussovska/Desktop/merged_significant_neurons_with_brain_regions.xlsx"

START_TIME = 0.2
END_TIME = 1.0
BOOTSTRAP_ITER = 1000
N_SHUFFLES = 1000
ALPHA = 0.05

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------
# Load data
# ---------------------------------------------------

df = pd.read_excel(input_path)
regions = pd.read_excel(region_path)

df.columns = df.columns.astype(str).str.strip()
regions.columns = regions.columns.astype(str).str.strip()

# ---------------------------------------------------
# Detect region column
# ---------------------------------------------------

possible_region_cols = [
    "Location"
]

region_col = None

for col in possible_region_cols:
    if col in regions.columns:
        region_col = col
        break

if region_col is None:
    raise KeyError(
        f"No brain-region column found. Available columns:\n{regions.columns.tolist()}"
    )

print(f"Using brain-region column: {region_col}")

# ---------------------------------------------------
# Standardize neuron ID columns
# ---------------------------------------------------

if "Neuron_ID_3" not in regions.columns:
    raise KeyError(
        f"'Neuron_ID_3' not found in region file. Available columns:\n{regions.columns.tolist()}"
    )

if "Neuron_ID_3" not in df.columns:
    if "Neuron_ID" in df.columns:
        df["Neuron_ID_3"] = df["Neuron_ID"]
    else:
        raise KeyError(
            f"Neither 'Neuron_ID_3' nor 'Neuron_ID' found in encoding data. "
            f"Available columns:\n{df.columns.tolist()}"
        )

if "Neuron_ID" not in df.columns:
    df["Neuron_ID"] = df["Neuron_ID_3"]

# ---------------------------------------------------
# Merge brain regions
# ---------------------------------------------------

region_keep = (
    regions[["subject_id", "Neuron_ID_3", region_col]]
    .drop_duplicates()
    .rename(columns={region_col: "Location"})
)

df = df.merge(
    region_keep,
    on=["subject_id", "Neuron_ID_3"],
    how="left"
)

df = df.dropna(subset=["Location"]).copy()

print("Region counts before analysis:")
print(df[["subject_id", "Neuron_ID_3", "Location"]].drop_duplicates()["Location"].value_counts())

# ---------------------------------------------------
# Keep top 5 regions by neuron count
# ---------------------------------------------------

top_regions = (
    df[["subject_id", "Neuron_ID_3", "Location"]]
    .drop_duplicates()["Location"]
    .value_counts()
    .head(5)
    .index
    .tolist()
)

print("\nTop 5 regions:")
print(top_regions)

df = df[df["Location"].isin(top_regions)].copy()

# ---------------------------------------------------
# Firing rate
# ---------------------------------------------------

def safe_parse_spikes(spikes):
    if pd.isna(spikes):
        return []

    if isinstance(spikes, str):
        spikes = spikes.strip()
        if spikes == "" or spikes == "[]":
            return []
        try:
            parsed = literal_eval(spikes)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    if isinstance(spikes, (list, tuple, np.ndarray)):
        return list(spikes)

    return []


def compute_firing_rate(df, start_time=0.2, end_time=1.0):
    df = df.copy()
    spike_rates = []

    for _, row in df.iterrows():
        spike_times = safe_parse_spikes(row["Standardized_Spikes"])

        spikes_in_window = [
            s for s in spike_times
            if start_time <= s <= end_time
        ]

        firing_rate = len(spikes_in_window) / (end_time - start_time)
        spike_rates.append(firing_rate)

    df["Spike_Rate_new"] = spike_rates
    return df


df = compute_firing_rate(df, START_TIME, END_TIME)

# ---------------------------------------------------
# Top two categories
# ---------------------------------------------------

def extract_top_two_categories(df, label_col="stimulus_index"):

    results = []

    for subj in df["subject_id"].dropna().unique():

        df_subj = df[df["subject_id"] == subj]

        for neuron in df_subj["Neuron_ID"].dropna().unique():

            df_neuron = df_subj[df_subj["Neuron_ID"] == neuron]

            if df_neuron.empty:
                continue

            brain_region = df_neuron["Location"].iloc[0]

            category_rows = []

            for image in df_neuron[label_col].dropna().unique():

                rates = df_neuron.loc[
                    df_neuron[label_col] == image,
                    "Spike_Rate_new"
                ].dropna()

                if len(rates) == 0:
                    continue

                category_rows.append({
                    "image": image,
                    "mean_rate": rates.mean(),
                    "rates": rates.tolist()
                })

            if len(category_rows) < 2:
                continue

            category_rows = sorted(
                category_rows,
                key=lambda x: x["mean_rate"],
                reverse=True
            )

            first = category_rows[0]
            second = category_rows[1]

            results.append({
                "subject_id": subj,
                "Neuron_ID": neuron,
                "Neuron_ID_3": df_neuron["Neuron_ID_3"].iloc[0],
                "Location": brain_region,
                "im_cat_1st": first["image"],
                "mean_1st": first["mean_rate"],
                "cat_1st": first["rates"],
                "im_cat_2nd": second["image"],
                "mean_2nd": second["mean_rate"],
                "cat_2nd": second["rates"]
            })

    return pd.DataFrame(results)

# ---------------------------------------------------
# Bootstrap significance test — no FDR
# ---------------------------------------------------

def resampling(arr1, arr2, iteration=1000):
    min_len = min(len(arr1), len(arr2))

    if min_len < 2:
        return np.nan, [np.nan, np.nan]

    diffs = []

    for _ in range(iteration):
        sample1 = rng.choice(arr1, min_len, replace=True)
        sample2 = rng.choice(arr2, min_len, replace=True)
        diffs.append(np.mean(sample1) - np.mean(sample2))

    CI = np.percentile(diffs, [2.5, 97.5])

    m1 = 2 * sum(i > 0 for i in diffs) / iteration
    m2 = 1 - sum(i < 0 for i in diffs) / iteration
    m3 = 2 * sum(i < 0 for i in diffs) / iteration
    m4 = 1 - sum(i > 0 for i in diffs) / iteration

    p_val = min(m1, m2, m3, m4)

    return p_val, CI


def run_significance_test(df_top_2, iteration=1000):

    df_top_2 = df_top_2.copy()

    p_vals = []
    CIs = []
    signs = []

    for _, row in df_top_2.iterrows():

        arr1 = [float(i) for i in row["cat_1st"]]
        arr2 = [float(i) for i in row["cat_2nd"]]

        p, ci = resampling(arr1, arr2, iteration)

        p_vals.append(p)
        CIs.append(ci)

        signs.append(
            "Y" if pd.notna(p) and p < ALPHA and len(arr1) > 1 and len(arr2) > 1 else "N"
        )

    df_top_2["p_val"] = p_vals
    df_top_2["CI"] = CIs
    df_top_2["Signi"] = signs

    return df_top_2

# ---------------------------------------------------
# Real data, by region
# ---------------------------------------------------

df_top_2_real = extract_top_two_categories(df, label_col="stimulus_index")
df_final_real = run_significance_test(df_top_2_real, iteration=BOOTSTRAP_ITER)

real_region_counts = (
    df_final_real
    .groupby("Location")
    .agg(
        n_neurons_tested=("Neuron_ID_3", "nunique"),
        n_significant_real=("Signi", lambda x: int((x == "Y").sum()))
    )
    .reset_index()
)

print("\nReal significant neurons by region:")
print(real_region_counts)

# ---------------------------------------------------
# Shuffle labels within each neuron
# ---------------------------------------------------

def shuffle_labels_within_each_neuron(df, label_col="stimulus_index"):

    df_shuff = df.copy()
    shuffled_labels = pd.Series(index=df_shuff.index, dtype=object)

    for (subj, neuron), idx in df_shuff.groupby(["subject_id", "Neuron_ID_3"]).groups.items():

        idx = list(idx)
        labels = df_shuff.loc[idx, label_col].values.copy()
        shuffled_labels.loc[idx] = rng.permutation(labels)

    df_shuff["stimulus_index_shuffled"] = shuffled_labels

    return df_shuff

# ---------------------------------------------------
# Region-wise shuffled null
# ---------------------------------------------------

shuffle_rows = []

for shuffle_i in range(N_SHUFFLES):

    print(f"Running shuffle {shuffle_i + 1}/{N_SHUFFLES}")

    df_shuff = shuffle_labels_within_each_neuron(
        df,
        label_col="stimulus_index"
    )

    df_top_2_shuff = extract_top_two_categories(
        df_shuff,
        label_col="stimulus_index_shuffled"
    )

    df_final_shuff = run_significance_test(
        df_top_2_shuff,
        iteration=BOOTSTRAP_ITER
    )

    region_counts = (
        df_final_shuff
        .groupby("Location")
        .agg(
            n_neurons_tested=("Neuron_ID_3", "nunique"),
            n_significant_shuffle=("Signi", lambda x: int((x == "Y").sum()))
        )
        .reset_index()
    )

    region_counts["shuffle_i"] = shuffle_i + 1

    shuffle_rows.append(region_counts)

df_shuffle_region_counts = pd.concat(shuffle_rows, ignore_index=True)

# ---------------------------------------------------
# Summary
# ---------------------------------------------------

summary_rows = []

for region in top_regions:

    real_count = real_region_counts.loc[
        real_region_counts["Location"] == region,
        "n_significant_real"
    ]

    real_count = int(real_count.iloc[0]) if len(real_count) else 0

    shuff_counts = df_shuffle_region_counts.loc[
        df_shuffle_region_counts["Location"] == region,
        "n_significant_shuffle"
    ]

    summary_rows.append({
        "Location": region,
        "real_significant_no_FDR": real_count,
        "mean_significant_under_shuffle_no_FDR": shuff_counts.mean(),
        "std_significant_under_shuffle_no_FDR": shuff_counts.std(),
        "median_significant_under_shuffle_no_FDR": shuff_counts.median(),
        "min_significant_under_shuffle_no_FDR": shuff_counts.min(),
        "max_significant_under_shuffle_no_FDR": shuff_counts.max(),
        "empirical_p_real_ge_shuffle": (
            (np.sum(shuff_counts >= real_count) + 1) / (N_SHUFFLES + 1)
            if len(shuff_counts) > 0 else np.nan
        )
    })

summary_by_region = pd.DataFrame(summary_rows)

print("\nRegion-wise shuffle null summary:")
print(summary_by_region)

# ---------------------------------------------------
# Save outputs
# ---------------------------------------------------

with pd.ExcelWriter("/Users/darikussovska/Desktop/PROJECT/Neuron_Check_Shuffled_Label_Null_by_Region_1000.xlsx", engine="openpyxl") as writer:
    summary_by_region.to_excel(writer, sheet_name="summary_by_region", index=False)
    df_shuffle_region_counts.to_excel(writer, sheet_name="shuffle_counts_by_region", index=False)
    df_final_real.to_excel(writer, sheet_name="real_neurons_by_region", index=False)
    real_region_counts.to_excel(writer, sheet_name="real_region_counts", index=False)

df_final_real.to_excel(
    "/Users/darikussovska/Desktop/PROJECT/Neuron_Check_Significant_All_REAL_by_Region_1000.xlsx",
    index=False
)

# ---------------------------------------------------
# Plot 5 region distributions
# ---------------------------------------------------

fig, axes = plt.subplots(
    nrows=1,
    ncols=len(top_regions),
    figsize=(5 * len(top_regions), 4),
    sharey=True
)

if len(top_regions) == 1:
    axes = [axes]

for ax, region in zip(axes, top_regions):

    shuff_counts = df_shuffle_region_counts.loc[
        df_shuffle_region_counts["Location"] == region,
        "n_significant_shuffle"
    ]

    real_count = summary_by_region.loc[
        summary_by_region["Location"] == region,
        "real_significant_no_FDR"
    ].iloc[0]

    ax.hist(
        shuff_counts,
        bins=15,
        alpha=0.8,
        edgecolor="black"
    )

    ax.axvline(
        real_count,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Real = {real_count}"
    )

    ax.set_title(region)
    ax.set_xlabel("Significant neurons")
    ax.legend(frameon=False)

axes[0].set_ylabel("Shuffle count")

plt.suptitle(
    "Null distributions by brain region\nlabels shuffled within each neuron, no FDR",
    fontsize=14
)

plt.tight_layout()

plt.savefig("/Users/darikussovska/Desktop/PROJECT/shuffled_label_null_distribution_by_region_000.png", dpi=300)
plt.savefig("/Users/darikussovska/Desktop/PROJECT/shuffled_label_null_distribution_by_region_000.eps", format="eps", dpi=300)
plt.show()

print("\nSaved region-wise shuffle-null outputs.")
