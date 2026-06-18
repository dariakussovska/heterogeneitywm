import pandas as pd
import numpy as np
from ast import literal_eval
import os
import matplotlib.pyplot as plt

input_path = "../clean_data/cleaned_Encoding1.xlsx"

START_TIME = 0.2
END_TIME = 1.0
BOOTSTRAP_ITER = 1000
N_SHUFFLES = 1000
ALPHA = 0.05

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

df = pd.read_excel(input_path)
df.columns = df.columns.astype(str).str.strip()

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


def extract_top_two_categories(df, label_col="stimulus_index"):

    results = []

    for subj in df["subject_id"].dropna().unique():

        df_subj = df[df["subject_id"] == subj]

        for neuron in df_subj["Neuron_ID"].dropna().unique():

            df_neuron = df_subj[df_subj["Neuron_ID"] == neuron]

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
                "im_cat_1st": first["image"],
                "mean_1st": first["mean_rate"],
                "cat_1st": first["rates"],
                "im_cat_2nd": second["image"],
                "mean_2nd": second["mean_rate"],
                "cat_2nd": second["rates"]
            })

    return pd.DataFrame(results)


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

df_top_2_real = extract_top_two_categories(df, label_col="stimulus_index")
df_final_real = run_significance_test(df_top_2_real, iteration=BOOTSTRAP_ITER)

n_real_significant = int((df_final_real["Signi"] == "Y").sum())

df_final_real.to_excel(
    "./Neuron_Check_Significant_All_NEW.xlsx",
    index=False
)

print("REAL DATA")
print(f"Significant neurons: {n_real_significant}")

def shuffle_labels_within_each_neuron(df, label_col="stimulus_index"):

    df_shuff = df.copy()
    shuffled_labels = pd.Series(index=df_shuff.index, dtype=object)

    for (subj, neuron), idx in df_shuff.groupby(["subject_id", "Neuron_ID"]).groups.items():

        idx = list(idx)
        labels = df_shuff.loc[idx, label_col].values.copy()

        shuffled_labels.loc[idx] = rng.permutation(labels)

    df_shuff["stimulus_index_shuffled"] = shuffled_labels

    return df_shuff

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

    n_significant = int((df_final_shuff["Signi"] == "Y").sum())

    shuffle_rows.append({
        "shuffle_i": shuffle_i + 1,
        "n_significant": n_significant,
        "n_neurons_tested": len(df_final_shuff)
    })

df_shuffle_counts = pd.DataFrame(shuffle_rows)

summary = pd.DataFrame([{
    "n_shuffles": N_SHUFFLES,
    "bootstrap_iterations_per_neuron": BOOTSTRAP_ITER,
    "alpha_uncorrected": ALPHA,
    "real_significant": n_real_significant,
    "mean_significant_under_shuffle": df_shuffle_counts["n_significant"].mean(),
    "std_significant_under_shuffle": df_shuffle_counts["n_significant"].std(),
    "median_significant_under_shuffle": df_shuffle_counts["n_significant"].median(),
    "min_significant_under_shuffle": df_shuffle_counts["n_significant"].min(),
    "max_significant_under_shuffle": df_shuffle_counts["n_significant"].max(),
    "empirical_p_real_ge_shuffle": (
        (np.sum(df_shuffle_counts["n_significant"] >= n_real_significant) + 1)
        / (N_SHUFFLES + 1)
    )
}])

with pd.ExcelWriter("./Neuron_Check_Shuffled_Label_Null_1000.xlsx", engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="summary", index=False)
    df_shuffle_counts.to_excel(writer, sheet_name="shuffle_counts", index=False)
    df_final_real.to_excel(writer, sheet_name="real_neurons", index=False)

print("\nSHUFFLE NULL SUMMARY")
print(summary.T)

plt.figure(figsize=(7, 5))

plt.hist(
    df_shuffle_counts["n_significant"],
    bins=15,
    alpha=0.8,
    edgecolor="black"
)

plt.axvline(
    n_real_significant,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Real count =58"
)

plt.xlabel("Number of significant neurons under shuffled labels")
plt.ylabel("Shuffle count")
plt.title("Null distribution without FDR")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./shuffled_label_null_distribution.eps", format="eps", dpi=300)
plt.show()
