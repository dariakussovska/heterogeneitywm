import os
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from scipy.stats import wilcoxon
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore")

random_seed = 42
rng = np.random.default_rng(random_seed)

n_simulations = 500
n_trials_per_group = 45
n_neurons = 50
firing_rate_hz = 3.0
bin_size = 0.07
sigma = 0.04
prominence_threshold_percentile = 90

# RT groups used to generate spikes
short_rt_min = 0.30
short_rt_max = 0.55

long_rt_min = 0.80
long_rt_max = 1.20

# Save directory
save_dir = "./"
os.makedirs(save_dir, exist_ok=True)

# =========================================================
# LOCAL GAUSSIAN KERNEL
# =========================================================
def build_local_gaussian_kernel(bin_size=0.07, sigma=0.04):
    kernel_width = max(3, int(np.ceil(6 * sigma / bin_size)))
    if kernel_width % 2 == 0:
        kernel_width += 1
    kernel = gaussian(kernel_width, std=sigma / bin_size)
    kernel = kernel / np.sum(kernel)
    return kernel

kernel = build_local_gaussian_kernel(bin_size=bin_size, sigma=sigma)

# POISSON SPIKE GENERATOR

def simulate_poisson_spikes(rate_hz, duration_s, rng):
    n_spikes = rng.poisson(rate_hz * duration_s)
    if n_spikes == 0:
        return np.array([], dtype=float)
    return np.sort(rng.uniform(0, duration_s, size=n_spikes))

def simulate_trials(n_trials, n_neurons, rate_hz, rt_min, rt_max, rng):
    trials = []
    rts = rng.uniform(rt_min, rt_max, size=n_trials)

    for trial_id, rt in enumerate(rts):
        neuron_spikes = {}
        for neuron_id in range(n_neurons):
            neuron_spikes[neuron_id] = simulate_poisson_spikes(rate_hz, rt, rng)

        trials.append({
            "trial_id": trial_id,
            "reaction_time": rt,
            "neuron_spikes": neuron_spikes
        })

    return trials

# FIXED 1 s SLOT BURST PIPELINE

def compute_bursts_fixed_1s_slots(trials,
                                  forced_trial_duration=1.0,
                                  bin_size=0.07,
                                  prominence_percentile=90,
                                  kernel=None):
    if kernel is None:
        raise ValueError("kernel must be provided")

    all_spikes = []

    for trial_idx, trial in enumerate(trials):
        slot_offset = trial_idx * forced_trial_duration
        for spikes in trial["neuron_spikes"].values():
            if len(spikes) > 0:
                all_spikes.extend(spikes + slot_offset)

    total_duration = len(trials) * forced_trial_duration

    if total_duration <= 0:
        return {
            "burst_count": 0,
            "threshold": np.nan,
            "total_duration": 0.0,
            "smooth_rate": None,
            "peaks": np.array([])
        }

    time_bins = np.arange(0.0, total_duration + bin_size, bin_size)
    counts, _ = np.histogram(all_spikes, bins=time_bins)
    smooth_rate = np.convolve(counts, kernel, mode="same")

    threshold = np.percentile(smooth_rate, prominence_percentile)
    peaks, _ = signal.find_peaks(smooth_rate, prominence=threshold)

    return {
        "burst_count": int(len(peaks)),
        "threshold": threshold,
        "total_duration": total_duration,
        "smooth_rate": smooth_rate,
        "peaks": peaks
    }

# =========================================================
# RUN SIMULATIONS
# =========================================================
results = []

for sim in range(n_simulations):
    short_trials = simulate_trials(
        n_trials=n_trials_per_group,
        n_neurons=n_neurons,
        rate_hz=firing_rate_hz,
        rt_min=short_rt_min,
        rt_max=short_rt_max,
        rng=rng
    )

    long_trials = simulate_trials(
        n_trials=n_trials_per_group,
        n_neurons=n_neurons,
        rate_hz=firing_rate_hz,
        rt_min=long_rt_min,
        rt_max=long_rt_max,
        rng=rng
    )

    out_short = compute_bursts_fixed_1s_slots(
        short_trials,
        forced_trial_duration=1.0,
        bin_size=bin_size,
        prominence_percentile=prominence_threshold_percentile,
        kernel=kernel
    )

    out_long = compute_bursts_fixed_1s_slots(
        long_trials,
        forced_trial_duration=1.0,
        bin_size=bin_size,
        prominence_percentile=prominence_threshold_percentile,
        kernel=kernel
    )

    results.extend([
        {
            "simulation": sim,
            "condition": "short_RT",
            "burst_count": out_short["burst_count"],
            "threshold": out_short["threshold"],
            "total_duration": out_short["total_duration"]
        },
        {
            "simulation": sim,
            "condition": "long_RT",
            "burst_count": out_long["burst_count"],
            "threshold": out_long["threshold"],
            "total_duration": out_long["total_duration"]
        }
    ])

df_results = pd.DataFrame(results)

excel_path = os.path.join(save_dir, "poisson_probe_rt_fixed_1s_results.xlsx")
df_results.to_excel(excel_path, index=False)
print(f"Saved results to: {excel_path}")

# =========================================================
# PAIRED STATS WITH FDR CORRECTION
# =========================================================
from statsmodels.stats.multitest import multipletests

def paired_test(df, metric):
    pivot = df.pivot_table(
        index="simulation",
        columns="condition",
        values=metric,
        aggfunc="first"
    )
    pivot = pivot[["short_RT", "long_RT"]].dropna()

    if len(pivot) < 2:
        return np.nan, np.nan, pivot

    stat, p = wilcoxon(
        pivot["short_RT"],
        pivot["long_RT"],
        alternative="two-sided"
    )
    return stat, p, pivot


print("\n================ FIXED 1 s SLOTS ================\n")

metrics = ["burst_count", "threshold"]

stats_results = []

for metric in metrics:
    stat, p, pivot = paired_test(df_results, metric)

    stats_results.append({
        "metric": metric,
        "statistic": stat,
        "p_uncorrected": p,
        "n_pairs": len(pivot)
    })

stats_df = pd.DataFrame(stats_results)

# Benjamini-Hochberg FDR correction

stats_df["p_fdr"] = np.nan
stats_df["significant_fdr_0.05"] = False

valid = stats_df["p_uncorrected"].notna()

reject, p_fdr, _, _ = multipletests(
    stats_df.loc[valid, "p_uncorrected"],
    alpha=0.05,
    method="fdr_bh"
)

stats_df.loc[valid, "p_fdr"] = p_fdr
stats_df.loc[valid, "significant_fdr_0.05"] = reject

print(stats_df)

stats_path = os.path.join(save_dir, "poisson_probe_rt_fixed_1s_stats_fdr.xlsx")
stats_df.to_excel(stats_path, index=False)
print(f"Saved FDR-corrected stats to: {stats_path}")

summary = (
    df_results
    .groupby("condition")[["burst_count", "threshold", "total_duration"]]
    .agg(["mean", "std"])
)

print("\n================ SUMMARY ================\n")
print(summary)

summary_path = os.path.join(save_dir, "poisson_probe_rt_fixed_1s_summary.xlsx")
summary.to_excel(summary_path)
print(f"Saved summary to: {summary_path}")

for metric in ["burst_count", "threshold"]:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df_results, x="condition", y=metric, showfliers=False)
    sns.stripplot(data=df_results, x="condition", y=metric,
                  alpha=0.35, color="black")
    plt.title(f"Poisson simulation: {metric} (fixed 1 s slots)")
    plt.tight_layout()

    fig_path = os.path.join(save_dir, f"{metric}_fixed_1s_slots.eps")
    plt.savefig(fig_path, format="eps", bbox_inches="tight")
    plt.show()

# EXAMPLE SINGLE SIMULATION TRACE

example_short_trials = simulate_trials(
    n_trials=n_trials_per_group,
    n_neurons=n_neurons,
    rate_hz=firing_rate_hz,
    rt_min=short_rt_min,
    rt_max=short_rt_max,
    rng=np.random.default_rng(123)
)

example_long_trials = simulate_trials(
    n_trials=n_trials_per_group,
    n_neurons=n_neurons,
    rate_hz=firing_rate_hz,
    rt_min=long_rt_min,
    rt_max=long_rt_max,
    rng=np.random.default_rng(456)
)

def get_trace_from_trials_fixed_1s(trials):
    all_spikes = []
    total_duration = len(trials) * 1.0

    for i, trial in enumerate(trials):
        offset = i * 1.0
        for spikes in trial["neuron_spikes"].values():
            if len(spikes) > 0:
                all_spikes.extend(spikes + offset)

    time_bins = np.arange(0.0, total_duration + bin_size, bin_size)
    counts, _ = np.histogram(all_spikes, bins=time_bins)
    smooth_rate = np.convolve(counts, kernel, mode="same")
    threshold = np.percentile(smooth_rate, prominence_threshold_percentile)
    peaks, _ = signal.find_peaks(smooth_rate, prominence=threshold)

    return smooth_rate, threshold, peaks

smooth_short, thr_short, peaks_short = get_trace_from_trials_fixed_1s(example_short_trials)
smooth_long, thr_long, peaks_long = get_trace_from_trials_fixed_1s(example_long_trials)

plt.figure(figsize=(14, 5))
plt.plot(smooth_short, label="Short RT")
plt.axhline(thr_short, linestyle="--", label=f"Threshold = {thr_short:.2f}")
plt.scatter(peaks_short, smooth_short[peaks_short], s=20)
plt.title("Example simulation: short RT, fixed 1 s slots")
plt.legend()
plt.tight_layout()
short_trace_path = os.path.join(save_dir, "example_short_RT_fixed_1s_trace.eps")
plt.savefig(short_trace_path, format="eps", bbox_inches="tight")
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(smooth_long, label="Long RT")
plt.axhline(thr_long, linestyle="--", label=f"Threshold = {thr_long:.2f}")
plt.scatter(peaks_long, smooth_long[peaks_long], s=20)
plt.title("Example simulation: long RT, fixed 1 s slots")
plt.legend()
plt.tight_layout()
long_trace_path = os.path.join(save_dir, "example_long_RT_fixed_1s_trace.eps")
plt.savefig(long_trace_path, format="eps", bbox_inches="tight")
plt.show()

print("\nSaved EPS figures:")
print(os.path.join(save_dir, "burst_count_fixed_1s_slots.eps"))
print(os.path.join(save_dir, "threshold_fixed_1s_slots.eps"))
print(short_trace_path)
print(long_trace_path)
