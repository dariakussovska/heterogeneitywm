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
warnings.filterwarnings('ignore')

# =========================
# Parameters
# =========================
bin_size = 0.07                # s
sigma = 0.04                   # Gaussian kernel SD (s)
prominence_threshold_percentile = 90
duration = {
    'Encoding 1': 1,
    'Encoding 2': 1,
    'Encoding 3': 1,
    'Delay': 3,
    'Probe': 1
}

# Per-subject low/high split knobs for ACG/Decay
PER_SUBJ = 10    # require >= 10 eligible neurons per subject
OVERLAP  = 1     

# =========================
# Load data
# =========================
df_metadata  = pd.read_excel('/home/daria/PROJECT/Clustering_3D.xlsx')
df_metadata2 = pd.read_excel('/home/daria/PROJECT/all_neuron_brain_regions_cleaned.xlsx')

df_enc1  = pd.read_excel('/home/daria/PROJECT/graph_data/graph_encoding1.xlsx')
df_enc2  = pd.read_excel('/home/daria/PROJECT/graph_data/graph_encoding2.xlsx')
df_enc3  = pd.read_excel('/home/daria/PROJECT/graph_data/graph_encoding3.xlsx')
df_delay = pd.read_excel('/home/daria/PROJECT/graph_data/graph_delay.xlsx')
df_probe = pd.read_excel('/home/daria/PROJECT/graph_data/graph_probe.xlsx')

# Convenience filtered/meta frames
df_metadata_decay_acg = df_metadata[df_metadata['R2'] > 0.3].copy()
df_metadata_concept   = df_metadata2[df_metadata2['Signi'] == 'Y'].copy()
df_metadata_in_py     = df_metadata[df_metadata['Cell_Type_New'].isin(['IN', 'PY'])].copy()
df_metadata_all_cells = df_metadata.copy()

encoding_map = {1: ('Encoding 1', df_enc1), 2: ('Encoding 2', df_enc2), 3: ('Encoding 3', df_enc3)}
period_data  = {'Delay': df_delay, 'Probe': df_probe}

# Keep your original category names for downstream compatibility
categories = [
    "All_cells",
    "5_lowest_ACG", "5_highest_ACG",
    "5_lowest_decay", "5_highest_decay",
    "Concept_cells",
    "Interneurons", "Pyramidal",
]

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
    Requires at least `per_subj` rows (after dropping NaNs in feature).
    Returns (low_ids, high_ids); each ~ per_subj//2 + overlap.
    Stable sort; tiebreaker on Neuron_ID_3.
    """
    if df_subj.empty or feature not in df_subj.columns:
        return [], []
    df = df_subj[["Neuron_ID_3", feature]].dropna(subset=[feature]).copy()
    if len(df) < per_subj:
        return [], []
    df_sorted = df.sort_values([feature, "Neuron_ID_3"], ascending=[True, True], kind="mergesort")
    m = (per_subj // 2) + overlap
    low_ids  = df_sorted.head(m)["Neuron_ID_3"].tolist()
    high_ids = df_sorted.tail(m)["Neuron_ID_3"].tolist()
    return low_ids, high_ids

subject_ids_per_category = {}
for category in categories:
    if category == "Concept_cells":
        subject_counts = df_metadata2[df_metadata2["Signi"] == "Y"]["subject_id"].value_counts()
        threshold = 3
    elif category in ["Pyramidal", "Interneurons"]:
        label = "PY" if category == "Pyramidal" else "IN"
        subject_counts = df_metadata[df_metadata["Cell_Type_New"] == label]["subject_id"].value_counts()
        threshold = 3
    elif category == "All_cells":
        subject_counts = df_metadata["subject_id"].value_counts()
        threshold = 10
    elif category in ["5_lowest_ACG", "5_highest_ACG"]:
        eligible = df_metadata[df_metadata["R2"] > 0.3].dropna(subset=["ACG_Norm"])
        subject_counts = eligible["subject_id"].value_counts()
        threshold = PER_SUBJ
    else:  # 5_lowest_decay / 5_highest_decay
        eligible = df_metadata[df_metadata["R2"] > 0.3].dropna(subset=["Decay"])
        subject_counts = eligible["subject_id"].value_counts()
        threshold = PER_SUBJ
    subject_ids_per_category[category] = subject_counts[subject_counts >= threshold].index.tolist()

burst_counts = {
    cat: {
        period: {load: [] for load in [1, 2, 3]}
        for period in ['Encoding', 'Delay_part1', 'Delay_part2', 'Delay_part3', 'Probe']
    }
    for cat in categories
}

for group in categories:
    for subject_id in subject_ids_per_category[group]:
        # Subject-specific metadata
        df_meta    = df_metadata[df_metadata['subject_id'] == subject_id].copy()
        df_meta_r2 = df_meta[df_meta['R2'] > 0.3].copy()

        # Per-subject low/high splits with overlap for ACG & Decay (from R2-filtered)
        low_acg,  high_acg  = split_low_high(df_meta_r2, "ACG_Norm", per_subj=PER_SUBJ, overlap=OVERLAP)
        low_dec,  high_dec  = split_low_high(df_meta_r2, "Decay",    per_subj=PER_SUBJ, overlap=OVERLAP)

        # Concept cells (from df_metadata2 to be safe)
        concept_ids = df_metadata2[(df_metadata2["subject_id"] == subject_id) &
                                   (df_metadata2["Signi"] == "Y")]["Neuron_ID_3"].tolist()

        neuron_groups = {
            "5_lowest_ACG":   low_acg,
            "5_highest_ACG":  high_acg,
            "5_lowest_decay": low_dec,
            "5_highest_decay":high_dec,
            "Interneurons":   df_meta[df_meta['Cell_Type_New'] == 'IN']['Neuron_ID_3'].tolist(),
            "Pyramidal":      df_meta[df_meta['Cell_Type_New'] == 'PY']['Neuron_ID_3'].tolist(),
            "All_cells":      df_meta['Neuron_ID_3'].tolist(),
            "Concept_cells":  concept_ids
        }

        neurons = neuron_groups.get(group, [])
        if len(neurons) == 0:
            continue

        # Iterate loads and periods
        for load in [1, 2, 3]:
            enc_label, df_enc = encoding_map[load]
            df_enc_sub = df_enc[(df_enc['subject_id'] == subject_id) &
                                (df_enc['num_images_presented'] == load)]

            for period_label in ['Encoding', 'Delay', 'Probe']:
                if period_label == 'Encoding':
                    df_sub = df_enc_sub
                    trial_duration = duration[enc_label]
                else:
                    dfP = period_data[period_label]
                    df_sub = dfP[(dfP['subject_id'] == subject_id) &
                                 (dfP['num_images_presented'] == load)]
                    trial_duration = duration[period_label]

                if df_sub.empty:
                    continue

                # Restrict to selected neurons
                df_group = df_sub[df_sub['Neuron_ID_3'].isin(neurons)]
                if df_group.empty:
                    continue

                trial_ids = sorted(df_group['trial_id'].unique())
                if len(trial_ids) == 0:
                    continue

                # Collect all spikes across trials (concatenated timeline)
                all_spikes = []
                for trial_idx, trial_id in enumerate(trial_ids):
                    df_trial = df_group[df_group['trial_id'] == trial_id]
                    col = 'Standardized_Spikes'
                    for neuron_id in sorted(df_trial['Neuron_ID_3'].unique()):
                        series = df_trial[df_trial['Neuron_ID_3'] == neuron_id][col].dropna()
                        # take all rows (if duplicates exist, they’ll just contribute spikes)
                        for s in series:
                            arr = parse_spike_list(s, trial_duration)
                            if arr.size:
                                all_spikes.extend(arr + trial_idx * trial_duration)

                if not all_spikes:
                    continue

                # Histogram -> smooth -> threshold -> peaks
                total_dur = len(trial_ids) * trial_duration
                time_bins = np.arange(0.0, total_dur + bin_size, bin_size)
                counts, _ = np.histogram(all_spikes, bins=time_bins)

                kernel = gaussian(len(counts), std=sigma / bin_size)
                kernel = kernel / np.sum(kernel)
                smooth_rate = np.convolve(counts, kernel, mode='same')

                threshold = np.percentile(smooth_rate, prominence_threshold_percentile)
                peaks, _ = signal.find_peaks(smooth_rate, prominence=threshold)

                # Save counts
                if period_label == 'Delay':
                    thirds = np.array_split(np.arange(len(smooth_rate)), 3)
                    for i, part in enumerate(thirds):
                        count = np.sum(np.isin(peaks, part))
                        burst_counts[group][f'Delay_part{i+1}'][load].append(int(count))
                else:
                    burst_counts[group][period_label][load].append(int(len(peaks)))

periods = ['Encoding', 'Delay_part1', 'Delay_part2', 'Delay_part3', 'Probe']

def symbol_from_p(p_corr):
    # '#' for p<0.01 ; '##' for p<0.005 ; else ''
    if np.isnan(p_corr):
        return ""
    if p_corr < 0.005:
        return "***"
    if p_corr < 0.01:
        return "**"
    if p_corr < 0.05:
        return "*"
    return ""

def add_hash(ax, text, x1, x2, base_y, height=0.6, text_pad=0.1, lw=1.5, color='k'):
    """Draw a bracket between x1 and x2 at given height and place text ('#' or '##')."""
    ax.plot([x1, x1, x2, x2], [base_y, base_y + height, base_y + height, base_y],
            lw=lw, c=color, clip_on=False)
    ax.text((x1 + x2) / 2, base_y + height + text_pad, text,
            ha='center', va='bottom', fontsize=12, color=color)

for load in [1, 2, 3]:
    plot_data = []
    for group in categories:
        for period in periods:
            for val in burst_counts[group][period][load]:
                plot_data.append({
                    'Cell Type': group,
                    'Period': period,
                    'Burst Count': val,
                    'Load': load
                })
    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(20, 10))
    for i, group in enumerate(categories, 1):
        ax = plt.subplot(2, 4, i)
        data = df_plot[df_plot['Cell Type'] == group]
        sns.boxplot(x='Period', y='Burst Count', data=data, palette='Set2',
                    showfliers=False, ax=ax)
        sns.stripplot(x='Period', y='Burst Count', data=data, color='black',
                      size=4, jitter=True, dodge=True, alpha=0.6, ax=ax)
        ax.set_title(f'{group} (Load {load})', fontsize=12)
        ax.set_ylim(0, 50)
        ax.set_xlabel('')
        ax.set_ylabel('')

        # pairwise Wilcoxon across periods (+ FDR) 
        try:
            comparisons, indices = [], []
            # pairwise over periods; paired only if equal-length vectors
            for j in range(len(periods)):
                for k in range(j + 1, len(periods)):
                    g1 = data[data['Period'] == periods[j]]['Burst Count'].values
                    g2 = data[data['Period'] == periods[k]]['Burst Count'].values
                    if len(g1) == len(g2) and len(g1) > 0:
                        try:
                            p = wilcoxon(g1, g2, alternative='two-sided',
                                         zero_method='wilcox').pvalue
                            comparisons.append(p)
                            indices.append((j, k))
                        except Exception:
                            continue

            if comparisons:
                # FDR-BH per group & load
                _, pvals_corr, _, _ = multipletests(comparisons, alpha=0.05, method='fdr_bh')

                # annotate significant pairs
                ymax = data['Burst Count'].max() if not data.empty else 0
                base = ymax + 1.0
                step = 0.6
                height = 0.45
                for (j, k), p_corr in zip(indices, pvals_corr):
                    sym = symbol_from_p(p_corr)
                    if sym:
                        level = (k - j - 1)
                        y = base + level * step
                        add_hash(ax, sym, j, k, y, height=height)
        except Exception as e:
            print(f"Skipping stats for {group} (load {load}) due to error: {e}")

    plt.suptitle(f"Burst Count Across Periods (Delay in 3 Parts) - Load {load}",
                 fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(f"/home/daria/PROJECT/combined_burst_load{load}.eps",
                format='eps', dpi=300)
    plt.show()
