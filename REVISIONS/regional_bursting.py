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

df_metadata  = pd.read_excel('../Neuron_Check_Significant_All.xlsx')
df_metadata2 = pd.read_excel('./all_neuron_brain_regions_merged.xlsx')

df_fix   = pd.read_excel('../clean_data/cleaned_Fixation.xlsx')   
df_enc1  = pd.read_excel('../graph_data/graph_encoding1.xlsx')
df_enc2  = pd.read_excel('../graph_data/graph_encoding2.xlsx')
df_enc3  = pd.read_excel('../graph_data/graph_encoding3.xlsx')
df_delay = pd.read_excel('../graph_data/graph_delay.xlsx')
df_probe = pd.read_excel('../graph_data/graph_probe.xlsx')

bin_size = 0.03           
sigma = 0.05                   
prominence_threshold_percentile = 90
duration = {
    'Fixation': 1,        
    'Encoding 1': 1,
    'Encoding 2': 1,
    'Encoding 3': 1,
    'Delay': 3,
   # 'Probe': 1
}

# Per-subject low/high split knobs for ACG/Decay
PER_SUBJ = 10    # require >= 10 eligible neurons per subject
OVERLAP  = 2    

encoding_map = {
    1: ('Encoding 1', df_enc1),
    2: ('Encoding 2', df_enc2),
    3: ('Encoding 3', df_enc3),
}
period_data = {
    'Fixation': df_fix,
    'Delay': df_delay,
 #   'Probe': df_probe
}

def collapse_lr(loc):
    if pd.isna(loc):
        return np.nan
    s = str(loc).strip().lower()
    if s.endswith('_left'):
        return s[:-5]
    if s.endswith('_right'):
        return s[:-6]
    return s

df_metadata2 = df_metadata2.copy()
df_metadata2['Region'] = df_metadata2['Region'].apply(collapse_lr)

# Optional: enforce only known regions to avoid typos sneaking in
KNOWN_REGIONS = {'hippocampus', 'amygdala', 'dorsal_anterior_cingulate_cortex', 'pre_supplementary_motor_area'}
df_metadata2 = df_metadata2[df_metadata2['Region'].isin(KNOWN_REGIONS)].copy()

regions = sorted(df_metadata2['Region'].dropna().unique().tolist())
print("Regions found:", regions)

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

    arr = arr[(arr >= 0.0) & (arr <= float(trial_dur))]
    return arr


MIN_NEURONS_PER_REGION = 4 

subject_ids_per_region = {}
for reg in regions:
    counts = df_metadata2[df_metadata2['Region'] == reg]['subject_id'].value_counts()
    subject_ids_per_region[reg] = counts[counts >= MIN_NEURONS_PER_REGION].index.tolist()

print("Subjects per region (after threshold):")
for reg in regions:
    print(reg, "->", len(subject_ids_per_region[reg]))

periods_for_counts = ['Fixation', 'Encoding', 'Delay_part1', 'Delay_part2', 'Delay_part3']

burst_counts = {
    reg: {
        period: {load: [] for load in [1, 2, 3]}
        for period in periods_for_counts
    }
    for reg in regions
}

# Normalization by neuron count is applied to avoid fake region differences
# (regions with more recorded neurons would otherwise inflate burst counts).

for reg in regions:
    for subject_id in subject_ids_per_region[reg]:

        # neuron IDs for this subject + region
        neurons = df_metadata2[
            (df_metadata2['subject_id'] == subject_id) &
            (df_metadata2['Region'] == reg)
        ]['Neuron_ID_3'].unique().tolist()

        if len(neurons) == 0:
            continue

        for load in [1, 2, 3]:
            enc_label, df_enc = encoding_map[load]

            df_enc_sub = df_enc[
                (df_enc['subject_id'] == subject_id) &
                (df_enc['num_images_presented'] == load)
            ]
            if df_enc_sub.empty:
                # still can have fixation/delay/probe, so don’t continue here
                pass

            for period_label in ['Fixation', 'Encoding', 'Delay']:

                # pick dataframe + duration
                if period_label == 'Encoding':
                    df_sub = df_enc_sub
                    trial_duration = duration[enc_label]
                else:
                    dfP = period_data[period_label]  # Fixation / Delay / Probe
                    df_sub = dfP[
                        (dfP['subject_id'] == subject_id) &
                        (dfP['num_images_presented'] == load)
                    ]
                    trial_duration = duration[period_label]

                if df_sub.empty:
                    continue

                # restrict to region neurons
                df_group = df_sub[df_sub['Neuron_ID_3'].isin(neurons)]
                if df_group.empty:
                    continue

                trial_ids = sorted(df_group['trial_id'].unique())
                if len(trial_ids) == 0:
                    continue

                # collect spikes across trials (concatenated timeline)
                all_spikes = []

                for trial_idx, trial_id in enumerate(trial_ids):
                    df_trial = df_group[df_group['trial_id'] == trial_id]

                    if period_label == 'Encoding':
                        col = 'Standardized_Spikes'
                    elif period_label == 'Fixation':
                        col = 'Standardized_Spikes'
                    else:
                        col = f'Standardized_Spikes'

                    # iterate neurons
                    for neuron_id in sorted(df_trial['Neuron_ID_3'].unique()):
                        series = df_trial[df_trial['Neuron_ID_3'] == neuron_id][col].dropna()
                        for s in series:
                            arr = parse_spike_list(s, trial_duration)
                            if arr.size:
                                all_spikes.extend(arr + trial_idx * trial_duration)

                if not all_spikes:
                    continue

                total_dur = len(trial_ids) * trial_duration
                time_bins = np.arange(0.0, total_dur + bin_size, bin_size)

                counts, _ = np.histogram(all_spikes, bins=time_bins)

                # normalize by number of neurons in region for this subject
                counts = counts / max(len(neurons), 1)

                # smooth
                std_bins = sigma / bin_size
                std_bins = max(1.0, std_bins)

                kernel_width = int(np.ceil(5 * std_bins))
                kernel_width = max(kernel_width, 3)
                if kernel_width % 2 == 0:
                    kernel_width += 1

                kernel = gaussian(kernel_width, std=std_bins)
                kernel /= kernel.sum()
                
                smooth_rate = np.convolve(counts, kernel, mode='same')

                # peaks
                threshold = np.percentile(smooth_rate, prominence_threshold_percentile)
                peaks, _ = signal.find_peaks(smooth_rate, prominence=threshold)

                # store counts (delay split into 3 equal parts)
                if period_label == 'Delay':
                    thirds = np.array_split(np.arange(len(smooth_rate)), 3)
                    for i, part in enumerate(thirds):
                        count = np.sum(np.isin(peaks, part))
                        burst_counts[reg][f'Delay_part{i+1}'][load].append(int(count))
                else:
                    burst_counts[reg][period_label][load].append(int(len(peaks)))

def symbol_from_p(p_corr):
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
    ax.plot([x1, x1, x2, x2],
            [base_y, base_y + height, base_y + height, base_y],
            lw=lw, c=color, clip_on=False)
    ax.text((x1 + x2) / 2, base_y + height + text_pad, text,
            ha='center', va='bottom', fontsize=12, color=color)

periods = periods_for_counts

for load in [1, 2, 3]:
    plot_data = []
    for reg in regions:
        for period in periods:
            for val in burst_counts[reg][period][load]:
                plot_data.append({
                    'Region': reg,
                    'Period': period,
                    'Burst Count': val,
                    'Load': load
                })
    df_plot = pd.DataFrame(plot_data)

    # layout: 2 rows, enough cols for regions
    n = len(regions)
    ncols = int(np.ceil(n / 2))
    nrows = 2 if n > 1 else 1

    plt.figure(figsize=(5 * ncols, 5 * nrows))

    for i, reg in enumerate(regions, 1):
        ax = plt.subplot(nrows, ncols, i)
        data = df_plot[df_plot['Region'] == reg]

        sns.boxplot(x='Period', y='Burst Count', data=data, showfliers=False, ax=ax)
        sns.stripplot(x='Period', y='Burst Count', data=data,
                      color='black', size=4, jitter=True, alpha=0.6, ax=ax)

        ax.set_title(f'{reg} (Load {load})', fontsize=12)
        ax.set_ylim(0, 50)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=30)

        # Period-wise Wilcoxon (paired only if equal lengths)
        try:
            comparisons, indices = [], []
            for j in range(len(periods)):
                for k in range(j + 1, len(periods)):
                    g1 = data[data['Period'] == periods[j]]['Burst Count'].values
                    g2 = data[data['Period'] == periods[k]]['Burst Count'].values

                    # only paired tests when vectors align (same subjects contributing)
                    if len(g1) == len(g2) and len(g1) > 0:
                        try:
                            p = wilcoxon(g1, g2, alternative='two-sided', zero_method='wilcox').pvalue
                            comparisons.append(p)
                            indices.append((j, k))
                        except Exception:
                            continue

            if comparisons:
                _, pvals_corr, _, _ = multipletests(comparisons, alpha=0.05, method='fdr_bh')

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
            print(f"Skipping stats for {reg} (load {load}) due to error: {e}")

    plt.suptitle(f"Region-based Burst Count Across Periods (Delay split into 3) - Load {load}",
                 fontsize=16, y=1.02)
    plt.tight_layout()

    # Save path (change if needed)
    plt.savefig(f"./region_burst_load{load}.eps",
                format='eps', dpi=300)
    plt.show()

summary_rows = []
stats_rows = []

for reg in regions:
    for load in [1, 2, 3]:

        # ---- descriptive stats ----
        for period in periods:
            vals = np.array(burst_counts[reg][period][load], dtype=float)

            summary_rows.append({
                'Region': reg,
                'Load': load,
                'Period': period,
                'n_subjects': len(vals),
                'mean_burst_count': np.nanmean(vals) if len(vals) else np.nan,
                'median_burst_count': np.nanmedian(vals) if len(vals) else np.nan,
                'std_burst_count': np.nanstd(vals, ddof=1) if len(vals) > 1 else np.nan,
                'sem_burst_count': (
                    np.nanstd(vals, ddof=1) / np.sqrt(len(vals))
                    if len(vals) > 1 else np.nan
                ),
                'min_burst_count': np.nanmin(vals) if len(vals) else np.nan,
                'max_burst_count': np.nanmax(vals) if len(vals) else np.nan
            })

        # ---- pairwise Wilcoxon tests within region/load ----
        temp_pvals = []
        temp_meta = []

        for j in range(len(periods)):
            for k in range(j + 1, len(periods)):

                p1 = periods[j]
                p2 = periods[k]

                g1 = np.array(burst_counts[reg][p1][load], dtype=float)
                g2 = np.array(burst_counts[reg][p2][load], dtype=float)

                # Your current script only allows paired tests if lengths match.
                # This assumes the values are aligned by subject/order.
                if len(g1) == len(g2) and len(g1) > 0:
                    try:
                        stat, p_raw = wilcoxon(
                            g1, g2,
                            alternative='two-sided',
                            zero_method='wilcox'
                        )

                        temp_pvals.append(p_raw)
                        temp_meta.append({
                            'Region': reg,
                            'Load': load,
                            'Comparison': f'{p1} vs {p2}',
                            'Period_1': p1,
                            'Period_2': p2,
                            'n_pairs': len(g1),
                            'mean_1': np.nanmean(g1),
                            'mean_2': np.nanmean(g2),
                            'median_1': np.nanmedian(g1),
                            'median_2': np.nanmedian(g2),
                            'mean_difference_2_minus_1': np.nanmean(g2 - g1),
                            'median_difference_2_minus_1': np.nanmedian(g2 - g1),
                            'wilcoxon_stat': stat,
                            'p_raw': p_raw
                        })

                    except Exception as e:
                        stats_rows.append({
                            'Region': reg,
                            'Load': load,
                            'Comparison': f'{p1} vs {p2}',
                            'Period_1': p1,
                            'Period_2': p2,
                            'n_pairs': len(g1),
                            'error': str(e),
                            'p_raw': np.nan,
                            'p_fdr': np.nan,
                            'significance': ''
                        })

        if temp_pvals:
            _, pvals_fdr, _, _ = multipletests(
                temp_pvals,
                alpha=0.05,
                method='fdr_bh'
            )

            for row, p_fdr in zip(temp_meta, pvals_fdr):
                row['p_fdr'] = p_fdr
                row['significance'] = symbol_from_p(p_fdr)
                row['error'] = ''
                stats_rows.append(row)


df_region_summary = pd.DataFrame(summary_rows)
df_region_stats = pd.DataFrame(stats_rows)

# Export summary + stats to Excel

out_stats_xlsx = "./Region_Burst_Summary_and_Pvalues.xlsx"

with pd.ExcelWriter(out_stats_xlsx, engine='openpyxl') as writer:
    df_region_summary.to_excel(writer, sheet_name='summary_counts', index=False)
    df_region_stats.to_excel(writer, sheet_name='wilcoxon_pvalues', index=False)
    df_burst_export.to_excel(writer, sheet_name='raw_burst_counts', index=False)

print("Saved region summary and p-values to:", out_stats_xlsx)

rows = []
for reg in regions:
    for period in periods:
        for load in [1, 2, 3]:
            vals = burst_counts[reg][period][load]
            for v in vals:
                rows.append({'Region': reg, 'Period': period, 'Load': load, 'Burst Count': v})

df_burst_export = pd.DataFrame(rows)
out_xlsx = "./Region_Burst_Counts_All.xlsx"
df_burst_export.to_excel(out_xlsx, index=False)
print("Saved burst counts to:", out_xlsx)
