"""
Population bursting per epoch, split into an alignment and a single-neuron component (Fig 4f-q, Fig 6c,d; Supp Figs 6, 7, 10a-d).

For each participant, neuronal group, load and epoch, real burst counts are compared with two surrogates (100 each, same
detection threshold as the real data):
  - trial shuffle: each neuron's spike train taken from another trial (independent permutation per neuron), which keeps every
    single-neuron property and removes cross-neuron alignment;
  - Poisson: each neuron's count per trial ~ Poisson(real count), spike times uniform in the window.
alignment     = log2((real + 1) / (shuffle + 1))      (Fig 4f-k, Fig 6c,d)
single_neuron = log2((shuffle + 1) / (Poisson + 1))   (Fig 4l-q)

Epochs: equal 1 s windows. Fixation 0-1 s, Encoding 0-1 s (image `load`), maintenance M1 0-1 s, M2 0.75-1.75 s, M3 1.5-2.5 s.
Each window is cut from every trial and stitched across trials.
Burst detection: 30 ms bins, Gaussian smoothing (sigma 50 ms, kernel 8 bins), peaks with prominence above the 90th
percentile of the real trace, at least 140 ms apart.
Statistics: one-sided paired Wilcoxon across participants (alignment: real > shuffle; single-neuron: shuffle > Poisson),
FDR-BH across the 5 epochs within each group and load.

REGION = False: cell groups (Fig 4, Fig 6c,d, Supp Figs 6, 10a-d).
REGION = True : brain regions, ACG-labelled neurons only, participants with >= 5 such neurons in the region (Supp Fig 7).
Outputs: burst_alignment_results[_region].xlsx and alignment / single_neuron box plots per load.
"""
import ast
import hashlib
import warnings

import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import seaborn as sns

warnings.filterwarnings('ignore')

# =========================
# Parameters
# =========================
REGION = False                    # True for Supp Fig 7 (brain regions)
bin_size = 0.03                   # s
sigma = 0.05                      # Gaussian kernel SD (s)
prominence_threshold_percentile = 90
min_inter_burst_interval = 0.14   # s
WIN = 1.0                         # window length in every epoch (s)
N_SURR = 100                      # surrogates per participant, load and epoch
MIN_TRIALS = 5                    # minimum trials per participant, load and epoch

PER_SUBJ, OVERLAP = 10, 2         # ACG / decay groups: >= 10 neurons with R2 > 0.3; 7 lowest / 7 highest
INPY_MIN = 5                      # minimum PY / IN per participant
CONCEPT_MIN = 3                   # minimum concept cells per participant
ALL_MIN = 10                      # minimum neurons per participant (all cells)
REG_MIN = 5                       # minimum ACG-labelled neurons in a region per participant

min_bin_separation = int(np.ceil(min_inter_burst_interval / bin_size))
kernel = gaussian(int(5 * sigma / bin_size), std=sigma / bin_size)   # 8 bins

M_STARTS = [k * (2.5 - WIN) / 2 for k in range(3)]                   # 0, 0.75, 1.5 s
WINDOWS = {'Fixation': ('fix', 0.0), 'Encoding': ('enc', 0.0),
           'M1': ('delay', M_STARTS[0]), 'M2': ('delay', M_STARTS[1]), 'M3': ('delay', M_STARTS[2])}
periods = list(WINDOWS)

df_metadata = pd.read_excel('../REVISIONS/revision_clustering_no_waveform_labels.xlsx')
df_metadata2 = pd.read_excel('../REVISIONS/all_neuron_brain_regions_merged.xlsx')

if REGION:
    # ACG-labelled neurons only; vmPFC left out (only 2 participants with >= 5 such neurons)
    df_metadata2 = df_metadata2[df_metadata2['Neuron_ID_3'].isin(df_metadata[df_metadata['Cell_Type_New'].notna()]['Neuron_ID_3'])]
    categories = ['amygdala', 'hippocampus', 'dorsal_anterior_cingulate_cortex', 'pre_supplementary_motor_area']
else:
    categories = ['All_cells', '5_lowest_ACG', '5_highest_ACG', '5_lowest_decay', '5_highest_decay',
                  'Concept_cells', 'Interneurons', 'Pyramidal']

cols = ['subject_id', 'trial_id', 'Neuron_ID_3', 'num_images_presented', 'Standardized_Spikes']
keys = ['subject_id', 'trial_id', 'Neuron_ID_3']
period_data = {'fix': pd.read_excel('../clean_data/cleaned_Fixation.xlsx', usecols=cols).drop_duplicates(keys),
               'delay': pd.read_excel('../graph_data/graph_delay.xlsx', usecols=cols).drop_duplicates(keys)}
encoding_data = {L: pd.read_excel(f'../graph_data/graph_encoding{L}.xlsx', usecols=cols).drop_duplicates(keys) for L in (1, 2, 3)}


def parse_spike_list(s):
    if not isinstance(s, str) or s.strip() in ('', '[]'):
        return np.array([])
    try:
        return np.atleast_1d(np.array(ast.literal_eval(s), dtype=float))
    except Exception:
        return np.array([])


def seed(*parts):
    """Fixed seed per participant / group / load / epoch, so results do not depend on run order."""
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little') % (2 ** 32 - 1)


def count_bursts(M, threshold=None):
    """M: trials x neurons list of spike arrays (0..WIN), stitched across trials. Returns (burst count, threshold)."""
    all_spikes = np.concatenate([np.concatenate(trial) + k * WIN for k, trial in enumerate(M)])
    counts, _ = np.histogram(all_spikes, np.arange(0, len(M) * WIN + bin_size, bin_size))
    smooth_rate = np.convolve(counts, kernel, mode='same')
    if threshold is None:
        threshold = np.percentile(smooth_rate, prominence_threshold_percentile)
    peaks, _ = signal.find_peaks(smooth_rate, prominence=threshold, distance=min_bin_separation)
    return len(peaks), threshold


def split_low_high(df_subj, feature):
    df = df_subj[['Neuron_ID_3', feature]].dropna(subset=[feature])
    if len(df) < PER_SUBJ:
        return [], []
    df = df.sort_values([feature, 'Neuron_ID_3'], kind='mergesort')
    m = PER_SUBJ // 2 + OVERLAP
    return df.head(m)['Neuron_ID_3'].tolist(), df.tail(m)['Neuron_ID_3'].tolist()


def eligible_subjects(cat):
    if REGION:
        counts, threshold = df_metadata2[df_metadata2['Region'] == cat]['subject_id'].value_counts(), REG_MIN
    elif cat == 'Concept_cells':
        counts, threshold = df_metadata2[df_metadata2['Signi'] == 'Y']['subject_id'].value_counts(), CONCEPT_MIN
    elif cat in ('Pyramidal', 'Interneurons'):
        label = 'PY' if cat == 'Pyramidal' else 'IN'
        counts, threshold = df_metadata[df_metadata['Cell_Type_New'] == label]['subject_id'].value_counts(), INPY_MIN
    elif cat == 'All_cells':
        counts, threshold = df_metadata['subject_id'].value_counts(), ALL_MIN
    else:
        feature = 'Mean_ACG' if 'ACG' in cat else 'Decay'
        counts = df_metadata[df_metadata['R2'] > 0.3].dropna(subset=[feature])['subject_id'].value_counts()
        threshold = PER_SUBJ
    return sorted(counts[counts >= threshold].index)


def group_neurons(cat, s):
    if REGION:
        return df_metadata2[(df_metadata2['subject_id'] == s) & (df_metadata2['Region'] == cat)]['Neuron_ID_3'].tolist()
    m = df_metadata[df_metadata['subject_id'] == s]
    m_r2 = m[m['R2'] > 0.3]
    low_acg, high_acg = split_low_high(m_r2, 'Mean_ACG')
    low_dec, high_dec = split_low_high(m_r2, 'Decay')
    return {'5_lowest_ACG': low_acg, '5_highest_ACG': high_acg, '5_lowest_decay': low_dec, '5_highest_decay': high_dec,
            'Interneurons': m[m['Cell_Type_New'] == 'IN']['Neuron_ID_3'].tolist(),
            'Pyramidal': m[m['Cell_Type_New'] == 'PY']['Neuron_ID_3'].tolist(),
            'All_cells': m['Neuron_ID_3'].tolist(),
            'Concept_cells': df_metadata2[(df_metadata2['subject_id'] == s) & (df_metadata2['Signi'] == 'Y')]['Neuron_ID_3'].tolist()}[cat]

# Real vs surrogate bursts
rows = []
for cat in categories:
    for s in eligible_subjects(cat):
        neurons = set(group_neurons(cat, s))
        if not neurons:
            continue
        for load in (1, 2, 3):
            for period, (source, t0) in WINDOWS.items():
                d = encoding_data[load] if source == 'enc' else period_data[source]
                d = d[(d['subject_id'] == s) & (d['num_images_presented'] == load) & d['Neuron_ID_3'].isin(neurons)]
                neu, tri = sorted(d['Neuron_ID_3'].unique()), sorted(d['trial_id'].unique())
                if len(tri) < MIN_TRIALS:
                    continue
                spikes = {}
                for r in d.itertuples():
                    x = parse_spike_list(r.Standardized_Spikes)
                    spikes[(r.trial_id, r.Neuron_ID_3)] = x[(x >= t0) & (x < t0 + WIN)] - t0
                M = [[spikes.get((t, n), np.array([])) for n in neu] for t in tri]
                if sum(a.size for trial in M for a in trial) == 0:
                    continue

                real, threshold = count_bursts(M)

                # trial-shuffle surrogate
                rng = np.random.default_rng(seed(cat, s, load, period))
                shuffle = []
                for _ in range(N_SURR):
                    perm = [rng.permutation(len(tri)) for _ in neu]
                    shuffle.append(count_bursts([[M[perm[j][i]][j] for j in range(len(neu))] for i in range(len(tri))], threshold)[0])
                shuffle = float(np.mean(shuffle))

                # Poisson surrogate
                rng = np.random.default_rng(seed(cat, s, load, period, 'poisson'))
                poisson = float(np.mean([count_bursts([[rng.uniform(0, WIN, rng.poisson(a.size)) for a in trial] for trial in M], threshold)[0]
                                         for _ in range(N_SURR)]))

                rows.append(dict(Load=load, Cell_Type=cat, Period=period, subject_id=s, n_neurons=len(neu), n_trials=len(tri),
                                 real_bursts=real, shuffle_mean_bursts=shuffle, poisson_mean_bursts=poisson,
                                 alignment=np.log2((real + 1) / (shuffle + 1)),
                                 single_neuron=np.log2((shuffle + 1) / (poisson + 1))))
    print(f'{cat}: done', flush=True)
long = pd.DataFrame(rows)


# Statistics: each epoch vs its control, FDR-BH over the 5 epochs within group x load
def symbol_from_p(p):
    if pd.isna(p):
        return ''
    return '***' if p < 0.005 else '**' if p < 0.01 else '*' if p < 0.05 else ''


TESTS = {'alignment': ('real_bursts', 'shuffle_mean_bursts'),          # real > trial shuffle
         'single_neuron': ('shuffle_mean_bursts', 'poisson_mean_bursts')}  # trial shuffle > Poisson
stats = []
for metric, (a, b) in TESTS.items():
    for (load, cat), g in long.groupby(['Load', 'Cell_Type']):
        block = []
        for period in periods:
            x = g[g['Period'] == period]
            if len(x) < 2:
                continue
            if (x[a] != x[b]).any():
                res = wilcoxon(x[a], x[b], alternative='greater')
                W, p = res.statistic, res.pvalue
            else:
                W, p = np.nan, 1.0
            block.append(dict(Metric=metric, Load=load, Cell_Type=cat, Period=period, n=len(x),
                              median_log2=x[metric].median(), wilcoxon_stat=W, raw_p=p))
        if block:
            for r, q in zip(block, multipletests([r['raw_p'] for r in block], method='fdr_bh')[1]):
                r['p_corrected_fdr_bh'], r['significance_symbol'] = q, symbol_from_p(q)
            stats += block
stats = pd.DataFrame(stats)

suffix = '_region' if REGION else ''
with pd.ExcelWriter(f'./burst_alignment_results{suffix}.xlsx', engine='openpyxl') as xw:
    long.to_excel(xw, sheet_name='per_subject', index=False)
    stats.to_excel(xw, sheet_name='epoch_vs_control', index=False)
    
for metric, ylabel in (('alignment', 'Alignment: burst excess over trial shuffle (log2)'),
                       ('single_neuron', 'Single-neuron: trial shuffle over Poisson (log2)')):
    lo, hi = long[metric].min(), long[metric].max()
    span = hi - lo
    for load in (1, 2, 3):
        df_plot = long[long['Load'] == load]
        nrow = int(np.ceil(len(categories) / 4))
        plt.figure(figsize=(20, 5 * nrow))
        for i, group in enumerate(categories, 1):
            ax = plt.subplot(nrow, 4, i)
            data = df_plot[df_plot['Cell_Type'] == group]
            sns.boxplot(order=periods, x='Period', y=metric, data=data, palette='Set2', showfliers=False, ax=ax)
            sns.stripplot(order=periods, x='Period', y=metric, data=data, color='black', size=4, jitter=True, alpha=0.6, ax=ax)
            ax.axhline(0, ls='--', color='0.4', lw=1, zorder=0)
            ax.set_title(f'{group} (Load {load}, n={data["subject_id"].nunique()})', fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel(ylabel if i % 4 == 1 else '')
            ax.set_ylim(lo - 0.04 * span, hi + 0.1 * span)
            sig = stats[(stats['Metric'] == metric) & (stats['Load'] == load) & (stats['Cell_Type'] == group)]
            for r in sig.itertuples():
                if r.significance_symbol:
                    top = data[data['Period'] == r.Period][metric].max()
                    ax.text(periods.index(r.Period), top + 0.02 * span, r.significance_symbol, ha='center', va='bottom', fontsize=12)
        plt.suptitle(f'{metric} across epochs (1 s windows) - Load {load}', fontsize=18, y=1.02)
        plt.tight_layout()
        plt.savefig(f'./{metric}_burst{suffix}_load{load}.eps', format='eps', dpi=300, bbox_inches='tight')
        plt.close()
