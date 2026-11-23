"""
Concept-cell recruitment into PY- and IN-defined population bursts (Supp Fig 10f, g).

Bursts per participant from the NON-concept neurons of one cell type (no-waveform ACG labels, >= 3 per participant),
all loads, trials concatenated: 30 ms bins, Gaussian smoothing (sigma 50 ms, 8-bin kernel), prominence above the 90th
percentile, >= 140 ms apart. Burst window = +-50 ms around each peak.
Epochs: encoding (first image, 0-0.9 s; preferred = the cell's preferred image on screen) and maintenance (0-2.5 s).
Concept cells: Signi == Y. Per concept cell and condition (preferred / nonpreferred trials, >= 3 trials):
  recruitment = log2((observed spikes inside burst windows + 1) / (expected + 1)), expected = mean over 200 draws in
  which each trial's spike train is taken from another trial of the same load and condition (bursts fixed).
Tests: Wilcoxon signed-rank against 0, BH across the 8 tests (PY/IN bursts x encoding/maintenance x preferred/nonpreferred);
preferred vs nonpreferred (Wilcoxon, uncorrected) reported in the stats file.

"""
import ast
import os

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal.windows import gaussian
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =========================
# Parameters
# =========================
bin_size, sigma, HW = 0.03, 0.05, 0.05
kernel = gaussian(int(5 * sigma / bin_size), std=sigma / bin_size)   # 8 bins
distance = int(np.ceil(0.14 / bin_size))                              # 140 ms
N_NULL = 200
EPOCHS = {'Encoding': ('../graph_data/graph_encoding1.xlsx', 0.9), 'Maintenance': ('../graph_data/graph_delay.xlsx', 2.5)}
COLORS = {'PY': ('#b5283c', '#e3a3ab'), 'IN': ('#235356', '#8fb3b5')}
OUT = './burst_concept/'
os.makedirs(OUT, exist_ok=True)


def parse(s, dur):
    if not isinstance(s, str) or s.strip() in ('', '[]'):
        return np.array([])
    x = np.atleast_1d(np.array(ast.literal_eval(s), float))
    return np.sort(x[(x >= 0) & (x < dur)])


def bursts(trains, tids, dur):
    """burst peak times per trial, trials concatenated"""
    allsp = np.concatenate([trains[t] + k * dur for k, t in enumerate(tids)])
    counts, _ = np.histogram(allsp, np.arange(0, len(tids) * dur + bin_size, bin_size))
    sm = np.convolve(counts, kernel, mode='same')
    pk, _ = signal.find_peaks(sm, prominence=np.percentile(sm, 90), distance=distance)
    t = (pk + 0.5) * bin_size
    return {tid: t[(t >= k * dur) & (t < (k + 1) * dur)] - k * dur for k, tid in enumerate(tids)}


def spikes_in_bursts(sp, pk, dur):
    m = np.zeros(int(round(dur * 1000)), bool)
    for a, b in np.clip(np.stack([pk - HW, pk + HW], 1), 0, dur) if len(pk) else []:
        m[int(a * 1000):int(b * 1000)] = True
    return int(m[np.minimum((sp * 1000).astype(int), len(m) - 1)].sum()) if sp.size else 0


def recruitment(pop, epoch, concept, pop_ids):
    path, dur = EPOCHS[epoch]
    rng = np.random.default_rng(4)
    d = pd.read_excel(path, usecols=['subject_id', 'trial_id', 'Neuron_ID_3', 'num_images_presented', 'Standardized_Spikes', 'Category'])
    d = d[d.num_images_presented.isin([1, 2, 3])].drop_duplicates(['subject_id', 'trial_id', 'Neuron_ID_3']).copy()
    d['sp'] = d.Standardized_Spikes.map(lambda s: parse(s, dur))
    rows = []
    for s, ds in d.groupby('subject_id'):
        cc = [n for n in ds.Neuron_ID_3.unique() if n in concept]
        p = ds[~ds.Neuron_ID_3.isin(concept) & ds.Neuron_ID_3.isin(pop_ids)]
        if not cc or p.Neuron_ID_3.nunique() < 3:
            continue
        tids = sorted(p.trial_id.unique())
        pk = bursts(p.groupby('trial_id').sp.apply(lambda v: np.concatenate(v.values)).to_dict(), tids, dur)
        load = ds.drop_duplicates('trial_id').set_index('trial_id').num_images_presented.to_dict()
        for c in cc:
            dc = ds[(ds.Neuron_ID_3 == c) & ds.Category.isin(['Preferred', 'Non-Preferred']) & ds.trial_id.isin(pk)].set_index('trial_id')
            res = {}
            for cat, key in (('Preferred', 'pref'), ('Non-Preferred', 'non')):
                tt = list(dc.index[dc.Category == cat])
                if len(tt) < 3:
                    continue
                count = lambda assign: sum(spikes_in_bursts(dc.loc[src, 'sp'], pk[t], dur) for t, src in assign)
                obs = count([(t, t) for t in tt])
                null = []
                for _ in range(N_NULL):
                    assign = []
                    for t in tt:
                        pool = [u for u in tt if u != t and load[u] == load[t]] or [u for u in tt if u != t]
                        assign.append((t, pool[rng.integers(len(pool))]))
                    null.append(count(assign))
                res[f'exc_{key}'] = np.log2((obs + 1) / (np.mean(null) + 1))
                res[f'n_in_{key}'], res[f'n_in_null_{key}'] = obs, np.mean(null)
            if len(res) == 6:
                rows.append(dict(bursts=pop, epoch=epoch, subject_id=s, cell=c, **res))
    print(pop, epoch, 'done', flush=True)
    return rows


def star(p):
    return '***' if p < 0.005 else '**' if p < 0.01 else '*' if p < 0.05 else ''


if __name__ == '__main__':
    wf = pd.read_excel('../Supplementary_Figures/revision_clustering_waveform_labels.xlsx')
    nw = pd.read_excel('../Supplementary_Figures/revision_clustering_no_waveform_labels.xlsx')
    concept = set(wf[wf.Signi == 'Y'].Neuron_ID_3)
    C = pd.DataFrame([r for pop in ('PY', 'IN') for e in EPOCHS
                      for r in recruitment(pop, e, concept, set(nw[nw.Cell_Type_New == pop].Neuron_ID_3))])
    C.to_csv(OUT + 'burst_concept_cells.csv', index=False)

    stats = []
    for (pop, e), x in C.groupby(['bursts', 'epoch'], sort=False):
        for c in ('exc_pref', 'exc_non'):
            w = wilcoxon(x[c])
            stats.append(dict(bursts=pop, epoch=e, test='preferred vs 0' if c == 'exc_pref' else 'nonpreferred vs 0', n=len(x),
                              n_participants=x.subject_id.nunique(), mean=x[c].mean(), W=w.statistic, p=w.pvalue))
    T = pd.DataFrame(stats)
    T['p_BH'] = multipletests(T.p, method='fdr_bh')[1]          # 8 tests
    for (pop, e), x in C.groupby(['bursts', 'epoch'], sort=False):
        w = wilcoxon(x.exc_pref, x.exc_non)
        T.loc[len(T)] = dict(bursts=pop, epoch=e, test='preferred vs nonpreferred (uncorrected)', n=len(x),
                             n_participants=x.subject_id.nunique(), mean=(x.exc_pref - x.exc_non).mean(), W=w.statistic, p=w.pvalue, p_BH=np.nan)
    T.to_csv(OUT + 'burst_concept_stats.csv', index=False)
    print(T.round(4).to_string(index=False))

    # figures: one panel per burst population
    for pop, letter, name in (('PY', 'f', 'pyramidal neurons'), ('IN', 'g', 'interneurons')):
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        D = C[C.bursts == pop]
        for ei, e in enumerate(EPOCHS):
            x = D[D.epoch == e]
            for j, (c, test) in enumerate((('exc_pref', 'preferred vs 0'), ('exc_non', 'nonpreferred vs 0'))):
                xp, v = ei * 2.5 + j, x[c].values
                ax.scatter(xp + np.random.default_rng(j).uniform(-0.15, 0.15, len(v)), v, s=7, color='0.55', alpha=0.45, lw=0)
                ax.errorbar(xp, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), fmt='o', color=COLORS[pop][j], mec='k', mew=0.5,
                            ms=7, lw=1.6, ecolor='k', capsize=3, zorder=3)
                q = T[(T.bursts == pop) & (T.epoch == e) & (T.test == test)].p_BH.iloc[0]
                ax.text(xp, 2.3, star(q), ha='center', va='bottom', fontsize=15)
            ax.text(ei * 2.5 + 0.5, -0.12, e, ha='center', va='top', fontsize=10, transform=ax.get_xaxis_transform())
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_xticks([0, 1, 2.5, 3.5]); ax.set_xticklabels(['Pref.', 'Nonpref.'] * 2, fontsize=10)
        ax.set_xlim(-0.5, 4); ax.set_ylim(-2.05, 2.75)
        n = D.drop_duplicates('cell')
        ax.set_title(f'Bursts defined by {name}\n({len(n)} concept cells, {n.subject_id.nunique()} participants)', fontsize=11)
        ax.set_ylabel('Concept-cell recruitment\ninto bursts (log$_2$ obs/exp)')
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT + f'S10{letter}_{pop}_bursts.eps', format='eps', bbox_inches='tight')
