"""
Components of burst excess: single-neuron structure vs alignment across neurons (Supp Fig 10e).

Maintenance 0-2.5 s, load 1. PY and IN: ACG-labelled neurons (no-waveform labels), participants with >= 3 of the type.
Bursts as in Fig 4: per participant, the group's spikes pooled, trials concatenated, 30 ms bins, Gaussian smoothing
(sigma 50 ms, 8-bin kernel), peaks with prominence above the 90th percentile of the real trace, >= 140 ms apart. The same
threshold is used for the surrogates. 100 surrogates per participant, mean count:
  Poisson       : per neuron and trial, count ~ Poisson(real count), times uniform in the window
  PSTH          : as Poisson, times drawn from the neuron's trial-averaged PSTH (10 ms bins, Gaussian sigma 30 ms)
  trial shuffle : each neuron's real spike train moved to another trial (independent permutation per neuron)
  (jitter +-50 ms is drawn as well, so the random sequence matches the published run; it is not used here)
Components: single-neuron = log2((shuffle + 1) / (Poisson + 1)); alignment = log2((real + 1) / (shuffle + 1)).
Tests: one-sided paired Wilcoxon (shuffle > Poisson; real > shuffle), BH across the 4 tests (PY, IN x 2 components);
shuffle > PSTH (one-sided paired, BH across PY, IN); IN vs PY per component, two-sided paired Wilcoxon over participants
with both types.
"""
import ast
import hashlib
import os

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal.windows import gaussian
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =========================
# Parameters
# =========================
WIN, LOAD, N_SURR = 2.5, 1, 100
bin_size, sigma = 0.03, 0.05
kernel = gaussian(int(5 * sigma / bin_size), std=sigma / bin_size)   # 8 bins
distance = int(np.ceil(0.14 / bin_size))                              # 5 bins = 140 ms
MIN_CELLS = 3
EPOCH = 'Maintenance_0-2.5s'          # part of the surrogate seed
GROUPS = {'Pyramidal': 'PY', 'Interneurons': 'IN'}
COLORS = {'PY': '#b5283c', 'IN': '#235356'}
OUT = './burst_components/'
os.makedirs(OUT, exist_ok=True)


def seed(*p):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, p)).encode()).digest()[:8], 'little') % (2 ** 32 - 1)


def parse(s):
    if not isinstance(s, str) or s.strip() in ('', '[]'):
        return np.array([])
    x = np.atleast_1d(np.array(ast.literal_eval(s), float))
    return x[(x >= 0) & (x < WIN)]


def detect(trains_by_trial, thr=None):
    """trains_by_trial: trials x neurons spike arrays. Returns (burst count, prominence threshold)."""
    n = len(trains_by_trial)
    allsp = np.concatenate([np.concatenate(tr) + k * WIN for k, tr in enumerate(trains_by_trial)])
    counts, _ = np.histogram(allsp, np.arange(0, n * WIN + bin_size, bin_size))
    sm = np.convolve(counts, kernel, mode='same')
    if thr is None:
        thr = np.percentile(sm, 90)
    return len(signal.find_peaks(sm, prominence=thr, distance=distance)[0]), thr


def surrogate_counts(M, thr, rng):
    """mean burst count over N_SURR surrogates (Poisson, PSTH, trial shuffle); M = trials x neurons"""
    n_tr, n_neu = len(M), len(M[0])
    edges = np.arange(0, WIN + 0.01, 0.01)
    psth = []
    for j in range(n_neu):
        h = gaussian_filter1d(np.histogram(np.concatenate([M[i][j] for i in range(n_tr)]), edges)[0].astype(float), 3) + 1e-9
        psth.append(h / h.sum())
    out = {'poisson': [], 'psth': [], 'shuffle': []}
    for _ in range(N_SURR):
        s1, s2 = [], []
        for i in range(n_tr):
            a1, a2 = [], []
            for j in range(n_neu):
                c = rng.poisson(M[i][j].size)
                a1.append(rng.uniform(0, WIN, c))
                k = rng.choice(len(psth[j]), c, p=psth[j])
                a2.append(edges[k] + rng.uniform(0, 0.01, c))
                rng.uniform(-0.05, 0.05, M[i][j].size)   # jitter surrogate of the published run (unused)
            s1.append(a1); s2.append(a2)
        perm = [rng.permutation(n_tr) for _ in range(n_neu)]
        s3 = [[M[perm[j][i]][j] for j in range(n_neu)] for i in range(n_tr)]
        for k, tr in (('poisson', s1), ('psth', s2), ('shuffle', s3)):
            out[k].append(detect(tr, thr)[0])
    return {k: np.mean(v) for k, v in out.items()}


def star(p):
    return '***' if p < 0.005 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


if __name__ == '__main__':
    labels = pd.read_excel('../Supplementary_Figures/revision_clustering_no_waveform_labels.xlsx')
    d = pd.read_excel('../graph_data/graph_delay.xlsx', usecols=['subject_id', 'trial_id', 'Neuron_ID_3', 'num_images_presented', 'Standardized_Spikes'])
    d = d[d.num_images_presented == LOAD].drop_duplicates(['trial_id', 'Neuron_ID_3', 'subject_id'])
    d['sp'] = d.Standardized_Spikes.map(parse)

    rows = []
    for group, typ in GROUPS.items():
        for s, ms in labels[labels.Cell_Type_New == typ].groupby('subject_id'):
            if len(ms) < MIN_CELLS:
                continue
            ds = d[(d.subject_id == s) & d.Neuron_ID_3.isin(set(ms.Neuron_ID_3))]
            neu, tri = sorted(ds.Neuron_ID_3.unique()), sorted(ds.trial_id.unique())
            if len(neu) < MIN_CELLS or len(tri) < 5:
                continue
            S = {(r.trial_id, r.Neuron_ID_3): r.sp for r in ds.itertuples()}
            M = [[S.get((t, n), np.array([])) for n in neu] for t in tri]
            real, thr = detect(M)
            surr = surrogate_counts(M, thr, np.random.default_rng(seed(EPOCH, group, s, 'no_waveform')))
            rows.append(dict(cell_type=typ, subject_id=s, n_neurons=len(neu), n_trials=len(tri), real=real, **surr))
            print(typ, s, 'done', flush=True)
    R = pd.DataFrame(rows)
    R['single_neuron'] = np.log2((R.shuffle + 1) / (R.poisson + 1))
    R['alignment'] = np.log2((R.real + 1) / (R.shuffle + 1))
    R.to_csv(OUT + 'burst_components_per_subject.csv', index=False)

    # statistics
    stats = []
    for typ in ('PY', 'IN'):
        x = R[R.cell_type == typ]
        for comp, (a, b) in (('single_neuron', ('shuffle', 'poisson')), ('alignment', ('real', 'shuffle'))):
            w = wilcoxon(x[a], x[b], alternative='greater')
            stats.append(dict(test=f'{typ} {comp} ({a} > {b})', family='components', n=len(x), W=w.statistic, p=w.pvalue))
    for typ in ('PY', 'IN'):
        x = R[R.cell_type == typ]
        w = wilcoxon(x.shuffle, x.psth, alternative='greater')
        stats.append(dict(test=f'{typ} shuffle > PSTH', family='psth', n=len(x), W=w.statistic, p=w.pvalue))
    T = pd.DataFrame(stats)
    T['p_BH'] = T.groupby('family').p.transform(lambda p: multipletests(p, method='fdr_bh')[1])
    both = R.pivot(index='subject_id', columns='cell_type', values=['single_neuron', 'alignment']).dropna()
    for comp in ('alignment', 'single_neuron'):
        w = wilcoxon(both[comp].IN, both[comp].PY)
        T.loc[len(T)] = dict(test=f'IN vs PY {comp} (two-sided)', family='IN vs PY', n=len(both), W=w.statistic, p=w.pvalue, p_BH=np.nan)
    T.to_csv(OUT + 'burst_components_stats.csv', index=False)
    print(T.round(5).to_string(index=False))

    # figure (S10e)
    q = {r.test.split(' (')[0]: r.p_BH for r in T[T.family == 'components'].itertuples()}
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for j, typ in enumerate(('PY', 'IN')):
        x = R[R.cell_type == typ]
        for off, comp, alpha, hatch in ((-0.18, 'single_neuron', 0.45, '///'), (0.18, 'alignment', 0.9, None)):
            v = x[comp].values
            m, e = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
            ax.bar(j + off, m, 0.32, color=COLORS[typ], alpha=alpha, hatch=hatch, edgecolor='w' if hatch else None, lw=0)
            ax.scatter(j + off + rng.uniform(-0.1, 0.1, len(v)), v, s=7, color='0.55', alpha=0.6, lw=0, zorder=2)
            ax.errorbar(j + off, m, e, color='k', lw=1.4, capsize=3, zorder=3)
            ax.text(j + off, m + e + 0.05, star(q[f'{typ} {comp}']), ha='center', fontsize=10)
        ax.text(j, -0.24, f'{typ} (n = {len(x)})', ha='center', fontsize=9, fontweight='bold', color=COLORS[typ], transform=ax.get_xaxis_transform())
    p_al = T[T.test == 'IN vs PY alignment (two-sided)'].p.iloc[0]
    ax.plot([0.18, 0.18, 1.18, 1.18], [1.15, 1.2, 1.2, 1.15], color='k', lw=0.8)
    ax.text(0.68, 1.22, star(p_al), ha='center', va='bottom', fontsize=10)
    ax.axhline(0, color='k', lw=0.7)
    ax.set_xticks([-0.18, 0.18, 0.82, 1.18]); ax.set_xticklabels(['Single-\nneuron', 'Align-\nment'] * 2, fontsize=8)
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(-0.4, 2.45)
    ax.set_ylabel('Component of burst excess (log$_2$)')
    ax.set_title('Components of burst excess (maintenance, load 1)', fontsize=9.5)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT + 'S10e_components.eps', format='eps', bbox_inches='tight')
