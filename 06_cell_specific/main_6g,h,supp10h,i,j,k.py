"""
Co-Firing Index (CFI): burst count -> CFI and CFI -> behaviour (Fig 6i,j; Supp Fig 10h-k).

CFI, per neuron pair and trial: coincidences within +-30 ms, expected count = mean over 50 circular trial shifts,
CFI = log2((observed + c) / (expected + c)), c = 0.5 (0.1 and 1 for the sensitivity panel). Averaged over pairs per
trial and z-scored across trials within participant. Pairings: IN-IN and PY-PY (ACG-labelled neurons, >= 3 per
participant) and all neurons (>= 10 per participant). Epochs: fixation 0-0.9 s, encoding 0-1 s (first image),
maintenance 0-2.8 s. Encoding and maintenance use the trials shared by both epochs for each pair.

Models (participant random intercept and random slope of the focal predictor):
  Burst -> CFI (Supp 10h): CFI_z ~ burst_count_z + firing_rate_z (mixed model). Bursts counted on each trial
      separately: spikes of the pairing's neurons pooled, 30 ms bins, Gaussian smoothing (sigma 50 ms, 8-bin kernel),
      peaks with prominence above the 50th percentile of that trial's trace, no minimum separation.
  CFI -> RT (Fig 6i,j; Supp 10i,k): log RT ~ CFI_z + load_z + firing_rate_z (mixed model; correct trials).
  CFI -> accuracy (Supp 10j): accuracy ~ CFI_z + load_z + firing_rate_z (GEE, binomial, logit; cluster = participant).
Significance: focal predictor permuted across trials within participant (5000), two-sided,
P = (1 + #|null| >= |observed|) / (1 + #valid permutations). BH correction: burst -> CFI over 6 tests (3 pairings x
encoding, maintenance); behaviour over 9 tests per outcome and constant (3 pairings x fixation, encoding, maintenance).

Run from 06_cell_specific. The permutations take long (about 200,000 model fits); set N_JOBS to run models in parallel.
Outputs: cfi_<pairing>[_eps<c>].csv, firing_rates.csv, burst_counts.csv, burst_cfi_stats.csv,
cfi_behaviour_stats.csv, Fig6ij_CFI_RT.eps, FigS10hk_CFI.eps
"""
import ast
import os
import warnings
import zlib
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal.windows import gaussian
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.simplefilter('ignore')

# =========================
# Parameters
# =========================
COINC_WIN = 0.030                   # coincidence window (s)
N_SHIFTS = 50                       # trial shifts for the expected count
CONSTANTS = [0.5, 0.1, 1.0]         # c in log2((obs + c) / (exp + c)); 0.5 used throughout
DUR = {'fixation': 0.9, 'encoding': 1.0, 'maintenance': 2.8}
bin_size, sigma, BURST_PCT = 0.03, 0.05, 50   # per-trial burst counts
N_PERM = 5000
N_JOBS = 1                          # parallel model fits 

CATS = ['Labelled_IN', 'Labelled_PY', 'All_cells']
NAMES = {'Labelled_IN': 'IN', 'Labelled_PY': 'PY', 'All_cells': 'All cells'}
COLORS = {'Labelled_PY': '#b5283c', 'Labelled_IN': '#235356', 'All_cells': '#f2e6b1'}
EPOCHS = ['fixation', 'encoding', 'maintenance']
FILES = {'fixation': '../clean_data/cleaned_Fixation.xlsx', 'encoding': '../graph_data/graph_encoding1.xlsx',
         'maintenance': '../graph_data/graph_delay.xlsx'}
LABELS = '../Supplementary_Figures/'
TRIAL_INFO = '../data/trial_info.xlsx'
OUT = './'
kernel = gaussian(int(5 * sigma / bin_size), std=sigma / bin_size)   # 8 bins


def parse(x):
    if isinstance(x, np.ndarray):
        return x
    if pd.isna(x):
        return np.array([])
    t = str(x).strip()
    if t in ('', '[]', 'nan', 'None'):
        return np.array([])
    try:
        return np.atleast_1d(np.asarray(ast.literal_eval(t), float))
    except Exception:
        return np.asarray([float(v) for v in t.replace('[', '').replace(']', '').split(',') if v.strip()])


def clip(a, dur):
    return a[(a >= 0) & (a <= dur)]


def zwithin(d, col, out, ddof=0):
    d[out] = d.groupby('subject_id')[col].transform(lambda v: (v - v.mean()) / v.std(ddof=ddof) if v.std(ddof=ddof) > 0 else 0.0)
    return d


# =========================
# 1. CFI, firing rate and burst count per trial
# =========================
def coinc(a, b):
    return 0 if len(a) == 0 or len(b) == 0 else int(np.sum(np.abs(a[:, None] - b[None, :]) <= COINC_WIN))


def obs_exp(am, bm, trials):
    """observed and expected (mean over N_SHIFTS circular trial shifts) coincidences per trial"""
    n = len(trials)
    return np.array([(coinc(am[t], bm[t]), np.mean([coinc(am[t], bm[trials[(i + s) % n]]) for s in range(1, N_SHIFTS + 1)]))
                     for i, t in enumerate(trials)])


def burst_count(spikes, dur):
    if len(spikes) == 0:
        return 0
    counts, _ = np.histogram(spikes, np.arange(0.0, dur + bin_size, bin_size))
    sm_ = np.convolve(counts, kernel, mode='same')
    return len(signal.find_peaks(sm_, prominence=np.percentile(sm_, BURST_PCT))[0])


def compute_trial_measures():
    no_wf = pd.read_excel(LABELS + 'revision_clustering_no_waveform_labels.xlsx')
    wf = pd.read_excel(LABELS + 'revision_clustering_waveform_labels.xlsx')
    members = {'Labelled_IN': set(no_wf.loc[no_wf.Cell_Type_New == 'IN', 'Neuron_ID_3']),
               'Labelled_PY': set(no_wf.loc[no_wf.Cell_Type_New == 'PY', 'Neuron_ID_3']),
               'All_cells': set(wf.Neuron_ID_3)}
    data = {}
    for e, f in FILES.items():
        x = pd.read_excel(f, usecols=['subject_id', 'trial_id', 'Neuron_ID_3', 'Standardized_Spikes'])
        x['subject_id'], x['trial_id'] = x.subject_id.astype(int), x.trial_id.astype(int)
        data[e] = x
    subj_neurons = {}
    for e in ('encoding', 'maintenance'):
        for s, n in data[e][['subject_id', 'Neuron_ID_3']].drop_duplicates().itertuples(index=False):
            subj_neurons.setdefault(s, set()).add(n)

    fr_rows, burst_rows = [], []
    for cat in CATS:
        min_n = 10 if cat == 'All_cells' else 3
        subs = {s: sorted(n for n in members[cat] if n in subj_neurons[s]) for s in sorted(subj_neurons)}
        subs = {s: v for s, v in subs.items() if len(v) >= min_n}
        cache = {e: {} for e in EPOCHS}
        for e in EPOCHS:
            x = data[e][data[e].Neuron_ID_3.isin(set().union(*subs.values()))]
            for (n, t), g in x.groupby(['Neuron_ID_3', 'trial_id']):
                cache[e].setdefault(int(n), {})[int(t)] = parse(g.Standardized_Spikes.iloc[0])
            for s, ids in subs.items():
                xs = x[(x.subject_id == s) & x.Neuron_ID_3.isin(ids)]
                nn = xs.Neuron_ID_3.nunique()
                for t, g in xs.groupby('trial_id'):
                    sp = np.concatenate([clip(parse(v), DUR[e]) for v in g.Standardized_Spikes])
                    fr_rows.append(dict(category=cat, epoch=e, subject_id=s, trial_id=int(t), firing_rate_hz=len(sp) / (nn * DUR[e])))
                    if e != 'fixation':
                        burst_rows.append(dict(category=cat, epoch=e, subject_id=s, trial_id=int(t), burst_count=burst_count(sp, DUR[e])))
        rows = []
        for s, ids in subs.items():
            for a, b in combinations(ids, 2):
                ae, be, am, bm = (cache[e].get(k, {}) for e, k in (('encoding', a), ('encoding', b), ('maintenance', a), ('maintenance', b)))
                shared = sorted(set(ae) & set(be) & set(am) & set(bm))
                if len(shared) >= 2:
                    for t, ce, cm in zip(shared, obs_exp(ae, be, shared), obs_exp(am, bm, shared)):
                        rows += [(s, t, 'encoding', *ce), (s, t, 'maintenance', *cm)]
                af, bf = cache['fixation'].get(a, {}), cache['fixation'].get(b, {})
                shared = sorted(set(af) & set(bf))
                if len(shared) >= 2:
                    for t, cf in zip(shared, obs_exp(af, bf, shared)):
                        rows.append((s, t, 'fixation', *cf))
        P = pd.DataFrame(rows, columns=['subject_id', 'trial_id', 'epoch', 'obs', 'exp'])
        for c in CONSTANTS:
            P['cfi'] = np.log2((P.obs + c) / (P.exp + c))
            T = P.groupby(['subject_id', 'trial_id', 'epoch']).cfi.mean().reset_index()
            T['cfi_z'] = T.groupby(['subject_id', 'epoch']).cfi.transform(lambda v: (v - v.mean()) / v.std(ddof=1))
            T.to_csv(OUT + f'cfi_{cat}' + ('' if c == 0.5 else f'_eps{c:g}') + '.csv', index=False)
        print(f'{cat}: {len(subs)} participants', flush=True)
    pd.DataFrame(fr_rows).to_csv(OUT + 'firing_rates.csv', index=False)
    pd.DataFrame(burst_rows).to_csv(OUT + 'burst_counts.csv', index=False)


# =========================
# 2. Models with within-participant permutation tests
# =========================
def fit_mixed(d, y, focal, covs):
    f = f'{y} ~ {focal}' + ''.join(f' + {c}' for c in covs)
    return smf.mixedlm(f, d, groups=d.subject_id, re_formula=f'~{focal}').fit(reml=True, method='lbfgs')


def fit_gee(d, y, focal, covs):
    f = f'{y} ~ {focal}' + ''.join(f' + {c}' for c in covs)
    return smf.gee(f, 'subject_id', d, family=sm.families.Binomial()).fit()


def permutation_p(d, fitter, y, focal, covs, obs, rng, cumulative):
    """cumulative=True re-permutes the previous permutation (burst models), False permutes the original order each time"""
    null, dd = [], d.copy()
    for _ in range(N_PERM):
        if not cumulative:
            dd = d.copy()
        dd[focal] = dd.groupby('subject_id')[focal].transform(lambda v: rng.permutation(v.values))
        try:
            null.append(fitter(dd, y, focal, covs).params[focal])
        except Exception:
            pass
    null = np.array(null)
    return (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1), len(null)


def burst_model(cat, epoch):
    B, FR = pd.read_csv(OUT + 'burst_counts.csv'), pd.read_csv(OUT + 'firing_rates.csv')
    C = pd.read_csv(OUT + f'cfi_{cat}.csv')
    key = ['subject_id', 'trial_id']
    d = (B[(B.category == cat) & (B.epoch == epoch)][key + ['burst_count']]
         .merge(FR[(FR.category == cat) & (FR.epoch == epoch)][key + ['firing_rate_hz']], on=key)
         .merge(C[C.epoch == epoch][key + ['cfi_z']], on=key))
    d = zwithin(zwithin(d, 'burst_count', 'burst_count_z'), 'firing_rate_hz', 'firing_rate_z')
    d = d.dropna(subset=['burst_count_z', 'firing_rate_z', 'cfi_z']).reset_index(drop=True)
    r = fit_mixed(d, 'cfi_z', 'burst_count_z', ['firing_rate_z'])
    b = r.params.burst_count_z
    p, n_valid = permutation_p(d, fit_mixed, 'cfi_z', 'burst_count_z', ['firing_rate_z'], b, np.random.default_rng(0), cumulative=True)
    return dict(category=cat, epoch=epoch, n_subjects=d.subject_id.nunique(), n_trials=len(d), coef=b, se=r.bse.burst_count_z,
                wald_p=r.pvalues.burst_count_z, perm_p=p, n_perm=n_valid)


def behaviour_model(outcome, cat, epoch, c):
    suffix = '' if c == 0.5 else f'_eps{c:g}'
    ti = pd.read_excel(TRIAL_INFO)[['subject_id', 'trial_id', 'RT', 'response_accuracy', 'num_images_presented']].drop_duplicates()
    FR = pd.read_csv(OUT + 'firing_rates.csv')
    C = pd.read_csv(OUT + f'cfi_{cat}{suffix}.csv')
    d = C[C.epoch == epoch].merge(ti, on=['subject_id', 'trial_id'])
    if outcome == 'rt':
        d = d[(d.response_accuracy == 1) & (d.RT > 0)].copy()
        d['log_rt'] = np.log(d.RT)
    d = zwithin(d.copy(), 'num_images_presented', 'load_z')
    d = d.merge(FR[(FR.category == cat) & (FR.epoch == epoch)][['subject_id', 'trial_id', 'firing_rate_hz']], on=['subject_id', 'trial_id'])
    d = zwithin(d, 'firing_rate_hz', 'fr_z')
    y, fitter, covs = ('log_rt', fit_mixed, ['load_z', 'fr_z']) if outcome == 'rt' else ('response_accuracy', fit_gee, ['load_z', 'fr_z'])
    d = d.dropna(subset=['cfi_z', y] + covs).reset_index(drop=True)
    r = fitter(d, y, 'cfi_z', covs)
    b = r.params.cfi_z
    rng = np.random.default_rng(zlib.crc32(f'fr{suffix}|{outcome}|{cat}|{epoch}'.encode()))
    p, n_valid = permutation_p(d, fitter, y, 'cfi_z', covs, b, rng, cumulative=False)
    return dict(outcome=outcome, eps=c, category=cat, epoch=epoch, n_subjects=d.subject_id.nunique(), n_trials=len(d),
                coef=b, se=r.bse.cfi_z, wald_p=r.pvalues.cfi_z, perm_p=p, n_perm=n_valid)


def run(job):
    return burst_model(*job[1:]) if job[0] == 'burst' else behaviour_model(*job[1:])
  
# 3. Figures
def star(p):
    return '***' if p < 0.005 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def bars(ax, S, epoch, title, ylabel=''):
    """Fig 6i,j style: one bar per pairing, filled when BH-significant"""
    for i, cat in enumerate(['Labelled_PY', 'Labelled_IN', 'All_cells']):
        r = S[(S.category == cat) & (S.epoch == epoch)].iloc[0]
        ax.bar(i, r.coef, 0.7, color=COLORS[cat], edgecolor='k', lw=0.8, label=NAMES[cat])
        ax.text(i, r.coef + (0.001 if r.coef >= 0 else -0.001), star(r.p_BH), ha='center', va='bottom' if r.coef >= 0 else 'top', fontsize=14)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks([1]); ax.set_xticklabels([epoch.capitalize()])
    ax.set_title(title); ax.set_ylabel(ylabel)


def points(ax, S, epochs, constants=(0.5,)):
    """Supp 10 style: coefficient +- SE per pairing and epoch; filled markers when BH-significant"""
    marker = {0.5: 'o', 0.1: 'v', 1.0: '^'}
    for gi, cat in enumerate(['Labelled_IN', 'Labelled_PY', 'All_cells']):
        for j, e in enumerate(epochs):
            for k, c in enumerate(constants):
                r = S[(S.category == cat) & (S.epoch == e) & (S.eps == c)].iloc[0] if 'eps' in S else S[(S.category == cat) & (S.epoch == e)].iloc[0]
                x = j + (gi - 1) * 0.25 + (k - (len(constants) - 1) / 2) * 0.07
                col = 'k' if cat == 'All_cells' else COLORS[cat]
                ax.errorbar(x, r.coef, yerr=r.se, fmt=marker[c], color=col, capsize=2, mfc=col if r.p_BH < 0.05 else 'white',
                            label=NAMES[cat] if (j == 0 and k == 0) else None)
                ax.text(x, r.coef + np.sign(r.coef or 1) * (r.se + 0.003), star(r.p_BH), ha='center', va='bottom' if r.coef >= 0 else 'top', fontsize=9, color=col)
    ax.axhline(0, color='k', lw=0.7)
    ax.set_xticks(range(len(epochs))); ax.set_xticklabels([e.capitalize() for e in epochs])


def figures():
    B = pd.read_csv(OUT + 'burst_cfi_stats.csv')
    S = pd.read_csv(OUT + 'cfi_behaviour_stats.csv')
    rt = S[(S.outcome == 'rt') & (S.eps == 0.5)]

    # Fig 6i,j: CFI -> RT, encoding and maintenance
    fig, axs = plt.subplots(1, 2, figsize=(6, 3.5), sharey=True)
    bars(axs[0], rt, 'encoding', 'CFI vs RT', r'$\beta$ coefficient (log RT)')
    bars(axs[1], rt, 'maintenance', 'CFI vs RT')
    axs[0].legend(frameon=False, loc='lower left')
    fig.tight_layout(); fig.savefig(OUT + 'Fig6ij_CFI_RT.eps', format='eps'); plt.close(fig)

    # Supp 10h-k: burst -> CFI; CFI -> RT and accuracy (fixation, encoding, maintenance); constant sensitivity (RT)
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    points(axs[0], B.assign(eps=0.5), ['encoding', 'maintenance'])
    axs[0].set_title('h  Burst count vs CFI'); axs[0].set_ylabel(r'$\beta$ (CFI s.d. per burst-count s.d.)')
    points(axs[1], rt, EPOCHS); axs[1].set_title('i  CFI vs reaction time'); axs[1].set_ylabel(r'$\beta$ (log RT)')
    points(axs[2], S[(S.outcome == 'accuracy') & (S.eps == 0.5)], EPOCHS); axs[2].set_title('j  CFI vs accuracy'); axs[2].set_ylabel(r'$\beta$ (log-odds)')
    points(axs[3], S[S.outcome == 'rt'], EPOCHS, constants=(0.1, 0.5, 1.0)); axs[3].set_title('k  Sensitivity to the constant (0.1, 0.5, 1)')
    axs[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT + 'FigS10hk_CFI.eps', format='eps'); plt.close(fig)


if __name__ == '__main__':
    compute_trial_measures()

    jobs = ([('burst', cat, e) for cat in CATS for e in ('encoding', 'maintenance')]
            + [('behaviour', 'rt', cat, e, c) for c in CONSTANTS for cat in CATS for e in EPOCHS]
            + [('behaviour', 'accuracy', cat, e, 0.5) for cat in CATS for e in EPOCHS])
    if N_JOBS > 1:
        with ProcessPoolExecutor(N_JOBS) as ex:
            results = list(ex.map(run, jobs))
    else:
        results = [run(j) for j in jobs]
    for j, r in zip(jobs, results):
        print(j, f"coef={r['coef']:.4f} perm_p={r['perm_p']:.4f}", flush=True)

    B = pd.DataFrame([r for j, r in zip(jobs, results) if j[0] == 'burst'])
    B['p_BH'] = multipletests(B.perm_p, method='fdr_bh')[1]                 # 6 tests
    B.to_csv(OUT + 'burst_cfi_stats.csv', index=False)
    S = pd.DataFrame([r for j, r in zip(jobs, results) if j[0] == 'behaviour'])
    S['p_BH'] = S.groupby(['outcome', 'eps']).perm_p.transform(lambda p: multipletests(p, method='fdr_bh')[1])   # 9 tests each
    S.to_csv(OUT + 'cfi_behaviour_stats.csv', index=False)
    print(B.round(4).to_string(), S.round(4).to_string(), sep='\n\n')

    figures()
