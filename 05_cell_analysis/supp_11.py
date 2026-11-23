"""
Single-neuron two-state dynamics (Supp Fig 11a-m): one- vs two-state Poisson HMMs and the link between HMM high states
and population bursts.

HMM fits (as in the R2 HMM.py), all neurons with >= 20 trials, 30 ms bins:
  encoding  : first image, 0-1 s (33 bins). Parameters from a 2-state fit to all trials; two-state preference = 5-fold
              held-out delta log-likelihood per bin (2-state - 1-state).
  maintenance: parameters from one fit to 1-2 s; preference = 5-fold held-out delta LL pooled over 5 resampled 1 s windows
              (start times 0-1.5 s). Firing rate (>= 1 Hz rule, rate matching) = spikes in 0-2.5 s / (trials x 2.5 s).
Exclusions: degenerate fits (both self-transitions >= 0.999; one saturated with the other state's rate at the 1e-3 Hz
floor), and rate < 1 Hz (all panels except a, b, where only degenerate fits are excluded).
Classes: persistent if delta LL <= 0, high/low rate ratio < 1.5 or only the high state saturated; intermittent if the
high-state dwell time < 0.3 s; otherwise hybrid.
Burst link (c): per participant (>= 6 neurons), population bursts from all OTHER neurons (leave-one-out; Fig 4 detection:
30 ms bins, sigma 50 ms, prominence > 90th percentile, >= 140 ms apart; all loads, trials concatenated). Posterior
P(high state) per 30 ms bin from the neuron's fit (encoding 0-1 s, maintenance 1-2 s); per trial, inside minus outside
+-50 ms burst windows, averaged over trials, minus a null with burst times from another trial of the same load (20 draws).
Statistics: a, b: none (descriptive); c: Wilcoxon signed-rank vs 0; d-g: Wilcoxon signed-rank on delta LL; h-k:
chi-square vs uniform; l, m: rate-matched pairs (each PY matched 1:1 to the IN with the nearest log rate, without
replacement, |difference| <= 0.25), Fisher's exact test per class, BH across the 3 classes.
The fits take about an hour on one core; set N_JOBS to fit neurons in parallel (results do not depend on it). Fits and
burst links are saved and reused on later runs.
Run from 05_cell_analysis. Outputs (./HMM): hmm_fits_encoding.csv, hmm_fits_maintenance.csv, hmm_burst_link_<epoch>.csv,
S11_stats.csv, S11a ... S11m panels (.eps)
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
import ast
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal.windows import gaussian
from scipy.stats import poisson, wilcoxon, chisquare, fisher_exact
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =========================
# Parameters
# =========================
BIN = 0.03
EDGES = np.arange(0.0, 1.0 + BIN, BIN)[:34]           # 33 bins, 0-0.99 s
K_FOLDS, N_RESAMPLES = 5, 5
VALID_STARTS = np.round(np.arange(0.0, 1.5 + 1e-9, BIN), 4)
NEAR_FLOOR = 1e-3
RATE_WIN = 2.5
bin_size, sigma, HW, N_NULL = 0.03, 0.05, 0.05, 20
kernel = gaussian(int(5 * sigma / bin_size), std=sigma / bin_size)
distance = int(np.ceil(0.14 / bin_size))
N_JOBS = int(os.environ.get('N_JOBS', 1))
CL = ['persistent', 'hybrid', 'intermittent']
ALLC, CONC, ONE, TWO = '#b56b8f', '#4fb3aa', '#8fa58f', '#8a4652'
CLS = {'persistent': '#b07a86', 'hybrid': '#f7cdc4', 'intermittent': '#dcc3ca'}
PYC, INC = '#c8323c', '#1d4a5a'
EC = dict(edgecolor='k', linewidth=0.9)
FILES = {'encoding': '../graph_data/graph_encoding1.xlsx', 'maintenance': '../graph_data/graph_delay.xlsx'}
OUT = os.environ.get('HMM_OUT', './HMM/')
os.makedirs(OUT, exist_ok=True)


def parse(x, lo, hi, right_closed=False):
    if pd.isna(x) or not isinstance(x, str) or x.strip() in ('', '[]'):
        return np.array([])
    try:
        a = np.atleast_1d(np.array(ast.literal_eval(x.strip()), float))
    except Exception:
        return np.array([])
    return a[(a >= lo) & ((a <= hi) if right_closed else (a < hi))]


# =========================
# 1. Poisson HMM
# =========================
def logpmf(y, lam):
    return poisson.logpmf(y, np.clip(lam * BIN, 1e-12, None))


def forward(Y, lam, A, pi):
    logB = np.stack([logpmf(Y, lam[0]), logpmf(Y, lam[1])], -1)
    B, Bmax = np.exp(logB - logB.max(-1, keepdims=True)), logB.max(-1)
    n, T = Y.shape
    al, c = np.zeros((n, T, 2)), np.zeros((n, T))
    a0 = pi[None] * B[:, 0]
    c[:, 0] = np.clip(a0.sum(1), 1e-300, None); al[:, 0] = a0 / c[:, 0, None]
    for t in range(1, T):
        a = (al[:, t - 1] @ A) * B[:, t]
        c[:, t] = np.clip(a.sum(1), 1e-300, None); al[:, t] = a / c[:, t, None]
    return B, al, c, float((np.log(c) + Bmax).sum())


def fit_2state(Y, n_restarts=1, max_iter=150, tol=1e-4, seed=0):
    """EM for a 2-state Poisson HMM on a trials x bins count matrix; states sorted low to high rate"""
    rng = np.random.default_rng(seed)
    rate = Y.mean(1) / BIN
    best = None
    for r in range(n_restarts):
        if r == 0:
            lam, A = np.array([np.percentile(rate, 25), np.percentile(rate, 75)]), np.array([[0.9, 0.1], [0.1, 0.9]])
        elif r == 1:
            lam, A = np.array([np.percentile(rate, 10), np.percentile(rate, 90)]), np.array([[0.97, 0.03], [0.03, 0.97]])
        else:
            lam, A = np.sort(max(rate.mean(), 1e-3) * rng.uniform(0.3, 2.5, 2)), rng.dirichlet([8, 1], 2)
        lam = np.clip(lam, 1e-3, None)
        if lam[0] == lam[1]:
            lam[1] += 1e-3
        pi, prev, it = np.array([0.5, 0.5]), -np.inf, 0
        for it in range(max_iter):
            B, al, c, ll = forward(Y, lam, A, pi)
            n, T = Y.shape
            be = np.zeros((n, T, 2)); be[:, T - 1] = 1.0
            for t in range(T - 2, -1, -1):
                be[:, t] = ((B[:, t + 1] * be[:, t + 1]) @ A.T) / c[:, t + 1, None]
            g = al * be; g /= g.sum(-1, keepdims=True)
            xi = np.zeros((2, 2))
            for t in range(T - 1):
                xi += (al[:, t, :, None] * A[None] * B[:, t + 1, None, :] * be[:, t + 1, None, :] / c[:, t + 1, None, None]).sum(0)
            pi = g[:, 0].mean(0)
            A = xi / np.clip(g[:, :-1].sum((0, 1))[:, None], 1e-300, None); A /= A.sum(1, keepdims=True)
            lam = np.clip((g * Y[:, :, None]).sum((0, 1)) / np.clip(g.sum((0, 1)), 1e-300, None) / BIN, 1e-4, None)
            if ll - prev < tol and it > 3:
                prev = ll
                break
            prev = ll
        if best is None or prev > best['ll']:
            o = np.argsort(lam)
            best = dict(lam=lam[o], A=A[np.ix_(o, o)], pi=pi[o], ll=prev)
    return best


def kfold_delta(Y, seed):
    """5-fold held-out log-likelihood, 2-state minus 1-state: returns (ll1, ll2, bins)"""
    idx = np.random.default_rng(seed).permutation(Y.shape[0])
    t1 = t2 = 0.0; nb = 0
    for fi, te in enumerate(np.array_split(idx, K_FOLDS)):
        tr = np.setdiff1d(idx, te, assume_unique=True)
        if len(te) == 0 or len(tr) < 5:
            continue
        lam1 = max(Y[tr].mean() / BIN, 1e-4)
        f2 = fit_2state(Y[tr], seed=seed + fi + 1)
        t1 += float(logpmf(Y[te], lam1).sum()); t2 += forward(Y[te], f2['lam'], f2['A'], f2['pi'])[3]; nb += Y[te].size
    return t1, t2, nb


def binned(trains, lo, hi, edges, right_closed=False):
    return np.array([np.histogram(parse(x, lo, hi, right_closed), edges)[0] for x in trains], int)


def fit_neuron(job):
    """job = (epoch, neuron, subject, spike strings sorted by trial)"""
    epoch, nid, sid, trains = job
    seed = (int(nid) * 2654435761) % (2 ** 32 - 1)
    if epoch == 'encoding':
        Y = binned(trains, 0.0, 1.0, EDGES, right_closed=True)
        l1, l2, nb = kfold_delta(Y, seed)
    else:
        Y = binned(trains, 1.0, 2.0, EDGES + 1.0)
        starts = np.random.default_rng(seed).choice(VALID_STARTS, size=N_RESAMPLES, replace=False)
        l1 = l2 = 0.0; nb = 0
        for ri, s in enumerate(starts):
            a, b, n = kfold_delta(binned(trains, float(s), float(s) + 1.0, np.linspace(float(s), float(s) + 1.0, 34)), seed + ri * 1000)
            l1 += a; l2 += b; nb += n
    f = fit_2state(Y, n_restarts=2, seed=seed + 1)
    return dict(neuron_id=int(nid), subject_id=int(sid), n_trials=Y.shape[0], total_spikes=int(Y.sum()),
                delta_ll_per_bin=(l2 - l1) / nb if nb else np.nan, lam_low_hz=f['lam'][0], lam_high_hz=f['lam'][1],
                rate_ratio=f['lam'][1] / max(f['lam'][0], 1e-6), dwell_high_s=BIN / max(1 - f['A'][1, 1], 1e-6),
                A_00=f['A'][0, 0], A_11=f['A'][1, 1], overall_rate_hz=Y.mean() / BIN)


def fit_all(epoch, data):
    f = OUT + f'hmm_fits_{epoch}.csv'
    if os.path.exists(f):
        return pd.read_csv(f)
    jobs = [(epoch, n, g.subject_id.iloc[0], list(g.sort_values('trial_id').Standardized_Spikes))
            for n, g in data.groupby('Neuron_ID_3') if len(g) >= 20]
    if N_JOBS > 1:
        with ProcessPoolExecutor(N_JOBS, mp_context=mp.get_context('spawn')) as ex:
            rows = list(ex.map(fit_neuron, jobs, chunksize=4))
    else:
        rows = [fit_neuron(j) for j in jobs]
    R = pd.DataFrame(rows)
    R.to_csv(f, index=False)
    print(epoch, len(R), 'neurons fitted', flush=True)
    return R


def derive(d, concept, py, in_):
    d = d.copy()
    a00, a11 = d.A_00 >= 0.999, d.A_11 >= 0.999
    lf, hf = d.lam_low_hz <= NEAR_FLOOR, d.lam_high_hz <= NEAR_FLOOR
    d['excl'] = (a00 & a11) | (a11 & ~a00 & lf) | (a00 & ~a11 & hf)
    fp = a11 & ~a00 & ~lf & ~d.excl
    d['clean_active'] = ~d.excl & (d.overall_rate_hz >= 1.0)
    d['fav2'] = d.delta_ll_per_bin.notna() & (d.delta_ll_per_bin > 0)
    d['cls'] = [np.nan if pd.isna(r.delta_ll_per_bin) or pd.isna(r.rate_ratio) else
                'persistent' if f_ or r.delta_ll_per_bin <= 0 or r.rate_ratio < 1.5 else
                'intermittent' if r.dwell_high_s < 0.3 else 'hybrid' for r, f_ in zip(d.itertuples(), fp)]
    d['lograte'] = np.log(d.overall_rate_hz.clip(lower=1e-3))
    d['concept'], d['PY'], d['IN'] = d.neuron_id.isin(concept), d.neuron_id.isin(py), d.neuron_id.isin(in_)
    return d


# =========================
# 2. HMM high state vs population bursts
# =========================
def posterior_high(y, lam, A):
    B = np.stack([poisson.pmf(y, lam[0] * BIN), poisson.pmf(y, lam[1] * BIN)], 1) + 1e-300
    pi = np.array([1 - A[1, 1], 1 - A[0, 0]]); pi = pi / pi.sum() if pi.sum() > 0 else np.array([.5, .5])
    T = len(y); al = np.zeros((T, 2)); be = np.ones((T, 2)); c = np.zeros(T)
    al[0] = pi * B[0]; c[0] = al[0].sum(); al[0] /= c[0]
    for t in range(1, T):
        al[t] = (al[t - 1] @ A) * B[t]; c[t] = al[t].sum(); al[t] /= c[t]
    for t in range(T - 2, -1, -1):
        be[t] = (A @ (B[t + 1] * be[t + 1])) / c[t + 1]
    g = al * be
    return (g / g.sum(1, keepdims=True))[:, 1]


def burst_link(epoch, fits, data):
    f = OUT + f'hmm_burst_link_{epoch}.csv'
    if os.path.exists(f):
        return pd.read_csv(f)
    dur, w0 = (1.0, 0.0) if epoch == 'encoding' else (2.5, 1.0)
    fits = fits[~fits.excl & (fits.overall_rate_hz >= 1.0) & (fits.lam_high_hz > fits.lam_low_hz)].set_index('neuron_id')
    rng = np.random.default_rng(12)
    dl = data.copy()
    dl['sp'] = dl.Standardized_Spikes.map(lambda s: np.sort(parse(s, 0.0, dur)))
    ctr = (np.arange(33) + 0.5) * BIN + w0
    rows = []
    for s, ds in dl.groupby('subject_id'):
        neu = [n for n in ds.Neuron_ID_3.unique() if n in fits.index]
        if not neu or ds.Neuron_ID_3.nunique() < 6:
            continue
        tids = sorted(ds.trial_id.unique())
        load = ds.drop_duplicates('trial_id').set_index('trial_id').num_images_presented.to_dict()
        S = {(r.trial_id, r.Neuron_ID_3): r.sp for r in ds.itertuples()}
        allN = list(ds.Neuron_ID_3.unique())
        same = {t: [u for u in tids if u != t and load[u] == load[t]] for t in tids}
        for n in neu:
            others = [m for m in allN if m != n]
            cat = np.concatenate([np.concatenate([S.get((t, m), np.array([])) for m in others]) + k * dur for k, t in enumerate(tids)])
            sm = np.convolve(np.histogram(cat, np.arange(0, len(tids) * dur + bin_size, bin_size))[0], kernel, mode='same')
            pk, _ = signal.find_peaks(sm, prominence=np.percentile(sm, 90), distance=distance)
            pt = (pk + 0.5) * bin_size
            peaks = {t: pt[(pt >= k * dur) & (pt < (k + 1) * dur)] - k * dur for k, t in enumerate(tids)}
            r = fits.loc[n]
            lam, A = np.array([r.lam_low_hz, r.lam_high_hz]), np.array([[r.A_00, 1 - r.A_00], [1 - r.A_11, r.A_11]])
            P = {}
            for t in tids:
                x = S.get((t, n), np.array([])); x = x[(x >= w0) & (x < w0 + 1.0)] - w0
                P[t] = posterior_high(np.bincount(np.minimum((x / BIN).astype(int), 32), minlength=33), lam, A)

            def in_minus_out(assign):
                v = []
                for t, src in assign:
                    inb = np.array([np.any(np.abs(peaks[src] - u) <= HW) for u in ctr])
                    if inb.any() and (~inb).any():
                        v.append(P[t][inb].mean() - P[t][~inb].mean())
                return np.mean(v) if v else np.nan

            obs = in_minus_out([(t, t) for t in tids])
            null = [in_minus_out([(t, same[t][rng.integers(len(same[t]))]) for t in tids if same[t]]) for _ in range(N_NULL)]
            rows.append(dict(subject_id=s, neuron=n, obs=obs, null=np.nanmean(null), link=obs - np.nanmean(null)))
    D = pd.DataFrame(rows).dropna(subset=['link'])
    D.to_csv(f, index=False)
    print(epoch, 'burst link:', len(D), 'neurons', flush=True)
    return D


# =========================
# 3. Panels
# =========================
def stars(p):
    return '***' if p < 0.005 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def save(fig, name):
    for ax in fig.axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT + name + '.eps', format='eps', bbox_inches='tight')
    plt.close(fig)


def match(targets, pool, cal=0.25):
    """each target matched 1:1 to the pool neuron with the nearest log rate, without replacement, within the caliper"""
    pool, out = pool.copy(), []
    for r in targets.sort_values('lograte').itertuples():
        if pool.empty:
            break
        j = (pool.lograte - r.lograte).abs().idxmin()
        if abs(pool.loc[j, 'lograte'] - r.lograte) <= cal:
            out.append((r.Index, j)); pool = pool.drop(j)
    return out


def panels(EP, links):
    S = []
    thr = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200]
    # a, b: spike-count threshold sweep (degenerate fits excluded only)
    for letter, (ep, d) in zip('ab', EP.items()):
        fig, ax = plt.subplots(figsize=(5, 3.6)); cl = d[~d.excl]
        for nm, m, col in (('All cells', cl.neuron_id.notna(), ALLC), ('Concept cells', cl.concept, CONC)):
            ax.plot(thr, [cl[m & (cl.total_spikes >= t)].fav2.mean() for t in thr], '-o', color=col, ms=5, lw=1.8, label=nm)
            for t in (0, 200):
                g = cl[m & (cl.total_spikes >= t)]
                S.append(dict(panel=letter, epoch=ep, test=f'{nm}, >= {t} spikes: fraction favouring two-state', n=len(g), value=g.fav2.mean()))
        ax.set_ylim(0.45, 0.95); ax.set_xlabel('Minimum total spike-count threshold'); ax.set_ylabel('Fraction favoring two-state')
        ax.set_title(ep.capitalize()); ax.legend(frameon=False, loc='lower right')
        save(fig, f'S11{letter}_spikecount_{ep}')
    # c: burst link, all cells
    fig, ax = plt.subplots(figsize=(3.2, 3.6))
    for x, ep in enumerate(EP):
        v = 100 * links[ep].link; p = wilcoxon(v).pvalue; m, e = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
        ax.bar(x, m, 0.62, color=TWO, **EC); ax.errorbar(x, m, e, color='k', lw=1.3, capsize=4)
        ax.text(x, m + e + 0.06, stars(p), ha='center', fontsize=16)
        S.append(dict(panel='c', epoch=ep, test='P(high) inside - outside bursts (pp), Wilcoxon vs 0', n=len(v), value=m, sem=e, W=wilcoxon(v).statistic, p=p))
    ax.axhline(0, color='k', lw=1); ax.set_xticks([0, 1]); ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 3.2)
    ax.set_xticklabels([f'{ep.capitalize()}\n(n={len(links[ep])})' for ep in EP])
    ax.set_ylabel('P(high state), inside − outside\nbursts (percentage points)'); ax.set_title('HMM high states and\npopulation bursts')
    save(fig, 'S11c_burst_link')
    # d-g: one- vs two-state; h-k: classes
    specs = [(ep, nm, g) for ep, d in EP.items() for nm, g in (('All cells', d[d.clean_active]), ('Concept cells', d[d.clean_active & d.concept]))]
    for k, (ep, nm, g) in enumerate(specs):
        fig, ax = plt.subplots(figsize=(2.8, 3.4)); f2 = g.fav2.mean(); p = wilcoxon(g.delta_ll_per_bin).pvalue
        ax.bar([0, 1], [1 - f2, f2], 0.65, color=[ONE, TWO], **EC); ax.text(1, f2 + 0.02, stars(p), ha='center', fontsize=16)
        ax.set_ylim(0, 1.05); ax.set_xticks([0, 1]); ax.set_xticklabels(['One-state', 'Two-state']); ax.set_ylabel('Fraction of neurons')
        ax.set_title(f'{ep.capitalize()}: {nm} (n={len(g)})', fontsize=10)
        S.append(dict(panel='defg'[k], epoch=ep, test=f'{nm}: two-state preference, Wilcoxon on delta LL', n=len(g), value=f2, p=p))
        save(fig, f'S11{"defg"[k]}_preference_{ep}_{nm.split()[0].lower()}')
        fig, ax = plt.subplots(figsize=(3.2, 3.4)); cnt = np.array([(g.cls == c).sum() for c in CL]); r = chisquare(cnt)
        ax.bar(range(3), cnt / len(g), 0.65, color=[CLS[c] for c in CL], **EC)
        if r.pvalue < 0.05:
            j = int(np.argmax(cnt)); ax.text(j, cnt[j] / len(g) + 0.02, stars(r.pvalue), ha='center', fontsize=16)
        ax.set_ylim(0, 0.65); ax.set_xticks(range(3)); ax.set_xticklabels([c.capitalize() for c in CL]); ax.set_ylabel('Fraction of neurons')
        ax.set_title(f'{ep.capitalize()}: {nm} (n={len(g)})', fontsize=10)
        S.append(dict(panel='hijk'[k], epoch=ep, test=f'{nm}: classes vs uniform, chi-square (df = 2)', n=len(g), value=r.statistic, p=r.pvalue,
                      **{f'n_{c}': int(v) for c, v in zip(CL, cnt)}))
        save(fig, f'S11{"hijk"[k]}_classes_{ep}_{nm.split()[0].lower()}')
    # l, m: PY vs IN, rate-matched
    for letter, (ep, d) in zip('lm', EP.items()):
        ca = d[d.clean_active]; P, I = ca[ca.PY], ca[ca.IN]
        pr = match(P, I); Pm, Im = P.loc[[a for a, _ in pr]], I.loc[[b for _, b in pr]]
        raw = [fisher_exact([[(Im.cls == c).sum(), (Im.cls != c).sum()], [(Pm.cls == c).sum(), (Pm.cls != c).sum()]])[1] for c in CL]
        q = multipletests(raw, method='fdr_bh')[1]
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for y, c in enumerate(CL):
            fp, fi = (Pm.cls == c).mean(), (Im.cls == c).mean()
            ax.barh(y, -fp, 0.72, color=PYC, **EC); ax.barh(y, fi, 0.72, color=INC, **EC)
            ax.text(fi + 0.02, y, stars(q[y]) or 'n.s.', va='center', ha='left', fontsize=14 if stars(q[y]) else 10)
            S.append(dict(panel=letter, epoch=ep, test=f'rate-matched PY vs IN, {c}, Fisher (BH over 3 classes)', n=len(Pm),
                          value=100 * fi - 100 * fp, pct_PY=100 * fp, pct_IN=100 * fi, p=raw[y], p_BH=q[y],
                          PY_rate_median_hz=Pm.overall_rate_hz.median(), IN_rate_median_hz=Im.overall_rate_hz.median()))
        ax.axvline(0, color='k', lw=1.1); ax.set_xlim(-0.75, 0.75)
        ax.set_xticks(np.arange(-0.6, 0.61, 0.2)); ax.set_xticklabels([f'{abs(v):.1f}' for v in np.arange(-0.6, 0.61, 0.2)])
        ax.set_yticks(range(3)); ax.set_yticklabels([c.capitalize() for c in CL]); ax.set_xlabel('Fraction of neurons')
        ax.set_title(f'{ep.capitalize()}, rate-matched ({len(Pm)} pairs)')
        ax.legend(handles=[Patch(color=PYC), Patch(color=INC)], frameon=False, loc='upper right',
                  labels=[f'PY ({Pm.overall_rate_hz.median():.1f} Hz)', f'IN ({Im.overall_rate_hz.median():.1f} Hz)'])
        save(fig, f'S11{letter}_PY_vs_IN_ratematched_{ep}')
    return pd.DataFrame(S)


if __name__ == '__main__':
    concept = set(pd.read_excel('../Supplementary_Figures/all_neuron_brain_regions_merged.xlsx').query("Signi == 'Y'").Neuron_ID_3)
    lab = pd.read_excel('../Supplementary_Figures/revision_clustering_no_waveform_labels.xlsx')
    py, in_ = set(lab[lab.Cell_Type_New == 'PY'].Neuron_ID_3), set(lab[lab.Cell_Type_New == 'IN'].Neuron_ID_3)
    data = {e: pd.read_excel(f, usecols=['subject_id', 'trial_id', 'Neuron_ID_3', 'num_images_presented', 'Standardized_Spikes'])
            for e, f in FILES.items()}
    fits = {e: fit_all(e, d) for e, d in data.items()}
    # maintenance firing rate over 0-2.5 s
    c = data['maintenance'].assign(n=data['maintenance'].Standardized_Spikes.map(lambda s: len(parse(s, 0.0, RATE_WIN))))
    g = c.groupby('Neuron_ID_3').n.agg(['sum', 'size'])
    fits['maintenance']['overall_rate_hz'] = fits['maintenance'].neuron_id.map(g['sum'] / (g['size'] * RATE_WIN))
    EP = {e: derive(f, concept, py, in_) for e, f in fits.items()}
    links = {e: burst_link(e, EP[e], data[e]) for e in EP}
    T = panels(EP, links)
    T.to_csv(OUT + 'S11_stats.csv', index=False)
    pd.set_option('display.width', 250)
    print(T.round(6).to_string(index=False))
