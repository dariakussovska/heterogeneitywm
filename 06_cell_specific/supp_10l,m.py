import os
os.environ['OMP_NUM_THREADS'] = '1'          
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests
from multiprocessing import Pool
import ast

trial_info = pd.read_excel('../data/new_trial_final.xlsx')
df_clustering = pd.read_excel('../REVISIONS/revision_clustering_waveform_labels.xlsx')
subject_trials = trial_info[trial_info['subject_id'] == 14][['trial_id_final', 'num_images_presented', 'stimulus_index_enc1', 'stimulus_index_enc2', 'stimulus_index_enc3', 'response_accuracy']]
y_matrix = subject_trials.reset_index(drop=True)

# 1. Load Spike Data (maintenance) and fixation rates
df_delay = pd.read_excel('../graph_data/graph_delay.xlsx')
df_delay = df_delay.merge(trial_info[['subject_id', 'trial_id', 'trial_id_final']], on=['subject_id', 'trial_id'], how='inner')
df_fixation = pd.read_excel('../clean_data/cleaned_Fixation.xlsx')

concept = df_clustering[(df_clustering['Signi'] == 'Y') & df_clustering['Cell_Type_New'].isin(['PY', 'IN'])]
available = np.intersect1d(df_delay['Neuron_ID_3'].unique(), df_fixation['Neuron_ID_3'].unique())
neuron_ids = {t: np.intersect1d(concept.loc[concept['Cell_Type_New'] == t, 'Neuron_ID_3'].unique(), available) for t in ['PY', 'IN']}
sample_size = min(len(neuron_ids['PY']), len(neuron_ids['IN']))      # 11
print({t: len(v) for t, v in neuron_ids.items()}, 'sample size', sample_size)

def parse_spike_times(spike_str):
    try:
        return ast.literal_eval(spike_str)
    except:
        return []

spikes = {(row.trial_id_final, row.Neuron_ID_3): np.array(parse_spike_times(row.Standardized_Spikes), dtype=float)
          for row in df_delay.itertuples()}

fixation = df_fixation[df_fixation['Neuron_ID_3'].isin(np.union1d(neuron_ids['PY'], neuron_ids['IN']))]
fixation_means = fixation.groupby('Neuron_ID_3')['Spikes_rate_Fixation'].mean().to_dict()
fixation_stds = fixation.groupby('Neuron_ID_3')['Spikes_rate_Fixation'].std().to_dict()

# label = last image shown
y_all = np.array([row[f"stimulus_index_enc{int(row['num_images_presented'])}"] for _, row in y_matrix.iterrows()])
load_trials = {load: np.where(y_matrix['num_images_presented'].values == load)[0] for load in [1, 2, 3]}

bin_sizes = [0.25, 0.5, 0.75, 1.0]
step_size = 0.025
duration = 2.5
n_bootstrap = 100
n_perm = 1000
N_JOBS = int(os.environ.get('N_JOBS', 1))

def create_time_bins(bin_size):
    return np.arange(0, duration - bin_size + step_size, step_size)

def count_spikes(spikes, bins, size):
    return np.array([np.sum((spikes >= t) & (spikes < t + size)) for t in bins])

def build_binned_matrix(load, bin_size, cell_type):
    """trials x neurons x bins, z-scored with each neuron's fixation rate"""
    bins = create_time_bins(bin_size)
    trials = load_trials[load]
    ids = neuron_ids[cell_type]
    binned_matrix = np.zeros((len(trials), len(ids), len(bins)))
    for i, t in enumerate(trials):
        for j, neuron in enumerate(ids):
            binned = count_spikes(spikes.get((y_matrix.loc[t, 'trial_id_final'], neuron), np.array([])), bins, bin_size)
            mean = fixation_means.get(neuron, 0)
            std = fixation_stds.get(neuron, 1)
            binned_matrix[i, j, :] = (binned - mean) / std if std > 0 else binned
    return binned_matrix

def decode(X, y, seed):
    """mean accuracy (%) over 10 repeats of stratified 3-fold CV"""
    accs = []
    for rep in range(10):
        for train, test in StratifiedKFold(3, shuffle=True, random_state=seed + rep).split(X, y):
            clf = SVC(kernel='linear', random_state=seed + rep)
            clf.fit(X[train], y[train])
            accs.append(np.mean(clf.predict(X[test]) == y[test]) * 100)
    return np.mean(accs)

def draw(X, columns):
    """features of one bootstrap draw (neurons x bins flattened)"""
    return X[:, columns, :].reshape(X.shape[0], -1)

matrices = {(load, b, t): build_binned_matrix(load, b, t) for load in [1, 2, 3] for b in bin_sizes for t in ['PY', 'IN']}

# 2. Null of means for one cell type x load x bin and one permutation
def null_mean(task):
    load, b_ms, cell_type, p = task
    X = matrices[(load, b_ms / 1000, cell_type)]
    rng_p = np.random.default_rng([load, b_ms, 0 if cell_type == 'PY' else 1, p + 1])
    y_perm = rng_p.permutation(y_all[load_trials[load]])
    accs = [decode(draw(X, rng_p.integers(0, X.shape[1], sample_size)), y_perm, 10000 * (p + 2) + d) for d in range(n_bootstrap)]
    return load, b_ms, cell_type, p, np.mean(accs)

if __name__ == '__main__':
    folder_path = './'
    os.makedirs(folder_path, exist_ok=True)

    # 3. Real decoding: 100 bootstrap draws (and one shuffled-label decode per draw)
    rng = np.random.default_rng(42)
    # the published run first went through an earlier loop on the same generator; repeat its draws (no fitting)
    # so the accuracies are identical to the published ones
    for load in [1, 2, 3]:
        for b in bin_sizes:
            for _ in range(n_bootstrap):
                rng.choice(neuron_ids['PY'], sample_size); rng.choice(neuron_ids['IN'], sample_size)
            for _ in range(n_bootstrap):
                rng.permutation(y_all[load_trials[load]]); rng.choice(neuron_ids['PY'], sample_size); rng.choice(neuron_ids['IN'], sample_size)

    rows = []
    for load in [1, 2, 3]:
        y = y_all[load_trials[load]]
        for b in bin_sizes:
            for i in range(n_bootstrap):
                X_py = draw(matrices[(load, b, 'PY')], np.searchsorted(neuron_ids['PY'], rng.choice(neuron_ids['PY'], sample_size)))
                X_in = draw(matrices[(load, b, 'IN')], np.searchsorted(neuron_ids['IN'], rng.choice(neuron_ids['IN'], sample_size)))
                real_py, real_in = decode(X_py, y, 42 + i), decode(X_in, y, 142 + i)
                y_shuf = rng.permutation(y)
                shuf_py, shuf_in = decode(X_py, y_shuf, 242 + i), decode(X_in, y_shuf, 342 + i)
                for cell_type, condition, acc in [('PY', 'Real', real_py), ('IN', 'Real', real_in), ('PY', 'Shuffled', shuf_py), ('IN', 'Shuffled', shuf_in)]:
                    rows.append(dict(load=load, bin_ms=int(b * 1000), cell_type=cell_type, condition=condition, draw=i, accuracy=acc))
            print(f'Load {load}, {int(b * 1000)} ms done')
    draws = pd.DataFrame(rows)
    draws.to_csv(folder_path + 'decoding_draws.csv', index=False)

    # 4. Null of means (resumes from null_values.csv)
    null_file = folder_path + 'null_values.csv'
    if not os.path.exists(null_file):
        open(null_file, 'w').write('load,bin_ms,cell_type,perm,value\n')
    done = set(map(tuple, pd.read_csv(null_file)[['load', 'bin_ms', 'cell_type', 'perm']].values.tolist()))
    tasks = [(load, int(b * 1000), t, p) for load in [1, 2, 3] for b in bin_sizes for t in ['PY', 'IN'] for p in range(n_perm)]
    tasks = [task for task in tasks if task not in done]
    with Pool(N_JOBS) as pool:
        for result in pool.imap_unordered(null_mean, tasks, chunksize=10):
            with open(null_file, 'a') as f:
                f.write(','.join(map(str, result)) + '\n')
    null = pd.read_csv(null_file)

    # 5. Statistics: P = (1 + #null >= real) / (1 + n_perm), BH across bins within cell type x load
    stats = []
    for (load, b_ms, cell_type), g in draws[draws.condition == 'Real'].groupby(['load', 'bin_ms', 'cell_type']):
        real = g.accuracy.mean()
        v = null[(null.load == load) & (null.bin_ms == b_ms) & (null.cell_type == cell_type)].value.values
        stats.append(dict(load=load, cell_type=cell_type, bin_ms=b_ms, accuracy=real, sem=g.accuracy.sem(), null_mean=v.mean(),
                          n_null=len(v), p=(1 + np.sum(v >= real)) / (1 + len(v))))
    stats = pd.DataFrame(stats)
    stats['p_BH'] = stats.groupby(['cell_type', 'load']).p.transform(lambda p: multipletests(p, method='fdr_bh')[1])
    stats = stats.sort_values(['cell_type', 'load', 'bin_ms'])
    stats.to_csv(folder_path + 'decoding_stats.csv', index=False)
    print(stats.round(4).to_string(index=False))

    # 6. Plots: accuracy vs bin width per load (mean +- SEM over draws), shuffled labels, chance
    def star(p):
        return '***' if p < 0.005 else '**' if p < 0.01 else '*' if p < 0.05 else ''

    colors = {'PY': ['#8e1c2e', '#cf5d6d', '#e9a9b2'], 'IN': ['#17393b', '#4f8a8d', '#9cc4c5']}
    x = [int(b * 1000) for b in bin_sizes]
    for cell_type, panel in [('PY', 'l'), ('IN', 'm')]:
        d = draws[draws.cell_type == cell_type]
        fig, ax = plt.subplots(figsize=(4.8, 3.8))
        shuf = d[d.condition == 'Shuffled'].groupby('bin_ms').accuracy
        ax.errorbar(range(4), shuf.mean().loc[x], shuf.sem().loc[x], color='#6b6b6b', ls='--', marker='o', ms=4, capsize=2, label='Shuffled (all loads)')
        ax.axhline(20, ls=':', color='0.4', lw=1, label='Chance (20%)')
        for k, load in enumerate([1, 2, 3]):
            real = d[(d.condition == 'Real') & (d.load == load)].groupby('bin_ms').accuracy
            ax.errorbar(range(4), real.mean().loc[x], real.sem().loc[x], color=colors[cell_type][k], lw=2, marker='o', ms=5, capsize=2)
            ax.text(3.09, real.mean().loc[x[-1]], f'Load {load}', va='center', fontsize=9)
            ax.text(-0.46, 58 - 4 * k, f'L{load}', va='center', fontsize=9)
            for j, b_ms in enumerate(x):
                p = stats[(stats.cell_type == cell_type) & (stats.load == load) & (stats.bin_ms == b_ms)].p_BH.iloc[0]
                ax.text(j, 58 - 4 * k, star(p), ha='center', va='center', fontsize=12)
        ax.set_xticks(range(4)); ax.set_xticklabels(x); ax.set_xlim(-0.5, 3.6)
        ax.set_ylim(0, 60); ax.set_xlabel('Bin size (ms)'); ax.set_ylabel('Decoding accuracy (%)')
        ax.set_title(f'{cell_type} concept cells (n = {len(neuron_ids[cell_type])})')
        ax.legend(frameon=False, loc='lower left', fontsize=8, ncol=2)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(folder_path + f'supp_10{panel}_{cell_type}_decoding.eps', format='eps', dpi=300)
        plt.close()
