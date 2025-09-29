import numpy as np
import ast
from dPCA import dPCA
import matplotlib.pyplot as plt
from ast import literal_eval
from scipy.ndimage import gaussian_filter1d
from dPCA.dPCA import dPCA
from collections import defaultdict
from scipy.spatial.distance import pdist
import scipy.stats as sps
import scikit_posthocs as sp

trial_info = pd.read_excel('/home/daria/new_trial_info.xlsx')
subject_trials = trial_info[(trial_info['subject_id'] == 14) & (trial_info['num_images_presented'] == 1)][['new_trial_id', 'num_images_presented', 'stimulus_index']]
y_matrix = subject_trials

target_trials_per_stimulus = 6  # Desired number of trials per stimulus

# Initialize an empty DataFrame for the balanced y_matrix
y_matrix_balanced = pd.DataFrame()

# Iterate through each stimulus index and sample the desired number of trials
for stimulus in y_matrix['stimulus_index'].unique():
    stimulus_trials = y_matrix[y_matrix['stimulus_index'] == stimulus]
    sampled_trials = stimulus_trials.sample(n=target_trials_per_stimulus, random_state=42)  # Set random_state for reproducibility
    y_matrix_balanced = pd.concat([y_matrix_balanced, sampled_trials]) 

y_matrix_balanced.reset_index(drop=True, inplace=True)
balanced_distribution = y_matrix_balanced['stimulus_index'].value_counts().sort_index()
print("\nBalanced trial distribution across stimuli:")
print(balanced_distribution)

balanced_y_matrix = y_matrix_balanced

fixation_data_with_brain_region = pd.read_excel('/home/daria/graph_data/graph_fixation.xlsx')
enc1_data_with_brain_region = pd.read_excel('/home/daria/clean_data/cleaned_Encoding1.xlsx')
delay_data_with_brain_region = pd.read_excel('/home/daria/graph_data/graph_delay.xlsx')

fixation_data = fixation_data_with_brain_region

def filter_data(data, subject_ids=None, brain_regions=None, neuron_ids_3=None):
    filtered_data = data[data['Significance'] == 'Y']  # Start with all significant neurons
    if subject_ids:
        filtered_data = filtered_data[filtered_data['subject_id'].isin(subject_ids)]
    if brain_regions:
        filtered_data = filtered_data[filtered_data['Location'].isin(brain_regions)]
    if neuron_ids_3:  
        filtered_data = filtered_data[filtered_data['Neuron_ID_3'].isin(neuron_ids_3)]
    return filtered_data

def calculate_baseline_stats(fixation_data, spike_column, subject_id, neuron_id):
    neuron_data = fixation_data[(fixation_data['subject_id'] == subject_id) & (fixation_data['Neuron_ID'] == neuron_id)]
    if neuron_data.empty or spike_column not in neuron_data.columns:
        return 0, 1 
    firing_rates = neuron_data[spike_column].dropna().values
    return np.mean(firing_rates), np.std(firing_rates)

def calculate_firing_rates(trials, spike_column, time_bins):
    trial_firing_rates = []
    for _, row in trials.iterrows():
        spike_times = literal_eval(str(row[spike_column])) if isinstance(row[spike_column], str) else []
        if not isinstance(spike_times, list):
            trial_firing_rates.append(np.zeros(len(time_bins) - 1))
            continue
        spike_counts, _ = np.histogram(spike_times, bins=time_bins)
        firing_rates = spike_counts / (time_bins[1] - time_bins[0]) 
        trial_firing_rates.append(firing_rates)
    return np.array(trial_firing_rates)  

def calculate_z_scores(firing_rates, mean_baseline, std_baseline):
    if std_baseline == 0:
        return np.zeros_like(firing_rates) 
    return (firing_rates - mean_baseline) / std_baseline

def construct_stimulus_tables_no_avg_balanced(enc1_data, fixation_data, balanced_y_matrix, time_bins_enc1, smoothing_sigma):
    subjects_in_data = enc1_data['subject_id'].unique()
    print(f"Subjects included in analysis: {subjects_in_data}")

    neuron_subject_combinations = enc1_data[['Neuron_ID', 'subject_id', 'Neuron_ID_3']].drop_duplicates()
    num_neurons_analyzed = neuron_subject_combinations.shape[0]
    print(f"Number of neurons analyzed: {num_neurons_analyzed}")

    stimulus_ids = balanced_y_matrix['stimulus_index'].unique()
    z_scores_by_stimulus = {stimulus: {} for stimulus in stimulus_ids}

    for _, row in neuron_subject_combinations.iterrows():
        neuron_id, subject_id, neuron_id_3 = row['Neuron_ID'], row['subject_id'], row['Neuron_ID_3']
        label = f"{subject_id}_{neuron_id}_{neuron_id_3}" 

        baseline_mean, baseline_std = calculate_baseline_stats(
            fixation_data, 'Spikes_rate_Fixation', subject_id, neuron_id
        )

        for stimulus in stimulus_ids:
            trial_ids = balanced_y_matrix[balanced_y_matrix['stimulus_index'] == stimulus]['new_trial_id']
            stimulus_trials = enc1_data[
                (enc1_data['new_trial_id'].isin(trial_ids)) &
                (enc1_data['Neuron_ID'] == neuron_id) &
                (enc1_data['subject_id'] == subject_id)
            ]

            trial_firing_rates = calculate_firing_rates(stimulus_trials, 'Standardized_Spikes', time_bins_enc1)
            trial_z_scores = np.array([calculate_z_scores(fr, baseline_mean, baseline_std) for fr in trial_firing_rates])
            smoothed_z_scores = np.array([gaussian_filter1d(z, sigma=smoothing_sigma) for z in trial_z_scores])

            if label not in z_scores_by_stimulus[stimulus]:
                z_scores_by_stimulus[stimulus][label] = smoothed_z_scores
            else:
                z_scores_by_stimulus[stimulus][label] = np.vstack(
                    [z_scores_by_stimulus[stimulus][label], smoothed_z_scores]
                )

    return z_scores_by_stimulus

subject_ids_to_include = []  
brain_regions_to_include = []  
neuron_ids_3_to_include = []  

fixation_filtered = filter_data(fixation_data_with_brain_region, subject_ids_to_include, brain_regions_to_include, neuron_ids_3_to_include)
enc1_filtered = filter_data(enc1_data_with_brain_region, subject_ids_to_include, brain_regions_to_include, neuron_ids_3_to_include)

time_bins_enc1 = np.arange(0, 1, 0.002) 
smoothing_sigma = 60 

z_scores_by_stimulus_balanced = construct_stimulus_tables_no_avg_balanced(enc1_filtered, fixation_filtered, balanced_y_matrix, time_bins_enc1, smoothing_sigma)

neurons = enc1_filtered[['Neuron_ID', 'subject_id', 'Neuron_ID_3']].drop_duplicates()
neuron_labels = neurons.apply(lambda x: f"{x['subject_id']}_{x['Neuron_ID']}_{x['Neuron_ID_3']}", axis=1).tolist()
num_neurons = len(neuron_labels) 
print(f"Number of neurons: {num_neurons}")

stimuli = balanced_y_matrix['stimulus_index'].unique()
num_stimuli = len(stimuli)
num_samples_per_stimulus = {stimulus: balanced_y_matrix[balanced_y_matrix['stimulus_index'] == stimulus].shape[0]
                            for stimulus in stimuli}
max_samples = max(num_samples_per_stimulus.values()) 

num_time_bins = len(time_bins_enc1) - 1
Xtrial = np.zeros((max_samples, num_neurons, num_stimuli, num_time_bins))

for stimulus_idx, stimulus in enumerate(stimuli):
    stimulus_trials = balanced_y_matrix[balanced_y_matrix['stimulus_index'] == stimulus]['new_trial_id'].tolist()
    for sample_idx, trial_id in enumerate(stimulus_trials):
        for neuron_idx, neuron_label in enumerate(neuron_labels):
            if neuron_label in z_scores_by_stimulus_balanced[stimulus]:
                trial_data = z_scores_by_stimulus_balanced[stimulus][neuron_label]
                if sample_idx < trial_data.shape[0]:
                    Xtrial[sample_idx, neuron_idx, stimulus_idx, :] = trial_data[sample_idx]
                else:
                    Xtrial[sample_idx, neuron_idx, stimulus_idx, :] = np.zeros(num_time_bins)

# check the shape of the constructed Xtrial matrix
print(f"Xtrial matrix shape: {Xtrial.shape}")  

### DELAY 

# Define the time bins for the delay period
time_bins_delay = np.arange(0, 2.5, 0.002)  
subject_ids_to_include = []  
brain_regions_to_include = [] 
neuron_ids_3_to_include = []

delay_filtered = filter_data(delay_data_with_brain_region, subject_ids_to_include, brain_regions_to_include, neuron_ids_3_to_include)
delay_filtered = delay_filtered[delay_filtered['Significance'] == 'Y']

time_bins_delay = np.arange(0, 2.5, 0.002)  

def construct_stimulus_tables_no_avg_delay(delay_data, fixation_data, balanced_y_matrix, time_bins_delay, smoothing_sigma):
    subjects_in_data = delay_data['subject_id'].unique()
    print(f"Subjects included in analysis: {subjects_in_data}")

    neuron_subject_combinations = delay_data[['Neuron_ID', 'subject_id', 'Neuron_ID_3']].drop_duplicates()
    num_neurons_analyzed = neuron_subject_combinations.shape[0]
    print(f"Number of neurons analyzed: {num_neurons_analyzed}")

    stimulus_ids = balanced_y_matrix['stimulus_index'].unique()
    z_scores_by_stimulus = {stimulus: {} for stimulus in stimulus_ids}

    for _, row in neuron_subject_combinations.iterrows():
        neuron_id, subject_id, neuron_id_3 = row['Neuron_ID'], row['subject_id'], row['Neuron_ID_3']
        label = f"{subject_id}_{neuron_id}_{neuron_id_3}" 

        # Baseline statistics from fixation period
        baseline_mean, baseline_std = calculate_baseline_stats(
            fixation_data, 'Spikes_rate_Fixation', subject_id, neuron_id
        )

        for stimulus in stimulus_ids:
            trial_ids = balanced_y_matrix[balanced_y_matrix['stimulus_index'] == stimulus]['new_trial_id']
            stimulus_trials = delay_data[
                (delay_data['trial_id'].isin(trial_ids)) &
                (delay_data['Neuron_ID'] == neuron_id) &
                (delay_data['subject_id'] == subject_id)
            ]

            trial_firing_rates = calculate_firing_rates(stimulus_trials, 'Standardized_Spikes_in_Delay', time_bins_delay)
            trial_z_scores = np.array([calculate_z_scores(fr, baseline_mean, baseline_std) for fr in trial_firing_rates])
            smoothed_z_scores = np.array([gaussian_filter1d(z, sigma=smoothing_sigma) for z in trial_z_scores])
            if label not in z_scores_by_stimulus[stimulus]:
                z_scores_by_stimulus[stimulus][label] = smoothed_z_scores
            else:
                z_scores_by_stimulus[stimulus][label] = np.vstack(
                    [z_scores_by_stimulus[stimulus][label], smoothed_z_scores]
                )

    return z_scores_by_stimulus

z_scores_by_stimulus_delay = construct_stimulus_tables_no_avg_delay(delay_filtered, fixation_filtered, balanced_y_matrix, time_bins_delay, smoothing_sigma)
neurons = delay_filtered[['Neuron_ID', 'subject_id', 'Neuron_ID_3']].drop_duplicates()
neuron_labels = neurons.apply(lambda x: f"{x['subject_id']}_{x['Neuron_ID']}_{x['Neuron_ID_3']}", axis=1).tolist()
num_neurons = len(neuron_labels) 
print(f"Number of neurons: {num_neurons}")

num_time_bins_delay = len(time_bins_delay) - 1
Dtrial = np.zeros((max_samples, num_neurons, num_stimuli, num_time_bins_delay))

# Populate the Dtrial matrix
for stimulus_idx, stimulus in enumerate(stimuli):
    stimulus_trials = balanced_y_matrix[balanced_y_matrix['stimulus_index'] == stimulus]['new_trial_id'].tolist()
    for sample_idx, trial_id in enumerate(stimulus_trials):
        for neuron_idx, neuron_label in enumerate(neuron_labels):
            if neuron_label in z_scores_by_stimulus_delay[stimulus]:
                trial_data = z_scores_by_stimulus_delay[stimulus][neuron_label]
                if sample_idx < trial_data.shape[0]:
                    Dtrial[sample_idx, neuron_idx, stimulus_idx, :] = trial_data[sample_idx]
                else:
                    Dtrial[sample_idx, neuron_idx, stimulus_idx, :] = np.zeros(num_time_bins_delay)

print(f"Dtrial matrix shape: {Dtrial.shape}")  

trialE = Xtrial
trialD = Dtrial 
print(trialE.shape)
print(trialD.shape)

###ENC AND MAINTENANCE

#dPCA distance-resampling  (Encoding vs Maintenance)

def run_dpca_distance_resampling_E_vs_D(
        trialE,                 # (n_reps, n_neurons, n_stimuli, n_timeE)
        trialD,                 # (n_reps, n_neurons, n_stimuli, n_timeD)
        n_iter=100,
        selected_dpcs=(0, 1, 2, 3, 4),
        stim_indices=(0, 1, 2, 3, 4),
        savefig_path=None,
        random_seed=5,
):
    """
    Returns
    -------
    measure_dict      dict(epoch → np.array)   distance matrices
    kw_p              float                    omnibus p-value
    dunn_df           pd.DataFrame | None      post-hoc p-values
    explvar_enc_arr   (n_iter, #dpcs)          variance explained
    explvar_maint_arr (n_iter, #dpcs)          variance explained
    """

    np.random.seed(random_seed)

    n_reps, n_neurons, n_stimuli, _ = trialE.shape
    n_train = n_reps // 2         # e.g. 3
    n_test  = n_reps - n_train     # e.g. 3

    epochs = [
        "Enc_train",
        "Maint_train",
        "Enc_test",
        "Maint_test",
        "Maint onto Enc",
    ]

    def center(x):
        """mean-center over neurons (axis=1)"""
        m = np.mean(x.reshape(x.shape[0], -1), axis=1, keepdims=True)
        return x - m[:, None]

    def stim_dist(Z_epoch):
        """pair-wise Euclidean distance between stimulus means
        in low-D dPCA space"""
        stim_means = []
        for s in stim_indices:
            traj = Z_epoch["s"][selected_dpcs, s, :]    # (n_dpcs, time)
            stim_means.append(np.mean(traj, axis=1))    # (n_dpcs,)
        stim_means = np.stack(stim_means, axis=0)
        return pdist(stim_means, metric="euclidean")    

    all_dists        = defaultdict(list)
    explvar_enc      = []
    explvar_maint    = []
    split_signatures = []

    for it in range(n_iter):

        # stratified split 
        train_idx_bystim, test_idx_bystim, split_signature = [], [], []
        for s in stim_indices:
            idx = np.random.permutation(n_reps)
            train_idx_bystim.append(idx[:n_train])
            test_idx_bystim.append(idx[n_train:])
            split_signature.append((tuple(idx[:n_train]), tuple(idx[n_train:])))
        split_signatures.append(tuple(split_signature))

        tE = trialE.shape[-1]
        tD = trialD.shape[-1]

        trialX_enc_train   = np.zeros((n_train, n_neurons, n_stimuli, tE))
        trialX_enc_test    = np.zeros((n_test,  n_neurons, n_stimuli, tE))
        trialX_maint_train = np.zeros((n_train, n_neurons, n_stimuli, tD))
        trialX_maint_test  = np.zeros((n_test,  n_neurons, n_stimuli, tD))

        for s_idx, stim in enumerate(stim_indices):
            tr_idx = train_idx_bystim[s_idx]
            te_idx = test_idx_bystim[s_idx]

            # encoding
            trialX_enc_train[:,   :, s_idx, :] = trialE[tr_idx, :, stim, :]
            trialX_enc_test[:,    :, s_idx, :] = trialE[te_idx, :, stim, :]

            # maintenance
            trialX_maint_train[:, :, s_idx, :] = trialD[tr_idx, :, stim, :]
            trialX_maint_test[:,  :, s_idx, :] = trialD[te_idx, :, stim, :]

        # trial averages (for dPCA fit/transform) 
        RD_enc_train   = np.mean(trialX_enc_train,   axis=0)
        RD_enc_test    = np.mean(trialX_enc_test,    axis=0)
        RD_maint_train = np.mean(trialX_maint_train, axis=0)
        RD_maint_test  = np.mean(trialX_maint_test,  axis=0)

        # mean-center neurons
        RD_enc_train   = center(RD_enc_train)
        RD_enc_test    = center(RD_enc_test)
        RD_maint_train = center(RD_maint_train)
        RD_maint_test  = center(RD_maint_test)

        # dPCA fits 
        dpca_enc = dPCA(labels="st", regularizer="auto")
        dpca_enc.opt_regularizer_flag = True
        dpca_enc.protect = []
        Z_enc_train = dpca_enc.fit_transform(RD_enc_train, trialX_enc_train)
        Z_enc_test  = dpca_enc.transform(RD_enc_test)
        Z_maint_test_on_enc = dpca_enc.transform(RD_maint_test)
        if hasattr(dpca_enc, "explained_variance_ratio_"):
            explvar_enc.append(dpca_enc.explained_variance_ratio_["s"])

        dpca_maint = dPCA(labels="st", regularizer="auto")
        dpca_maint.opt_regularizer_flag = True
        dpca_maint.protect = []
        Z_maint_train = dpca_maint.fit_transform(RD_maint_train, trialX_maint_train)
        Z_maint_test  = dpca_maint.transform(RD_maint_test)
        if hasattr(dpca_maint, "explained_variance_ratio_"):
            explvar_maint.append(dpca_maint.explained_variance_ratio_["s"])

        # pair-wise distances 
        all_dists["Enc_train"].append(stim_dist(Z_enc_train))
        all_dists["Maint_train"].append(stim_dist(Z_maint_train))
        all_dists["Enc_test"].append(stim_dist(Z_enc_test))
        all_dists["Maint_test"].append(stim_dist(Z_maint_test))
        all_dists["Maint onto Enc"].append(stim_dist(Z_maint_test_on_enc))

    assert len(set(split_signatures)) == n_iter, "Duplicate splits detected!"

    # STATISTICS  (KW + Dunn)
    measure_dict = {k: np.stack(v) for k, v in all_dists.items()}  # (n_iter, n_pairs)

    # one scalar per split/epoch: mean of all pair-wise distances
    box_datas = [measure_dict[ep].mean(axis=1) for ep in epochs]  

    # 1 ) Kruskal-Wallis
    H_kw, kw_p = sps.kruskal(*box_datas)
    print(f"\nKruskal–Wallis  H = {H_kw:.4f},   p = {kw_p:.4g}")

    # 2 ) Dunn post-hoc  (only if omnibus is significant)
    if kw_p < 0.05:
        mat = np.vstack(box_datas)                      
        dunn_df = sp.posthoc_dunn(mat, p_adjust="fdr_bh")
        dunn_df.index = epochs
        dunn_df.columns = epochs
        print("\nDunn post-hoc p-values (FDR-BH corrected)")
        print(dunn_df)
    else:
        dunn_df = None
        print("Omnibus test not significant → skipping post-hoc.")

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # TRAIN PLOT 
    train_data = [box_datas[epochs.index("Enc_train")], 
                 box_datas[epochs.index("Maint_train")]]
    bp_train = ax1.boxplot(
        train_data,
        labels=["Encoding Train", "Maintenance Train"],
        patch_artist=True,
        medianprops={"color": "k"},
        showfliers=False,
    )
    ax1.set_ylabel("Euclidean distance (a.u.)")
    ax1.set_title("Train Periods (n={} splits)".format(n_iter))
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Overlay individual splits for train
    for i, y in enumerate(train_data):
        x = np.random.normal(i + 1, 0.08, size=len(y))
        ax1.plot(x, y, "o", color="k", alpha=0.7, ms=6)
    
    # TEST PLOT (right)
    test_data = [box_datas[epochs.index("Enc_test")],
                box_datas[epochs.index("Maint_test")],
                box_datas[epochs.index("Maint onto Enc")]]
    bp_test = ax2.boxplot(
        test_data,
        labels=["Encoding Test", "Maintenance Test", "Maint→Enc"],
        patch_artist=True,
        medianprops={"color": "k"},
        showfliers=False,
    )
    ax2.set_ylabel("Euclidean distance (a.u.)")
    ax2.set_title("Test Periods (n={} splits)".format(n_iter))
    ax2.set_ylim(0, 10)  
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    
    for i, y in enumerate(test_data):
        x = np.random.normal(i + 1, 0.08, size=len(y))
        ax2.plot(x, y, "o", color="k", alpha=0.7, ms=6)
    
    plt.tight_layout()
    
    if savefig_path:
        plt.savefig(savefig_path, bbox_inches="tight")
        print(f"Figure saved to {savefig_path}")
    plt.show()

    return measure_dict, kw_p, dunn_df, explvar_enc, explvar_maint

measure_dict, kw_p, dunn_df, explE, explD = run_dpca_distance_resampling_E_vs_D(
                trialE,
                trialD,
                n_iter        = 100,              
                selected_dpcs = (0,1,2,3,4),      # which dPCs enter the distance
                stim_indices  = (0,1,2,3,4),      # which of the 5 stimuli to keep
                savefig_path  = "/home/daria/kms_enc_2.eps",
                random_seed   = 1
        )


n_iter, n_dpcs = explE.shape
labels = [f"dPC{i+1}" for i in range(n_dpcs)]

fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# Encoding
ax[0].boxplot(explE * 100, labels=labels, patch_artist=True,
              medianprops={"color": "k"})
ax[0].set_title("Encoding")
ax[0].set_ylabel("explained variance  (%)")

# Maintenance
ax[1].boxplot(explD * 100, labels=labels, patch_artist=True,
              medianprops={"color": "k"})
ax[1].set_title("Maintenance")

plt.suptitle(f"Variance explained across {n_iter} splits")
plt.tight_layout()
plt.savefig("/home/daria/enc_maintenance_training_var.eps", format='eps', bbox_inches='tight', dpi=300)
plt.show()

# Same but for Early vs Late maintenance

def run_dpca_distance_resampling_stratified(
    trialD,
    n_iter=100,
    time_split=624,
    selected_dpcs=(0, 1, 2, 3, 4),
    stim_indices=(0, 1, 2, 3, 4),
    savefig_path=None,
    random_seed=5,
):
    """
    trialD shape  : (n_reps, n_neurons, n_stimuli, n_time)
    Returns       :
        measure_dict       dict(epoch → np.array)   distance matrices
        kw_p               float                    omnibus p-value
        dunn_df            pd.DataFrame             post-hoc p-values
        explvar_early_arr  (n_iter, #dpcs)          variance explained
        explvar_late_arr   (n_iter, #dpcs)          variance explained
    """

    np.random.seed(random_seed)

    n_reps, n_neurons, n_stimuli, n_time = trialD.shape
    n_train = n_reps // 2          # e.g. 3
    n_test  = n_reps - n_train     # e.g. 3

    epochs = [
        "Early_train",
        "Late_train",
        "Early_test",
        "Late_test",
        "Late onto Early",
    ]

    def center(x):
        """mean-center over neurons (axis=1)"""
        m = np.mean(x.reshape(x.shape[0], -1), axis=1, keepdims=True)
        return x - m[:, None]

    def stim_dist(Z_epoch):
        """pair-wise Euclidean distance between stimulus means
        in low-D dPCA space"""
        stim_means = []
        for s in stim_indices:
            traj = Z_epoch["s"][selected_dpcs, s, :]  # (n_dpcs, time)
            stim_means.append(np.mean(traj, axis=1))  # (n_dpcs,)
        stim_means = np.stack(stim_means, axis=0)
        return pdist(stim_means, metric="euclidean") 

    all_dists      = defaultdict(list)
    explvar_early  = []
    explvar_late   = []
    split_signatures = []

    for it in range(n_iter):
        train_idx_bystim, test_idx_bystim, split_signature = [], [], []
        for s in stim_indices:
            idx = np.random.permutation(n_reps)
            train_idx_bystim.append(idx[:n_train])
            test_idx_bystim.append(idx[n_train:])
            split_signature.append((tuple(idx[:n_train]), tuple(idx[n_train:])))
        split_signatures.append(tuple(split_signature))

        t0 = time_split
        t1 = n_time - time_split

        trialX_early_train = np.zeros((n_train, n_neurons, n_stimuli, t0))
        trialX_early_test  = np.zeros((n_test,  n_neurons, n_stimuli, t0))
        trialX_late_train  = np.zeros((n_train, n_neurons, n_stimuli, t1))
        trialX_late_test   = np.zeros((n_test,  n_neurons, n_stimuli, t1))

        for s_idx, stim in enumerate(stim_indices):
            tr_idx = train_idx_bystim[s_idx]
            te_idx = test_idx_bystim[s_idx]

            # early
            trialX_early_train[:, :, s_idx, :] = trialD[tr_idx, :, stim, :t0]
            trialX_early_test[:,  :, s_idx, :] = trialD[te_idx, :, stim, :t0]
            # late
            trialX_late_train[:,  :, s_idx, :] = trialD[tr_idx, :, stim, t0:]
            trialX_late_test[:,   :, s_idx, :] = trialD[te_idx, :, stim, t0:]

        # trial averages (for dPCA fit/transform) 
        RD_early_train = np.mean(trialX_early_train, axis=0)
        RD_early_test  = np.mean(trialX_early_test,  axis=0)
        RD_late_train  = np.mean(trialX_late_train,  axis=0)
        RD_late_test   = np.mean(trialX_late_test,   axis=0)

        # mean-center neurons 
        RD_early_train = center(RD_early_train)
        RD_early_test  = center(RD_early_test)
        RD_late_train  = center(RD_late_train)
        RD_late_test   = center(RD_late_test)

        # dPCA fits 
        dpca_early = dPCA(labels="st", regularizer="auto")
        dpca_early.opt_regularizer_flag = True
        dpca_early.protect = ['t']
        Z_early_train = dpca_early.fit_transform(RD_early_train, trialX_early_train)
        Z_early_test  = dpca_early.transform(RD_early_test)
        Z_late_test_on_early = dpca_early.transform(RD_late_test)
        if hasattr(dpca_early, "explained_variance_ratio_"):
            explvar_early.append(dpca_early.explained_variance_ratio_["s"])

        dpca_late = dPCA(labels="st", regularizer="auto")
        dpca_late.opt_regularizer_flag = True
        dpca_late.protect = ['t']
        Z_late_train = dpca_late.fit_transform(RD_late_train, trialX_late_train)
        Z_late_test  = dpca_late.transform(RD_late_test)
        if hasattr(dpca_late, "explained_variance_ratio_"):
            explvar_late.append(dpca_late.explained_variance_ratio_["s"])

        # pair-wise distances 
        all_dists["Early_train"].append(stim_dist(Z_early_train))
        all_dists["Late_train"].append(stim_dist(Z_late_train))
        all_dists["Early_test"].append(stim_dist(Z_early_test))
        all_dists["Late_test"].append(stim_dist(Z_late_test))
        all_dists["Late onto Early"].append(stim_dist(Z_late_test_on_early))

    assert len(set(split_signatures)) == n_iter, "Duplicate splits detected!"

    # STATISTICS  (KW + Dunn)
    measure_dict = {k: np.stack(v) for k, v in all_dists.items()}  # (n_iter, n_pairs)

    # one scalar per split/epoch: mean of all pair-wise distances
    box_datas = [measure_dict[ep].mean(axis=1) for ep in epochs]   # list of 5 arrays

    # 1 ) Kruskal-Wallis 
    H_kw, kw_p = sps.kruskal(*box_datas)
    print(f"\nKruskal–Wallis  H = {H_kw:.4f},   p = {kw_p:.4g}")

    # 2 ) Dunn post-hoc  (only if omnibus is significant)
    if kw_p < 0.05:
        mat = np.vstack(box_datas)                      # shape (5, n_iter)
        dunn_df = sp.posthoc_dunn(mat, p_adjust="fdr_bh")
        dunn_df.index = epochs
        dunn_df.columns = epochs
        print("\nDunn post-hoc p-values (FDR-BH corrected)")
        print(dunn_df)
    else:
        dunn_df = None
        print("Omnibus test not significant → skipping post-hoc.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # TRAIN PLOT 
    train_data = [box_datas[epochs.index("Early_train")], 
                 box_datas[epochs.index("Late_train")]]
    bp_train = ax1.boxplot(
        train_data,
        labels=["Early Train", "Late Train"],
        patch_artist=True,
        medianprops={"color": "k"},
        showfliers=False,
    )
    ax1.set_ylabel("Euclidean distance (a.u.)")
    ax1.set_ylim(0, 10) 
    ax1.set_title("Train Periods (n={} splits)".format(n_iter))
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Overlay individual splits for train
    for i, y in enumerate(train_data):
        x = np.random.normal(i + 1, 0.08, size=len(y))
        ax1.plot(x, y, "o", color="k", alpha=0.7, ms=6)
    
    # TEST PLOT 
    test_data = [box_datas[epochs.index("Early_test")],
                box_datas[epochs.index("Late_test")],
                box_datas[epochs.index("Late onto Early")]]
    bp_test = ax2.boxplot(
        test_data,
        labels=["Early Test", "Late Test", "Late→Early"],
        patch_artist=True,
        medianprops={"color": "k"},
        showfliers=False,
    )
    ax2.set_ylabel("Euclidean distance (a.u.)")
    ax2.set_title("Test Periods (n={} splits)".format(n_iter))
    ax2.set_ylim(0, 5) 
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Overlay individual splits for test
    for i, y in enumerate(test_data):
        x = np.random.normal(i + 1, 0.08, size=len(y))
        ax2.plot(x, y, "o", color="k", alpha=0.7, ms=6)
    
    plt.tight_layout()
    
    if savefig_path:
        plt.savefig(savefig_path, bbox_inches="tight")
        print(f"Figure saved to {savefig_path}")
    plt.show()

    explvar_early_arr = np.vstack(explvar_early)
    explvar_late_arr  = np.vstack(explvar_late)

    return measure_dict, kw_p, dunn_df, explvar_early_arr, explvar_late_arr

measure_dict, kw_p, dunn_df, explvar_early, explvar_late = run_dpca_distance_resampling_stratified(
     trialD,
     n_iter=100,
     time_split=624,
     selected_dpcs=[0, 1, 2, 3, 4],
     stim_indices=[0, 1, 2, 3, 4],
     savefig_path='/home/daria/icant_2.eps',
)
# Plot variance explained
plt.boxplot(explvar_early[:, :5]) # Or explvar_late
plt.title('Explained variance: Early')
plt.show()
