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

# Same but for Early vs Late maintenance
trialE = np.load(f"/./trialE.npy")
trialD = np.load(f"/./trialD.npy")

trialX = trialE

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
     savefig_path='/./03_dpca/var_explained_late_early.eps',
)
# Plot variance explained
plt.boxplot(explvar_early[:, :5]) # Or explvar_late
plt.title('Explained variance: Early')
plt.show()
