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

trialE = np.load(f"/./trialE.npy")
trialD = np.load(f"/./trialD.npy")

trialX = trialE

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
                savefig_path  = "/./03_dpca/variance.eps",
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
plt.savefig("/./03_dpca/enc_maintenance_training_var.eps", format='eps', bbox_inches='tight', dpi=300)
plt.show()
