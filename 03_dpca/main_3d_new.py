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

trialE = np.load(f"../trialE.npy")
trialD = np.load(f"../trialD.npy")

trialX = trialE

### ENCODING AND MAINTENANCE

# dPCA distance-resampling (Encoding vs Maintenance)

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

    # 1) Kruskal-Wallis
    H_kw, kw_p = sps.kruskal(*box_datas)
    print(f"\nKruskal–Wallis  H = {H_kw:.4f},   p = {kw_p:.4g}")

    # 2) Dunn post-hoc  (only if omnibus is significant)
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
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    
    for i, y in enumerate(test_data):
        x = np.random.normal(i + 1, 0.08, size=len(y))
        ax2.plot(x, y, "o", color="k", alpha=0.7, ms=6)
    
    # Add significance stars if post-hoc exists
    if dunn_df is not None:
        # Define pairs to test for significance
        pairs_to_test = [
            (0, 1),  # Enc_train vs Maint_train
            (0, 2),  # Enc_train vs Enc_test
            (1, 3),  # Maint_train vs Maint_test
        ]
        
        y_max_train = max([max(d) for d in train_data])
        y_max_test = max([max(d) for d in test_data])
        
        # Add significance bars for train plot
        for pair in pairs_to_test[:1]:  # Only train comparison for left plot
            i, j = pair
            if i < len(train_data) and j < len(train_data):
                p_val = dunn_df.iloc[i, j]
                if p_val < 0.001:
                    star = '***'
                elif p_val < 0.01:
                    star = '**'
                elif p_val < 0.05:
                    star = '*'
                else:
                    star = ''
                
                if star:
                    # Draw line
                    ax1.plot([i+1, j+1], [y_max_train*1.05, y_max_train*1.05], 'k-', lw=1)
                    # Add star
                    ax1.text((i+1 + j+1)/2, y_max_train*1.08, star, 
                            ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if savefig_path:
        plt.savefig(savefig_path, bbox_inches="tight")
        print(f"Figure saved to {savefig_path}")
    plt.show()

    return measure_dict, kw_p, dunn_df, explvar_enc, explvar_maint

# Run the main analysis
measure_dict, kw_p, dunn_df, explE, explD = run_dpca_distance_resampling_E_vs_D(
                trialE,
                trialD,
                n_iter        = 100,              
                selected_dpcs = (0,1,2,3,4),      # which dPCs enter the distance
                stim_indices  = (0,1,2,3,4),      # which of the 5 stimuli to keep
                savefig_path  = "./variance.eps",
                random_seed   = 1
        )

# Variance explained plots with significance testing
n_iter, n_dpcs = explE.shape
labels = [f"dPC{i+1}" for i in range(n_dpcs)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Convert to percentages
explE_pct = explE * 100
explD_pct = explD * 100

# Encoding variance
bp1 = ax1.boxplot(explE_pct, labels=labels, patch_artist=True,
                  medianprops={"color": "k"}, showfliers=False)
ax1.set_title("Encoding Period")
ax1.set_ylabel("Explained Variance (%)")
ax1.grid(axis="y", linestyle="--", alpha=0.5)

# Maintenance variance  
bp2 = ax2.boxplot(explD_pct, labels=labels, patch_artist=True,
                  medianprops={"color": "k"}, showfliers=False)
ax2.set_title("Maintenance Period")
ax2.grid(axis="y", linestyle="--", alpha=0.5)

# Add individual points
for i in range(n_dpcs):
    # Encoding
    x_enc = np.random.normal(i+1, 0.1, size=n_iter)
    ax1.plot(x_enc, explE_pct[:, i], 'o', color='k', alpha=0.6, ms=4)
    
    # Maintenance
    x_maint = np.random.normal(i+1, 0.1, size=n_iter)
    ax2.plot(x_maint, explD_pct[:, i], 'o', color='k', alpha=0.6, ms=4)

# Statistical testing between Encoding and Maintenance for each dPC
significance_results = []
for i in range(n_dpcs):
    # Wilcoxon signed-rank test for paired data
    stat, p_val = sps.wilcoxon(explE_pct[:, i], explD_pct[:, i])
    significance_results.append((i, stat, p_val))

# Add significance stars above each dPC pair
y_max = max(np.max(explE_pct), np.max(explD_pct))
for i, stat, p_val in significance_results:
    if p_val < 0.001:
        stars = '***'
    elif p_val < 0.01:
        stars = '**'
    elif p_val < 0.05:
        stars = '*'
    else:
        stars = 'ns'
    
    # Draw line connecting the two boxes
    ax1.plot([i+1, i+1], [y_max*0.95, y_max*1.05], 'k-', lw=1, alpha=0.7)
    ax2.plot([i+1, i+1], [y_max*0.95, y_max*1.05], 'k-', lw=1, alpha=0.7)
    
    # Add stars in the middle
    mid_x = (i+1) + 2.5  # Position between the two subplots
    fig.text(mid_x/6.5, 0.92, stars, ha='center', va='bottom', 
             fontsize=10, fontweight='bold', transform=ax1.transAxes)

plt.suptitle(f"Variance Explained by Stimulus Dimension (n={n_iter} splits)", fontsize=14)
plt.tight_layout()
plt.savefig("./enc_maintenance_training_var.eps", format='eps', bbox_inches='tight', dpi=300)
plt.show()

# Print statistical results
print("\n" + "="*60)
print("STATISTICAL COMPARISON: Encoding vs Maintenance Variance")
print("="*60)
for i, stat, p_val in significance_results:
    print(f"dPC{i+1}: W = {stat:.4f}, p = {p_val:.4g}")
