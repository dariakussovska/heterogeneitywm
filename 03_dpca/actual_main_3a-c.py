


# ------------------------------------------------------------------
#  (1)  data arrive here
# ------------------------------------------------------------------
trialE = Xtrial        # encoding   (n_rep, n_neurons, n_stim, n_tE)
trialD = Dtrial        # maintenance(n_rep, n_neurons, n_stim, n_tD)

n_rep, n_neurons, n_stim, tE = trialE.shape
_,      _,           _,     tD = trialD.shape
n_train = n_rep // 2
n_test  = n_rep - n_train

# ------------------------------------------------------------------
#  (2)  one 50/50 split – stratified over stimuli
# ------------------------------------------------------------------
rng = np.random.default_rng(seed=0)

train_idx_by_stim, test_idx_by_stim = [], []
for s in range(n_stim):
    perm          = rng.permutation(n_rep)
    train_idx_by_stim.append(perm[:n_train])
    test_idx_by_stim.append(perm[n_train:])

# helper to pick the right repetitions
def gather(arr, idx_list):
    """stack the selected repetition indices for every stimulus"""
    out = []
    for stim, idx in enumerate(idx_list):
        out.append(arr[idx, :, stim, :])            # shape (n_sel, n_neurons, t)
    return np.stack(out, axis=2)                    # (n_sel, n_neurons, n_stim, t)

trialE_train   = gather(trialE, train_idx_by_stim)  # (n_train, n_neurons, n_stim, tE)
trialE_test    = gather(trialE, test_idx_by_stim)
trialD_train   = gather(trialD, train_idx_by_stim)  # (n_train, n_neurons, n_stim, tD)
trialD_test    = gather(trialD, test_idx_by_stim)

# ------------------------------------------------------------------
#  (3)  trial averages  + mean-centering over neurons
# ------------------------------------------------------------------
def centre_over_neurons(x):
    m = np.mean(x.reshape(x.shape[0], -1), axis=1, keepdims=True)
    return x - m[:, None]

RD_E_train = centre_over_neurons(np.mean(trialE_train, axis=0))   # (n_neurons, n_stim, tE)
RD_E_test  = centre_over_neurons(np.mean(trialE_test,  axis=0))
RD_D_train = centre_over_neurons(np.mean(trialD_train, axis=0))   # (n_neurons, n_stim, tD)
RD_D_test  = centre_over_neurons(np.mean(trialD_test,  axis=0))

dpca_enc = dPCA(labels='st', regularizer='auto')
dpca_enc.opt_regularizer_flag = True  # Enable regularization optimization
dpca_enc.protect = []  # No axes are protected
Z = dpca_enc.fit_transform(RD_E_train, trialE_train)

Z_test = dpca_enc.transform(RD_E_test)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Marginalization key
marginalization_key = 's'

# Extract projections
Z_encoding_train = Z_test[marginalization_key]

# Define conditions to plot (e.g., stimuli 2-5)
condition_range = [1, 2, 3, 4] 
colors = ['red', 'blue', 'green', 'black', 'purple']

# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot training data (thin solid lines)
for condition_idx in condition_range:
    traj_train = Z_encoding_train[:3, condition_idx, :]  # dPC1–3

    ax.plot(traj_train[0], traj_train[1], traj_train[2],
            color=colors[condition_idx], linewidth=1.5,
            label=f'Train Stimulus {condition_idx+1}')

    ax.scatter(traj_train[0, 0], traj_train[1, 0], traj_train[2, 0],
               color=colors[condition_idx], s=30, edgecolor='k', alpha=0.7)

# Labels and aesthetics
ax.set_xlabel('dPC 1', fontsize=12)
ax.set_ylabel('dPC 2', fontsize=12)
ax.set_zlabel('dPC 3', fontsize=12)
ax.set_title('dPCA Neural Trajectories (Delay on Delay)', fontsize=15)
ax.legend(fontsize=9)
ax.view_init(elev=20, azim=40)
ax.grid(True)

# Save or show
plt.savefig("/./03_dpca/enc_on_encoding_test.eps", format='eps', bbox_inches='tight', dpi=300)
plt.show()
