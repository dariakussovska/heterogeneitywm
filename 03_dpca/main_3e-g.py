import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from dPCA.dPCA import dPCA 

trialE = np.load(f"/home/daria/PROJECT/trialE.npy")
trialD = np.load(f"/home/daria/PROJECT/trialD.npy")

trialX = trialE

# 1. Split trialD into early and late maintenance
trialD_early = trialD[:, :, :, :624]   # shape: (6, 58, 5, 624)
trialD_late  = trialD[:, :, :, 624:]   # shape: (6, 58, 5, 625)

RD_early = np.mean(trialD_early, axis=0)  
RD_late  = np.mean(trialD_late, axis=0)  

RD_early_mean = np.mean(RD_early.reshape(58, -1), axis=1)[:, None, None] 
RD_late_mean  = np.mean(RD_late.reshape(58, -1), axis=1)[:, None, None]  

RD_early_centered = RD_early - RD_early_mean  # shape: (58, 5, 624)
RD_late_centered  = RD_late - RD_late_mean    # shape: (58, 5, 625)

print("Early Maintenance:", RD_early_centered.shape)
print("Late Maintenance:", RD_late_centered.shape)

n_trials = trialD.shape[0]
n_train = int(n_trials * 0.5)
n_test = n_trials - n_train

# Create shuffled indices for train/test split
np.random.seed(2)  # for reproducibility
indices = np.random.permutation(n_trials)
train_idx, test_idx = indices[:n_train], indices[n_train:]

# Split trialD into early and late maintenance 
trialD_early = trialD[:, :, :, :624]   
trialD_late  = trialD[:, :, :, 624:]  

early_train = trialD_early[train_idx] 
early_test  = trialD_early[test_idx]

late_train = trialD_late[train_idx]
late_test  = trialD_late[test_idx]

RD_early_train = np.mean(early_train, axis=0)  
RD_early_test  = np.mean(early_test, axis=0)

RD_late_train = np.mean(late_train, axis=0)
RD_late_test  = np.mean(late_test, axis=0)

RD_early_train_mean = np.mean(RD_early_train.reshape(RD_early_train.shape[0], -1), axis=1)[:, None, None]
RD_early_test_mean  = np.mean(RD_early_test.reshape(RD_early_test.shape[0], -1), axis=1)[:, None, None]

RD_late_train_mean = np.mean(RD_late_train.reshape(RD_late_train.shape[0], -1), axis=1)[:, None, None]
RD_late_test_mean  = np.mean(RD_late_test.reshape(RD_late_test.shape[0], -1), axis=1)[:, None, None]

RD_early_train_centered = RD_early_train - RD_early_train_mean
RD_early_test_centered  = RD_early_test - RD_early_test_mean

RD_late_train_centered = RD_late_train - RD_late_train_mean
RD_late_test_centered  = RD_late_test - RD_late_test_mean

dpca_early = dPCA(labels='st', regularizer='auto')
dpca_early.opt_regularizer_flag = True  # Enable regularization optimization
dpca_early.protect = []  # No axes are protected
Z_early_again = dpca_early.fit_transform(RD_early_train_centered, early_train)
Z_early_test = dpca_early.transform(RD_early_test_centered)

for key in dpca_early.explained_variance_ratio_:
    variance_components = dpca_early.explained_variance_ratio_[key]
    total = np.sum(variance_components)
    print(f"Total variance explained by '{key}': {total:.4f}")


# Marginalization key
marginalization_key = 's'

# Extract projections
Z_encoding_train = Z_early_test[marginalization_key]

# Define conditions to plot (e.g., stimuli 2-5)
condition_range = [0, 1, 2, 3, 4] 
colors = ['red', 'blue', 'green', 'black', 'purple']

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for condition_idx in condition_range:
    traj_train = Z_encoding_train[:3, condition_idx, :]  

    ax.plot(traj_train[0], traj_train[1], traj_train[2],
            color=colors[condition_idx], linewidth=1.5,
            label=f'Train Stimulus {condition_idx+1}')

    ax.scatter(traj_train[0, 0], traj_train[1, 0], traj_train[2, 0],
               color=colors[condition_idx], s=30, edgecolor='k', alpha=0.7)

ax.set_xlabel('dPC 1', fontsize=12)
ax.set_ylabel('dPC 2', fontsize=12)
ax.set_zlabel('dPC 3', fontsize=12)
ax.set_title('dPCA Neural Trajectories (Early)', fontsize=15)
ax.legend(fontsize=9)
ax.view_init(elev=20, azim=60)
ax.grid(True)

# Save or show
plt.savefig("/home/daria/PROJECT/late_on_early.eps", format='eps', bbox_inches='tight', dpi=300)
plt.show()
