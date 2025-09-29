import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

trialX = Xtrial
trialD = Dtrial

# Trial-averaged data 
R_train = np.mean(trialX, axis=0)  # (58, 5, 199)
RD_train = np.mean(trialD, axis=0)  # (58, 5, 499)
# Compute training mean per neuron
R_train_mean = np.mean(R_train.reshape(58, -1), axis=1)[:, None, None]
RD_train_mean = np.mean(RD_train.reshape(58, -1), axis=1)[:, None, None]
# Subtract mean from train and test data 
R_train_centered = R_train - R_train_mean
RD_train_centered = RD_train - RD_train_mean

print("Train Centered Shapes:")
print(R_train_centered.shape, RD_train_centered.shape)

dpca_maintenance = dPCA(labels='st', regularizer='auto')
dpca_maintenance.opt_regularizer_flag = True  # Enable regularization optimization
dpca_maintenance.protect = []  # No axes are protected
Z_maintenance = dpca_maintenance.fit_transform(RD_train_centered, trialD)

ZD_test = dpca_maintenance.transform(RD_train_centered)

for key in dpca_maintenance.explained_variance_ratio_:
    variance_components = dpca_maintenance.explained_variance_ratio_[key]
    total = np.sum(variance_components)
    print(f"Total variance explained by '{key}': {total:.4f}")

# Marginalization key
marginalization_key = 's'

# Extract projections
Z_encoding_train = ZD_test[marginalization_key]

# Define conditions to plot (e.g., stimuli 2-5)
condition_range = [1, 2, 3, 4] 
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
ax.set_title('dPCA Neural Trajectories: Train vs Test (Maintenance)', fontsize=15)
ax.legend(fontsize=9)
ax.view_init(elev=20, azim=40)
ax.grid(True)

# Save or show
plt.savefig("/home/daria/maintenance_onto_maintenance.eps", format='eps', bbox_inches='tight', dpi=300)
plt.show()

