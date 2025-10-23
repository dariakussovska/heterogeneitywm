import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

mat_file_path = '../data/cell_analysis/acg.mat'  
data = loadmat(mat_file_path)
num_neurons = 902
acg_narrow = data['acg']['narrow'][0,0]  

# Create time axis from -50 ms to 50 ms for 201 points
time_axis = np.linspace(-50, 50, 201)

# Plot only neurons 10-20 (11 neurons total)
start_neuron = 20
end_neuron = 40
neurons_to_plot = range(start_neuron, end_neuron + 1)
num_neurons_to_plot = len(neurons_to_plot)

plots_per_row = 5
num_rows = (num_neurons_to_plot + plots_per_row - 1) // plots_per_row

fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(15, num_rows * 3))
if num_rows == 1:
    axes = [axes] if plots_per_row == 1 else axes.flatten()
else:
    axes = axes.flatten()

# Plot ACG for each selected neuron
for i, neuron_index in enumerate(neurons_to_plot):
    neuron_acg = acg_narrow[:, neuron_index]
    axes[i].plot(time_axis, neuron_acg, linewidth=1.5)
    axes[i].set_title(f'Neuron {neuron_index + 1}', fontsize=10)
    axes[i].set_xlabel('Time (ms)', fontsize=8)
    axes[i].set_ylabel('ACG', fontsize=8)
    axes[i].set_xlim([-50, 50])
    axes[i].grid(True)

# Hide any unused subplots
for j in range(num_neurons_to_plot, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
