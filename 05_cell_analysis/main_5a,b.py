import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

mat_file_path = '../data/cell_analysis/acg.mat'  
data = loadmat(mat_file_path)
num_neurons = 902
acg_narrow = data['acg']['narrow'][0,0]  

plots_per_row = 5  
num_rows = (num_neurons + plots_per_row - 1) // plots_per_row 

fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(20, num_rows * 3))  
axes = axes.flatten() 

# create time axis from -50 ms to 50 ms for 201 points
time_axis = np.linspace(-50, 50, 201)  # generate time axis for 201 points (for wide ACG) 

mean_values = np.zeros(num_neurons)

# plot ACG for each neuron and calculate mean values
for neuron_index in range(num_neurons):
    # extract the ACG for the selected neuron (column)
    neuron_acg = acg_narrow[:, neuron_index]  # get the column corresponding to the selected neuron
    axes[neuron_index].plot(time_axis, neuron_acg, linewidth=1.5)
    axes[neuron_index].set_title(f'Neuron {neuron_index + 1})', fontsize=8)
    axes[neuron_index].set_xlabel('Time (ms)', fontsize=6)
    axes[neuron_index].set_ylabel('ACG', fontsize=6)
    axes[neuron_index].set_xlim([-50, 50])  # Set x-axis limits
    axes[neuron_index].grid(True)

for ax in axes[num_neurons:]:
    ax.axis('off')

plt.tight_layout()
plt.show()
