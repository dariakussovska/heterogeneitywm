% Mean ACG per neuron (raw, in spikes/s), as used for the spectral clustering.
% Run from data/cell_analysis/. acg.mat contains acg.wide (1001 x 902: ±500 ms, 1 ms bins)

load('acg.mat');                                          % loads the struct 'acg'
acg_values = acg.wide;

% Mean of the raw wide ACG for each neuron (column-wise mean) -> Cell_metrics.xlsx, column mean_acg
mean_acg_values = mean(acg_values, 1, 'omitnan');

% Normalized mean ACG (each ACG divided by its maximum, zero-lag bin = 0) -> column norm_acg
norm_acg_values = mean(normalize_acg_to_unit(acg_values), 1, 'omitnan');

disp('Mean ACG values for first 5 neurons:');
disp(mean_acg_values(1:5));
