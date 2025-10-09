% Extract the ACG matrix (wide or narrow)
acg_values = spikes.acg.wide_norm;  % Use acg_narrow if needed

% Compute the mean ACG for each neuron (column-wise mean)
mean_acg_values = mean(acg_values, 1, 'omitnan');  % Ignore NaNs if present

% Display the first few mean ACG values
disp('Mean ACG values for first 5 neurons:');
disp(mean_acg_values(1:5));
