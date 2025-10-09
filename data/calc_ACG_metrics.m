function acg_metrics = calc_ACG_metrics(spikes, sr)
    % Inputs:
    %   spikes - structure containing 'times' (cell array of spike times) and 'total' (number of spikes)
    %   sr - sampling rate
    
    % Ensure that spikes.times is a cell array
    if ~iscell(spikes.times)
        error('spikes.times must be a cell array');
    end

    numcells = length(spikes.times);  % Number of neurons

    % Initialize metrics
    ThetaModulationIndex = nan(1, numcells);
    BurstIndex_Royer2012 = nan(1, numcells);
    BurstIndex_Doublets = nan(1, numcells);

    % Define bins for ACG calculation
    bins_wide = 1001;  % Total 1001 bins for wide ACG (1s, 1ms bins)
    bins_narrow = 201;  % Total 201 bins for narrow ACG (100ms, 0.5ms bins)

    % Initialize matrices for storing ACGs
    acg_wide = zeros(bins_wide, numcells);  % 1001 bins
    acg_narrow = zeros(bins_narrow, numcells);  % 201 bins

    disp('Calculating narrow ACGs (100ms, 0.5ms bins) and wide ACGs (1s, 1ms bins)');
    tic
    for i = 1:numcells
        spike_times = spikes.times{i};  % Access spike times for this neuron

        if length(spike_times) > 1
            % Calculate narrow ACG (100ms, 0.5ms bins)
            acg_narrow(:, i) = autocorrelogram(spike_times, sr, 0.0005, 0.1, bins_narrow);

            % Calculate wide ACG (1s, 1ms bins)
            acg_wide(:, i) = autocorrelogram(spike_times, sr, 0.001, 1, bins_wide);

            % Metrics from narrow ACG (BurstIndex_Doublets)
            BurstIndex_Doublets(i) = max(acg_narrow(106:116, i)) / mean(acg_narrow(116:123, i));

            % Metrics from wide ACG (ThetaModulationIndex)
            ThetaModulationIndex(i) = (mean(acg_wide(601:640, i)) - mean(acg_wide(550:570, i))) / ...
                                      (mean(acg_wide(550:570, i)) + mean(acg_wide(601:640, i)));

            % Metrics from wide ACG (BurstIndex_Royer2012)
            BurstIndex_Royer2012(i) = mean(acg_wide(503:505, i)) / mean(acg_wide(700:800, i));
        else
            % If less than 2 spikes, the metrics are NaN
            acg_narrow(:, i) = NaN;
            acg_wide(:, i) = NaN;
            BurstIndex_Doublets(i) = NaN;
            ThetaModulationIndex(i) = NaN;
            BurstIndex_Royer2012(i) = NaN;
        end
    end
    toc

    % Store the results in a structure
    acg_metrics.acg_wide = acg_wide;
    acg_metrics.acg_narrow = acg_narrow;
    acg_metrics.thetaModulationIndex = ThetaModulationIndex;
    acg_metrics.burstIndex_Royer2012 = BurstIndex_Royer2012;
    acg_metrics.burstIndex_Doublets = BurstIndex_Doublets;
end

function acg = autocorrelogram(spike_times, Fs, bin_size, duration, expected_num_bins)
    % Initialize ACG
    acg = zeros(expected_num_bins, 1);  % Initialize the ACG with correct size
    
    % Compute time differences between spikes (for autocorrelation)
    for i = 1:length(spike_times)
        diffs = spike_times - spike_times(i);
        diffs(i) = [];  % Remove self-coincidence (lag = 0)
        
        % Convert differences to bins
        bins = round(diffs / bin_size);
        
        % Keep only valid bins within the expected range
        valid_bins = bins(abs(bins) <= floor(expected_num_bins / 2));  % Half for positive and half for negative lags
        
        % Increment the ACG at those valid bins
        for bin = valid_bins'
            acg(bin + floor(expected_num_bins / 2) + 1) = acg(bin + floor(expected_num_bins / 2) + 1) + 1;
        end
    end
    
    % Normalize by spike count and bin size
    acg = acg / (length(spike_times) * bin_size);
end
