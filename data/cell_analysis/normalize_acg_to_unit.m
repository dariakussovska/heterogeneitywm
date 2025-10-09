function [acg_unit] = normalize_acg_to_unit(acg)
% acg: [numBins x numNeurons] (e.g., acg_wide or acg_narrow). Here, we need acg_wide.
% Returns acg_unit with each column divided by its column max.
% Zero-lag bin is forced to 0 after normalization.
% acg = spikes.acg.acg_wide; You can put that in the command window to extract the acg_wide from each neuron. 

    acg_unit = acg;                       % copy
    [numBins, ~] = size(acg);
    zeroBin = ceil(numBins/2);            % middle bin (0 lag)

    % Column-wise max ignoring NaNs
    cmax = max(acg, [], 1, 'omitnan');

    % Avoid division by 0: mark columns with nonpositive/NaN max to skip
    good = cmax > 0 & ~isnan(cmax);

    % Normalize good columns
    acg_unit(:, good) = acg(:, good) ./ cmax(ones(numBins,1), good);

    % For bad columns (all zeros/NaNs), return NaNs to make it explicit
    acg_unit(:, ~good) = NaN;

    % Force zero-lag bin to 0 (prevents tiny numerical bumps)
    acg_unit(zeroBin, :) = 0;
end
