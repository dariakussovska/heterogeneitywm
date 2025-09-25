cv2 = zeros(1, 902);

% Loop through each cell to calculate the Cv2
for i = 1:length(project.neuron_spikes)
    % Get the spike times for the current neuron
    spikeTimes = project.neuron_spikes{i};
    
    % Check if there are enough spikes to compute ISIs
    if length(spikeTimes) > 2
        % Calculate the interspike intervals (ISI)
        ISI = diff(spikeTimes); % Time differences between consecutive spikes
        
        % Initialize array for Cv2 values
        Cv2_values = zeros(1, length(ISI) - 1);
        
        % Loop through each pair of adjacent ISIs
        for j = 1:length(ISI) - 1
            % Compute Cv2 using the formula
            Cv2_values(j) = 2 * abs(ISI(j+1) - ISI(j)) / (ISI(j+1) + ISI(j));
        end
        
        % Compute the average Cv2 for the current neuron
        cv2(i) = mean(Cv2_values);
    else
        % If not enough spikes to calculate Cv2, set it to NaN
        cv2(i) = NaN;
    end
end
