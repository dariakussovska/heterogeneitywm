% Extract relevant metrics from project struct
mean_acg_values = project.Mean_ACG;  % Mean ACG values
peakA = project.peakA;  % First peak amplitude
peakB = project.peakB;  % Second peak amplitude
acg_tau_rise = project.burstIndex_Royer2012;  % Tau rise values

% Initialize classification arrays
num_neurons = length(mean_acg_values);
cell_types = strings(1, num_neurons);  % Classification array

% Loop through each neuron and classify
for i = 1:num_neurons
    if mean_acg_values(i) > 6
        % Classify as Interneuron if Mean ACG is high
        if peakA(i) < peakB(i) %|| acg_tau_rise(i) < 1
            cell_types(i) = "Interneuron";
        else
            cell_types(i) = "Unknown";  % Mean ACG suggests Interneuron, but other features don't match
        end
    elseif mean_acg_values(i) < 4
        % Classify as Pyramidal if Mean ACG is low
        if peakA(i) > peakB(i) || acg_tau_rise(i) > 2
            cell_types(i) = "Pyramidal";
        else
            cell_types(i) = "Unknown";  % Mean ACG suggests Pyramidal, but other features don't match
        end
    else
        % If Mean ACG is in the 5-6 range, label as unknown
        cell_types(i) = "Unknown";
    end
end

% Store classifications in the project struct
project.cell_types = cell_types;

% Display classification counts
num_interneurons = sum(cell_types == "Interneuron");
num_pyramidal = sum(cell_types == "Pyramidal");
num_unknown = sum(cell_types == "Unknown");

fprintf('Number of Interneurons: %d\n', num_interneurons);
fprintf('Number of Pyramidal Cells: %d\n', num_pyramidal);
fprintf('Number of Unknown Cells: %d\n', num_unknown);
