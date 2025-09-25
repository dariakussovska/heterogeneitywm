function plot_electrodes(electrodes, templatePath, x_slice)
% plotElectrodesSagittal plots electrodes on a sagittal MRI slice.
%
% Inputs:
%   electrodes   - structure array with fields:
%                     .coords : [x, y, z] in MNI mm
%                     .region : string (e.g., 'dACC', 'Amygdala', etc.)
%   templatePath - string; full path to the MRI NIfTI file 
%                  (e.g., '\Users\darikussovska\Desktop\PROJECT\electrodes plotting\CIT168_T1w_1mm_MNI.nii')
%   x_slice      - sagittal slice index (in voxel space) at which to display electrodes
%
% Example usage:
%   electrodes = [
%       struct('coords',[3.96,29.04,25.04],'region','dACC');
%       struct('coords',[3.81,23.59,58.16],'region','pre-SMA');
%       struct('coords',[16.44,-8.80,-19.37],'region','Amygdala');
%       ... % etc.
%   ];
%   templatePath = 'C:\Users\darikussovska\Desktop\CIT168_T1w_1mm_MNI.nii';
%   x_slice = 90;
%   plotElectrodesSagittal(electrodes, templatePath, x_slice);

%% 1) Define region-to-color mapping (as HEX strings)
% Use the HEX codes provided:
regionMap = containers.Map;
regionMap('OFC')       = '#794252';  % Red
regionMap('ACC')   = '#B8385E';  % Blue
regionMap('pre-SMA')    = '#1E427E';  % Dark Red
regionMap('HPC')= '#394E3A';  % Dark Blue
regionMap('vmPFC')      = '#3EA248';  % Purple

%% 2) Inline HEX-to-RGB conversion (no external helper needed)
% This lambda function converts a HEX string like '#EF4743' to an [R G B] vector in [0,1]
hex2rgb = @(hexStr) [hex2dec(hexStr(2:3)), hex2dec(hexStr(4:5)), hex2dec(hexStr(6:7))] / 255;

%% 3) Load the MRI volume using SPM
V = spm_vol(templatePath);
mri_data = spm_read_vols(V);
affine_mat = V.mat;   % transforms voxel -> MNI
affine_inv = inv(affine_mat);  % transforms MNI -> voxel

%% 4) Convert electrode MNI coordinates to voxel space
N = length(electrodes);
voxel_coords = zeros(N, 3);
for i = 1:N
    mni_coords = electrodes(i).coords;  % [x y z] in MNI mm
    mni_hom = [mni_coords, 1]';          % 4x1 homogeneous coordinate
    voxel_hom = affine_inv * mni_hom;      % convert to voxel space
    voxel_coords(i,:) = round(voxel_hom(1:3))';
end

%% 5) Select electrodes near the chosen sagittal slice
% Adjust the threshold as needed; here we use ±6 voxels.
threshold = 10;
indices = find(abs(voxel_coords(:,1) - x_slice) <= threshold);

%% 6) Extract the sagittal slice from the MRI volume
if x_slice < 1 || x_slice > size(mri_data,1)
    error('x_slice is out of the volume range.');
end
sagittal_slice = squeeze(mri_data(x_slice,:,:));  % sagittal slice: dimensions [Y, Z]
% Adjust orientation: remove or modify these lines until you get the desired view.
sagittal_slice = rot90(sagittal_slice);   
sagittal_slice = flipud(sagittal_slice);

%% 7) Create figure and plot the sagittal slice with axes visible
figure('Color', 'w');  % white background for the figure
imagesc(sagittal_slice);
colormap(gray);
axis equal;
axis on;          % show axis ticks
set(gca, 'Color', 'w');  % white background for the axes (optional)
set(gca, 'YDir', 'normal');
title(sprintf('Sagittal Slice @ x = %d', x_slice), 'FontSize', 14, 'FontWeight', 'bold');
hold on;

%% 8) Overlay electrodes on the sagittal slice (vectorized per region)
%% 8) Fully vectorized: plot all electrodes near the sagittal slice at once (no alpha)
marker_size = 50;

% Find all electrodes near the x_slice
near_mask = abs(voxel_coords(:,1) - x_slice) <= threshold;
electrodes_near = electrodes(near_mask);
voxel_near = voxel_coords(near_mask, :);

% Preallocate
vy_all = voxel_near(:, 2);
vz_all = voxel_near(:, 3);
N_near = length(electrodes_near);
colors_all = zeros(N_near, 3);  % RGB colors

% Assign RGB color based on region
for i = 1:N_near
    reg = electrodes_near(i).region;
    if isKey(regionMap, reg)
        colors_all(i, :) = hex2rgb(regionMap(reg));
    else
        colors_all(i, :) = [1, 1, 0];  % Yellow for unknown regions
    end
end

% Plot all at once (no transparency)
scatter(vy_all, vz_all, marker_size, colors_all, 'filled', 'MarkerEdgeColor', 'k');

%% 9) Relabel the axes in MNI coordinates
% Define desired MNI Y and Z coordinates (e.g., -80 to 80 in steps of 20)
desired_MNI_y = -100:20:80;
desired_MNI_z = -60:20:80;

% Convert to voxel indices using inverse affine:
voxel_y = round((desired_MNI_y - V.mat(2,4)) / V.mat(2,2)) + 1;
voxel_z = round((desired_MNI_z - V.mat(3,4)) / V.mat(3,3)) + 1;

% Set tick positions and labels
set(gca, 'XTick', voxel_y, 'XTickLabel', arrayfun(@num2str, desired_MNI_y, 'UniformOutput', false));
set(gca, 'YTick', voxel_z, 'YTickLabel', arrayfun(@num2str, desired_MNI_z, 'UniformOutput', false));
%% 10) Add legend using dummy off-screen points
hold on;
regionHandles = zeros(1, length(keys(regionMap)));
regionNames = keys(regionMap);
for j = 1:length(regionNames)
    regName = regionNames{j};
    regionHandles(j) = scatter(-100, -100, marker_size, hex2rgb(regionMap(regName)), 'filled', 'DisplayName', regName);
end
legend(regionHandles, regionNames, 'Location', 'bestoutside');

%% 11) Save the figure (optional)
saveas(gcf, sprintf('Sagittal_x%d.eps', x_slice), 'epsc');
end
