# DOCUMENTATION 

## DATA EXTRACTION

1) Install the dataset (from https://www.nature.com/articles/s41597-024-02943-8) and put it in a folder on your local computer. Here, I am using the folder PROJECT, and the dataset files are in the folder 000469. 
2) Run the scripts data_extraction.py and data_extraction_2.py. 

- These scripts are aimed to make the extraction of necessary data for the analysis easier and more convenient.
- They take subject id, trial id, start time and end time of each trial, neuron id, stimulus identity id, spike rate, and individual spikes for every neuron in each trial from the fixation, encoding, delay, and probe periods.
- Running the first code extracts data in easy to use Excel sheets for every period (Fixation, Encoding (1-3), Delay, and Probe). We are only using session 2 (the Sternberg task, not the Screening task) for our analyses.
- Running the second code extracts trial information data. This includes the stimulus identity of the image in every trial, and the number of images presented in every trial to each subject (1-3). Image identity of 5 means that there was no image presented (in Encoding 2 and Encoding 3 only).
- Running the second code also extracts neuron metrics data, useful for future analysis. These will be useful for neuron classification afterwards. 
For plotting electrodes locations on an anatomical image (sagittal plane) of the brain, open the electrodes folder in the PROJECT folder. It includes a matlab struct with the locations of the electrodes and the area in the brain they come from. It also includes the MNI anatomical template used and a function for plotting the electrodes. Using x_slice, you need to specify the slice of the brain you want to see, and then call the function.
- Finally, the code extracts the corresponding brain areas from each of the electrodes. 

**Inputs:**

- nwb files from all subjects (taken directly from the dataset). They are in the form sub-{i+1}/sub-{i+1}_ses-2_ecephys+image.nwb and all of them are in a folder named 000469. 
- electrodes_locations.mat: a MATLAB file containing x, y, z coordinates of all electrode locations, as well as their regions in the brain 
- plot_electrodes.m: a MATLAB function used to plot the electrode coordinates on an anatomical image of the brain in the sagittal plane. 
- CIT168_T1w_1mm_MNI.nii: the anatomical image of the brain used for plotting the electrode coordinates 

**Outputs:**

- all_spike_rate_data_encoding1.xlsx: Excel file with the following characteristics: subject id (1-21), trial id (1-135), neuron id, stimulus id (0-4), image id (how the people who created the dataset generated the ids), spikes for every trial and neuron, spike rate for every trial and neuron, start and end time of the encoding 1 period of the corresponding trial. 

- all_spike_rate_data_encoding2.xlsx :everything like the previous file, but for encoding 2 period

- all_spike_rate_data_encoding3.xlsx :everything like the previous file, but for encoding 3 period

- all_spike_rate_data_fixation.xlsx: subject id (1-21), trial id (1-135), neuron id, spikes for every trial and neuron, spike rate for every trial and neuron, start and end time of the fixation period of the corresponding trial. 

- all_spike_rate_data_delay.xlsx: subject id (1-21), trial id (1-135), neuron id, spikes for every trial and neuron, spike rate for every trial and neuron, start and end time of the delay period of the corresponding trial. 

- all_spike_rate_data_probe.xlsx: subject id (1-21), trial id (1-135), neuron id, spikes for every trial and neuron, spike rate for every trial and neuron, start and end time of the probe period of the corresponding trial, and probe image identity (0-4) shown in that period. 

- trial_info.xlsx: an Excel file, containing the image identity presented in each trial in each subject, as well as the numbers of images presented during the specified trial. 

- all_spike_rate_metrics.xlsx: an Excel file, containing different metrics (CV2, firing rate, ISI, waveform peak and mean SNR projection distance, and isolation distance) for every recorded cell. 

**Figures:**

- response_accuracy_ranked.eps: an EPS file showing the response accuracy, ranked from lowest to highest, for all the subjects in the datasets (21) 

- response_time.eps: an EPS file showing the response time in accurate trials across subjects, separated into Loads 1, 2, and 3   

- MTL_plot.eps: an EPS file with the electrode locations in the amygdala and the hippocampus

- MFC_plot.eps: an EPS file with the electrode locations in the dACC, vmPFC, and dACC           

## STANDARDIZING THE DATA 

Here, we are preparing the data for subsequent analyses. We are standardizing, normalizing, aligning the data, and coming up with a new neuron id, so that we can pool all the neurons together for more statistical power. Important note: we are generating two types of excel files: ones that start with cleaned_{task period} and ones that start with graph_{task period}. We are doing this because the decoding also requires a different standardizing method for some of the figures, and we do not want to mess up with the original Excel sheets too much. 

- Run the code standardization.py
- This code standardizes the data by the start time of encoding period 1 to the existing Excel. This type of standardization is needed for our cross-temporal decoding analysis, where we run continuous decoding throughout the entire trial (and not only the delay period). 
- Make sure that you have downloaded new_trial_info.xlsx and trial_info_final.xlsx, as these are needed for the code. It is important to balance the design matrix and y matrix by the identity of images that are presented. For that, we need the new_trial_final.xlsx excel sheet with our new trial identities. As we will also use the balanced by load trial identities, we also add a new_trial_id column using the new_trial_info.xlsx file. 
- This code adds a new neuron id (subject id + 0 + neuron id) for easier pooling of the neurons.
- It saves all files in a new clean_data folder.
- The code creates another folder: graph_data, used in PSTH graphs. It merges data with trial information and standardizes the spikes, this time, based on the start of the corresponding period (delay standardized with start of delay, etc.)

**Inputs:** 

- all_spike_rate_data_encoding1.xlsx: Excel file with the following characteristics: subject id (1-21), trial id (1-135), neuron id, stimulus id (0-4), image id (how the people who created the dataset generated the ids), spikes for every trial and neuron, spike rate for every trial and neuron, start and end time of the encoding 1 period of the corresponding trial. 

- all_spike_rate_data_encoding2.xlsx :everything like the previous file, but for encoding 2 period

- all_spike_rate_data_encoding3.xlsx :everything like the previous file, but for encoding 3 period

- all_spike_rate_data_fixation.xlsx: subject id (1-21), trial id (1-135), neuron id, spikes for every trial and neuron, spike rate for every trial and neuron, start and end time of the fixation period of the corresponding trial. 

- all_spike_rate_data_delay.xlsx: subject id (1-21), trial id (1-135), neuron id, spikes for every trial and neuron, spike rate for every trial and neuron, start and end time of the delay period of the corresponding trial. 

- all_spike_rate_data_probe.xlsx: subject id (1-21), trial id (1-135), neuron id, spikes for every trial and neuron, spike rate for every trial and neuron, start and end time of the probe period of the corresponding trial, and probe image identity (0-4) shown in that period. 

- trial_info.xlsx: an Excel file, containing the image identity presented in each trial in each subject, as well as the numbers of images presented during the specified trial. 

- new_trial_info.xlsx: trial identities, ensuring pooling across neurons for GLM encoder. 

- new_trial_final.xlsx: new trial identities, ensuring easy pooling across neurons for the decoder. Generated as mentioned above.

**Outputs:** 

Folder clean_data containing: 

- cleaned_Encoding1.xlsx
- cleaned_Encoding2.xlsx
- cleaned_Encoding3.xlsx
- cleaned_Fixation.xlsx
- cleaned_Delay.xlsx
- cleaned_Probe.xlsx

These files contain standardized spikes (relative to the start time of encoding 1 period in each trial, or the fixation start time in each trial for the fixation period only), as well as new neuron identities (Neuron_ID_3) and new trial identities to all allow for easier access by the decoder. 

- graph_encoding1.xlsx : an Excel file with standardized spikes from Encoding 1 per subject, trial and neuron, with merged data for brain region, concept vs non-concept cell, preferred image identity of every neuron. 
- graph_delay.xlsx : an Excel file with standardized spikes from the Delay (relative to start of the corresponding trial in the Delay) per subject, trial and neuron, with merged data for brain region, concept vs non-concept cell, preferred image identity of every neuron (relative to Encoding 1 period). 
- graph_probe.xlsx: an Excel file with standardized spikes from the Probe (relative to start of the corresponding trial in the Probe) per subject, trial and neuron, with merged data for brain region, concept vs non-concept cell, preferred image identity of every neuron (relative to Encoding 1 period), with information about preferred vs non-preferred image identity in the probe itself. 
- graph_fixation.xlsx:  an Excel file with standardized spikes from the Fixation period (relative to the start of the corresponding trial in the Fixation) per subject, trial, and neuron. 

## CONCEPT CELLS

- Run the code concept_cell_defition.py
- This code performs statistical analysis on the spikings of all neurons in the encoding 1 period of all the trials. 
- It plots the concept cells firing rate with mean and standard error of the mean, throughout the five different image identities they react to. 
- It merges all_neuron_brain_regions_cleaned.xlsx (brain regions for each of the neurons) to the concept cell definition (specified from the Neuron_Check_Significant_All.xlsx file as Signi == Y for concept cell and Signi == N for a non-concept cell)
- Based on the preferred image identities, the code also adds preferred vs non-preferred stimulus columns to see how concept cells might react differently for their selective stimulus vs the rest.
- 05_merge_data.py then merges the significance of the neurons to all dataframes, adds brain region locations, as well as trial info. 

Run the code to calculate the total number of concept cells and the number in each region and plot a pie chart with the percentages. 
Here, we will use the spike data gathered from Encoding period 1, in order to classify neurons as concept cells or non-concept cells. We will also look at the brain regions of all the cells in the dataset. 

**Inputs:** 

all_spike_rate_data_encoding1.xlsx: Used for statistical analysis to classify the neurons as concept vs non-concept cells.

all_neuron_brain_regions_cleaned.xlsx. Given as a cleaned version of all the neurons in all the subjects and which region of the brain they are from. 

**Outputs:**

Neuron_Check_Significant_All.xlsx. An excel file containing the statistical analysis conducted of every neuron, using its individual spikes in all encoding 1 period trials. Contains p values, confidence intervals, top two image categories from which the statistical analysis is conducted and determines significance (Y if significant and N if non-significant – aka non concept cell). 

merged_significant_braing_regions_

**Figures:** 

concept_cell_1.eps and concept_cell_2.eps: two figures showing the firing rates for two concept cells across all stimulus identities in the Encoding 1 period. 

region_distribution.eps: an EPS file showing a pie chart the distribution of cells recorded from different brain areas.

concept_cell_distribution.eps: an EPS file showing a pie chart of the percentage of concept cells per brain region. 

## FIRING RATE GRAPHS 

Here, we are preparing the existing files for plotting the firing rates across concept and non-concept cells for preferred image identities versus non-preferred image identities. 

- First, we need to run the code 06_add_category to add the corresponding trial type ("Category" column) to say whether the trial was preferred or non-preferred for that particular neuron and subject id combination. 
- Under the 02_psth_decoding >> main_2a,b, we choose a neuron to plot. We run the code to load the data and filter by the number of images presented (to show only load 1 trials). We count spikes and convert to firing rates in the corresponding time bins, and subsequently smooth the data. We calculate calculate z-scores (subtracting the mean firing rate in the baseline and dividing by the standard deviation), and the confidence intervals for plotting later. Finally, we plot the z-scores and confidence intervals for the encoding, delay and probe periods.
- The script also includes the generation of a raster with the spike times of this particular neuron over the task epochs. 

**Inputs:**

- graph_encoding1.xlsx : an Excel file with standardized spikes from Encoding 1 per subject, trial and neuron, with merged data for brain region, concept vs non-concept cell, preferred image identity of every neuron. 
- graph_delay.xlsx : an Excel file with standardized spikes from the Delay (relative to start of the corresponding trial in the Delay) per subject, trial and neuron, with merged data for brain region, concept vs non-concept cell, preferred image identity of every neuron (relative to Encoding 1 period). 
- graph_probe.xlsx: an Excel file with standardized spikes from the Probe (relative to start of the corresponding trial in the Probe) per subject, trial and neuron, with merged data for brain region, concept vs non-concept cell, preferred image identity of every neuron (relative to Encoding 1 period), with information about preferred vs non-preferred image identity in the probe itself. 
- graph_fixation.xlsx:  an Excel file with standardized spikes from the Fixation period (relative to the start of the corresponding trial in the Fixation) per subject, trial, and neuron. 
- trial_info.xlsx: an Excel file, containing the image identity presented in each trial in each subject, as well as the numbers of images presented during the specified trial. 

**Outputs (Figures):** 

- single_PSTH_4052_load1.eps: A peri-stimulus time histogram of a concept cell during load 1 trials where the preferred image was presented vs not-presented in the encoding, maintenance, and probe task epochs. 
- raster_plots_4052.eps: a raster plot of the same concept cell with trials ordered by preferred vs non-preferred in the encoding, maintenance, and probe task epochs. 

## DECODING 

Here, we are trying to decoding stimulus identity from the maintenance period to see how information gets maintained. In addition, we are doing cross-temporal decoding to see how memoranda evolve through time, and we are doing within-subject decoding on brain regions. 

- We construct the y matrix from individual subject trial labels in the new_trial_final.xlsx file as shown in the code.
- Run 02_psth_decoding >> main_2d.py to create a design matrix for decoding analysis. Data is binned, normalized by z-scoring with individual neuron’s activity during baseline (fixation). You can choose which neurons to include in the data. The code also creates a corresponding y matrix with labels, performs a leave-one-out cross-validation, and a permutation test for significance testing. By varying the neurons that participate in the decoding analysis, as well as the loads, this code generates the .eps files that are used in Fig. 2f, Fig. 6i,j as well as Fig. S1 and Fig S5i,j.
- The script 02_psth_decoding >> main_2e.py performs a similar design matrix creation and permutation testing, but this time with all task phases. This is for the purpose of achieving cross-temporal decoding maps. Thresholds for clustering are performed as well. By varying the neurons that participate in the decoding analysis, as well as the loads, this code generates the .eps files that are used in Fig. 2d as well as Fig. S1d, e.
- The script 02_psth_decoding >> main_2f.py performs an analysis across participants. It filters participants with 3 or more concept cells identified and runs decoding with concept cells from different brain regions to see how stable the decoding accuracy is across participants. This uses the initial trial info excel sheet to ensure as many trials in each participant included as possible. This code generates the .eps files used in Fig. 2e and Fig. S1f.

**Inputs:**
- trial_info.xlsx: an Excel file, containing the image identity presented in each trial in each subject, as well as the numbers of images presented during the specified trial. 
- new_trial_final.xlsx: new trial identities, ensuring easy pooling across neurons for the decoder. Generated as mentioned above.

Folder clean_data containing: 

cleaned_Encoding1.xlsx
cleaned_Encoding2.xlsx
cleaned_Encoding3.xlsx
cleaned_Fixation.xlsx
cleaned_Delay.xlsx
cleaned_Probe.xlsx

These files contain standardized spikes (relative to the start time of encoding 1 period in each trial, or the fixation start time in each trial for the fixation period only), as well as new neuron identities (Neuron_ID_3) and new trial identities to all allow for easier access by the decoder. These are used for cross-temporal decoding. 

graph_delay.xlsx : an Excel file with standardized spikes from the Delay (relative to start of the corresponding trial in the Delay) per subject, trial and neuron, with merged data for brain region, concept vs non-concept cell, preferred image identity of every neuron (relative to Encoding 1 period). This is used for decoding from the maintenance period only.

**Outputs (Figures):** 

- decoding_timebins.eps, decoding_timebins_py, decoding_timebins_in: .eps files that include decoding accuracy during the maintenance period, using neural activity from all concept cells, or just PY and just IN concept cells, and varying the time window used for decoding.
- cross-temporal_decoding.eps, cross_temporal_decoding_py.eps, cross_temporal_decoding_in.eps: .eps files that include cross temporal decoding accuracy across task periods, using neural activity from all concept cells, or just PY and just IN concept cells. 
- brain_regions_decoding.eps: an .eps file that includes decoding accuracy during the maintenance period, using concept cells in different regions (MTL and AMY) and across participants


## POISSON ENCODING MODEL

Here, we fit the spike trains of individual neurons to a Poisson GLM model to see how they are modulated with task parameters such as load and reaction time.  

- The script 01_task >> main_e,f.py loads trial data and task relevant features. For this script we are using the trials, balanced by load, as this is our main point of interest.
- It prepares an X matrix with the relevant neurons that you want to use. There are options for concept cells vs all cells, as well as just PY or IN cells. You can also load the neuronal locations if you want to do a region-specific analysis (as done in our Fig. 1).
- It defines region labels, combining left and right neurons from the same brain regions (neurons with amygdala_left and amygdala_right locations are put under the same underlying label -- amygdala). 
- Lastly, the script identifes brain regions where the neural activity is significantly modulated by a task variable. See our methods and the description in the notebook for detailed explanation of the procedure and statistical analysis. This notebook allows the creation of the following figures in our paper: Fig. 1e,f ; Fig. 6e,f; Fig. S5 e;f. 

**Inputs:**

- new_trial_info.xlsx: trial identities, ensuring pooling across neurons for GLM encoder. 
- graph_delay.xlsx : an Excel file with standardized spikes from the Delay (relative to start of the corresponding trial in the Delay) per subject, trial and neuron, with merged data for brain region, concept vs non-concept cell, preferred image identity of every neuron (relative to Encoding 1 period). This is used for decoding from the maintenance period only.
- merged_significant_neurons_with_brain_regions.xlsx.: an Excel file with all cells, including notation for the concept cells and their location in the brain. Use this if you want to achieve the across brain-region analysis. 
- Clustering_3D.xlsx: an Excel file with labels for putative cell-type classification (PY/IN). Use this if you want to achieve analysis across neuronal types. 

**Outputs (Figures):**

RT_all.eps, RT_concept.eps, load_all.eps, load_concept.eps: .eps files that (depending on the neurons used), portray plots, showing the significantly modulated neurons (by brain region or neuronal type) by task parameters such as load and reaction time. 

## DIMENSIONALITY REDUCTION

Here, we are preparing the data for dimensionality reduction and fitting it to dPCA. Then, we are projecting the data onto a 3D plane and visualizing the trajectories of each stimulus identity. We quantify separability by calculating pairwise Euclidean distances between stimulus identities in different conditions and perform stats between conditions. 

- The script filters and balances the data, bins and standardizes the spike times in encoding and maintenance, and creates X matrices.
- It performs stratified splitting (100 splits) on Encoding and Maintenance data, fits the train data to dPCA and transforms the test data onto the established fit. It also calculates pairwise distances between stimulus trajectories in each iteration and performs statistical analysis. This function also plots the explained variance by dPCs in each of the 100 iterations. 
- Then it does the same, but on Early and Late Maintenance data. 
- the script is also designed to repeat the same analsis but with just a single split, used for visualization. Running this script returns all panels associated with Fig.3 and Fig. S2. 

**Inputs:** 

- new_trial_info.xlsx: trial identities, ensuring pooling across neurons across load. 
- cleaned_Encoding1.xlsx : an Excel file with standardized spikes from Encoding 1 per subject, trial and neuron, with merged data for brain region, concept vs non-concept cell, preferred image identity of every neuron. 
- graph_delay.xlsx : an Excel file with standardized spikes from the Delay (relative to start of the corresponding trial in the Delay) per subject, trial and neuron, with merged data for brain region, concept vs non-concept cell, preferred image identity of every neuron (relative to Encoding 1 period). 
cleaned_Fixation.xlsx:  an Excel file with standardized spikes from the Fixation period (relative to the start of the corresponding trial in the Fixation) per subject, trial, and neuron. 

**Outputs (Figures):** 
- enc_maint_boxplots, early_late_boxplots.eps…etc: .eps files that include the pairwise distances between stimulus identities in training and testing of 100 stratified samples in encoding, maintenance, maintenance projected onto encoding, early maintenance, late maintenance, and late maintenance projected onto early maintenance. 
- enc_maint_var.eps, early_var.eps, late_var.eps: .eps files that include the explained variance from fitting the training data in the dPCA algorithm in each of the training conditions: encoding, maintenance, early maintenance, late maintenance
- encoding_on_encoding, encoding_on_maintenance, maintenance_on_maintenance.eps…etc. : .eps files with example projections of stimulus trajectories in state space in each of the conditions of training/testing the data.  


## BURSTING

Here, we are performing a bursting analysis in all cells and across subjects to look for alternative methods of carrying memoranda. We are grouping cells into different groups of neurons and determining statistical significance between them. 

- The script concatenates spike times of individual neurons across time. You can do that for encoding and for the delay. If you want to do it for the delay, you simply have to change the TRIAL_WIN parameter to 2.8 (duration of the delay in seconds); load the delay spiking data in path_trials and change the Standardized_Spikes_New to Standardized_Spikes_in_Delay. This code also groups neurons based on their ACG features, and also based on their mechanistic properties (pyramidal vs interneurons vs concept cells) and it does it at a subject level. 
- The script then visualizes the burst counts for real vs poisson data and determines statistical significance. 
- The nect script performs a bursting analysis across task epochs. This code separates the delay period into three parts and measures statistical significance in each group of neurons across subjects. 
- It also performs multiple comparisons correction (FDR) and saves all p values in the relevant excel file. The figures generated in this notebook are mostly ones from Figure 4 and Figure S3 in the paper. 

**Inputs:**
- graph_encoding1.xlsx, graph_encoding2.xlsx, graph_encoding3.xlsx : an Excel file with standardized spikes from Encoding 1, 2, and 3 per subject, trial and neuron, with merged data for brain region, concept vs non-concept cell, preferred image identity of every neuron. 
- graph_delay.xlsx : an Excel file with standardized spikes from the Delay (relative to start of the corresponding trial in the Delay) per subject, trial and neuron, with merged data for brain region, concept vs non-concept cell, preferred image identity of every neuron (relative to Encoding 1 period). 
- graph_probe.xlsx: an Excel file with standardized spikes from the Probe (relative to start of the corresponding trial in the Probe) per subject, trial and neuron, with merged data for brain region, concept vs non-concept cell, preferred image identity of every neuron (relative to Encoding 1 period), with information about preferred vs non-preferred image identity in the probe itself. 
- Clustering_3D.xlsx: an Excel file with labels for putative cell-type classification (PY/IN). Use this if you want to achieve analysis across neuronal types. 
- merged_significant_neurons_with_brain_regions.xlsx.: an Excel file with all cells, including notation for the concept cells and their location in the brain

**Outputs:**

- enc_vs_poisson.eps; delay_vs_possion.eps: .eps files showing comparisons between bursts in the real data vs bursts generated by chance (by a poisson process).
- burst_load1.eps, burst_load2.eps, burst_load3.eps: .eps files showing comparisons between bursts across task epochs and subjects.
- wilxocon_pvalues_across_periods.xlsx: excel file outlining the p values across task periods. For easier plugging into the paper. 

## CELL-TYPE CLASSIFICATION

Here, we are extracting waveforms and spike times and using Cell Explorer functions to find properties of the waveforms and attempt to label the neurons by Interneurons and Pyramidal cells.

- We run the script to extract the spike times of each neuron from the original .nwb files. Save with UIDs to make it easier for subsequent analysis. We extract the brain regions for every UID.

ON MATLAB: 
- Now we need to organise a struct to feed into CellExplorer. First, create the struct. It needs to have these exact names so it runs smoothly with CellExplorer. We need to reorganize the waveforms from 256x902 to 1x902 cell with 1x32 doubles. Open the one_reshape_waveforms.m file and follow the instructions to reshape the data. Create a struct within the struct you created, called waveforms. In waveforms, we have to have raw, filt and time cells. In our case, raw and filt are going to be the same, as the waveforms are already filtered. Parse the reshaped data into the raw and filt cells. Create a time 1x902 cell with 1x32 doubles with timesteps from -0.75 to + 0.75. Add the UIDs and the spike times (with the name neuron_spikes) into the general struct. We are giving you the entire struct, but this documentation outlines exactly how we extracted and processed the data. 
We also have to create cells with general information, sampling rate (32 000 Hz), subject id, animal for each neuron. We also have some required by CellExplorer matrices like deepSuperficial_num, SWR_modulation index, etc, but we fill those with NaN or 0, as first, it is not required for our type of analysis, and second, we don’t have access to all of that info from the dataset. All of this is already into the struct. 
- CV2, Firing rate, Peak and Mean SNR, as well as projection and isolation distance matrices have already been calculated. 
- We can use the calc_ACG_metrics.m function (by CellExplorer) to calculate narrow and wide ACG values, as well as Theta modulation and burst indices (Royer and Doublets). 
- We can use the calc_waveform_metrics.m function (by CellExplorer) to calculate waveform metrics (peak to Trough, AB ratio, etc.) 
- We can use use the fit_ACG.m function (by CellExplorer) to fit a function to calculate ACG values to use for further analysis. 
- We can use = the three_Mean_ACG.m file and copy paste in MATLAB to calculate mean ACG values
- Then, run CellExplorer >> CellExplorer('metrics', project);

ON PYTHON 
- Now we can continue with our .ipynb file. Run the next block of code to visualize the ACG. You can use that to inspect the cells visually.
- We can create an Excel sheet for easier analysis of cell metrics. We provided one in the waveforms folder (Cell_analysis.xlsx). You can use that to try out different combinations of cell metrics, plot histograms, and explore how different metrics relate to each other. 
- First, however, we use the Cell_analysis.xlsx file. We filter cells with an R squared value below 0.3 (less than 30% explained variance) and we run a spectral clustering method on the following metrics – Mean ACG, firing_rate, Tau_rise. We want to see how the cells get separated. We save the spectral clustering labels (0,1) in a new excel file (Clustering_3D.xlsx). The script assigns cell types to the labeled clusters. The cluster that includes cells with higher firing rates is labeled the interneuron cluster, and the other one is labeled the pyramidal cluster. It adds a new column “Cell_Type_New”, with the new identity of each cell, based on the spectral clustering. Then print the number of IN/PY classified neurons and we visualise the UMAP. 
- The script can then run descriptive statistics on the already classified cells → histograms, distributions and comparison across cell types. 
- Finally, this script compares bursting (CV2) metrics for IN and PY concept cells in their preferred vs non-preferred trials for Encoding and then for Maintenance. We do it in the following way: we concatenate the spike times for the preferred vs for the non-preferred trials, separately for Encoding and Maintenance for each of the neurons. Then, we calculate a CV2 value for each neuron in each of the four conditions (pref enc, nonpref enc, pref maintenance, nonpref maintenance), and we plot those on a figure. We compare within and across conditions for IN and PY. 

**Inputs:**

- nwb files from all subjects (taken directly from the dataset). They are in the form sub-{i+1}/sub-{i+1}_ses-2_ecephys+image.nwb and all of them are in a folder named 000469. 
- all_neuron_brain_regions_cleaned.xlsx. Given as a cleaned version of all the neurons in all the subjects and which region of the brain they are from. 
- project.mat: this is the complete struct, if you want to use it directly to feed into CellExplorer. If not, there are some helpful functions and a .ipynb file for extraction of data and calculation of metrics. 
- calc_ACG_metrics.m: calculates wide and narrow ACG, as well as burst indices and theta modulation indices. 
- fit_ACG.m: fits the ACG using a function and calculates various acg values that can help for further analysis. 
- three_Mean_ACG.m: calculates mean ACG values from wide acg matrices of each neuron 
- neuron_spikes.m: has all the spikes in a 1x902, but for easier use in the .ipynb file 
- acg.mat: has all the wide and narrow acg values of each neuron (in a 1x902) for easier use in the .ipynb file (for visualisation purposes) 
- Cell_analysis.xlsx: this is an Excel file that provides a visual interpretation of the ACGs of all the neurons, as well as different metrics that can be used to explore cell types. Columns labelled in pink are directly taken from the dataset. Other columns are provided after running CellExplorer functions.

**Outputs:** 

- Clustering_3D.xlsx: an Excel file containing the cell type labels provided by the spectral clustering
- waveforms_average.mat: a 256x902 matrix with the waveform values over time for each neuron
- spike_times_and_uids.mat: a 1x902 cell with all the spikes of all the neurons in the dataset
- brainRegion.mat: all the neuron brain regions for each UID 

**Figures:** 

- clustering.eps: an EPS file that contains the clustering output of the spectral clustering analysis 
- Cell_Metrics.eps: an EPS file that contains box plots and significance testing between the mean ACG, ACG tau rise, firing rate, CV2, and burst index values between cells labeled as interneurons vs pyramidal cells. 
- preferred_vs_nonpreferred_cv2.eps: an .eps file showing the differences in CV2 between IN and PY concept cells in their preferred vs non-preferred trials, separately for Encoding and Maintenance. 
