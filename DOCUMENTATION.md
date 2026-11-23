# DOCUMENTATION 

## DATA EXTRACTION

1) Install the dataset (from https://www.nature.com/articles/s41597-024-02943-8) and put it in a folder on your local computer. Directions are given in the README.md file. Here, we are using the folder heterogeneity_wm, and the dataset files are in the folder 000469. 
2) Run the scripts 01_data_extraction.py and 02_trial_info.py. 

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

- Run the code 03_standardization.py
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

- Run the code 04_concept_cells.py
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

merged_significant_neurons_with_brain_regions.xlsx.: an Excel file with all cells, including notation for the concept cells and their location in the brain. 

**Figures:** 

concept_cell_1.eps and concept_cell_2.eps: two figures showing the firing rates for two concept cells across all stimulus identities in the Encoding 1 period. 

region_distribution.eps: an EPS file showing a pie chart the distribution of cells recorded from different brain areas.

concept_cell_distribution.eps: an EPS file showing a pie chart of the percentage of concept cells per brain region. 

## FIRING RATE GRAPHS 

Here, we are preparing the existing files for plotting the firing rates across concept and non-concept cells for preferred image identities versus non-preferred image identities. 

- First, we need to run the code 06_add_category.py to add the corresponding trial type ("Category" column) to say whether the trial was preferred or non-preferred for that particular neuron and subject id combination. 
- Under the 02_psth_decoding >> main_2a,b, we choose a neuron to plot. We run the code to load the data and filter by the number of images presented (to show only load 1 trials). We count spikes and convert to firing rates in the corresponding time bins, and subsequently smooth the data. We calculate calculate z-scores (subtracting the mean firing rate in the baseline and dividing by the standard deviation), and the confidence intervals for plotting later. Finally, we plot the z-scores and confidence intervals for the encoding, delay and probe periods.
- The script also includes the generation of a raster with the spike times of this particular neuron over the task epochs. 
- 02_psth_decoding >> main_2c,5k,l,supp_1b,c.py plots the population PSTH of concept cells (Fig. 2c; Fig. S1b,c). Set CELL_TYPE=PY or CELL_TYPE=IN to use only the concept cells of one type (ACG labels, Supplementary_Figures >> revision_clustering_no_waveform_labels.xlsx) for Fig. 5k and 5l. Output: PSTH_concept_<all|PY|IN>.eps.

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

Here, we are decoding stimulus identity from the maintenance period to see how information gets maintained. In addition, we are doing cross-temporal decoding to see how memoranda evolve through time, and we are doing within-subject decoding on brain regions.

- We construct the y matrix from individual subject trial labels in the new_trial_final.xlsx file as shown in the code. Data is binned and normalized by z-scoring with each neuron's activity during baseline (fixation). You can choose which neurons to include in the data.
- The script 02_psth_decoding >> main_2d,supp_1d,e.py builds the design matrix across all task phases (encoding, delay, probe) and performs cross-temporal decoding (train on one time bin, test on another). Significant clusters are outlined after FDR correction. By varying the neurons that participate in the decoding analysis, as well as the loads, this code generates the .eps files that are used in Fig. 2d as well as Fig. S1d,e.
- The script 02_psth_decoding >> main_2e,supp_1f.py performs an analysis across participants. It filters participants with 3 or more concept cells identified and runs decoding with concept cells from different brain regions to see how stable the decoding accuracy is across participants. This uses the initial trial info excel sheet (trial_info.xlsx, from 02_trial_info.py) to ensure as many trials in each participant included as possible. This code generates the .eps files used in Fig. 2e and Fig. S1f.
- The script 02_psth_decoding >> main_2f,supp_1g.py decodes stimulus identity during maintenance from concept cells for different bin widths (250, 500, 750, 1000 ms), with repeated stratified cross-validation and a label-permutation test. This code generates the .eps files used in Fig. 2f and Fig. S1g.

**Inputs:**
- trial_info.xlsx: an Excel file, containing the image identity presented in each trial in each subject, as well as the numbers of images presented during the specified trial. 
- new_trial_final.xlsx: new trial identities, ensuring easy pooling across neurons for the decoder. Generated as mentioned above.
- Neuron_Check_Significant_All.xlsx: concept-cell definition (Signi == Y), from 04_concept_cells.py.

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

- decoding_timebins_repeatedcv.eps: decoding accuracy during the maintenance period from all concept cells, for each bin width (with decoding_comparisons_repeatedcv.xlsx and bin_width_pairwise_comparisons_repeatedcv.xlsx).
- cross_temporal_decoding_MTL.eps (name set in the script): cross-temporal decoding accuracy across task periods for the chosen set of concept cells.
- brain_regions_decoding.eps: an .eps file that includes decoding accuracy during the maintenance period, using concept cells in different regions (MTL and AMY) and across participants


## POISSON ENCODING MODEL

Here, we fit the spike trains of individual neurons to a Poisson GLM model to see how they are modulated with task parameters such as load and reaction time.  

- The script 01_task >> main_1e,f.py (Fig. 1e,f) loads trial data and task relevant features; 06_cell_specific >> main_6e,f,supp_e,f.py runs the same model by putative cell type, using the waveform labels (all 902 units) (Fig. 6e,f; Fig. S5e,f). Set PREDICTOR=num_images_presented (load; Fig. 6e, S5e) or PREDICTOR=RT (Fig. 6f, S5f), CONCEPT_ONLY=1 to restrict to concept cells, and TAG to name the output (TAG.eps, TAG_summary.csv). For this script we are using the trials, balanced by load, as this is our main point of interest.
- It prepares an X matrix with the relevant neurons that you want to use. There are options for concept cells vs all cells, as well as just PY or IN cells. You can also load the neuronal locations if you want to do a region-specific analysis (as done in our Fig. 1).
- It defines region labels, combining left and right neurons from the same brain regions (neurons with amygdala_left and amygdala_right locations are put under the same underlying label -- amygdala). 
- Lastly, the script identifes brain regions where the neural activity is significantly modulated by a task variable. See our methods and the description in the notebook for detailed explanation of the procedure and statistical analysis. This notebook allows the creation of the following figures in our paper: Fig. 1e,f ; Fig. 6e,f; Fig. S5 e;f.
- CURRENTLY THE SCRIPTS ARE SET UP FOR FIG. 1E and Fig. 6E only. IN ORDER TO OBTAIN FIG. 1F, CHANGE LINE 149 OF THE CODE (num_images_presented >> probe). 

**Inputs:**

- new_trial_info.xlsx: trial identities, ensuring pooling across neurons for GLM encoder. 
- graph_delay.xlsx : an Excel file with standardized spikes from the Delay (relative to start of the corresponding trial in the Delay) per subject, trial and neuron, with merged data for brain region, concept vs non-concept cell, preferred image identity of every neuron (relative to Encoding 1 period). This is used for decoding from the maintenance period only.
- merged_significant_neurons_with_brain_regions.xlsx.: an Excel file with all cells, including notation for the concept cells and their location in the brain. Use this if you want to achieve the across brain-region analysis. 
- Supplementary_Figures >> revision_clustering_waveform_labels.xlsx: putative cell-type labels (PY/IN) for all units, with Signi for concept cells (used by main_6e,f,supp_e,f.py). 

**Outputs (Figures):**

RT_all.eps, RT_concept.eps, load_all.eps, load_concept.eps: .eps files that (depending on the neurons used), portray plots, showing the significantly modulated neurons (by brain region or neuronal type) by task parameters such as load and reaction time. 

## DIMENSIONALITY REDUCTION

Here, we are preparing the data for dimensionality reduction. We prepare the data matrices in 08_dpca_matrices.py. When we have the matrices, then we will use 03_dpca >> main_3(any) to fit the data to dPCA. We can also use the codes for visualizing the trajectories of each stimulus identity. We quantify separability by calculating pairwise Euclidean distances between stimulus identities in different conditions and perform stats between conditions. 

- The script filters and balances the data, bins and standardizes the spike times in encoding and maintenance, and creates X matrices.
- It performs stratified splitting (100 splits) on Encoding and Maintenance data, fits the train data to dPCA and transforms the test data onto the established fit. It also calculates pairwise distances between stimulus trajectories in each iteration and performs statistical analysis. This function also plots the explained variance by dPCs in each of the 100 iterations. 
- Then it does the same, but on Early and Late Maintenance data. 
- The script is also designed to repeat the same analsis but with just a single split, used for visualization. Running this script returns all panels associated with Fig.3 and Fig. S2. 

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

Here, we are performing a bursting analysis across participants to look for alternative methods of carrying memoranda. Neurons are grouped per participant (all cells, ACG/decay splits, concept cells, putative pyramidal neurons and interneurons), and population bursts are detected from each group's pooled spikes: 30 ms bins, Gaussian smoothing (sigma 50 ms, 8-bin kernel), peaks with prominence above the 90th percentile of the real trace, at least 140 ms apart. Surrogates use the same detection threshold as the real data.

- 04_bursting >> main_4a.py: example population trace with detected bursts, real vs Poisson (Fig. 4a).
- 04_bursting >> main_4c,d,6a,b.py: burst counts in the real data vs Poisson surrogates for each neuronal group, encoding (0-1 s) or maintenance (set TRIAL_WIN to 2.8 and load the delay data, as noted in the script); one-sided paired Wilcoxon (Fig. 4c,d; Fig. 6a,b). PY/IN groups use participants with at least 3 labelled cells of the type.
- 04_bursting >> main_4e.py: ACG decay constant by putative cell type (ACG labels; units with R squared ≥ 0.3 and a fitted decay) (Fig. 4e).
- 04_bursting >> main_4f-q,6c,d.py: bursting per epoch in equal 1 s windows (fixation, encoding, maintenance M1 0-1 s, M2 0.75-1.75 s, M3 1.5-2.5 s), split into an alignment component, log2((real + 1)/(trial shuffle + 1)), and a single-neuron component, log2((trial shuffle + 1)/(Poisson + 1)). One-sided paired Wilcoxon, FDR-BH across the 5 epochs within group x load (Fig. 4f-q; Fig. 6c,d; Fig. S6, S10a-d). PY/IN groups use participants with at least 5 labelled cells. With REGION = True, the same analysis by brain region (Fig. S7).
- 04_bursting >> supp_10e.py: components of burst excess for PY and IN during maintenance (0-2.5 s, load 1), with Poisson, PSTH (rate-profile) and trial-shuffle surrogates; one-sided paired Wilcoxon (BH across the 4 component tests) and IN vs PY (two-sided, participants with both types) (Fig. S10e).
- 04_bursting >> supp_10f,g.py: recruitment of concept-cell spikes (within ±50 ms of burst peaks) into bursts defined by non-concept PY or IN, on preferred vs nonpreferred trials, encoding (0-0.9 s) and maintenance (0-2.5 s), against a trial-shuffle expectation; Wilcoxon vs 0, BH across the 8 tests (Fig. S10f,g).

**Inputs:**
- graph_encoding1.xlsx, graph_encoding2.xlsx, graph_encoding3.xlsx : an Excel file with standardized spikes from Encoding 1, 2, and 3 per subject, trial and neuron, with merged data for brain region, concept vs non-concept cell, preferred image identity of every neuron. 
- graph_delay.xlsx : an Excel file with standardized spikes from the Delay (relative to start of the corresponding trial in the Delay) per subject, trial and neuron, with merged data for brain region, concept vs non-concept cell, preferred image identity of every neuron (relative to Encoding 1 period). 
- cleaned_Fixation.xlsx: standardized spikes from the fixation period.
- Supplementary_Figures >> revision_clustering_no_waveform_labels.xlsx (ACG-based PY/IN labels) and revision_clustering_waveform_labels.xlsx (ACG + spike-width labels, and Signi for concept cells); see CELL-TYPE CLASSIFICATION.
- Supplementary_Figures >> all_neuron_brain_regions_merged.xlsx and data >> all_neuron_brain_regions_cleaned.xlsx: all cells, with concept-cell notation and brain region.

**Outputs:**

- Burst counts real vs Poisson per group (.eps) and the corresponding p values (main_4c,d,6a,b.py).
- burst_alignment_results.xlsx (sheets per_subject and epoch_vs_control) and alignment / single-neuron box plots per load (main_4f-q,6c,d.py; _region with REGION = True).
- burst_components/: burst_components_per_subject.csv, burst_components_stats.csv, S10e_components.eps (supp_10e.py).
- burst_concept/: burst_concept_cells.csv, burst_concept_stats.csv, S10f_PY_bursts.eps, S10g_IN_bursts.eps (supp_10f,g.py).

## CELL-TYPE CLASSIFICATION

Here, we are extracting waveforms and spike times and using Cell Explorer functions to find properties of the waveforms and attempt to label the neurons as Interneurons and Pyramidal cells.

- We have provided an intial matrix with spike times and total number of neurons under data >> cell_analysis >> spikes_struct.m. We have also provided the functions needed to calculate ACGs (raw and normalized), calculate mean ACG, fit the ACG (CellExplorer function) to get parameters such as decay time constant, ACG tau rise, and others. 

IN MATLAB: 
- In order for Cell Explorer to run smoothly, we need to organize a struct, which include the neurons' full spike trains, as well as the number of spikes for each neuron throughout the entire session. This is already provided under data >> cell_analysis >> spikes_struct.m. The struct also includes a UID (Neuron_ID_3) and some other important parameters that will be useful for subsequent analyses.
- As we already have the full spike trains and spike times, we can run the function calc_ACG_metrics.m, provided under data >> cell_analysis. This function calculates narrow and wide ACG values, as well as the burst indices (Royer and Doublets) used in subsequent cell-type related analyses.
- After calculating the wide and narrow ACGs, we can calculate the normalized ACG values with the function normalize_acg_to_unit.m to calculate normalized wide and narrow ACGs.
- After calculating those, we can also calculate a mean ACG by using instructions provided in cell_analysis >> Mean_ACG.m. 
- Subsequently, we can use the fit_ACG.m function (by CellExplorer) to fit a triple-exponential function to calculate certain ACG parameters, such as ACG tau decay, tau rise, and variance explained (R squared) of the function fit, which are all needed for our cell-type classification. 
- cv2 and firing rate were already calculated in the second script in the repository (02_trial_info.py).
- After extracting these values, they can be neatly put in one Excel file. We have provided this under data >> Cell_metrics.xlsx >> Cell_analysis.xlsx

IN PYTHON: 
- To visualise the autocorrelograms, we can use the script main_5a,b. This returns all autocorrelograms (one for each neuron).  
- For our cell classification (07_spectral_clustering.py), we use data >> cell_analysis >> Cell_metrics.xlsx. We remove cells with an R squared value below 0.3 (less than 30% explained variance) and run spectral clustering on Mean ACG, firing_rate and Tau_rise. The cluster with the higher firing rate is labeled the interneuron cluster, and the other one the pyramidal cluster, in a new column “Cell_Type_New”. The labels are saved in Clustering_3D.xlsx (same labels as Supplementary_Figures >> revision_clustering_no_waveform_labels.xlsx), the number of IN/PY neurons is printed and the UMAP is plotted. 
- The label files used for the revised analyses are in Supplementary_Figures: revision_clustering_no_waveform_labels.xlsx (ACG-based labels only, R squared ≥ 0.3: 127 PY, 167 IN) and revision_clustering_waveform_labels.xlsx (all 902 units; units with R squared < 0.3 labelled by spike width: 621 PY, 281 IN). See data >> cell_analysis >> README.md.
- The main_5d-h.py script runs descriptive statistics on the ACG-classified cells (revision_clustering_no_waveform_labels.xlsx) and compares cell types on firing rate, mean ACG (raw, as used for clustering), tau rise, spike width and CV2 (Fig. 5d-h); Mann–Whitney U, FDR-BY across metrics. Set ACG_DISPLAY=norm to plot the unit-normalized mean ACG instead. Outputs main_5dh.eps and main_5dh_stats_script.csv.
- The supp_4a-e.py script repeats the clustering on the validation data (data >> cell_analysis >> validation_data.xlsx) and compares the resulting labels (Fig. S4a-e). 

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
- main_5dh.eps: box plots and significance testing of firing rate, mean ACG, ACG tau rise, spike width and CV2 between cells labeled as interneurons vs pyramidal cells. 

## CO-FIRING INDEX (CFI)

06_cell_specific >> main_6g,h,supp10h,i,j,k.py relates pairwise synchrony to population bursting and to behaviour.

- CFI per neuron pair and trial: coincidences within ±30 ms, expected count = mean over 50 circular trial shifts, CFI = log2((observed + c)/(expected + c)), c = 0.5 (0.1 and 1 for the sensitivity panel). Averaged over pairs per trial and z-scored within participant. Pairings: IN–IN and PY–PY (ACG-labelled neurons, at least 3 per participant) and all neurons (at least 10 per participant). Epochs: fixation 0-0.9 s, encoding 0-1 s, maintenance 0-2.8 s.
- Burst count → CFI (Fig. S10h): mixed model CFI ~ burst count + firing rate.
- CFI → reaction time (Fig. 6g,h; Fig. S10i,k): mixed model log RT ~ CFI + load + firing rate, correct trials. CFI → accuracy (Fig. S10j): GEE (binomial).
- Significance: 5000 within-participant permutations of the predictor; BH over the 6 burst–CFI tests, and over the 9 behaviour tests (3 pairings x 3 epochs) per outcome and constant.
- The permutations take long (about 200,000 model fits); set N_JOBS to run the models in parallel.

**Inputs:** cleaned_Fixation.xlsx, graph_encoding1.xlsx, graph_delay.xlsx, the two label files in Supplementary_Figures, and data >> trial_info.xlsx (trial information with RT and response accuracy; this is not the trial_info.xlsx written by 02_trial_info.py, which has no RT or accuracy columns).

**Outputs:** cfi_<pairing>[_eps<c>].csv, firing_rates.csv, burst_counts.csv, burst_cfi_stats.csv, cfi_behaviour_stats.csv, Fig6gh_CFI_RT.eps, FigS10hk_CFI.eps.

## TWO-STATE (HMM) DYNAMICS

05_cell_analysis >> supp_11.py fits one- and two-state Poisson hidden Markov models to every neuron (30 ms bins) and produces the panels of Fig. S11.

- Encoding (first image, 0-1 s): parameters from a two-state fit to all trials; two-state preference = 5-fold held-out log-likelihood difference per bin (two-state − one-state).
- Maintenance: parameters from one fit to 1-2 s; preference from 5-fold held-out log-likelihoods pooled over 5 resampled 1 s windows (start times 0-1.5 s). Firing rate = spikes in 0-2.5 s / (trials x 2.5 s).
- Exclusions: degenerate fits, and neurons firing < 1 Hz (except in the spike-count sweep, a, b). Classes: persistent (held-out ΔLL ≤ 0, high/low rate ratio < 1.5, or saturated high state), intermittent (high-state dwell < 0.3 s), otherwise hybrid.
- HMM high state vs population bursts (c): bursts detected from all other neurons of the participant (leave-one-out); P(high state) inside minus outside ±50 ms burst windows, minus a null with burst times from another trial of the same load.
- Statistics: c, Wilcoxon vs 0; d-g, Wilcoxon signed-rank on held-out ΔLL; h-k, chi-square vs uniform; l, m, PY vs IN in firing-rate-matched pairs, Fisher's exact test per class, BH across the 3 classes.
- The fits take about an hour on one core; set N_JOBS to fit neurons in parallel. Fits and burst links are saved and reused on later runs.

**Inputs:** graph_encoding1.xlsx, graph_delay.xlsx, Supplementary_Figures >> all_neuron_brain_regions_merged.xlsx and revision_clustering_no_waveform_labels.xlsx.

**Outputs (in ./HMM):** hmm_fits_encoding.csv, hmm_fits_maintenance.csv, hmm_burst_link_encoding.csv, hmm_burst_link_maintenance.csv, S11_stats.csv, and one .eps per panel (S11a ... S11m).

## SUPPLEMENTARY_FIGURES FOLDER

Supplementary_Figures holds the cell-type label files and brain-region file used above, and the scripts for the additional analyses in the supplementary material (Bayesian and RNN decoding comparisons, Fano factor and probe-burst simulations, behavioural GEE analysis, concept-cell shuffle controls, spike widths). The scripts read their input files from this folder, so run them from inside it. 
