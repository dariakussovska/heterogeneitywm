# Neural-heterogeneity-shapes-the-temporal-structure-of-human-working-memory

This is a repository that includes all the codes necessary to create all main and supplementary figures of the Kussovska et al., 2026 paper. 
There are some notebooks for pre-processing steps, but we have also provided extracted and cleaned data for analysis as part of this repo. 

# Installing requirements and downloading the dataset

Clone this repository on your local computer or server by running this line of code in your terminal: 

```
git clone https://github.com/dariakussovska/heterogeneity_wm/
cd heterogeneity_wm
```
This will create a folder "heterogeneity_wm" which will have all the necessary files and python codes to generate all the figures in the Kussovska et al., 2026 paper. 

Install the dataset (from https://www.nature.com/articles/s41597-024-02943-8) and put it in a folder on your computer. There are multiple ways of installing this dataset, all of them detailed in the github repository of the Kyzar et al., 2024 paper. 

One way is to use dandi (install with conda or pip), which is added to the requirements.txt file 

```
pip install -r requirements.txt
dandi download DANDI:000469/0.240123.1806
```
Detailed description of the dataset is included in the original paper by Kyzar et al., 2024. 

If you don't have dCPA installed, you can do it like this:
```
git clone https://github.com/machenslab/dPCA.git
cd dPCA
cd python
pip install -e .
% If it gives you a mistake with the sklearn, you can do the following:
% nano setup.py
% Change sklearn to scikit-learn
```

# Data extraction and organization

To extact data for main analysis, you can use the code 1_data_extraction.py by going into the heterogeneity_wm folder on your terminal and running

```
python 01_data_extraction.py
python 02_trial_info.py
```
01_data_extraction.py extracts spike times for encoding 1, 2, and 3 periods, as well as the maintenance, and probe periods, and other relevant information, such as start and end time of each trial, subject id, neurond id, and stimulus presented. With 02_trial_info.py you can extract cell_metrics (calculation for firing rate, cv2, ISI, etc. and trial_info excel sheet, where each subject and trial id has its corresponding stimulus identity (or identities). 

# Important - Install dependencies 

For ease of use, some of the dependent files are given here, under the folder "data". new_trial_info.xlsx and trial_info_final.xlsx are files which make sure that stimulus ids are balanced based on load, and on number of image identities, respectively. all_neuron_brain_regions_cleaned.xlsx is a file where each neuron id has its correspodning brain region in a cleaned and organized way (because the extraction from the NWB file was not as straightforward). 

Under cell_analysis, there are matlab structs and functions to run the cell classification method from scratch, suing the software Cell Explorer. 

Under electrodes_plotting, there is a function to plot all the electrode coordinates from the Kyzar et al, 2024 paper in MNI space. 

Detailed explanation of each of these and how to run them is found under "Documentation". 

# After data extraction 

After data extraction, you need just two more steps before you have all necessary files to run all analyses and get all figures. First, standardize the spikes. Run the code standardization.py 

```
python 01_standardization.py
```

In order to standardize the spikes in two different ways: for PSTH graphs (standardizing by start of each corresponding period); and for cross-temporal decoding (standardizing the spikes from the encoding period). This outputs two folders with standardized spikes column, as well as adds new neuron identities (subject id + 0 + neuron id) for ease of use. 

Then, identify concept cells by running the python code 04_concept_cells.py. Add concept cell or not to the existing dataframes (Signi == Y if concept cell and Signi == N if not), as well as number of images, presented in each trial, and the brain region locations with 05_merge_data. 

```
python 04_concept_cells.py
python 05_merge_data.py
```
Now, based on the the image that elicits the highest response in all the cells, we add Preferred vs Non-preferred image identity in all of our files. We use them to add a column with "Category" for each trial -- Preferred vs Non-preferred trials. We will use those to plot PSTHs. 

```
python 06_add_category.py
```
# Inferring cell types

This step needed to recreate all figures with ease is to perform the cell-classification analysis. For preprocessing and how we get the cell metrics needed to classify neurons into pyramidal cells and interneurons, you can read the corresponding section in the DOCUMENTATION.md file. This analysis is done in Matlab with the help of the software CellExplorer, and functions required for it are listed in the "data" folder. However, here, we have provided the outputs of the CellExplorer function under data >> Cell_analysis.xlsx. With the code 07_cell_types.py, we will just run the spectral clustering on those metrics and assign neurons as pyramidal (PY) or interneurons (IN) for subsequent analysis.

```
python 07_cell_types.py
```
# Dimensionality reduction matrices

This step is needed to create the matrices used in dimensionality reduction (dPCA) analyses. It creates two .npy files -- one for encoding and one for maintenance, and saves them to your designated folder. After running this, you will be able to run all the analyses for Fig.3 and Fig.S2. 

```
python 08_dpca_matrices.py
```
Now that we've ran all of the initial files for data extraction, standardization, and concept cell definition we can proceed with the main analyses used in each of our figures. Each main figure and the panels associated with it have a folder (ex. 01_task, 02_psth_decoding). They include all scripts, needed to recreate the main figures in the manuscript. Note that some figures just require running the same code twice, but with a different set of neurons. If that is the case, then this will be specified as a comment in the specific code. Run each .py file to get the corresponding panel from a main figure.  
