# Neural-heterogeneity-shapes-the-temporal-structure-of-human-working-memory

This is a repository that includes all the codes necessary to create all main and supplementary figures of the Kussovska et al., 2025 paper. 
There are some notebooks for pre-processing steps, but we have also provided extracted and cleaned data for analysis as part of this repo. 

# Downloading the dataset
Install the dataset (from https://www.nature.com/articles/s41597-024-02943-8) and put it in a folder on your computer. There are multiple ways of installing this dataset, all of them detailed in the github repository of the Kyzar et al., 2024 paper. 

One way is to use dandi (install with conda or pip): 

```
pip install "dandi>=0.60.0"
dandi download DANDI:000469/0.240123.1806
```

Detailed description of the dataset is included in the original paper by Kyzar et al., 2024. 

# Data extraction and organization

After you have downloaded the dataset from the original paper by Kyzar et al., 2024, you can use clone this repository on your local computer or server by running this line of code in your terminal: 

```
git clone https://github.com/dariakussovska/heterogeneity_wm/
```
This will create a folder "heterogeneity_wm" which will have all the necessary files and python codes to generate all the figures in the Kussovska et al., 2026 paper. 

I highly recommend that you also have a separate directory for all your newly generated files. In my case, I use a folder called "PROJECT" as you will see on all of the codes. That is where the raw NWB files are stored, and that is where I move all the dependencies installed from this repository. 

To extact data for main analysis, you can use the code 1_data_extraction.py by going into the heterogeneity_wm folder on your terminal and running

```
python 1_data_extraction.py
```
1_data_extraction.py extracts spike times for encoding 1, 2, and 3 periods, as well as the maintenance, and probe periods, and other relevant information, such as start and end time of each trial, subject id, neurond id, and stimulus presented. 

If you run 2_data_extraction.py then you can extract cell_metrics (calculation for firing rate, cv2, ISI, etc. and trial_info excel sheet, where each subject and trial id has its corresponding stimulus identity (or identities). 

# Important - Install dependencies 

For ease of use, some of the dependent files are given here, under the folder "data". new_trial_info.xlsx and trial_info_final.xlsx are files which make sure that stimulus ids are balanced based on load, and on number of image identities, respectively. all_neuron_brain_regions_cleaned.xlsx is a file where each neuron id has its correspodning brain region in a cleaned and organized way (because the extraction from the NWB file was not as straightforward). 

Under cell_analysis, there are matlab structs and functions to run the cell classification method from scratch, suing the software Cell Explorer. 

Under electrodes_plotting, there is a function to plot all the electrode coordinates from the Kyzar et al, 2024 paper in MNI space. 

Detailed explanation of each of these and how to run them is found under "Documentation". 

# After data extraction 

After data extraction, you need just two more steps before you have all necessary files to run all analyses and get all figures. First, standardize the spikes. Run the code standardization.py 

```
python standardization.py
```

In order to standardize the spikes in two different ways: for PSTH graphs (standardizing by start of each corresponding period); and for cross-temporal decoding (standardizing the spikes from the encoding period). This outputs two folders with standardized spikes column, as well as adds new neuron identities (subject id + 0 + neuron id) for ease of use. 

Then, identify concept cells by running the python code concept_cell_definition.py. 

```
python concept_cell_definition.py

```
This includes our method for concept cell identification and adds the corresponding identity of each neuron to each of our standardized files (Signi == Y if concept cell and Signi == N if not). 
