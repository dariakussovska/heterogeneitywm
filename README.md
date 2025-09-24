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

# Data extraction

In order to extact data for main analysis, you can use the code 1_data_extraction. Note that this is not needed for you to recreate all analyses and figures, as Excel sheets for main analyses will be provided as part of every code for main/supplementary analysis figure. 
