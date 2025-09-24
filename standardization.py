{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "aada5e4a-5e83-4b2d-a32f-de4d41d3cb6a",
   "metadata": {},
   "source": [
    "# STANDARDIZING THE DATA"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8b9664e0-71f1-4eac-86eb-180ed5fe128a",
   "metadata": {},
   "source": [
    "**Standardise the spikes by the start time of the encoding 1 period. This ensures continuous representation of the spikes.** "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "df56d840-90fd-424a-b82a-708672739619",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Standardization completed for all filtered files.\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import ast\n",
    "\n",
    "filtered_files = {\n",
    "    'Encoding1': '/Users/darikussovska/Desktop/PROJECT/spike_rate_data_Encoding1.xlsx',\n",
    "    'Encoding2': '/Users/darikussovska/Desktop/PROJECT/spike_rate_data_Encoding2.xlsx',\n",
    "    'Encoding3': '/Users/darikussovska/Desktop/PROJECT/spike_rate_data_Encoding3.xlsx',\n",
    "    'Delay': '/Users/darikussovska/Desktop/PROJECT/all_spike_rate_data_delay.xlsx',\n",
    "    'Probe': '/Users/darikussovska/Desktop/PROJECT/all_spike_rate_data_probe.xlsx'\n",
    "}\n",
    "\n",
    "# Load Encoding1 start times\n",
    "def load_enc1_start_times(enc1_path):\n",
    "    enc1_df = pd.read_excel(enc1_path)\n",
    "    return enc1_df[['subject_id', 'trial_id', 'start_time']].drop_duplicates().rename(columns={'start_time': 'start_time_enc1'})\n",
    "\n",
    "enc1_start_times = load_enc1_start_times(filtered_files['Encoding1'])\n",
    "\n",
    "# Standardize spikes \n",
    "def standardize_spikes(file_path, spikes_column, period_name):\n",
    "    \"\"\"\n",
    "    Standardizes spikes using the start_time_enc1 column without saving intermediate files.\n",
    "    \"\"\"\n",
    "    df = pd.read_excel(file_path)\n",
    "\n",
    "    # Merge Encoding1 start times\n",
    "    df = pd.merge(\n",
    "        df,\n",
    "        enc1_start_times,\n",
    "        on=['subject_id', 'trial_id'],\n",
    "        how='left'\n",
    "    )\n",
    "\n",
    "    if 'start_time_enc1' in df.columns and spikes_column in df.columns:\n",
    "        # Standardize spikes\n",
    "        def standardize_spikes(row):\n",
    "            spikes = ast.literal_eval(row[spikes_column]) if isinstance(row[spikes_column], str) else row[spikes_column]\n",
    "            if isinstance(spikes, list) and not pd.isna(row['start_time_enc1']):\n",
    "                return [spike - row['start_time_enc1'] for spike in spikes]\n",
    "            return []\n",
    "\n",
    "        df['Standardized_Spikes'] = df.apply(standardize_spikes, axis=1)\n",
    "\n",
    "        return df\n",
    "    else:\n",
    "        print(f\"Missing required columns in {file_path}. Skipping standardization.\")\n",
    "        return None\n",
    "\n",
    "standardized_data = {}\n",
    "for period_name, file_path in filtered_files.items():\n",
    "    spikes_column = 'Spikes' \n",
    "    if period_name == 'Delay':\n",
    "        spikes_column = 'Spikes_in_Delay'\n",
    "    elif period_name == 'Probe':\n",
    "        spikes_column = 'Spikes_in_Probe'\n",
    "\n",
    "    standardized_df = standardize_spikes(\n",
    "        file_path=file_path,\n",
    "        spikes_column=spikes_column,\n",
    "        period_name=period_name\n",
    "    )\n",
    "    if standardized_df is not None:\n",
    "        standardized_data[period_name] = standardized_df\n",
    "\n",
    "print(\"Standardization completed for all filtered files.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b40bba23-b848-406b-b858-845665352731",
   "metadata": {},
   "source": [
    "**Filter by concept cells and non-concept cells. Significance = Y means that this neuron is a concept cell.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "5edad1bf-0de1-41d9-9652-48ebc48c88aa",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Significance column added to all standardized DataFrames.\n"
     ]
    }
   ],
   "source": [
    "# Load concept cell data\n",
    "significant_neurons = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/Neuron_Check_Significant_All.xlsx')\n",
    "significant_neurons.rename(columns={'Signi': 'Significance'}, inplace=True)\n",
    "\n",
    "significant_neurons['subject_id'] = significant_neurons['subject_id'].astype(str).str.strip()\n",
    "significant_neurons['Neuron_ID'] = significant_neurons['Neuron_ID'].astype(int)\n",
    "\n",
    "for period_name, df in standardized_data.items():\n",
    "    df['subject_id'] = df['subject_id'].astype(str).str.strip()\n",
    "    df['Neuron_ID'] = df['Neuron_ID'].astype(int)\n",
    "\n",
    "    df = pd.merge(\n",
    "        df,\n",
    "        significant_neurons[['subject_id', 'Neuron_ID', 'Significance']],\n",
    "        on=['subject_id', 'Neuron_ID'],\n",
    "        how='left'  \n",
    "    )\n",
    "\n",
    "    df['Significance'] = df['Significance'].fillna('N')\n",
    "    standardized_data[period_name] = df\n",
    "\n",
    "print(\"Significance column added to all standardized DataFrames.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "866d8bb7-be1a-4da5-b8b1-14fd5f993813",
   "metadata": {},
   "source": [
    "**Now, because some subjects experience 108 trials, and some experience 135 trials, and they have different number of images presented, we created a new trial_info dataset, which alignes the trials so that there are the same types of trials (aka with 1, 2, and 3 images presented) in every subject. This is done in Excel, and the new_trial_info is directly pasted in the PROJECT folder. The decoder is created as all subjects are pooled together as 1. The image IDs are arbitrarily labeled (0-4). The decoder is designed with the aligned data across all subjects.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "756498f9-035a-40c0-81a3-9835277f00d5",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "new_trial_id column added to all standardized DataFrames.\n"
     ]
    }
   ],
   "source": [
    "trial_info = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/new_trial_final.xlsx')\n",
    "\n",
    "trial_info['subject_id'] = trial_info['subject_id'].astype(str).str.strip()\n",
    "trial_info['trial_id'] = trial_info['trial_id'].astype(int)\n",
    "\n",
    "# Add trial_id_final column\n",
    "for period_name, df in standardized_data.items():\n",
    "    df['subject_id'] = df['subject_id'].astype(str).str.strip()\n",
    "    df['trial_id'] = df['trial_id'].astype(int)\n",
    "\n",
    "    df = pd.merge(\n",
    "        df,\n",
    "        trial_info[['subject_id', 'trial_id', 'trial_id_final']],\n",
    "        on=['subject_id', 'trial_id'],\n",
    "        how='left'  \n",
    "    )\n",
    "\n",
    "    standardized_data[period_name] = df\n",
    "\n",
    "print(\"new_trial_id column added to all standardized DataFrames.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6aa3a488-02de-466d-84c6-fbd3ae733527",
   "metadata": {},
   "source": [
    "**Remove unnecessary columns. Also, add a new neuron identity to allow for pooling across subjects. Do it with subject id + 0 + original neuron id.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "c8095ce7-6807-40db-a3d0-7cd625ec06ba",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cleaned data for Encoding1 saved to: /Users/darikussovska/Desktop/PROJECT/clean_data/cleaned_Encoding1.xlsx\n",
      "Cleaned data for Encoding2 saved to: /Users/darikussovska/Desktop/PROJECT/clean_data/cleaned_Encoding2.xlsx\n",
      "Cleaned data for Encoding3 saved to: /Users/darikussovska/Desktop/PROJECT/clean_data/cleaned_Encoding3.xlsx\n",
      "Cleaned data for Delay saved to: /Users/darikussovska/Desktop/PROJECT/clean_data/cleaned_Delay.xlsx\n",
      "Cleaned data for Probe saved to: /Users/darikussovska/Desktop/PROJECT/clean_data/cleaned_Probe.xlsx\n",
      "All standardized data has been cleaned and saved in the new_clean_data folder.\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import os\n",
    "\n",
    "new_cleaned_data_dir = '/Users/darikussovska/Desktop/PROJECT/clean_data'\n",
    "os.makedirs(new_cleaned_data_dir, exist_ok=True)\n",
    "\n",
    "# Columns to remove\n",
    "columns_to_remove = [\n",
    "    'mean_1st', 'cat_1st',\n",
    "    'im_cat_2nd', 'mean_2nd', 'cat_2nd',\n",
    "    'p_val', 'CI', 'start_time_enc1'\n",
    "]\n",
    "\n",
    "# Clean and save each standardized DataFrame\n",
    "def clean_standardized_data():\n",
    "    for period_name, df in standardized_data.items():\n",
    "        # Create Neuron_ID_3 column as a numeric value\n",
    "        df['Neuron_ID_3'] = (\n",
    "            df['subject_id'].astype(str) + '0' + df['Neuron_ID'].astype(str)\n",
    "        ).astype(int)\n",
    "\n",
    "        # Drop unnecessary columns\n",
    "        df_cleaned = df.drop(columns=columns_to_remove, errors='ignore')\n",
    "\n",
    "        # Save the cleaned file\n",
    "        output_file = os.path.join(new_cleaned_data_dir, f'cleaned_{period_name}.xlsx')\n",
    "        df_cleaned.to_excel(output_file, index=False)\n",
    "        print(f\"Cleaned data for {period_name} saved to: {output_file}\")\n",
    "\n",
    "# Clean and save all standardized data\n",
    "clean_standardized_data()\n",
    "print(\"All standardized data has been cleaned and saved in the clean_data folder.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "853e6f90-5624-48f4-92a6-e3b25f67c099",
   "metadata": {},
   "source": [
    "**Do the same for the fixation period, but use the start of the fixation to standardise the spikes.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e949c292-1623-4ef1-94fa-7751dfa46bf7",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fixation data standardized successfully.\n",
      "Cleaned fixation data saved to: /Users/darikussovska/Desktop/PROJECT/clean_data/cleaned_Fixation.xlsx\n",
      "Fixation period cleaning and saving completed.\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import ast\n",
    "import os\n",
    "\n",
    "fixation_file_path = '/Users/darikussovska/Desktop/PROJECT/all_spike_rate_data_fixation.xlsx'\n",
    "trial_info_path = '/Users/darikussovska/Desktop/PROJECT/trial_info copy.xlsx'\n",
    "output_dir = '/Users/darikussovska/Desktop/PROJECT/clean_data'\n",
    "os.makedirs(output_dir, exist_ok=True)\n",
    "\n",
    "def standardize_fixation_period(file_path, spikes_column, start_time_column):\n",
    "    \"\"\"\n",
    "    Standardizes spikes in the fixation period by its own start time.\n",
    "    Returns the standardized DataFrame.\n",
    "    \"\"\"\n",
    "    df = pd.read_excel(file_path)\n",
    "\n",
    "    if start_time_column in df.columns and spikes_column in df.columns:\n",
    "        def standardize_spikes(row):\n",
    "            spikes = ast.literal_eval(row[spikes_column]) if isinstance(row[spikes_column], str) else row[spikes_column]\n",
    "            if isinstance(spikes, list) and not pd.isna(row[start_time_column]):\n",
    "                return [spike - row[start_time_column] for spike in spikes]\n",
    "            return []\n",
    "\n",
    "        df['Standardized_Spikes'] = df.apply(standardize_spikes, axis=1)\n",
    "        print(\"Fixation data standardized successfully.\")\n",
    "        return df\n",
    "    else:\n",
    "        print(f\"Missing required columns in {file_path}. Skipping standardization.\")\n",
    "        return None\n",
    "\n",
    "standardized_fixation_data = standardize_fixation_period(\n",
    "    file_path=fixation_file_path,\n",
    "    spikes_column='Spikes_in_Fixation',\n",
    "    start_time_column='Fixation_Start'\n",
    ")\n",
    "\n",
    "trial_info = pd.read_excel(trial_info_path)\n",
    "trial_info['subject_id'] = trial_info['subject_id'].astype(str).str.strip()\n",
    "trial_info['trial_id'] = trial_info['trial_id'].astype(int)\n",
    "\n",
    "if standardized_fixation_data is not None:\n",
    "    standardized_fixation_data['subject_id'] = standardized_fixation_data['subject_id'].astype(str).str.strip()\n",
    "    standardized_fixation_data['trial_id'] = standardized_fixation_data['trial_id'].astype(int)\n",
    "    standardized_fixation_data = pd.merge(\n",
    "        standardized_fixation_data,\n",
    "        trial_info[['subject_id', 'trial_id', 'trial_id_final']],\n",
    "        on=['subject_id', 'trial_id'],\n",
    "        how='left'\n",
    "    )\n",
    "\n",
    "    standardized_fixation_data['Neuron_ID_3'] = (\n",
    "        standardized_fixation_data['subject_id'].astype(str) + '0' + \n",
    "        standardized_fixation_data['Neuron_ID'].astype(str)\n",
    "    ).astype(int)\n",
    "\n",
    "    output_file = os.path.join(output_dir, 'cleaned_Fixation.xlsx')\n",
    "    standardized_fixation_data.to_excel(output_file, index=False)\n",
    "    print(f\"Cleaned fixation data saved to: {output_file}\")\n",
    "\n",
    "print(\"Fixation period cleaning and saving completed.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "47aed244-5f98-4ac2-9a06-25783f6824a0",
   "metadata": {},
   "source": [
    "# STANDARDIZING DATA FOR PSTH PLOTS "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "36bc0091-035d-4a60-9a49-ec2c6f9f23e5",
   "metadata": {},
   "source": [
    "**Here we add preferred vs non-preferred stimulus column based on our concept cell analysis. This will be needed in order to see how the concept cells react to their preferred stimulus identity vs to their non-preferred stimulus identity.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "05cb15c2-6da7-4172-9263-6fe17aac7085",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "enc1_data = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/new_clean_data/cleaned_Encoding1.xlsx')\n",
    "print(enc1_data[['Neuron_ID_3', 'trial_id', 'start_time', 'Spikes', 'Standardized_Spikes']].head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "70c5071e-6a15-45fe-8f89-f9ec5eba4c5f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# IMPORTANT: FIX THE NEURON_ID_2 TO NEURON_ID_3 IN THE SERVER AND THE TRIAL INFO GENERATION (ORIGINAL ONE TO START FROM 1)\n",
    "significant_neurons = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/merged_significant_neurons_with_brain_regions.xlsx')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ba6e5da9-7cef-4359-87b0-b31657aee5d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Add a column for the preferred image ID to the Enc1 data\n",
    "def add_preferred_image_id(enc1_df, significant_neurons_df):\n",
    "    merged_df = pd.merge(enc1_df, significant_neurons_df[['subject_id', 'Neuron_ID', 'im_cat_1st']], on=['subject_id', 'Neuron_ID'], how='inner')\n",
    "    merged_df.rename(columns={'im_cat_1st': 'preferred_image_id'}, inplace=True)\n",
    "    return merged_df\n",
    "\n",
    "enc1_data_with_preferred = add_preferred_image_id(enc1_data, significant_neurons)\n",
    "print(enc1_data_with_preferred[['subject_id', 'Neuron_ID', 'trial_id', 'preferred_image_id', 'Standardized_Spikes']].head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "59da560b-f4c1-4ab3-9965-1951b2ab28a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the trial information data\n",
    "trial_info = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/trial_info.xlsx')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5f3efb37-0fdf-41b1-89d0-21cb9c444ed6",
   "metadata": {},
   "outputs": [],
   "source": [
    "enc1_data_filtered = pd.merge(enc1_data_with_preferred, trial_info, on=['subject_id', 'trial_id'], how='inner')\n",
    "# Define the file path where the filtered Enc1 data will be saved\n",
    "filtered_enc1_file_path = '/Users/darikussovska/Desktop/PROJECT/graph_encoding1.xlsx'\n",
    "enc1_data_filtered.to_excel(filtered_enc1_file_path, index=False)\n",
    "print(f\"Filtered Enc1 data saved to {filtered_enc1_file_path}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "97c14efd-4ba6-454a-ad22-070cecce8974",
   "metadata": {},
   "outputs": [],
   "source": [
    "def categorize_trials_by_preference(df):\n",
    "    df['Category'] = df.apply(lambda row: 'Preferred' if row['preferred_image_id'] == row['stimulus_index'] else 'Non-Preferred', axis=1)\n",
    "    return df\n",
    "\n",
    "enc1_data_categorized = categorize_trials_by_preference(enc1_data_filtered)\n",
    "\n",
    "print(enc1_data_categorized[['subject_id', 'Neuron_ID', 'trial_id', 'preferred_image_id', 'stimulus_index', 'Category']].head())\n",
    "enc1_data_categorized.to_excel(filtered_enc1_file_path, index=False)\n",
    "print(f\"Categorized Enc1 data saved to {filtered_enc1_file_path}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a9e21c1f-b50d-4006-b6df-9d1f1b1588a7",
   "metadata": {},
   "source": [
    "**Here we add brain region information (where each neuron comes from)**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f3325eba-1123-4e44-b2f9-40ff2a327550",
   "metadata": {},
   "outputs": [],
   "source": [
    "categorized_enc1_file_path = '/Users/darikussovska/Desktop/PROJECT/graph_encoding1.xlsx'\n",
    "\n",
    "enc1_data_categorized = pd.read_excel(categorized_enc1_file_path)\n",
    "significant_neurons = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/merged_significant_neurons_with_brain_regions.xlsx')\n",
    "\n",
    "# Function to add brain region information\n",
    "def add_brain_region_info(enc1_df, significant_neurons_df):\n",
    "    # Merge Enc1 data with significant neurons data on 'subject_id' and 'Neuron_ID'\n",
    "    merged_df = pd.merge(enc1_df, significant_neurons_df[['subject_id', 'Neuron_ID', 'Location']], on=['subject_id', 'Neuron_ID'], how='inner')\n",
    "    return merged_df\n",
    "\n",
    "enc1_data_with_brain_region = add_brain_region_info(enc1_data_categorized, significant_neurons)\n",
    "print(enc1_data_with_brain_region[['subject_id', 'Neuron_ID', 'trial_id', 'preferred_image_id', 'stimulus_index', 'Category', 'Location']].head())\n",
    "updated_enc1_file_path = '/Users/darikussovska/Desktop/PROJECT/graph_encoding1.xlsx'\n",
    "enc1_data_with_brain_region.to_excel(updated_enc1_file_path, index=False)\n",
    "print(f\"Updated Enc1 data with brain region saved to {updated_enc1_file_path}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9aaa6c16-4ab4-470e-ac21-8f7d93b46748",
   "metadata": {},
   "source": [
    "**Repeat the same for fixation, delay and probe. This time, standardize based on the start of the corresponding period.** "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "490f8207-2cfc-4dd3-b3f1-61525935af39",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Load the Enc1 and delay data with brain region from the Excel file\n",
    "enc1_data_with_brain_region = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/graph_encoding1.xlsx')\n",
    "delay_data = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/all_spike_rate_data_delay.xlsx')\n",
    "\n",
    "# Standardize spike times by subtracting the start time of the delay period\n",
    "def standardize_spike_times_delay(df):\n",
    "    df['Standardized_Spikes_in_Delay'] = df.apply(\n",
    "        lambda row: [spike - row['Delay_Start'] for spike in eval(row['Spikes_in_Delay'])] if pd.notna(row['Spikes_in_Delay']) else [],\n",
    "        axis=1\n",
    "    )\n",
    "    return df\n",
    "\n",
    "delay_data = standardize_spike_times_delay(delay_data)\n",
    "\n",
    "columns_to_keep_from_enc1 = [\n",
    "    'subject_id', 'Neuron_ID', 'trial_id', 'Significance', 'new_trial_id',\n",
    "    'Neuron_ID_3', 'preferred_image_id', 'image_id_enc1', 'stimulus_index_enc1',\n",
    "    'image_id_enc2', 'stimulus_index_enc2', 'image_id_enc3', 'stimulus_index_enc3',\n",
    "    'num_images_presented', 'Category', 'Location'\n",
    "]\n",
    "\n",
    "enc1_data_filtered = enc1_data_with_brain_region[columns_to_keep_from_enc1]\n",
    "\n",
    "merged_data = pd.merge(\n",
    "    delay_data,\n",
    "    enc1_data_filtered,\n",
    "    on=['subject_id', 'Neuron_ID', 'trial_id'],\n",
    "    how='inner'\n",
    ")\n",
    "\n",
    "merged_data.to_excel('/Users/darikussovska/Desktop/PROJECT/graph_delay.xlsx', index=False)\n",
    "print(\"Merged data saved\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2ad4d7aa-eef9-4a0f-97d1-624045f1f3ce",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "enc1_data_with_brain_region = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/graph_encoding1.xlsx')\n",
    "probe_data = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/all_spike_rate_data_probe.xlsx')\n",
    "\n",
    "def standardize_spike_times_probe(df):\n",
    "    df['Standardized_Spikes_in_Probe'] = df.apply(\n",
    "        lambda row: [spike - row['Probe_Start_Time'] for spike in eval(row['Spikes_in_Probe'])] if pd.notna(row['Spikes_in_Probe']) else [],\n",
    "        axis=1\n",
    "    )\n",
    "    return df\n",
    "\n",
    "probe_data = standardize_spike_times_probe(probe_data)\n",
    "\n",
    "columns_to_keep_from_enc1 = [\n",
    "    'subject_id', 'Neuron_ID', 'trial_id', 'Significance', 'new_trial_id',\n",
    "    'Neuron_ID_3', 'preferred_image_id', 'image_id_enc1', 'stimulus_index_enc1',\n",
    "    'image_id_enc2', 'stimulus_index_enc2', 'image_id_enc3', 'stimulus_index_enc3',\n",
    "    'num_images_presented', 'Category', 'Location'\n",
    "]\n",
    "\n",
    "enc1_data_filtered = enc1_data_with_brain_region[columns_to_keep_from_enc1]\n",
    "merged_data_probe = pd.merge(\n",
    "    probe_data,\n",
    "    enc1_data_filtered,\n",
    "    on=['subject_id', 'Neuron_ID', 'trial_id'],\n",
    "    how='inner'\n",
    ")\n",
    "\n",
    "merged_data_probe.to_excel('/Users/darikussovska/Desktop/PROJECT/graph_probe.xlsx', index=False)\n",
    "print(\"Merged data saved\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5f6c7ba6-3154-40ac-8be1-be304016ca8a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy.ndimage import gaussian_filter1d\n",
    "\n",
    "merged_data_probe = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/graph_probe.xlsx')\n",
    "\n",
    "# Adjust `Probe_Image_ID` to subtract 1 for consistency (RUN CODE ONLY ONCE)\n",
    "merged_data_probe['Probe_Image_ID'] = merged_data_probe['Probe_Image_ID'] - 1\n",
    "\n",
    "# Add a new column for categorizing trials based on the preferred vs non-preferred images in Enc1 and Probe periods\n",
    "def categorize_trial(row):\n",
    "    if row['Category'] == 'Preferred' and row['Probe_Image_ID'] == row['preferred_image_id']:\n",
    "        return 'Preferred Encoded'\n",
    "    elif row['Category'] == 'Preferred' and row['Probe_Image_ID'] != row['preferred_image_id']:\n",
    "        return 'Preferred Nonencoded'\n",
    "    elif row['Category'] == 'Non-Preferred' and row['Probe_Image_ID'] == row['preferred_image_id']:\n",
    "        return 'Nonpreferred Encoded'\n",
    "    else:\n",
    "        return 'Nonpreferred Nonencoded'\n",
    "\n",
    "merged_data_probe['Trial_Type'] = merged_data_probe.apply(categorize_trial, axis=1)\n",
    "merged_data_probe.to_excel('/Users/darikussovska/Desktop/PROJECT/graph_probe.xlsx', index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "59074fa8-726d-4cf1-99e4-f6d92536667c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "fixation_data = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/all_spike_rate_data_fixation.xlsx')\n",
    "significant_neurons = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/merged_significant_neurons_with_brain_regions.xlsx')\n",
    "trial_info = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/trial_info.xlsx')\n",
    "\n",
    "# Function to standardize spike times by subtracting the start time of the period\n",
    "def standardize_spike_times_fixation(df):\n",
    "    df['Standardized_Spikes_in_Fixation'] = df.apply(lambda row: [spike - row['Fixation_Start'] for spike in eval(row['Spikes_in_Fixation'])] if pd.notna(row['Spikes_in_Fixation']) else [], axis=1)\n",
    "    return df\n",
    "\n",
    "fixation_data = standardize_spike_times_fixation(fixation_data)\n",
    "print(fixation_data[['Neuron_ID', 'trial_id', 'Fixation_Start', 'Spikes_in_Fixation', 'Standardized_Spikes_in_Fixation']].head())\n",
    "\n",
    "filtered_fixation_data = pd.merge(fixation_data, significant_neurons[['subject_id', 'Neuron_ID', 'Location']], on=['subject_id', 'Neuron_ID'], how='inner')\n",
    "filtered_fixation_data = pd.merge(filtered_fixation_data, trial_info, on=['subject_id', 'trial_id'], how='inner')\n",
    "\n",
    "print(filtered_fixation_data.head())\n",
    "filtered_fixation_data.to_excel('/Users/darikussovska/Desktop/PROJECT/graph_fixation.xlsx', index=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "55d8fe25-8f38-47bf-83f8-86d42a62ef78",
   "metadata": {},
   "source": [
    "**Add the other id (final_trial_id) and save the files to the clean_data folder**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5d9814bf-e9d9-4b9e-a24a-249a2a677c5f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from ast import literal_eval\n",
    "\n",
    "fixation_data = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/new_clean_data/cleaned_Fixation.xlsx')\n",
    "enc1 = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/graph_encoding1.xlsx')\n",
    "enc2 = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/graph_encoding2.xlsx')\n",
    "enc3 = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/graph_encoding3.xlsx')\n",
    "delay = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/graph_delay.xlsx')\n",
    "probe = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/graph_probe.xlsx')\n",
    "\n",
    "new_trials = pd.read_excel('/Users/darikussovska/Desktop/PROJECT/new_trial_final.xlsx')\n",
    "def merge_with_new_trial_id(df, new_trials):\n",
    "    return df.merge(new_trials[['trial_id', 'subject_id', 'trial_id_final']],\n",
    "                    on=['trial_id', 'subject_id'],\n",
    "                    how='left')\n",
    "enc1 = merge_with_new_trial_id(enc1, new_trials)\n",
    "enc2 = merge_with_new_trial_id(enc2, new_trials)\n",
    "enc3 = merge_with_new_trial_id(enc3, new_trials)\n",
    "delay = merge_with_new_trial_id(delay, new_trials)\n",
    "probe = merge_with_new_trial_id(probe, new_trials)\n",
    "fixation_data = merge_with_new_trial_id(fixation_data, new_trials) \n",
    "\n",
    "enc1.to_excel('/Users/darikussovska/Desktop/PROJECT/clean_data/graph_encoding1.xlsx', index=False)\n",
    "enc2.to_excel('/Users/darikussovska/Desktop/PROJECT/clean_data/graph_encoding2.xlsx', index=False)\n",
    "enc3.to_excel('/Users/darikussovska/Desktop/PROJECT/clean_data/graph_encoding3.xlsx', index=False)\n",
    "delay.to_excel('/Users/darikussovska/Desktop/PROJECT/clean_data/graph_dela.xlsx', index=False)\n",
    "probe.to_excel('/Users/darikussovska/Desktop/PROJECT/clean_data/graph_probe.xlsx', index=False)\n",
    "fixation_data.to_excel('/Users/darikussovska/Desktop/PROJECT/clean_data/cleaned_Fixation.xlsx', index=False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
