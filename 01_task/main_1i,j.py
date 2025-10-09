import pandas as pd
import matplotlib.pyplot as plt

merged_df = pd.read_excel('/home/daria/PROJECT/all_neuron_brain_regions_cleaned.xlsx')

def categorize_region(location):
    vmPFC = ['ventral_medial_prefrontal_cortex_left', 'ventral_medial_prefrontal_cortex_right']
    dACC = ['dorsal_anterior_cingulate_cortex_right', 'dorsal_anterior_cingulate_cortex_left']
    hippocampus = ['hippocampus_left', 'hippocampus_right']
    amygdala = ['amygdala_left', 'amygdala_right']
    pre_SMA = ['pre_supplementary_motor_area_left', 'pre_supplementary_motor_area_right']
    
    if location in vmPFC:
        return 'vmPFC'
    elif location in dACC:
        return 'dACC'
    elif location in hippocampus:
        return 'Hippocampus'
    elif location in amygdala:
        return 'Amygdala'
    elif location in pre_SMA:
        return 'pre-SMA'
    else:
        return 'Other'

merged_df['Region_Category'] = merged_df['Location'].apply(categorize_region)
summary = merged_df.groupby('Region_Category')['Signi'].value_counts().unstack(fill_value=0)
summary.columns = summary.columns.astype(str)  # Ensure columns are strings like 'Y' and 'N'
summary['Total_Neurons'] = summary.sum(axis=1)
summary['Significant_Neurons'] = summary.get('Y', 0)
summary['Percentage'] = (summary['Significant_Neurons'] / summary['Total_Neurons']) * 100

summary_table = summary[['Significant_Neurons', 'Total_Neurons', 'Percentage']]
summary_table = summary_table.sort_values(by='Percentage', ascending=False)
print(summary_table.round(2))

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# First subplot: Distribution of all neurons
region_counts = merged_df['Region_Category'].value_counts()
ax1.pie(region_counts, labels=region_counts.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
ax1.set_title("Distribution of Neurons Across Brain Regions")

# Second subplot: Distribution of concept cells
desired_order = ['Amygdala', 'vmPFC', 'dACC', 'Hippocampus', 'pre-SMA']
pie_data = summary_table[summary_table['Significant_Neurons'] > 0]
pie_data = pie_data.reindex(desired_order).dropna()

ax2.pie(
    pie_data['Significant_Neurons'],
    labels=pie_data.index,
    autopct='%1.1f%%',
    startangle=90,           
    counterclock=False,      
    colors=plt.get_cmap('tab20').colors
)
ax2.set_title('Distribution of Concept Cells Across Brain Regions')

plt.tight_layout()
output_path = "/home/daria/PROJECT/01_task/combined_distributions.eps"
plt.savefig(output_path, format='eps', dpi=300)
plt.show()
print(f"EPS file saved at: {output_path}")
