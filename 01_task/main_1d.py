import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

reaction_data = pd.read_excel('/home/daria/PROJECT/graph_data/graph_probe.xlsx')
reaction_data['response_time'] = reaction_data['Probe_End_Time'] - reaction_data['Probe_Start_Time']
df_correct = reaction_data[reaction_data['response_accuracy'] == 1]
subject_medians = df_correct.groupby(['subject_id', 'num_images_presented'])['response_time'].median().reset_index()

plt.figure(figsize=(5, 6))
sns.boxplot(data=subject_medians, x='num_images_presented', y='response_time', palette='Purples', width=0.6, showfliers=False)
sns.stripplot(data=subject_medians, x='num_images_presented', y='response_time',
              color='black', size=6, jitter=True, alpha=0.8)
plt.xlabel("Load")
plt.ylabel("Median Reaction Time (s)")
plt.ylim(0, 2)
plt.title("Reaction Time Across Subjects and Memory Loads")
plt.xticks(ticks=[0, 1, 2], labels=[1, 2, 3])
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("/home/daria/PROJECT/response_time_boxes.eps", format="eps", dpi=300)
plt.show()
