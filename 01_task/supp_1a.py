import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Excel file
file_path = "/home/daria/PROJECT/all_spike_rate_data_probe.xlsx"  
df = pd.read_excel(file_path)

accuracy_df = df.groupby('subject_id')['response_accuracy'].agg(['sum', 'count'])
accuracy_df['response_accuracy'] = accuracy_df['sum'] / accuracy_df['count']  # Correct trials / Total trials

# Rank subjects by accuracy (from lowest to highest)
accuracy_df = accuracy_df.sort_values(by='response_accuracy').reset_index()
accuracy_df['rank'] = range(1, len(accuracy_df) + 1)  

mean_accuracy = accuracy_df['response_accuracy'].mean()
sem_accuracy = accuracy_df['response_accuracy'].std() / np.sqrt(len(accuracy_df))  # SEM = std / sqrt(N)

print("=== Subject Response Accuracy Summary ===")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Standard Error of the Mean (SEM): {sem_accuracy:.4f}")

plt.figure(figsize=(10, 5))
plt.scatter(accuracy_df['rank'], accuracy_df['response_accuracy'], c=accuracy_df['rank'], cmap='viridis', s=100, label="Subjects")
plt.axhline(y=mean_accuracy, color='red', linestyle='--', linewidth=2, label=f"Mean Accuracy ({mean_accuracy:.2f})")
plt.fill_between(accuracy_df['rank'], mean_accuracy - sem_accuracy, mean_accuracy + sem_accuracy, 
                 color='red', alpha=0.2, label=f"SEM ± {sem_accuracy:.2f}")
plt.xlabel("Subject Rank (Lowest to Highest Accuracy)")
plt.ylabel("Response Accuracy")
plt.title("Response Accuracy per Subject (Rank Ordered) with Mean ± SEM")
plt.colorbar(label="Rank Order")  
plt.xticks(accuracy_df['rank']) 
plt.ylim(0.5, 1) 
plt.grid(True)
plt.legend()
plt.savefig("/home/daria/PROJECT/01_task/supp_1a.eps", format="eps", dpi=300)
plt.show()
