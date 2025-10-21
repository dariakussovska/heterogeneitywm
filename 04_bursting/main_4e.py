import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "../Clustering_3D.feather"
data = pd.read_excel(file_path)

# Clean column names
data.columns = data.columns.str.strip()

data_filtered = data[(data["R2"] >= 0.3)]

# Drop rows where Decay is NaN
data_filtered = data.dropna(subset=["Decay"])

# Filter again to only keep neurons with defined Cell Type
#data_filtered = data_filtered.dropna(subset=["Cell_Type_New"])

# === Plot Decay only ===
plt.figure(figsize=(8, 6))

sns.histplot(
    data=data_filtered,
    x="Decay",
    bins=50,
    kde=True,
    alpha=0.7
)

plt.title("Histogram of Decay", fontsize=14)
plt.xlabel("Decay (ms)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.ylim(0, 50)  # Adjust as needed depending on your data range
plt.grid(True, linestyle="--", alpha=0.8)

plt.tight_layout()
plt.savefig("./main_4e.eps", format='eps', dpi=300)
plt.show()
