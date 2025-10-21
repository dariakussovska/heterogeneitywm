import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from mpl_toolkits.mplot3d import Axes3D
import umap.umap_ as umap
import warnings
warnings.filterwarnings("ignore")

# Load and preprocess data
df = pd.read_feather("/./data/cell_analysis/Cell_metrics.feather")

df_filtered = df[(df["R2"] >= 0.3)].dropna(subset=["firing_rate", "acg_norm", "tau_rise"]).copy()

hex_palette = {  
    "Spectral": ['#3254A2', '#941A37']
}

# Feature matrix
X = df_filtered[["firing_rate", "acg_norm", "tau_rise"]].values
X_scaled = StandardScaler().fit_transform(X)

# Spectral clustering
df_filtered["Spectral"] = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42).fit_predict(X_scaled)

# UMAP for Visualization
reducer = umap.UMAP(n_components=3, random_state=42)
embedding = reducer.fit_transform(X_scaled)
df_filtered["UMAP_1"] = embedding[:, 0]
df_filtered["UMAP_2"] = embedding[:, 1]
df_filtered["UMAP_3"] = embedding[:, 2]

# Assign cell types based on Spectral clustering
def map_spectral_to_type(spectral_label):
    if pd.isna(spectral_label):
        return None
    if int(spectral_label) == 1:
        return "PY"
    elif int(spectral_label) == 0:
        return "IN"
    else:
        return None

df_filtered["Cell_Type_New"] = df_filtered["Spectral"].apply(map_spectral_to_type)

# Count results
count_py = len(df_filtered[df_filtered["Cell_Type_New"] == "PY"])
count_in = len(df_filtered[df_filtered["Cell_Type_New"] == "IN"])
count_unlabeled = len(df_filtered[df_filtered["Cell_Type_New"].isna()])

print(f"Total neurons: {len(df_filtered)}")
print(f"Putative Pyramidal cells (PY): {count_py}")
print(f"Putative Interneurons (IN): {count_in}")
print(f"Unlabeled cells: {count_unlabeled}")

# 3D UMAP Plot
color_map = {
    'PY': '#8B0000',  # dark red
    'IN': '#00008B',  # dark blue
    None: 'gray'      # for unlabeled
}

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

for idx, row in df_filtered.iterrows():
    label = row["Cell_Type_New"]
    color = color_map.get(label, 'gray')
    
    ax.scatter(
        row["UMAP_1"],
        row["UMAP_2"],
        row["UMAP_3"],
        c=color,
        s=30,
        edgecolor='black' if pd.notna(label) else 'none',
        linewidths=0.4
    )
    
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w',
           label=f'PY ({count_py})',
           markerfacecolor=color_map['PY'], markeredgecolor='black', markersize=9),
    Line2D([0], [0], marker='o', color='w',
           label=f'IN ({count_in})',
           markerfacecolor=color_map['IN'], markeredgecolor='black', markersize=9)
]

ax.legend(handles=legend_elements, loc='upper left', frameon=True)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_zlabel("UMAP 3")
ax.set_title("3D UMAP: Spectral Cluster-Based Cell Type Assignment")
ax.view_init(elev=34, azim=-61)
plt.tight_layout()
plt.savefig("/./spectral_clustering.eps", format='eps', dpi=300)
plt.show()

# Save feather file with all results
output_path = "/./Clustering_3D.feather"
df_filtered.to_feather(output_path, index=False)
print(f"\nClustered data saved to: {output_path}")
print("Added 'Cell_Type_New' based on Spectral clustering and saved updated file.")
