import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
import umap
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ===================== PATHS =====================
in_path  = "../data/cell_analysis/validation_data.xlsx"
out_feather = "./Validation_clustering.xlsx"
fig_dir  = "./"
os.makedirs(fig_dir, exist_ok=True)

# ===================== LOAD ONCE =====================
df = pd.read_excel(in_path)
df.columns = df.columns.str.strip()

# Columns we’ll need
features_for_clustering = ["firing_rate", "norm_acg", "tau_rise"]
metrics_for_tests       = ["firing_rate", "norm_acg", "tau_rise", "tau_decay"]

# Drop rows missing anything we’ll use downstream
need_cols = list(set(features_for_clustering + metrics_for_tests + ["cell_type"]))
df = df.dropna(subset=[c for c in need_cols if c in df.columns]).copy()

# ===================== CLUSTERING =====================
X_scaled = StandardScaler().fit_transform(df[features_for_clustering].values)
df["Spectral"] = SpectralClustering(
    n_clusters=2, affinity="nearest_neighbors", random_state=42
).fit_predict(X_scaled)

# ===================== UMAP (3D for viz) =====================
embedding = umap.UMAP(n_components=3, random_state=42).fit_transform(X_scaled)
df["UMAP_1"], df["UMAP_2"], df["UMAP_3"] = embedding[:,0], embedding[:,1], embedding[:,2]

# ===================== MAJORITY LABEL & Cell_Type_New =====================
# majority label per spectral cluster (mode; ties -> first)
cluster_majority = (
    df.groupby("Spectral")["cell_type"]
      .agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)
      .to_dict()
)
df["cluster_majority"] = df["Spectral"].map(cluster_majority)
df["Cell_Type_New"] = np.where(df["cell_type"] == df["cluster_majority"], df["cell_type"], np.nan)

# ===================== STATS (incl. purity) =====================
total = len(df)
matched = df["Cell_Type_New"].notna().sum()
mismatched = total - matched

print(f"Total neurons: {total}")
print(f"Matched (cell_type == cluster majority): {matched}")
print(f"Mismatched/Unassigned: {mismatched}\n")

print("Matched breakdown:")
print(df.loc[df["Cell_Type_New"].notna(), "Cell_Type_New"].value_counts(), "\n")

per_cluster = (
    df.assign(_is_match=(df["cell_type"] == df["cluster_majority"]))
      .groupby("Spectral")
      .agg(n=("cell_type", "size"), matches=("_is_match", "sum"))
      .sort_index()
)
per_cluster["purity"] = per_cluster["matches"] / per_cluster["n"]
weighted_purity = per_cluster["matches"].sum() / per_cluster["n"].sum()
macro_purity    = per_cluster["purity"].mean()

print("Per-cluster counts & purity:")
print(per_cluster[["n","purity"]], "\n")
print(f"Weighted purity: {weighted_purity:.3f}")
print(f"Macro purity:    {macro_purity:.3f}\n")

print("Crosstab (cluster majority vs actual cell_type):")
print(pd.crosstab(df["cluster_majority"], df["cell_type"], dropna=False), "\n")

# ===================== 3D UMAP: matched vs unmatched =====================
color_map = {
    "PY": ("#8B0000", "#F4A6A6"),  # (matched, unmatched)
    "IN": ("#00008B", "#A6C8F4")
}
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

for label in ["PY", "IN"]:
    m = (df["cluster_majority"] == label) & (df["Cell_Type_New"].notna())
    ax.scatter(df.loc[m,"UMAP_1"], df.loc[m,"UMAP_2"], df.loc[m,"UMAP_3"],
               s=22, c=color_map[label][0], edgecolor="black", linewidths=0.2, label=f"{label} (match)")
    u = (df["cluster_majority"] == label) & (df["Cell_Type_New"].isna())
    ax.scatter(df.loc[u,"UMAP_1"], df.loc[u,"UMAP_2"], df.loc[u,"UMAP_3"],
               s=18, c=color_map[label][1], edgecolor="none", label=f"{label} (unmatched)")

ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.set_zlabel("UMAP 3")
ax.set_title("3D UMAP • Spectral majority vs. actual cell_type")
ax.legend(loc="best", fontsize=9)
plt.tight_layout()
umap_fig_path = os.path.join(fig_dir, "supp_4e.eps")
plt.savefig(umap_fig_path, format="eps", dpi=300)
plt.show()

# ===================== METRICS: MW-U + FDR-BY =====================
labels_pretty = {
    "firing_rate": "Firing Rate",
    "norm_acg":   "ACG (Norm)",
    "tau_rise":   "τ Rise",
    "tau_decay":  "τ Decay",
}

# Only rows with new type assigned
data = df[df["Cell_Type_New"].isin(["PY","IN"])].copy()
data["Neuron_Type"] = data["Cell_Type_New"]

# Mann–Whitney U per metric (PY vs IN)
raw_pvals = []
for metric in metrics_for_tests:
    py_vals = data.loc[data["Neuron_Type"]=="PY", metric].dropna()
    in_vals = data.loc[data["Neuron_Type"]=="IN", metric].dropna()
    if len(py_vals) > 1 and len(in_vals) > 1:
        _, p = mannwhitneyu(py_vals, in_vals, alternative="two-sided")
        raw_pvals.append(p)
    else:
        raw_pvals.append(np.nan)

raw_pvals = np.array(raw_pvals, dtype=float)
valid = ~np.isnan(raw_pvals)
pvals_corr = np.full_like(raw_pvals, np.nan, dtype=float)
rejected   = np.zeros_like(valid, dtype=bool)

if valid.sum() > 0:
    rej_v, p_corr_v, _, _ = multipletests(raw_pvals[valid], alpha=0.05, method="fdr_by")
    pvals_corr[valid] = p_corr_v
    rejected[valid]   = rej_v

def stars(p):
    if np.isnan(p): return ""
    if p < 0.005: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

fdr_results = {m: (bool(r), pc, stars(pc)) for m, r, pc in zip(metrics_for_tests, rejected, pvals_corr)}

# Plot grid
sns.set_context("talk")
palette = {"PY": "#B22436", "IN": "#3254A2"}

# long-form for seaborn
data_long = data.melt(
    id_vars=["Neuron_Type"], value_vars=metrics_for_tests,
    var_name="Metric", value_name="Value"
)
data_long["Metric_Label"] = data_long["Metric"].map(labels_pretty)

n = len(metrics_for_tests)
ncols = min(3, n)
nrows = ceil(n / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
axes = np.atleast_1d(axes).ravel()

def annotate_sig(ax, text="*", x1=0, x2=1, h_frac=0.06, top_margin_frac=0.15):
    if not text: return
    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1e-6)
    y = ymax - span * top_margin_frac
    h = span * h_frac
    if y + h*1.6 > ymax:
        ax.set_ylim(ymin, y + h*1.8)
        ymin, ymax = ax.get_ylim()
        span = ymax - ymin
        y = ymax - span * top_margin_frac
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c="k", clip_on=False)
    ax.text((x1+x2)/2, y + h + span*0.01, text, ha="center", va="bottom", fontsize=14, fontweight="bold", color="k")

for i, metric in enumerate(metrics_for_tests):
    ax = axes[i]
    sub = data_long[data_long["Metric"] == metric]
    sns.boxplot(x="Neuron_Type", y="Value", data=sub, showfliers=False, palette=palette, ax=ax)
    sns.stripplot(x="Neuron_Type", y="Value", data=sub, dodge=False, size=3, alpha=1, color="black", ax=ax)
    # friendly floor
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(min(ymin, -0.1), ymax)
    ax.set_title(labels_pretty[metric]); ax.set_xlabel(""); ax.set_ylabel("")
    annotate_sig(ax, fdr_results[metric][2])

# turn off any extra axes
for j in range(i+1, len(axes)):
    axes[j].axis("off")

fig.supylabel("Value", fontsize=16)
fig.subplots_adjust(wspace=0.25, hspace=0.38)
handles = [
    plt.Line2D([0],[0], marker='s', color='w', label='PY', markerfacecolor=palette["PY"], markersize=10),
    plt.Line2D([0],[0], marker='s', color='w', label='IN', markerfacecolor=palette["IN"], markersize=10),
]
fig.legend(handles=handles, title="Neuron Type", loc="upper right", fontsize=11)
plt.tight_layout(rect=[0, 0, 0.98, 0.98])

metrics_fig_path = os.path.join(fig_dir, "supp_4ad.eps")
plt.savefig(metrics_fig_path, format="eps", dpi=300)
plt.show()

df.to_feather(out_feather, index=False)
print(f"\nSaved once to: {out_feather}")
print(f"UMAP fig:    {umap_fig_path}")
print(f"Metrics fig: {metrics_fig_path}")

summary = pd.DataFrame({
    "Metric": metrics_for_tests,
    "p_raw": raw_pvals,
    "p_FDR_BY": pvals_corr,
    "Reject@0.05_BY": rejected
})
print("\nMann–Whitney PY vs IN per metric (raw & FDR-BY):")
print(summary)
