import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Load data
file_path = "/home/daria/PROJECT/Clustering_3D.xlsx"
data = pd.read_excel(file_path)
data.columns = data.columns.str.strip()

metrics = ["firing_rate", "acg_norm","tau_rise", "Burst"]
labels = {
    "firing_rate": "Firing Rate",
    "acg_norm": "ACG (Norm)",
    "tau_rise": "τ Rise",
    "Burst": "burst"
}

# Group separation
data_py = data[data["Cell_Type_New"] == "PY"].copy()
data_in = data[data["Cell_Type_New"] == "IN"].copy()
data_py["Neuron_Type"] = "PY"
data_in["Neuron_Type"] = "IN"
data_combined = pd.concat([data_py, data_in], ignore_index=True)

# Melt for seaborn
data_long = pd.melt(
    data_combined,
    id_vars=["Neuron_Type"],
    value_vars=metrics,
    var_name="Metric",
    value_name="Value"
)
data_long["Metric_Label"] = data_long["Metric"].map(labels)

# -----------------------------
# Mann–Whitney U per metric
# and FDR-BY across metrics
# -----------------------------
raw_pvals = []
metric_has_test = []
for metric in metrics:
    in_vals = data_in[metric].dropna()
    py_vals = data_py[metric].dropna()
    if len(in_vals) > 1 and len(py_vals) > 1:
        _, p = mannwhitneyu(in_vals, py_vals, alternative="two-sided")
        raw_pvals.append(p)
        metric_has_test.append(True)
    else:
        raw_pvals.append(np.nan)
        metric_has_test.append(False)

raw_pvals = np.array(raw_pvals, dtype=float)
valid_mask = ~np.isnan(raw_pvals)

# Apply BY only to valid p-values
pvals_corr = np.full_like(raw_pvals, np.nan, dtype=float)
rejected = np.zeros_like(valid_mask, dtype=bool)
if valid_mask.sum() > 0:
    rej_v, p_corr_v, _, _ = multipletests(raw_pvals[valid_mask], alpha=0.05, method='fdr_by')
    pvals_corr[valid_mask] = p_corr_v
    rejected[valid_mask] = rej_v

def stars_from_p(p):
    if np.isnan(p): return ""
    if p < 0.005:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return ""

# Build dict: metric -> (reject, p_corr, stars)
fdr_results = {}
for metric, rej, p_corr in zip(metrics, rejected, pvals_corr):
    fdr_results[metric] = (bool(rej), p_corr, stars_from_p(p_corr))

custom_palette = {"PY": "#B22436", "IN": "#3254A2"}
sns.set_context("talk")

fig, axs = plt.subplots(2, 3, figsize=(10, 8))
axs = axs.flatten()

# Helper to draw a bracket with stars between x=0 (PY) and x=1 (IN)
def annotate_sig(ax, x1=0, x2=1, text="*", h_frac=0.04, top_margin_frac=0.12, lw=1.5, color="k"):
    if not text:
        return
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin if ymax > ymin else 1.0
    # Start the bracket somewhat below the top, and leave room above
    y = ymax - span * top_margin_frac
    h = span * h_frac
    # If bracket would go out, lift the ceiling
    if y + h * 1.6 > ymax:
        ax.set_ylim(ymin, y + h * 1.8)
        ymin, ymax = ax.get_ylim()
        span = ymax - ymin
        y = ymax - span * top_margin_frac
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, c=color, clip_on=False)
    ax.text((x1 + x2) / 2, y + h + span * 0.01, text,
            ha="center", va="bottom", fontsize=14, fontweight="bold", color=color)

for i, metric in enumerate(metrics):
    ax = axs[i]
    metric_label = labels[metric]
    data_sub = data_long[data_long["Metric"] == metric]

    # Box + strip
    sns.boxplot(
        x="Neuron_Type", y="Value", data=data_sub,
        showfliers=False, palette=custom_palette, ax=ax
    )
    sns.stripplot(
        x="Neuron_Type", y="Value", data=data_sub,
        dodge=False, size=3, alpha=1, color="black", ax=ax
    )

    # Metric-specific y-limits first (so annotation can adapt)
    if metric == "burst":
        ax.set_ylim(-1, 10)
    elif metric == "ACG_Norm":
        ax.set_ylim(-0.1, 1)
    else:
        # default: start at -1
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(min(-0.1, ymin), ymax)

    ax.set_title(metric_label)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # FDR-BY stars
    reject_m, p_corr_m, stars = fdr_results[metric]
    annotate_sig(ax, 0, 1, text=stars)

    # Only bottom row shows x tick labels
    if i // 3 < 1:
        ax.set_xticklabels([])

    # Remove per-axes legend if any
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

# Hide the unused 6th subplot (since we have 5 metrics)
if len(axs) > len(metrics):
    axs[-1].axis("off")

fig.supylabel("Value", fontsize=16)
fig.subplots_adjust(wspace=0.25, hspace=0.38)

# Single figure legend
handles = [
    plt.Line2D([0],[0], marker='s', color='w', label='PY',
               markerfacecolor=custom_palette["PY"], markersize=10),
    plt.Line2D([0],[0], marker='s', color='w', label='IN',
               markerfacecolor=custom_palette["IN"], markersize=10)
]
fig.legend(handles=handles, title="Neuron Type", loc="upper right", fontsize=11)

plt.tight_layout(rect=[0, 0, 0.98, 0.98])
# Save if you want:
plt.savefig("/home/daria/PROJECT/metrics_grid.eps",
             format='eps', dpi=300)
plt.show()

summary = pd.DataFrame({
    "Metric": metrics,
    "p_raw": raw_pvals,
    "p_FDR_BY": pvals_corr,
    "Reject@0.05_BY": rejected
})
print("\nMann–Whitney PY vs IN per metric (raw & FDR-BY):")
print(summary)
