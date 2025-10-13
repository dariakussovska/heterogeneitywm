import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

# Load Excel
df = pd.read_excel("/home/daria/heterogeneity_wm/data/cell_analysis/decay_constant_change.xlsx")
df.columns = df.columns.str.strip()

# Ensure required columns
required_cols = ['Cell_Type', 'Nonpref Enc', 'Pref Enc', 'Nonpref D', 'Pref D']
assert all(col in df.columns for col in required_cols), "Missing required columns"

# Clean cell type strings
df['Cell_Type'] = df['Cell_Type'].astype(str).str.strip().str.upper()

# Split into cell types
df_py = df[df['Cell_Type'].str.startswith("PY")]
df_in = df[df['Cell_Type'].str.startswith("IN")]

# Function to compute mean, sem, and p-values
def get_stats(df, enc_non, enc_pref, d_non, d_pref):
    means = [df[enc_non].mean(), df[enc_pref].mean(), df[d_non].mean(), df[d_pref].mean()]
    sems = [df[enc_non].sem(), df[enc_pref].sem(), df[d_non].sem(), df[d_pref].sem()]
    p_enc = ttest_rel(df[enc_non], df[enc_pref]).pvalue
    p_delay = ttest_rel(df[d_non], df[d_pref]).pvalue
    return means, sems, [p_enc, p_delay]

# Get stats for PY and IN
means_py, sems_py, pvals_py = get_stats(df_py, 'Nonpref Enc', 'Pref Enc', 'Nonpref D', 'Pref D')
means_in, sems_in, pvals_in = get_stats(df_in, 'Nonpref Enc', 'Pref Enc', 'Nonpref D', 'Pref D')

# Plot setup
labels = ['Encoding', 'Delay']
x = np.arange(len(labels))  # [0, 1]
width = 0.2

fig, ax = plt.subplots(figsize=(6, 4))

# Plot bars: order is Nonpref then Pref for each Cell Type
# PY
ax.bar(x - 1.5*width, [means_py[0], means_py[2]], width, yerr=[sems_py[0], sems_py[2]], label='PY Nonpref', color='lightcoral', capsize=4)
ax.bar(x - 0.5*width, [means_py[1], means_py[3]], width, yerr=[sems_py[1], sems_py[3]], label='PY Pref', color='red', capsize=4)

# IN
ax.bar(x + 0.5*width, [means_in[0], means_in[2]], width, yerr=[sems_in[0], sems_in[2]], label='IN Nonpref', color='lightblue', capsize=4)
ax.bar(x + 1.5*width, [means_in[1], means_in[3]], width, yerr=[sems_in[1], sems_in[3]], label='IN Pref', color='blue', capsize=4)

# Axis and labels
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Mean Tau")
ax.set_title("Preferred vs Nonpreferred Activity by Cell Type")
ax.legend()

# Annotate significance stars (Encoding and Delay separately)
offset = 0.15  # vertical offset
# PY
for i, p in enumerate(pvals_py):
    y = max(means_py[2*i], means_py[2*i+1]) + sems_py[2*i] + sems_py[2*i+1] + offset
    ax.text(x[i] - width, y, '*' if p < 0.05 else 'n.s.', ha='center', color='blue', fontsize=12)
# IN
for i, p in enumerate(pvals_in):
    y = max(means_in[2*i], means_in[2*i+1]) + sems_in[2*i] + sems_in[2*i+1] + offset
    ax.text(x[i] + width, y, '*' if p < 0.05 else 'n.s.', ha='center', color='red', fontsize=12)

plt.tight_layout()
plt.show()
print(pvals_py)
print(pvals_in)
