import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu
import itertools

spike_data_path = "../all_spike_rate_data_probe.feather"
trial_info_path = "../trial_info.feather"

df = pd.read_feather(spike_data_path)
trial_info = pd.read_feather(trial_info_path)
df = df.merge(trial_info[['subject_id', 'trial_id', 'num_images_presented']], 
              on=['subject_id', 'trial_id'], how='left')
df = df.dropna(subset=['num_images_presented', 'response_accuracy'])
grouped = df.groupby(['subject_id', 'num_images_presented'])['response_accuracy'].mean().reset_index()

# Global Kruskal–Wallis across loads ---
groups = [grouped.loc[grouped['num_images_presented'] == L, 'response_accuracy']
          for L in sorted(grouped['num_images_presented'].unique())]
H, p_global = kruskal(*groups)
print(f"Global Kruskal–Wallis: H={H:.3f}, p={p_global:.4g}")

# Post-hoc pairwise comparisons
pairs = list(itertools.combinations(sorted(grouped['num_images_presented'].unique()), 2))
results = []
for a, b in pairs:
    ga = grouped.loc[grouped['num_images_presented'] == a, 'response_accuracy']
    gb = grouped.loc[grouped['num_images_presented'] == b, 'response_accuracy']
    stat, p = mannwhitneyu(ga, gb, alternative='two-sided')
    results.append({'pair': f'{a} vs {b}', 'p_raw': p})

# Bonferroni correction
m = len(results)
for r in results:
    r['p_adj'] = min(r['p_raw'] * m, 1.0)

def p_to_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'

for r in results:
    r['stars'] = p_to_stars(r['p_adj'])
    print(f"{r['pair']}: p_raw={r['p_raw']:.4g}, p_adj={r['p_adj']:.4g}, stars={r['stars']}")

def add_sig_bracket(ax, x1, x2, y, h, stars, lw=1.5):
    """
    x1, x2: positions on x-axis (0,1,2...)
    y: baseline height for the bracket
    h: vertical height of the bracket
    stars: string with significance stars
    """
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=lw, c='k')
    ax.text((x1 + x2) * 0.5, y + h, stars, ha='center', va='bottom', fontsize=12)

plt.figure(figsize=(6, 7))
ax = plt.gca()
sns.boxplot(data=grouped, x='num_images_presented', y='response_accuracy', 
            palette='pastel', showfliers=False, ax=ax)
sns.stripplot(data=grouped, x='num_images_presented', y='response_accuracy', 
              hue='subject_id', color='black', size=6, jitter=True, alpha=0.8)
plt.xlabel("Memory Load (Number of Images Presented)")
plt.ylabel("Subject Accuracy")
plt.title("Subject-wise Response Accuracy by Memory Load")
plt.ylim(0.5, 1.2)
plt.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend_.remove()
group_max = grouped.groupby('num_images_presented')['response_accuracy'].max()
y_base = group_max.max()
y_range = grouped['response_accuracy'].max() - grouped['response_accuracy'].min()
if y_range == 0:
    y_range = 0.1  
step = max(0.05 * y_range, 0.03)

load_to_idx = {load: i for i, load in enumerate(sorted(grouped['num_images_presented'].unique()))}
for i, r in enumerate(results):
    a, _, b = r['pair'].partition(' vs ')
    a = int(a); b = int(b)
    x1 = load_to_idx[a]; x2 = load_to_idx[b]
    y = y_base + (i+1) * step
    add_sig_bracket(ax, x1, x2, y, h=0.02 + 0.1*step, stars=r['stars'])

plt.tight_layout()
plt.savefig("./main_1c.eps", format="eps", dpi=300)
plt.show()
