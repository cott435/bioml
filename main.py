import pandas as pd
import numpy as np
from data_parse import get_geoparse, get_enrichment_groups
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import asyncio
import seaborn as sns
from utils import *

expression_df, phenotype_df, gene_annot = get_geoparse()
phenotype_df["hours"] = (
    phenotype_df["title"]
    .str.extract(r"(\d+(?:\.\d+)?)\s*hours", expand=False)
    .astype(float)
)
phenotype_df["treated"] = ~phenotype_df["title"].str.contains(
    "negative control", case=False, na=False
)
phenotype_df = phenotype_df[['hours', 'treated']]


# minimal filtering
MIN_EXPR = 4.5  # background cutoff
background_filter = (expression_df > MIN_EXPR).any(axis=1)
exp_filter = expression_df.loc[background_filter]
print(f"Genes after background filter: {exp_filter.shape[0]}")

MIN_VAR = 0.01
var_filter = exp_filter.var(axis=1) > MIN_VAR
exp_filter = exp_filter.loc[var_filter]
print(f"Genes after variance filter: {exp_filter.shape[0]}")


samples = phenotype_df.index
if not set(samples).issubset(exp_filter.columns):
    missing = set(samples) - set(exp_filter.columns)
    raise ValueError(f"Samples in pheno missing from expr columns: {sorted(list(missing))[:10]} ...")
expr = exp_filter.loc[:, samples]

design_df = pd.DataFrame(index=phenotype_df.index)
design_df['time'] = phenotype_df['hours'] - phenotype_df['hours'].mean()
design_df['treatment'] = phenotype_df['treated'].astype(int)
design_df['treat_time'] = design_df['time'] * design_df['treatment']

names = ['treatment', 'treat_time']
results = fit_ols_df(design_df, expr, fdr=names)

fdr = results['fdr']

hits = results[(fdr['treatment'] < 0.05) | (fdr['treat_time'] < 0.05)].sort_values(by=[('fdr', 'treat_time'), ('fdr', 'treatment')])
time_independent = results[(fdr['treatment'] < 0.05) & (fdr['treat_time'] >= 0.05)].sort_values(by=('fdr', 'treatment'))
time_dependent = results[(fdr['treat_time'] < 0.05) & (fdr['treatment'] >= 0.05)].sort_values(by=('fdr', 'treat_time'))
both = results[(fdr['treatment'] < 0.05) & (fdr['treat_time'] < 0.05)].sort_values(by=[('fdr', 'treat_time'), ('fdr', 'treatment')])

print(f"Total responsive genes: {len(hits)}")
print(f"Time Independent only: {len(time_independent)}")
print(f"Time Dependent only: {len(time_dependent)}")
print(f"Both: {len(both)}")

plot_compare_controls(phenotype_df, exp_filter, time_dependent.index)
plot_compare_controls(phenotype_df, exp_filter, time_independent.index)
plot_compare_controls(phenotype_df, exp_filter, both.index)

samples = phenotype_df[phenotype_df['treated']].index
time = phenotype_df[phenotype_df['treated']]['hours']
interaction = results[results[('fdr', 'treat_time')] < 0.05]
all_time_dependent = exp_filter.loc[interaction.index, samples]

popts = fit_trajectories(all_time_dependent, time)

def group_trajectories(s):
    direction = np.sign(s['lin_slope'])
    r2_improvement = s['r2_quad'] - s['r2_linear']
    if r2_improvement > 0.05 and s['quad_importance'] > 0.1:
        if direction * s['q_c'] < 0:
            response = "early_response"
        else:
            response = "late_response"
    else:
        response = "sustained_linear"
    return pd.Series({'Direction':  f'{'positive' if direction>0 else 'negative'} response', 'Type': response})

interaction_traj_groups = popts.apply(group_trajectories, axis=1)

gene_map = gene_annot.set_index('ID')['Gene Symbol']
gene_map = hits.index.to_series().map(gene_map).str.split(r' /// ').str[0]

all_groups = interaction_traj_groups.apply(lambda g: f'{g['Direction']}_{g['Type']}', axis=1)
t_i_groups = time_independent[('beta', 'treatment')].apply(lambda x: f'Time Independent {'positive' if x>0 else 'negative'}')
all_groups = pd.concat([all_groups, t_i_groups])
groups = {g: gene_map.loc[all_groups.index[all_groups == g]].dropna().tolist() for g in all_groups.unique()}

enr_results = asyncio.run(get_enrichment_groups(groups))
enr_results = enr_results[enr_results['Adjusted P-value'] < 0.25].sort_values('Adjusted P-value')
plot_enrichment_dotplot(enr_results)

f=1

hero_genes = interaction.index[:6].to_list()
n_genes = len(hero_genes)
rows = (n_genes + 1) // 2
fig, axes = plt.subplots(rows, 2, figsize=(10, 3.5 * rows), sharex=True)
axes = axes.flatten()  # Flattens the 2D grid to 1D for easy looping

# Aesthetic settings for "Light" look
sns.set_style("whitegrid", {'grid.linestyle': ':'})
sns.set_context("paper", font_scale=1.2)

for i, gene in enumerate(hero_genes):
    ax = axes[i]

    # 1. Prepare data for this specific gene
    # Join expression of this gene with metadata
    gene_data = phenotype_df.copy()
    gene_data['Expression'] = exp_filter.loc[gene, :].values

    # 2. Plot the Trajectory Line (Mean + Confidence Interval)
    # This acts as the "Fitted Line" visual
    sns.lineplot(
        data=gene_data,
        x='hours',
        y='Expression',
        hue='treated',
        palette=['#999999', '#E41A1C'],  # Grey (Control) vs Red (Treated)
        style='treated',
        dashes=False,
        linewidth=2,
        ax=ax,
        legend=False  # We add a master legend later
    )

    # 3. Plot the Raw Data Points (The Reality Check)
    sns.scatterplot(
        data=gene_data,
        x='hours',
        y='Expression',
        hue='treated',
        palette=['#4d4d4d', '#800000'],  # Darker versions for points
        style='treated',
        s=50,  # Size of dots
        alpha=0.7,
        ax=ax,
        legend=False
    )

    ax.set_title(gene, fontweight='bold')
    ax.set_ylabel("Log2 Expr")
    ax.set_xlabel("")  # Hide x-label for clarity until bottom

# Clean up empty subplots if odd number of genes
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Add a single Legend at the top
handles, labels = axes[0].get_legend_handles_labels()
# We only need the first 2 handles (Line Control, Line Treat)
# usually lineplot returns handles for lines+CI, tricky, so we build custom proxy if needed
# But usually just grabbing the first 2 works if configured right.
fig.legend(handles=[plt.Line2D([0], [0], color='#999999', lw=2),
                    plt.Line2D([0], [0], color='#E41A1C', lw=2)],
           labels=['Control', 'Treated'],
           loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))

plt.tight_layout()
plt.show()












