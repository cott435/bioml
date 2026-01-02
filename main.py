import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
import gseapy as gp

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

results = fit_timecourse_ols(
     expr_gene_by_sample=exp_filter,
     pheno=phenotype_df,
     time_col="hours",
     type_col="treated",
     center_time=True,
 )


res = results.copy()
res["abs_beta_treat"] = res["beta_treat"].abs()
res["abs_beta_interaction"] = res["beta_time_c:treat"].abs()
res["log10_fdr_treat"] = -np.log10(res["fdr_treat"] + 1e-300)
res["log10_fdr_interaction"] = -np.log10(res["fdr_interaction"] + 1e-300)

plt.figure(figsize=(6, 5))
plt.scatter(
    res["se_treat"],
    res["beta_treat"],
    c=res["log10_fdr_treat"],
    cmap="viridis",
    s=10
)
plt.axhline(0, color="black", lw=1)
plt.xlabel("Standard Error (SE)")
plt.ylabel("beta_treat (log2 effect)")
plt.title("Effect size vs uncertainty (colored by -log10 FDR)")
plt.colorbar(label="-log10(FDR)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.scatter(
    res["beta_treat"],
    res["log10_fdr_treat"],
    s=10,
    alpha=0.6
)
plt.axvline(0, color="black", lw=1)
plt.axhline(-np.log10(0.05), color="red", ls="--", lw=1)
plt.xlabel("beta_treat (log2 effect)")
plt.ylabel("-log10(FDR)")
plt.title("Volcano plot: miR-124 main effect")
plt.tight_layout()
plt.show()



plt.figure(figsize=(6, 5))
plt.scatter(
    res["beta_time_c:treat"],
    res["log10_fdr_interaction"],
    s=10,
    alpha=0.6
)
plt.axvline(0, color="black", lw=1)
plt.axhline(-np.log10(0.05), color="red", ls="--", lw=1)
plt.xlabel("beta_time:treat")
plt.ylabel("-log10(FDR)")
plt.title("Time-dependent miR-124 effects")
plt.tight_layout()
plt.show()


hits_main = res[
    (res["fdr_treat"] < 0.05) &
    (res["abs_beta_treat"] > 0.2)
].sort_values("fdr_treat")
hits_time = res[
    (res["fdr_interaction"] < 0.05) &
    (res["abs_beta_interaction"] > 0.01)
].sort_values("fdr_interaction")

top_n = 200
candidates = res.nsmallest(top_n, ['fdr_interaction', 'fdr_treat']).index

# Z-score expression per gene
expr_z = exp_filter.loc[candidates].T
expr_z = (expr_z - expr_z.mean()) / expr_z.std()

# Order samples by time, then treatment
sample_order = phenotype_df.sort_values(['treated', 'hours']).index
expr_z = expr_z.loc[sample_order]

plt.figure(figsize=(10, 12))
sns.heatmap(expr_z, cmap='RdBu_r', center=0, xticklabels=False, cbar_kws={'label': 'Z-score'})
plt.title(f'Top {top_n} Treatment-Responsive Genes (Z-scored expression)')
plt.ylabel('Genes (ordered by FDR)')
plt.xlabel('Samples (ordered by time then treatment)')
plt.show()


hits = res[(res['fdr_treat'] < 0.05) | (res['fdr_interaction'] < 0.05)].copy()
main_only = res[(res['fdr_treat'] < 0.05) & (res['fdr_interaction'] >= 0.05)]
interaction_only = res[(res['fdr_interaction'] < 0.05) & (res['fdr_treat'] >= 0.05)]
both = res[(res['fdr_treat'] < 0.05) & (res['fdr_interaction'] < 0.05)]


interaction = res[res['fdr_interaction'] < 0.05]
samples = phenotype_df[phenotype_df['treated']].index
time = phenotype_df[phenotype_df['treated']]['hours']
interaction_genes = interaction.index
df = exp_filter.loc[interaction_genes, samples]

popts = fit_trajectories(df, time)

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
    return pd.Series({'direction': 'positive' if direction>0 else 'negative', 'response': response})

interaction_traj_groups = popts.apply(group_trajectories, axis=1)

final_main = main_only[['fdr_treat', 'fdr_interaction']].copy()
final_main['response_class'] = 'treatment'

final_interaction = interaction[['fdr_interaction', 'fdr_treat']].copy()
final_interaction['response_class'] = interaction_traj_groups['response']

final = pd.concat([final_main, final_interaction], axis=0)
supp = popts[['lin_slope', 'peak_time']].copy()
supp['response_class'] = interaction_traj_groups['response']
supp['direction'] = interaction_traj_groups['direction']


unique_response = interaction_traj_groups['response'].unique()
lut_row = dict(zip(unique_response, sns.color_palette("Set1", len(unique_response))))
response_colors = interaction_traj_groups['response'].map(lut_row)

unique_dir = interaction_traj_groups['direction'].unique()
lut_dir = dict(zip(unique_dir, sns.color_palette("Set2", len(unique_dir))))
direction_colors = interaction_traj_groups['direction'].map(lut_dir)
row_colors = pd.DataFrame({'Direction': direction_colors, 'Response': response_colors})

# 4. Create Column Colors (Time and Treatment)
# Map Treatment
treatments = phenotype_df['treated']
unique_treat = treatments.unique()
lut_treat = dict(zip(unique_treat, ["#d9d9d9", "#525252"]))  # Light grey vs Dark grey
col_colors_treat = treatments.map(lut_treat)

norm = plt.Normalize(time.min(), time.max())
cmap = sns.cubehelix_palette(as_cmap=True)
col_colors_time = time.map(lambda x: cmap(norm(x)))

col_colors = pd.DataFrame({'Time': col_colors_time})

# 5. Plotting
# We set col_cluster=False to respect the Time order
# We set row_cluster=False (optional) if we want to strictly sort by Group
# OR we set row_cluster=True to see hierarchy within groups, but we need to sort the data first.

# Strategy: Sort data by Group first, then disable row_cluster to keep blocks clean
plot_data = df.loc[interaction_traj_groups.sort_values(by=['response', 'direction']).index]
row_colors = row_colors.loc[plot_data.index]

g = sns.clustermap(
    plot_data,
    z_score=0,  # Normalize rows (genes) to Z-scores
    cmap="vlag",  # Blue-White-Red diverging palette
    center=0,  # Center the colormap at Z=0
    col_cluster=False,  # KEEP columns ordered by Time!
    row_cluster=False,  # KEEP rows ordered by Group!
    row_colors=row_colors,  # Add the trajectory bar
    col_colors=col_colors,  # Add Time/Treat bars
    yticklabels=False,  # Hide gene names (too cluttered usually)
    xticklabels=True,
    figsize=(10, 12),
    cbar_pos=(0.02, 0.8, 0.03, 0.15)  # Move colorbar to top left
)

# Add a legend for the Trajectory Groups
from matplotlib.patches import Patch

handles = [Patch(facecolor=lut_row[name], edgecolor='w', label=name) for name in unique_response]
handles.extend([Patch(facecolor=lut_dir[name], edgecolor='w', label=f'{name} response') for name in unique_dir])
plt.legend(handles=handles, title='Trajectory', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title("Heatmap of Significant Genes (FDR < 0.05)", y=1.05)
plt.show()

gene_map = gene_annot.set_index('ID')['Gene Symbol']
gene_map = interaction_traj_groups.index.to_series().map(gene_map).str.split(r' /// ').str[0]

all_groups = interaction_traj_groups.apply(lambda g: f'{g['direction']}_{g['response']}', axis=1)
enr_results = []
for g in all_groups.unique():
    group_genes = gene_map[all_groups.index[all_groups == g]].dropna().tolist()
    enr = gp.enrichr(
        gene_list=group_genes,
        gene_sets=['GO_Biological_Process_2023', 'KEGG_2021_Human'],  # Select databases
        organism='Human',
        cutoff=0.05  # P-value cutoff
    )
    sub_res = enr.results
    sub_res['Group'] = g
    enr_results.append(sub_res)

enr_results = pd.concat(enr_results)

def plot_enrichment_dotplot(enrichment_df, top_n=5):
    """
    Plots a Dot Plot suitable for publication.

    enrichment_df: The output from run_enrichment
    top_n: How many top pathways to show per group
    """

    if enrichment_df.empty:
        print("No enrichment results to plot.")
        return

    # 1. Filter Top N pathways per Group based on Adjusted P-value
    # Sort by Group and P-value
    plot_df = enrichment_df.sort_values(['Group', 'Adjusted P-value'])
    plot_df = plot_df.groupby('Group').head(top_n).copy()

    # 2. Calculate Gene Ratio (fraction of input genes found in the pathway)
    # The 'Overlap' column usually looks like "10/500"
    def calculate_ratio(val):
        num, den = val.split('/')
        return float(num) / float(den)

    plot_df['GeneRatio'] = plot_df['Overlap'].apply(calculate_ratio)

    # 3. Create a log-scale significance column for color (Log10 P-val)
    # We use -log10 so that higher values = more significant
    plot_df['LogP'] = -np.log10(plot_df['Adjusted P-value'])

    # 4. Clean up Term names (remove the GO:12345 suffix for cleaner plot)
    plot_df['Term_Clean'] = plot_df['Term'].apply(lambda x: x.split(' (GO:')[0])

    # 5. Plotting using Seaborn
    plt.figure(figsize=(10, len(plot_df) * 0.4 + 2))  # Dynamic height

    # We create a categorical Scatter Plot
    sns.set_style("whitegrid")

    # Scatterplot args:
    # x = Effect size (Gene Ratio)
    # y = The Categories (Terms)
    # size = Number of genes (Count)
    # hue = Significance (Adjusted P-value)

    g = sns.scatterplot(
        data=plot_df,
        x="GeneRatio",
        y="Term_Clean",
        size="Genes",  # Size of dot = number of genes from your list
        hue="Adjusted P-value",  # Color of dot = significance
        style="Group",  # Shape of dot = Trajectory Group (Early/Late)
        palette="viridis_r",  # _r reverses it so Purple=0 (Significant) and Yellow=High
        sizes=(50, 400),
        edgecolor='black',
        linewidth=0.5
    )

    # Improving the layout
    plt.title("Functional Enrichment by Trajectory Group", fontsize=15)
    plt.xlabel("Gene Ratio (Genes in List / Genes in Pathway)")
    plt.ylabel("")

    # Move legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.tight_layout()
    plt.show()



