import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

default_colors=['blue', 'red','yellow']

def plot_grouped_response_heatmap(main_df, row_df, column_df, colormap=None):
    """column df assumed to be treated as continuous row assumed to be categorical
        row colors come from sns.color_palette; column colors from sns.light_palette"""
    colormap = {} if colormap is None else colormap
    colormap = {col: f'Set{i%3+1}' for i, col in enumerate(row_df.columns) if col not in colormap}
    colormap.update({col: default_colors[i%3+1] for i, col in enumerate(column_df.columns) if col not in colormap})
    handles=[]
    row_colors = pd.DataFrame(index=row_df.index)
    for col in row_df.columns:
        unique = row_df[col].unique()
        cc_map = dict(zip(unique, sns.color_palette(colormap[col], len(unique))))
        row_colors[col] = row_df[col].map(cc_map)
        handles.extend([Patch(facecolor=cc_map[name], edgecolor='w', label=name) for name in unique])

    col_colors = pd.DataFrame(index=column_df.index)
    for col in column_df.columns:
        series = column_df[col]
        norm = plt.Normalize(series.min(), series.max())
        cc_map = sns.light_palette(colormap[col], as_cmap=True)
        col_colors[col] = series.map(lambda x: cc_map(norm(x)))

    plot_data = main_df.loc[row_df.sort_values(by=row_df.columns.to_list()).index]
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
        xticklabels=False,
        figsize=(10, 12),
        cbar_pos=(0.02, 0.8, 0.03, 0.15)  # Move colorbar to top left
    )

    plt.legend(handles=handles, title='Trajectory', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title("Heatmap of Response Trajectories (FDR < 0.05)", y=1.05)
    plt.show()

def plot_enrichment_dotplot(enrichment_df, top_n=5, max_adj_p=0.25):
    """
    Plots a Dot Plot suitable for publication.

    enrichment_df: The output from run_enrichment
    top_n: How many top pathways to show per group
    """

    if enrichment_df.empty:
        print("No enrichment results to plot.")
        return

    plot_df = enrichment_df.sort_values(['Group', 'Adjusted P-value'])
    plot_df = plot_df.groupby('Group').head(top_n)
    plot_df = plot_df[plot_df['Adjusted P-value']<max_adj_p].copy()

    def calculate_ratio(val):
        num, den = val.split('/')
        return float(num) / float(den)

    plot_df['GeneRatio'] = plot_df['Overlap'].apply(calculate_ratio)
    plot_df['LogP'] = -np.log10(plot_df['Adjusted P-value'])
    plot_df['Term_Clean'] = plot_df['Term'].apply(lambda x: x.split(' (GO:')[0])

    plt.figure(figsize=(10, len(plot_df) * 0.4 + 2))  # Dynamic height
    sns.set_style("whitegrid")

    g = sns.scatterplot(
        data=plot_df,
        x="GeneRatio",
        y="Term_Clean",
        size="Genes",
        hue="Adjusted P-value",
        style="Group",
        palette="viridis_r",  # _r reverses it so Purple=0 (Significant) and Yellow=High
        sizes=(50, 400),
        edgecolor='black',
        linewidth=0.5
    )

    plt.title("Functional Enrichment by Trajectory Group", fontsize=15)
    plt.xlabel("Gene Ratio (Genes in List / Genes in Pathway)")
    plt.ylabel("")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.show()

def plot_compare_controls(pheno_df, exp_df, genes, top_n=6, time_col='hours', control_col='treated', title=None):
    genes = genes[:top_n] if len(genes)>top_n else genes
    rows = (len(genes) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(10, 3.5 * rows), sharex=True)
    axes = axes.flatten()

    sns.set_style("whitegrid", {'grid.linestyle': ':'})
    sns.set_context("paper", font_scale=1.2)

    for i, gene in enumerate(genes):
        ax = axes[i]

        gene_data = pheno_df.copy()
        gene_data['Expression'] = exp_df.loc[gene, :].values

        sns.lineplot(
            data=gene_data,
            x=time_col,
            y='Expression',
            hue=control_col,
            palette=['#999999', '#E41A1C'],
            style=control_col,
            dashes=False,
            linewidth=2,
            ax=ax,
            legend=False
        )

        sns.scatterplot(
            data=gene_data,
            x=time_col,
            y='Expression',
            hue=control_col,
            palette=['#4d4d4d', '#800000'],
            style=control_col,
            s=50,
            alpha=0.7,
            ax=ax,
            legend=False
        )

        ax.set_title(gene, fontweight='bold')
        ax.set_ylabel("Log2 Expr")
        ax.set_xlabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=[plt.Line2D([0], [0], color='#999999', lw=2),
                        plt.Line2D([0], [0], color='#E41A1C', lw=2)],
               labels=['Control', 'Treated'],
               loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.97))
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
