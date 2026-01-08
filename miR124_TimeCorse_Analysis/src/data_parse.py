import GEOparse
import os
import gseapy as gp
import asyncio
import pandas as pd
from pathlib import Path

raw_data_folder = os.path.join(Path.cwd().parents[0], "raw_data")

def get_geoparse(
    gse_id: str = "GSE6207",
    file_dir: str = raw_data_folder,
):
    os.makedirs(file_dir, exist_ok=True)
    gse = GEOparse.get_GEO(
        geo=gse_id,
        destdir=file_dir,
        how="full",
        silent=True
    )
    expression_df = gse.pivot_samples("VALUE")
    phenotype_df = gse.phenotype_data
    annot = next(iter(gse.gpls.values())).table
    return expression_df, phenotype_df, annot


async def gp_call(group_genes, gene_set=None, organism='Human', cutoff=0.05):
    if  gene_set is None:
        gene_set = ['GO_Biological_Process_2023', 'KEGG_2021_Human']
    return gp.enrichr(
        gene_list=group_genes,
        gene_sets=gene_set,
        organism=organism,
        cutoff=cutoff
    )

async def get_enrichment_groups(groups):
    groups = [(k, v) for k, v in groups.items()]
    tasks = [gp_call(group[1]) for group in groups]
    results = await asyncio.gather(*tasks)
    final=[]
    for i, group in enumerate(groups):
        sub_res = results[i].results
        sub_res['Group'] = group[0]
        final.append(sub_res)
    return pd.concat(final)

