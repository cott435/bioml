import GEOparse
import os

def get_geoparse(
    gse_id: str = "GSE6207",
    geo_dir: str = "./geo_data",
    force_download: bool = False,
):
    os.makedirs(geo_dir, exist_ok=True)
    gse_soft_file = os.path.join(geo_dir, f"{gse_id}_family.soft.gz")
    if force_download or not os.path.exists(gse_soft_file):
        gse = GEOparse.get_GEO(
            geo=gse_id,
            destdir=geo_dir,
            how="full",
        )
    else:
        gse = GEOparse.get_GEO(
            filepath=gse_soft_file,
        )
    expression_df = gse.pivot_samples("VALUE")
    phenotype_df = gse.phenotype_data
    return expression_df, phenotype_df