from os import path, makedirs
from .datasets import ESMEmbeddingDataset
from .parse import data_dir, make_sequence_fasta, cluster_fasta, add_clusters_to_df, df_save
from .embed import esm_extract_sequences
"""
data must be downloaded clustered and saved from tdc
    dataset will recieve Y from here
    KFold will need clusters from here too
will also be used 


get data from tdc
write fasta file
    Likely need to save df too for Y
use fasta for clustering
use fasta for extraction of embeddings

have a load clusters from fasta method
dataset should be able to load embeddings and Y given model and data name

"""

def get_esm_embedded_dataset(data_name, model_name, data_dir=data_dir):

    """
    check for fasta, raise error if not found (colab does not have tdc and slow to install there)

    check for df file, raise error if not found

    if both found, continue to embedding from fasta file

    move to dataset
    """
    data_dir = path.join(data_dir, data_name)
    df_path = path.join(data_dir, 'dataframe.parquet')
    fasta_path = path.join(data_dir, 'sequences.fasta')
    embedding_dir = path.join(data_dir, model_name)

    if path.exists(embedding_dir):
        # TODO: add df check for Y
        return ESMEmbeddingDataset(data_name, model_name, root_dir=data_dir)


    if path.exists(fasta_path):
        esm_extract_sequences()
    else:
        raise FileNotFoundError(f'fasta file not found: {fasta_path}; please add to directory')


# TODO may need new pipeline for multi; below is for single only

def cluster_sequences(data, data_name, cluster_coef=0.5, data_dir=data_dir, force=False):
    dataset_dir = path.join(data_dir, data_name)
    makedirs(dataset_dir, exist_ok=True)
    fasta_path = make_sequence_fasta(data['X'], data['ID'], file_dir=dataset_dir, force=force)
    clstr_path = cluster_fasta(fasta_path, cluster_coef=cluster_coef, force=force)
    clustered_data = add_clusters_to_df(data, clstr_path)
    df_save(clustered_data, file_dir=dataset_dir, force=force)
    return clustered_data




