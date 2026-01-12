from os import path
from datasets import ESMEmbeddingDataset
from parse import root_dir
from embed import esm_extract_sequences

def get_dataset(data_name, model_name, root_dir=root_dir):
    embedding_dir = path.join(root_dir, data_name, model_name)
    if path.exists(embedding_dir):
        return ESMEmbeddingDataset(data_name, model_name, root_dir=root_dir)
    esm_extract_sequences(sequences, data_name, model_name, root_dir=root_dir)



