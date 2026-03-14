from pathlib import Path
import torch
from data import ESMCSingleDS


data_name = 'IEDB_Jespersen'
model_name = 'esmc_300m'
base_data_dir = Path.cwd() / 'data' / 'data_files'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
dataset=ESMCSingleDS(data_name, model_name, save_dir=base_data_dir)

"""I want to add an ml pipeline to my protein sequence training. populate the directory ml with everything needed such as pipelines, main training scripts, and any other file needed to modularize this well. See ml/train_main.py for the start. My dataset can index all esm embeddings of proteins and has dataset.data that contains clusters which you will use for k fold cross validation. """
