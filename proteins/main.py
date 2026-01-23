from pathlib import Path
from data.datasets import ESMCSingleDS, SingleSequenceDS
from proteins.models.model import SequenceActiveSiteHead
from proteins.training.model_selection import run_cross_validation
import torch

data_name = 'IEDB_Jespersen'
model_name = 'esmc_300m'
base_data_dir = Path.cwd() / 'data' / 'data_files'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
dataset=ESMCSingleDS(data_name, model_name, save_dir=base_data_dir)

model = SequenceActiveSiteHead(dataset.embed_dim)
run_cross_validation(model, dataset, device=device)


