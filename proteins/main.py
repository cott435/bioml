import sys
from pathlib import Path
project_root = Path.cwd().parents[0]
sys.path.append(str(project_root))
import torch
from os import path
from data.parse import get_tdc_epitope
from data.datasets import ESM2EmbeddingDS
from model import ESMActiveSite
from training import run_cross_validation
from data.embed import ESMCForge

data_name = 'IEDB_Jespersen'
model_name = 'esmc-300m-2024-12'
save_dir = path.join(Path.cwd(), 'data', 'data_files')
data = get_tdc_epitope(data_name, file_dir=save_dir)

ef = ESMCForge(model_name, save_dir=path.join(save_dir, data_name))



dataset = ESM2EmbeddingDS(data_name, model_name, df=data, save_dir=save_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ESMActiveSite(dataset.embed_dim)
run_cross_validation(model, dataset, device=device)


