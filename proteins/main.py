"""import sys
from pathlib import Path
project_root = Path.cwd().parents[0]
sys.path.append(str(project_root))"""
from pathlib import Path
import torch
from data.parse import get_tdc_epitope
from data.datasets import ESM2EmbeddingDS, ESMCEmbeddingDS
from model import ESMActiveSite
from training import run_cross_validation
from data.embed import ESMCForgeEmbedder, ESMCEmbedder

data_name = 'IEDB_Jespersen'
model_name = 'esmc-300m-2024-12'
base_data_dir = Path.cwd() / 'data' / 'data_files'
data = get_tdc_epitope(data_name, file_dir=base_data_dir)

sequences = dict(zip(data['ID'], data['Sequence']))

el = ESMCEmbedder('esmc_300m', save_dir=base_data_dir / data_name)
r = el.batch_save(sequences)

ef = ESMCForgeEmbedder(model_name, save_dir=base_data_dir / data_name)
r = ef.batch_save(sequences)

dataset=ESMCEmbeddingDS(data_name, model_name, df=data, save_dir=base_data_dir)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ESMActiveSite(dataset.embed_dim)
run_cross_validation(model, dataset, device=device)


