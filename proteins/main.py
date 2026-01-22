"""import sys
from pathlib import Path
project_root = Path.cwd().parents[0]
sys.path.append(str(project_root))"""
from pathlib import Path
import torch
from data.parse import get_tdc_epitope, get_tdc_ppi
from data.datasets import ESMCEmbeddingDS, SingleSequenceDS
from model import SequenceActiveSiteHead
from training import run_cross_validation
from data.embed import ESMCBatchEmbedder



data_name = 'IEDB_Jespersen'
model_name = 'esmc_300m'
base_data_dir = Path.cwd() / 'data' / 'data_files'

data=get_tdc_ppi(file_dir=base_data_dir)

data = get_tdc_epitope(data_name, file_dir=base_data_dir)
lens = data['Sequence'].apply(lambda x: len(x))
ssd = SingleSequenceDS(data_name, df=data, save_dir=base_data_dir)

sequences = ssd.get_data_dict()

#forge_embedder = ESMCForgeEmbedder(model_name, save_dir=base_data_dir / data_name)
#forge_embedder.batch_save(sequences)

#single_embedder = ESMCEmbedder(model_name, save_dir=base_data_dir / data_name)
#single_embedder.save_sequence_embedding('Protein4', sequences['Protein4'])
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
el = ESMCBatchEmbedder(model_name, save_dir=base_data_dir / data_name, device=device)
el.batch_save(sequences)


dataset=ESMCEmbeddingDS(data_name, model_name, df=data, save_dir=base_data_dir)

model = SequenceActiveSiteHead(dataset.embed_dim)
run_cross_validation(model, dataset, device=device)


