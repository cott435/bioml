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

data_name = 'IEDB_Jespersen'
model_name = 'esm2_t6_8M_UR50D'
save_dir = path.join(Path.cwd(), 'data', 'data_files')
data = get_tdc_epitope(data_name, file_dir=save_dir)
dataset = ESM2EmbeddingDS(data_name, model_name, df=data, save_dir=save_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ESMActiveSite(dataset.embed_dim)
run_cross_validation(model, dataset, device=device)

def temp(row):
    emb_file = row['ID'] + ".pt"
    filepath = path.join(dataset.embedding_dir, emb_file)
    data = torch.load(filepath, map_location="cpu")
    reps = data["representations"]
    repr_layer = max(reps.keys())
    emb = reps[repr_layer]
    return {'emb_len': len(emb), 'seq_len': len(row['X'])}

d = dataset.data.apply(temp, axis=1)

