import sys
from pathlib import Path
project_root = Path.cwd().parents[0]
sys.path.append(str(project_root))
import torch
from data import get_tdc_epitope, plot_viz, cluster_sequences, SequenceEpitopeTokenizer
import numpy as np
from model import ESMActiveSite
from training import run_cross_validation
import esm

data_name = 'IEDB_Jespersen'
data = get_tdc_epitope(data_name, split=False)
data = cluster_sequences(data, data_name).dropna(how='any')

data = data[data['Antigen'].apply(lambda x: len(x) < 5000)]
plot_viz(data['Antigen'], data['Y'])

esm2, esm2_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
tokenizer = SequenceEpitopeTokenizer(esm2_alphabet)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ESMActiveSite(esm2)
run_cross_validation(model, tokenizer, data, device=device)


