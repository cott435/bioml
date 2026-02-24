from pathlib import Path
from data import ESMCSingleDS
from models import SequenceActiveSiteHead
import torch
from training import OptunaSearch, TrainingPipeline, EPTrainer, SinglePipeline, GridSearch
from training.params import ModelParamSpace, TrainerParamSpace

data_name = 'IEDB_Jespersen'
model_name = 'esmc_300m'
base_data_dir = Path.cwd() / 'data' / 'data_files'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

dataset=ESMCSingleDS(data_name, model_name, save_dir=base_data_dir)

results_dir = Path.cwd() / 'experiments'
model_param_space = ModelParamSpace(hidden_dim=128, activation='gelu', layers=5, kernel_size=3, expansion_ratio=2, block_type='ConvNeXt1DBlock')
trainer_param_space = TrainerParamSpace(max_tokens=30000, gamma=2, alpha=0.5)

pipeline = TrainingPipeline(dataset, SequenceActiveSiteHead, EPTrainer, device=device)

ss = GridSearch

op = OptunaSearch(pipeline, model_param_space, trainer_param_space, base_save_dir=results_dir, study_name='test2')
op.optimize(20)


from plotting import hists

idx = torch.randint(0, len(dataset), (30,))

embs = [dataset[i.item()][0] for i in idx]
ln = torch.concat([(emb - emb.mean(-1, keepdim=True)) / emb.std(-1, keepdim=True) for emb in embs])
fn = torch.concat([(emb - emb.mean(-2, keepdim=True)) / emb.std(-2, keepdim=True) for emb in embs])
bn = torch.concat(embs)
bn = (bn - bn.mean(0, keepdim=True)) / bn.std(0, keepdim=True)

hists([torch.concat(embs), ln, fn, bn], tails=True)

import numpy as np
from matplotlib import pyplot as plt

fnn = fn.flatten().numpy()
arc_fnn = np.arcsinh(fnn)
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
axs[0].hist(fnn.clip(-4,4), bins=100)
axs[1].hist(arc_fnn, bins=100)


fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
axs[0].hist(fn.flatten().numpy(), bins=100)
axs[1].hist(bn.flatten().numpy(), bins=100)
