from pathlib import Path
from data import ESMCSingleDS
from models import SequenceActiveSiteHead
import torch
from testing.model_test import Tester
from training.losses import BinaryFocalLoss
from training import OptunaSearch, TrainingPipeline, EPTrainer, SinglePipeline
from training.params import ModelParamSpace, TrainerParamSpace

data_name = 'IEDB_Jespersen'
model_name = 'esmc_300m'
base_data_dir = Path.cwd() / 'data' / 'data_files'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

dataset=ESMCSingleDS(data_name, model_name, save_dir=base_data_dir)

results_dir = Path.cwd() / 'experiments'
file_dir = Path('/Users/connorott/Downloads/add_alpha')

tester = Tester(SequenceActiveSiteHead, dataset, file_dir, BinaryFocalLoss(), trial_name="trial_0003")

import numpy as np

t4 = np.load('/Users/connorott/Downloads/004.npz')
t5 = np.load('/Users/connorott/Downloads/005.npz')
t4 = {f: t4[f] for f in t4.files}
t5 = {f: t5[f] for f in t5.files}
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import f1_score, matthews_corrcoef, average_precision_score

mcc = np.array([matthews_corrcoef(t4['labels'][i], t4['logits'][i]>0.5) for i in range(len(t4['logits']))])
f1 = np.array([f1_score(t4['labels'][i], t4['logits'][i]>0.5) for i in range(len(t4['logits']))])
prc = np.array([average_precision_score(t4['labels'][i], t4['logits'][i]) for i in range(len(t4['logits']))])

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ax=axes[0]
ax.plot(mcc, label='MCC')
ax.plot(f1, label='F1')
ax.plot(prc, label='PRC')
ax.legend()

ax=axes[1]
sns.kdeplot(t4['logits'][0][t4['labels'][0] == 0], fill=True, color='red', label='Non-Binding (Sampled)', ax=ax)
sns.kdeplot(t4['logits'][0][t4['labels'][0] == 1], fill=True, color='green', label='Binding (All)', ax=ax)
ax.set_title(f'Logit Separation (Epoch {len(t4['labels']) - 1})')
ax.set_xlabel('Logits')
ax.legend()



