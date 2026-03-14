from pathlib import Path
from data import ESMCSingleDS, PackedSequenceDataset
from models import SequenceActiveSiteHead, TokenActivationHead
import torch
from training import OptunaSearch, TrainingPipeline, EPTrainer, SinglePipeline, GridSearch
from training.params import ModelParamSpace, TrainerParamSpace

data_name = 'IEDB_Jespersen'
model_name = 'esmc_300m'
base_data_dir = Path.cwd() / 'data' / 'data_files'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
#dataset=ESMCSingleDS(data_name, model_name, save_dir=base_data_dir)
#dataset.save_full_embedding()
dataset = PackedSequenceDataset(data_name, model_name, save_dir=base_data_dir)

results_dir = Path.cwd() / 'experiments'
model_param_space = ModelParamSpace(hidden_dim=128, activation='gelu', layers=5, kernel_size=3, expansion_ratio=2, block_type='ConvNeXt1DBlock')
trainer_param_space = TrainerParamSpace(max_tokens=30000, gamma=2, alpha=0.5)

pipeline = TrainingPipeline(dataset, TokenActivationHead, EPTrainer, device=device, epochs=40, small_batch=15, stop_overfit=False)

params = {'hidden_dim': 256, 'base_lr': [2e-3, 4e-3], 'kernel_size': [1, 3, 5],
          'gamma': 0, 'alpha': 0.5, 'dropout': 0.0, "layers": 3, 'weight_decay': 0.0,
          'jitter': 0.000, 'warmup_len': 0.2, 'token_dropout':0.0, 'loss_start_ratio': 0.25,
          'max_tokens': 60000, 'max_norm': 5, 'inp_norm': True}


ss = GridSearch(pipeline, params, 'test_batch_10_6', base_save_dir=results_dir)

ss.optimize()



import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_excel(ss.trial_dir / 'results.xlsx')

X = df.drop(columns=["train_score",'trial','val_score'])
y = df["train_score"]

model = RandomForestRegressor()
model.fit(X, y)

importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print(importance)

import seaborn as sns

sns.scatterplot(data=df, x="params.gamma", y="train_score")

"""
See my scrpts/IEDB_Jespersen.ipynb to see the structure of embedding my data. I need to parse it and pre_embed it and then main.py to see how to run the training pipeline. I build it all for a single sequence dataset where there is one x for each y. Now I want to use pre_embed.MultiSequenceDS to do my training where there will be 2 x for each y. Modify ESMCSingleDS so that two can be handled. For the multi sequence dataset, it is only y=1 if the two proteins interact and all data values are y=1 so the data.sampler.ClusterPairSplitter must be able to cluster the data and resample. Find a way to work this since it will be adding data to the original dataset. Maybe it need modified. In trainers.py build a base trainer and two child classes for each training type if needed. Do not modify my models. I only want you working or data and training logic. Assume the loss function can handle the x and y and masks. Make things modular. The multi should inherit from the singles or both from a base for the datasets and trainers
"""