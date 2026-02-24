


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

params = {'hidden_dim': 128, 'dropout': 0.25, 'activation': 'gelu', 'layers': 3, 'kernel_size': 3,
          'expansion_ratio': 2, 'block_type': 'ConvNeXt1DBlock', 'layer_scale_init_value': 1e-6,
          'drop_path_rate': 0.25, 'lr': 1e-3, 'weight_decay': 5e-2, 'gamma': 2, 'alpha': 0.5,
          'lr_restarts': [True, False], 'jitter': 0.003, 'feature_dropout_first':False,
          'feature_dropout':0.3, 'token_dropout':0.1, 'max_tokens': 30000, 'max_norm': 0.5}

ss = GridSearch(pipeline, params, 'test_grid', base_save_dir=base_data_dir)
ss.optimize()














from pathlib import Path
from models import SequenceActiveSiteHead
import torch
from training.search import OptunaSearch
from training.trainers import EPTrainer
from sklearn.model_selection import GroupKFold
from training.params import ModelParamSpace, TrainerParamSpace


data_name = 'HuRi'
model_name = 'esmc_300m'
base_data_dir = Path.cwd() / 'data' / 'data_files'

#dataset = ESMCMultiDS(data_name, model_name, save_dir=base_data_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

results_dir = Path.cwd() / 'experiments'
model_param_space = ModelParamSpace()
trainer_param_space = TrainerParamSpace()

from data.parse import get_tdc_antibody_aff, get_tdc_protein_pep, get_tdc_ppi, get_tdc_epitope_binding

aff = get_tdc_antibody_aff(file_dir=base_data_dir)
pep = get_tdc_protein_pep(file_dir=base_data_dir)
ppi = get_tdc_ppi(file_dir=base_data_dir)
epi = get_tdc_epitope_binding(file_dir=base_data_dir)


"""
See my scrpts/IEDB_Jespersen.ipynb to see the structure of embedding my data. I need to parse it and pre_embed it and then main.py to see how to run the training pipeline. I build it all for a single sequence dataset where there is one x for each y. Now I want to use pre_embed.MultiSequenceDS to do my training where there will be 2 x for each y. Modify ESMCSingleDS so that two can be handled. For the multi sequence dataset, it is only y=1 if the two proteins interact and all data values are y=1 so the data.sampler.ClusterPairSplitter must be able to cluster the data and resample. Find a way to work this since it will be adding data to the original dataset. Maybe it need modified. In trainers.py build a base trainer and two child classes for each training type if needed. Do not modify my models. I only want you working or data and training logic. Assume the loss function can handle the x and y and masks. Make things modular. The multi should inherit from the singles or both from a base for the datasets and trainers
"""