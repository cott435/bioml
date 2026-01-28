from pathlib import Path
from data.datasets import ESMCSingleDS
from proteins.models.model import SequenceActiveSiteHead
import torch
from training.param_search import OptunaGroupedCV
from training.trainers import Trainer
from sklearn.model_selection import GroupKFold
from training.params import ModelParamSpace, TrainerParamSpace


data_name = 'IEDB_Jespersen'
model_name = 'esmc_300m'
base_data_dir = Path.cwd() / 'data' / 'data_files'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
dataset=ESMCSingleDS(data_name, model_name, save_dir=base_data_dir)

results_dir = Path.cwd() / 'experiments'
model_param_space = ModelParamSpace()
trainer_param_space = TrainerParamSpace()
op = OptunaGroupedCV(dataset, GroupKFold, SequenceActiveSiteHead, Trainer, device=device,
                     base_save_dir=results_dir, study_name='test1',
                     trainer_params=trainer_param_space, model_params=model_param_space, n_splits=10)
op.optimize(5)





