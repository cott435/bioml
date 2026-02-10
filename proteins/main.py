from pathlib import Path
from data.datasets import ESMCSingleDS
from proteins.models.model import SequenceActiveSiteHead
import torch
from training.param_search import OptunaSearch
from training.trainers import EPTrainer
from training.params import ModelParamSpace, TrainerParamSpace

data_name = 'IEDB_Jespersen'
model_name = 'esmc_300m'
base_data_dir = Path.cwd() / 'data' / 'data_files'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

dataset=ESMCSingleDS(data_name, model_name, save_dir=base_data_dir)

results_dir = Path.cwd() / 'experiments'
model_param_space = ModelParamSpace()
trainer_param_space = TrainerParamSpace()
op = OptunaSearch(dataset, SequenceActiveSiteHead, EPTrainer, model_param_space, trainer_param_space,
                  device=device, base_save_dir=results_dir, study_name='test2', test_size=0.2)
op.optimize(20)


from proteins.plotting import hist
