from pathlib import Path
from data import ESMCSingleDS
from models import SequenceActiveSiteHead
import torch
from training import OptunaSearch, TrainingPipeline, EPTrainer, SinglePipeline
from training.params import ModelParamSpace, TrainerParamSpace

data_name = 'IEDB_Jespersen'
model_name = 'esmc_300m'
base_data_dir = Path.cwd() / 'data' / 'data_files'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

dataset=ESMCSingleDS(data_name, model_name, save_dir=base_data_dir)

results_dir = Path.cwd() / 'experiments'
model_param_space = ModelParamSpace()
trainer_param_space = TrainerParamSpace(max_tokens=10000)

pipeline = TrainingPipeline(dataset, SequenceActiveSiteHead, EPTrainer, device=device)

op = OptunaSearch(pipeline, model_param_space, trainer_param_space, base_save_dir=results_dir, study_name='test2')
op.optimize(20)

from plotting import hist


