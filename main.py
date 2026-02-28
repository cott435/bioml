from pathlib import Path
from data import ESMCSingleDS
from models import SequenceActiveSiteHead
import torch
from training import OptunaSearch, TrainingPipeline, EPTrainer
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

op = OptunaSearch(pipeline, model_param_space, trainer_param_space, base_save_dir=results_dir, study_name='local')
op.optimize(20)





