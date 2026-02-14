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






