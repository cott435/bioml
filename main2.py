from pathlib import Path
from data.datasets import ESMCMultiDS
from models import SequenceActiveSiteHead
import torch
from training.optuna_search import OptunaSearch
from training.trainers import EPTrainer
from sklearn.model_selection import GroupKFold
from training.params import ModelParamSpace, TrainerParamSpace


data_name = 'HuRi'
model_name = 'esmc_300m'
base_data_dir = Path.cwd() / 'data' / 'data_files'

dataset = ESMCMultiDS(data_name, model_name, save_dir=base_data_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

results_dir = Path.cwd() / 'experiments'
model_param_space = ModelParamSpace()
trainer_param_space = TrainerParamSpace()

from data.parse import get_tdc_antibody_aff, get_tdc_protein_pep, get_tdc_ppi, get_tdc_epitope_binding

aff = get_tdc_antibody_aff(file_dir=base_data_dir)
pep = get_tdc_protein_pep(file_dir=base_data_dir)
ppi = get_tdc_ppi(file_dir=base_data_dir)
epi = get_tdc_epitope_binding(file_dir=base_data_dir)


