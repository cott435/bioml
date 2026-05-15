from pathlib import Path
from data import SequenceProcessingPipeline
from models import TokenActivationHead
import torch
from training import OptunaSearch, TrainingPipeline, TokenTrainer, SinglePipeline, GridSearch
from training.params import ModelParamSpace, TrainerParamSpace
import numpy as np

data_name = 'IEDB_Jespersen'
model_name = 'esmc_300m'
base_data_dir = Path.cwd() / 'data' / 'data_files'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

pipe = SequenceProcessingPipeline(
        data_name=data_name,
        sequence_kind="single",
        save_dir=base_data_dir,
)

dataset = pipe.build_training_dataset(
        storage="lmdb",
        include_embedding=True,
        include_structure=True,
        representation = 'concat',
        hidden_layers=[10, 20]
)

results_dir = Path.cwd() / 'experiments'
model_param_space = ModelParamSpace(hidden_dim=128, activation='gelu', layers=5, kernel_size=3, expansion_ratio=2, block_type='ConvNeXt1DBlock')
trainer_param_space = TrainerParamSpace(max_tokens=30000, gamma=2, alpha=0.5)

pipeline = TrainingPipeline(dataset, TokenActivationHead, TokenTrainer, device=device, epochs=40, small_batch=100, stop_overfit=False)

params = {'max_tokens': 60000, 'hidden_dim': 128, 'layers':3, 'max_norm': 0.5, 'inp_norm': 'instance',
          'feature_dropout': 0.15, 'dropout': 0.2, 'weight_decay': 0.03, 'token_dropout':0.1, 'drop_path': 0.3,
          'base_lr': 1e-3, 'inp_dropout':0.4,
          'loss_selection': 'dice_bce',
          'block_type': 'ConvNeXt1DBlock',
          'inp_intermediate_dim': 512, 'in_proj_norm': True, 'inp_activation': 'gelu',
}

ss = GridSearch(pipeline, params, 'dice_bce', base_save_dir=results_dir)

ss.optimize()





