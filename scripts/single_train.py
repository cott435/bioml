import torch
from pathlib import Path
from data import ESMSingleDS
from data.utils import pad_collate_fn
from models.crf import LinearChainCRF
from training import OptunaSearch, TrainingPipeline, TokenTrainer, SinglePipeline, GridSearch
from models import TokenActivationHead
from data import SequenceProcessingPipeline


data_name = 'IEDB_Jespersen'
base_data_dir = Path.cwd().parents[0]  / 'data'/ 'data_files'

pipe = SequenceProcessingPipeline(
        data_name=data_name,
        sequence_kind="single",
        save_dir=base_data_dir,
)

dataset = pipe.build_training_dataset(
        storage="lmdb",
        include_embedding=True,
        include_structure=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
results_dir = Path.cwd().parents[0] / 'experiments' / 'lstm'

pipeline = SinglePipeline(dataset, TokenActivationHead, TokenTrainer, results_dir, device=device, epochs=20,
                          small_batch=100, stop_overfit=False)

params = {'max_tokens': 60000, 'hidden_dim': 128, 'layers':3, 'max_norm': 5.5, 'inp_norm': 'instance',
          'feature_dropout': 0.0, 'dropout': 0.0, 'weight_decay': 0.00, 'token_dropout':0.0, 'drop_path': 0.0,
          'base_lr': 3e-3, 'pos_weight': 2, 'probs_weight': 0.00,
          'loss_selection': 'dice_bce', 'block_type': 'ResConvFFN'
         }

pipeline.run(params)





