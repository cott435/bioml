from pathlib import Path
from data.datasets import ESMCSingleDS
import torch
from proteins.data.utils import pad_collate_fn

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(True)

data_name = 'IEDB_Jespersen'
model_name = 'esmc_300m'
base_data_dir = Path.cwd()/ 'proteins' / 'data' / 'data_files'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
dataset=ESMCSingleDS(data_name, model_name, save_dir=base_data_dir)
dl = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=pad_collate_fn)

in_dim = 960


ln = torch.nn.LayerNorm(in_dim).to(device)
optim = torch.optim.Adam(ln.parameters(), lr=1e-3)

x = next(iter(dl))
x = x[0].to(device)

out = ln(x)
out.mean().backward()
for name, p in ln.named_parameters():
    print(p.grad.detach().clone())


x1 = torch.rand((1, 20, in_dim))
mask1 = torch.ones((1, 20), dtype=torch.bool)
x2 = torch.concat([x1, torch.zeros(1, 100, 960)], dim=1)
mask2 = torch.concat([mask1, torch.zeros(1, 100, dtype=mask1.dtype)], dim=1)

out1, out2 = model(x1.transpose(1, 2), x2.transpose(1, 2), mask1=mask1.unsqueeze(1), mask2=mask2.unsqueeze(1))


from pathlib import Path
from data.datasets import ESMCSingleDS
from proteins.models.model import SequenceActiveSiteHead
import torch
from training.param_search import OptunaSearch
from training.trainers import EPTrainer
from training.params import ModelParamSpace, TrainerParamSpace
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(True)

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





from pathlib import Path
from data.datasets import ESMCSingleDS, MultiSequenceDS, ESMCMultiDS
from proteins.models.model import SequenceActiveSiteHead
import torch
from training.param_search import OptunaSearch
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
op = OptunaSearch(dataset, GroupKFold, SequenceActiveSiteHead, EPTrainer, device=device,
                     base_save_dir=results_dir, study_name='test1',
                     trainer_params=trainer_param_space, model_params=model_param_space, n_splits=10)
op.optimize(20)


from proteins.plotting import hist


