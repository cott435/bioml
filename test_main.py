from pathlib import Path
from data import ESMCSingleDS
from models import TokenActivationHead
import torch
from training.losses import BinaryFocalLoss
from testing.model_test import Tester

data_name = 'IEDB_Jespersen'
model_name = 'esmc_300m'
base_data_dir = Path.cwd() / 'data' / 'data_files'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

dataset=ESMCSingleDS(data_name, model_name, save_dir=base_data_dir)


criterion = BinaryFocalLoss(reduction="none")
data_name = "IEDB_Jespersen"
model_name = "esmc_300m"
run_dir = Path.cwd() / "experiments" / "cur"
trial_name = 'trial_0000'
tester = Tester(
    TokenActivationHead,
    dataset,
    run_dir,
    criterion,
    trial_name=trial_name,
    device=device,
)

train_results = tester.evaluate_split("train")
val_results = tester.evaluate_split("val")

from plotting import hists, plt
hists([x_norm, x_proj, x], mask=mask)
hists([x1,x2,x3,x4,x5], mask=mask.squeeze(-1))



x_dim = x_flat.mean(-1).cpu().detach().numpy()
x_tok = x_flat.mean(0).cpu().detach().numpy()
plt.figure()
plt.plot(x_dim)
plt.figure()
plt.plot(x_tok)

plt.figure()
plt.imshow(x_flat.cpu().detach().numpy().T)

emb = np.array(self.dataset.embeddings[:40000])
plt.figure()
plt.imshow(emb)