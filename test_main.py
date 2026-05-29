from pathlib import Path
from data import ESMSingleDS
from models import TokenActivationHead
import torch
from training.losses import FocalLoss
from testing.model_test import Tester

data_name = 'IEDB_Jespersen'
model_name = 'esmc_300m'
base_data_dir = Path.cwd() / 'data' / 'data_files'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

dataset=ESMSingleDS(data_name, model_name, save_dir=base_data_dir)


criterion = FocalLoss(reduction="none")
data_name = "IEDB_Jespersen"
model_name = "esmc_300m"
run_dir = Path.cwd() / "experiments" / "dice2"
trial_name = 'trial_0001'
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
hists([x, x1, x2, x3, x4, x5], mask=mask.squeeze(-1))


from plotting import hists, plt
x1=self.dw(x.transpose(1, 2)).transpose(1, 2)
x2=self.norm(x1)
x3=self.exp(x2)
x4 = self.grn(self.activation(x3), mask=mask)
x5=self.cmp(x4)
hists([x, x1, x2, x3, x4, x5], mask=mask.squeeze(-1))
plt.tight_layout()





plt.plot(all_logits[:10000])
plt.plot(all_labels[:10000])
plt.plot(all_losses[:10000])