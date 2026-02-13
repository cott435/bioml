from inspect import signature
import pandas as pd
import torch

class Tester:

    def __init__(self, model_inst, dataset, file_dir, criterion):
        params = pd.read_csv(file_dir/'params.csv').set_index('Parameter').to_dict()['Value']
        model_params = signature(model_inst).parameters.keys()
        model_params = {k: v for k, v in params.items() if k in model_params}
        self.model = model_inst(dataset.embed_dim, **model_params)
        self.dataset = dataset
        self.criterion = criterion
        sd = torch.load(file_dir/'best_model.pth', map_location=torch.device('cpu'))
        self.model.load_state_dict(sd)
        self.model.eval()

    def predict(self, x):
        pass

