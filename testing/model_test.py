from inspect import signature
import pandas as pd
import torch
from training.pipeline import TrainingPipeline
import json


class Tester:

    def __init__(self, model_inst, dataset, file_dir, criterion, trial_name=None):
        data_dir = file_dir / "data"
        ckpt_dir = file_dir / "checkpoints"
        if trial_name:
            data_dir = data_dir / trial_name
            ckpt_dir = ckpt_dir / trial_name
        with open(data_dir /"params.json") as f:
            params = json.load(f)
        model_params = signature(model_inst).parameters.keys()
        model_params = {k: v for k, v in params.items() if k in model_params}
        self.model = model_inst(dataset.embed_dim, **model_params)
        self.dataset = dataset
        self.criterion = criterion
        sd = torch.load(ckpt_dir/'best_model.pth', map_location=torch.device('cpu'))
        self.model.load_state_dict(sd)
        self.model.eval()

    def predict(self, x):
        pass










