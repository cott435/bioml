from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional
from torch.utils.data import Subset, DataLoader
from data.utils import save_params_as_csv, bucket_collate_fn
import numpy as np
from data.sampling import RandomTokenBatchSampler, SortedTokenBatchSampler
import torch
from sklearn.model_selection import GroupShuffleSplit
from multiprocessing import cpu_count


class TrainingPipeline:
    def __init__(
            self,
            dataset,
            model_class: Callable,
            trainer_class: Callable,
            device: torch.device | str = 'cpu',
            test_size: float = 0.2,
            epochs: int = 25,
            base_seed: int = 42
    ):
        self.dataset = dataset
        self.model_class = model_class
        self.trainer_class = trainer_class
        self.epochs = epochs
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        groups = self.dataset.get_data_groups() if hasattr(self.dataset, 'get_data_groups') else None
        cv_splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=base_seed)
        self.train_idx, self.val_idx = next(cv_splitter.split(self.dataset.data, groups=groups))

        self.out_bias = self._calculate_output_bias()

    def _calculate_output_bias(self):
        training_data = self.dataset.data.iloc[self.train_idx]
        pos = training_data['Y'].apply(lambda x: len(x)).sum()
        neg = training_data['Sequence'].apply(lambda x: len(x)).sum() - pos
        return np.log(pos / neg)

    def _get_loaders(self, max_tokens):
        train_ds = Subset(self.dataset, self.train_idx)
        val_ds = Subset(self.dataset, self.val_idx)

        num_workers = cpu_count() // 2 if self.device.type == 'cuda' else 0
        prefetch_factor = 2 if self.device.type == 'cuda' else None
        lengths = self.dataset.get_lengths()

        train_loader_sampler = RandomTokenBatchSampler(
            train_ds, lengths[self.train_idx], max_tokens=max_tokens, drop_last=True
        )
        val_loader_sampler = SortedTokenBatchSampler(
            val_ds, lengths[self.val_idx], max_tokens=max_tokens * 3
        )

        train_loader = DataLoader(
            train_ds, collate_fn=bucket_collate_fn, batch_sampler=train_loader_sampler,
            pin_memory=torch.cuda.is_available(), num_workers=num_workers,
            prefetch_factor=prefetch_factor, persistent_workers=self.device.type == 'cuda'
        )
        val_loader = DataLoader(
            val_ds, collate_fn=bucket_collate_fn, batch_sampler=val_loader_sampler,
            prefetch_factor=prefetch_factor, num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), persistent_workers=self.device.type == 'cuda'
        )
        return train_loader, val_loader

    def run(
            self,
            params: Dict[str, Any],
            ckpt_dir: Path,
            log_dir: Path,
            data_dir: Path,
            trial=None
    ) -> float:

        with open(data_dir / "params.json", "w") as f:
            json.dump(params, f, indent=2)
        model_kwargs = {k: v for k, v in params.items() if k in self.model_class.__init__.__code__.co_varnames}
        trainer_kwargs = {k: v for k, v in params.items() if k not in model_kwargs}

        # Handle special params
        max_tokens = trainer_kwargs.pop("max_tokens", 10000)


        # Init components
        train_loader, val_loader = self._get_loaders(max_tokens)
        model = self.model_class(
            self.dataset.embed_dim,
            **model_kwargs,
            out_bias=self.out_bias
        )

        trainer = self.trainer_class(
            model,
            train_loader,
            val_loader,
            device=self.device,
            ckpt_dir=ckpt_dir,
            log_dir=log_dir,
            data_dir=data_dir,
            epochs=self.epochs,
            **trainer_kwargs,
        )

        score = trainer.train(trial)
        return score

class SinglePipeline(TrainingPipeline):

    def __init__(
            self,
            dataset,
            model_class: Callable,
            trainer_class: Callable,
            save_dir: Path,
            device: torch.device | str = 'cpu',
            test_size: float = 0.2,
            epochs: int = 25,
            base_seed: int = 42
    ):
        super().__init__(dataset, model_class, trainer_class, device=device, test_size=test_size, epochs=epochs, base_seed=base_seed)
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def run(self, params: Dict[str, Any], trial=None):
        ckpt_dir = self.save_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir = self.save_dir / "logging"
        log_dir.mkdir(parents=True, exist_ok=True)
        data_dir = self.save_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return super().run(params, ckpt_dir, log_dir, data_dir, trial=trial)
