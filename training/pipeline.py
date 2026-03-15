from __future__ import annotations
import gc
import json
from inspect import signature
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional
from torch.utils.data import Subset, DataLoader
from data.utils import bucket_collate_fn
import numpy as np
from data.sampling import RandomTokenBatchSampler, SortedTokenBatchSampler
import torch
from sklearn.model_selection import GroupShuffleSplit
from multiprocessing import cpu_count
from .losses import DynamicBinaryFocalLoss, BinaryFocalLoss, BCELoss


class TrainingPipeline:
    def __init__(
            self,
            dataset,
            model_class: Callable,
            trainer_class: Callable,
            device: torch.device | str = 'cpu',
            test_size: float = 0.2,
            epochs: int = 15,
            base_seed: int = 42,
            small_batch=None,
            stop_overfit: bool = True,
    ):
        self.dataset = dataset
        self.model_class = model_class
        self.trainer_class = trainer_class
        self.epochs = epochs
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.stop_overfit = stop_overfit

        groups = self.dataset.get_data_groups() if hasattr(self.dataset, 'get_data_groups') else None
        cv_splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=base_seed)
        self.train_idx, self.val_idx = next(cv_splitter.split(self.dataset.data, groups=groups))

        self.final_bias = self._calculate_output_bias()
        if small_batch:
            if small_batch > 1:
                self.train_idx = self.train_idx[:small_batch]
                self.val_idx = self.val_idx[:small_batch]
            else:
                self.train_idx, self.val_idx = np.array([2]), np.array([3])

    def _calculate_output_bias(self):
        training_data = self.dataset.data.iloc[self.train_idx]
        pos = training_data['Y'].apply(lambda x: len(x)).sum()
        neg = training_data['Sequence'].apply(lambda x: len(x)).sum() - pos
        return np.log(pos / neg)

    def _get_loaders(self, max_tokens):
        train_ds = Subset(self.dataset, self.train_idx)
        val_ds = Subset(self.dataset, self.val_idx)

        num_workers = max(2, cpu_count() - 2) if self.device.type == 'cuda' else 0
        prefetch_factor = 4 if num_workers > 0 else None
        persistent_workers =  num_workers > 0
        lengths = self.dataset.get_lengths()

        train_loader_sampler = RandomTokenBatchSampler(
            train_ds, lengths[self.train_idx], max_tokens=max_tokens, drop_last=len(self.train_idx) > 300
        )
        train_eval_loader_sampler = SortedTokenBatchSampler(
            train_ds, lengths[self.train_idx], max_tokens=max_tokens * 3
        )
        val_loader_sampler = SortedTokenBatchSampler(
            val_ds, lengths[self.val_idx], max_tokens=max_tokens * 3
        )

        train_loader = DataLoader(
            train_ds, collate_fn=bucket_collate_fn, batch_sampler=train_loader_sampler,
            pin_memory=torch.cuda.is_available(), num_workers=num_workers,
            prefetch_factor=prefetch_factor, persistent_workers=persistent_workers
        )
        train_eval_loader = DataLoader(
            train_ds, collate_fn=bucket_collate_fn, batch_sampler=train_eval_loader_sampler,
            prefetch_factor=prefetch_factor, num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), persistent_workers=persistent_workers
        )
        val_loader = DataLoader(
            val_ds, collate_fn=bucket_collate_fn, batch_sampler=val_loader_sampler,
            prefetch_factor=prefetch_factor, num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), persistent_workers=persistent_workers
        )
        return train_loader, train_eval_loader, val_loader

    def _save_split_indices(self, data_dir: Path):
        np.savez(
            data_dir / "splits.npz",
            train_idx=np.array(self.train_idx),
            val_idx=np.array(self.val_idx),
        )

    @staticmethod
    def _shutdown_loader_workers(loader: Optional[DataLoader]):
        if loader is None:
            return
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
            loader._iterator = None

    def _get_loss_criterion(self, loss_selection, pos_weight=10):
        if loss_selection == 'BCE':
            pos_weight = torch.tensor([pos_weight], device=self.device)
            loss_criterion = BCELoss(reduction='none', pos_weight=pos_weight)
        elif loss_selection == 'focal':
            loss_criterion = BinaryFocalLoss()
        else:
            raise NotImplementedError
        return loss_criterion

    def run(
            self,
            params: Dict[str, Any],
            ckpt_dir: Path,
            log_dir: Path,
            data_dir: Path,
            trial=None
    ) -> float:
        data_dir.mkdir(parents=True, exist_ok=True)

        with open(data_dir / "params.json", "w") as f:
            json.dump(params, f, indent=2)
        self._save_split_indices(data_dir)
        max_tokens = params.pop("max_tokens", 10000)
        loss_selection = params.pop("loss_selection", 'BCE')
        pos_weight = params.pop("pos_weight", 10)
        trainer_kwargs = {k: v for k, v in params.items() if k in self.trainer_class.__init__.__code__.co_varnames}
        model_kwargs = {k: v for k, v in params.items() if k not in trainer_kwargs}


        train_loader = train_eval_loader = val_loader = None
        model = None
        trainer = None
        try:
            train_loader, train_eval_loader, val_loader = self._get_loaders(max_tokens)
            model = self.model_class(
                self.dataset.embed_dim,
                **model_kwargs,
                final_bias=self.final_bias
            )

            loss_criterion = self._get_loss_criterion(loss_selection, pos_weight)

            trainer_extra = {}
            if 'train_eval_loader' in signature(self.trainer_class.__init__).parameters:
                trainer_extra['train_eval_loader'] = train_eval_loader

            trainer = self.trainer_class(
                model,
                loss_criterion,
                train_loader,
                val_loader,
                device=self.device,
                ckpt_dir=ckpt_dir,
                log_dir=log_dir,
                data_dir=data_dir,
                epochs=self.epochs,
                **trainer_extra,
                **trainer_kwargs,
            )

            return trainer.train(trial, stop_overfit=self.stop_overfit)
        finally:
            self._shutdown_loader_workers(train_loader)
            self._shutdown_loader_workers(train_eval_loader)
            self._shutdown_loader_workers(val_loader)

            del trainer, model, train_loader, train_eval_loader, val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            gc.collect()

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
