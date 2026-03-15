from __future__ import annotations

from inspect import signature
import json
from multiprocessing import cpu_count
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from data.sampling import SortedTokenBatchSampler
from data.utils import bucket_collate_fn
from training import EPTrainer


class Tester:

    def __init__(
            self,
            model_inst,
            dataset,
            file_dir: Path,
            criterion,
            trial_name: str | None = None,
            device: torch.device | str = 'cpu',
            checkpoint_name: str = 'final_model.pth',
            max_tokens: int = 10000,
    ):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.data_dir, self.ckpt_dir = self._resolve_run_dirs(Path(file_dir), trial_name)
        self.params = self._load_params(self.data_dir)

        embed_dim = self._resolve_embed_dim(dataset)
        self.model = model_inst(embed_dim, **self.params)
        self.dataset = dataset
        self.criterion = criterion
        self.max_tokens = self.params.get('max_tokens', max_tokens)
        self.train_idx, self.val_idx = self._load_split_indices(self.data_dir)

        checkpoint = torch.load(self.ckpt_dir / checkpoint_name, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def build_loader(self, split: str = 'val', max_tokens: int | None = None):
        max_tokens = max_tokens or self.max_tokens
        dataset = self._get_split_dataset(split)
        lengths = self._resolve_lengths(dataset)

        sampler = SortedTokenBatchSampler(dataset, lengths, max_tokens=max_tokens * 3)
        num_workers = cpu_count() // 2 if self.device.type == 'cuda' else 0
        prefetch_factor = 2 if self.device.type == 'cuda' else None
        return DataLoader(
            dataset,
            collate_fn=bucket_collate_fn,
            batch_sampler=sampler,
            prefetch_factor=prefetch_factor,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.device.type == 'cuda',
        )

    def evaluate_split(self, split: str = 'val', max_tokens: int | None = None):
        loader = self.build_loader(split=split, max_tokens=max_tokens)
        labels, logits, losses, batch_losses = EPTrainer.collect_outputs(
            self.model, loader, self.criterion, self.device
        )
        probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
        score, metrics = EPTrainer.compute_val_metric(probs, labels)
        avg_loss = float(np.mean(batch_losses)) if len(batch_losses) else float('nan')
        metrics['Loss'] = avg_loss
        """
        from plotting import hists
        hists([layer.gamma for layer in self.model.stack.stack])
        """

        return {
            'labels': labels,
            'logits': logits,
            'losses': losses,
            'metrics': metrics,
            'score': score,
        }

    def save_split_metrics(self, split: str, out_path: Path, max_tokens: int | None = None):
        results = self.evaluate_split(split=split, max_tokens=max_tokens)
        np.savez(
            out_path,
            labels=results['labels'],
            logits=results['logits'],
            losses=results['losses'],
        )
        return results

    def load_evals(self, split: str | None = None, path: Path | None = None):
        if path is None:
            if split is None:
                raise ValueError("Provide a split or path to load evals.")
            path = self.data_dir / f"{split}_metrics.npz"
        data = np.load(path)
        return {key: data[key] for key in data.files}

    def predict(self, loader: DataLoader):
        _, logits, _, _ = EPTrainer.collect_outputs(
            self.model, loader, self.criterion, self.device
        )
        return logits

    def _get_split_dataset(self, split: str):
        split = split.lower()
        if split in ('val', 'valid', 'test'):
            if self.val_idx is None:
                return self.dataset
            return Subset(self.dataset, self.val_idx)
        if split in ('train', 'training'):
            if self.train_idx is None:
                return self.dataset
            return Subset(self.dataset, self.train_idx)
        raise ValueError(f"Unknown split: {split}")

    @staticmethod
    def _resolve_run_dirs(file_dir: Path, trial_name: str | None):
        data_dir = file_dir / "data"
        ckpt_dir = file_dir / "checkpoints"
        if trial_name:
            data_dir = data_dir / trial_name
            ckpt_dir = ckpt_dir / trial_name
        return data_dir, ckpt_dir

    @staticmethod
    def _load_params(data_dir: Path):
        params_path = data_dir / "params.json"
        if not params_path.exists():
            raise FileNotFoundError(f"Missing params.json at {params_path}")
        with open(params_path) as f:
            return json.load(f)

    @staticmethod
    def _load_split_indices(data_dir: Path):
        split_path = data_dir / "splits.npz"
        if not split_path.exists():
            return None, None
        split_data = np.load(split_path)
        return split_data['train_idx'], split_data['val_idx']

    @staticmethod
    def _resolve_embed_dim(dataset):
        if hasattr(dataset, 'embed_dim'):
            return dataset.embed_dim
        if isinstance(dataset, Subset):
            return Tester._resolve_embed_dim(dataset.dataset)
        sample = dataset[0][0]
        return sample.shape[-1]

    @staticmethod
    def _resolve_lengths(dataset):
        if isinstance(dataset, Subset):
            base_lengths = Tester._resolve_lengths(dataset.dataset)
            return base_lengths[dataset.indices]
        if hasattr(dataset, 'get_lengths'):
            return dataset.get_lengths()
        lengths = []
        for i in range(len(dataset)):
            lengths.append(len(dataset[i][0]))
        return np.array(lengths)


class MetricsLoader:
    @staticmethod
    def _data_dir(run_dir: Path, trial_name: str | None = None) -> Path:
        data_dir = run_dir / "data"
        if trial_name:
            data_dir = data_dir / trial_name
        return data_dir

    @staticmethod
    def list_trials(run_dir: Path) -> list[str]:
        data_dir = run_dir / "data"
        if not data_dir.exists():
            return []
        trials = [p.name for p in data_dir.iterdir() if p.is_dir()]
        trials = [t for t in trials if t.startswith("trial_")]
        return sorted(trials)

    @staticmethod
    def load_metrics(path: Path):
        data = np.load(path)
        return {key: data[key] for key in data.files}

    @staticmethod
    def load_split(run_dir: Path, split: str, trial_name: str | None = None):
        data_dir = MetricsLoader._data_dir(run_dir, trial_name)
        path = data_dir / f"{split}_metrics.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics file at {path}")
        return MetricsLoader.load_metrics(path)

    @staticmethod
    def load_run(run_dir: Path, split: str, trial_names: list[str] | None = None):
        if trial_names is None:
            trial_names = MetricsLoader.list_trials(run_dir)
        if not trial_names:
            return [MetricsLoader.load_split(run_dir, split)], None
        metrics = [MetricsLoader.load_split(run_dir, split, t) for t in trial_names]
        return metrics, trial_names







