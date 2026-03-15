from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models.blocks import ConvNeXt1DBlock,ResConvFFN


def _rms(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return torch.sqrt(torch.mean(tensor.float().pow(2))).item()


def _param_grad_rms(module: nn.Module) -> float | None:
    sq_sum = 0.0
    count = 0
    for param in module.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        sq_sum += grad.pow(2).sum().item()
        count += grad.numel()
    if count == 0:
        return None
    return (sq_sum / count) ** 0.5


class ConvNeXtTelemetry:
    """
    Hook-based ConvNeXt block logger for TensorBoard.

    Logs:
    - Forward residual-vs-skip stats (input RMS, residual delta RMS, delta/skip ratio)
    - Backward flow stats (grad input/output RMS, in/out ratio)
    - Branch parameter grad RMS compared against skip-path gradient signal
    """

    def __init__(
        self,
        model: nn.Module,
        writer: SummaryWriter,
        log_every_steps: int = 10,
    ) -> None:
        self.model = model
        self.writer = writer
        self.log_every_steps = max(1, int(log_every_steps))
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.blocks: Dict[str, ConvNeXt1DBlock] = {}
        self.forward_stats: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.backward_stats: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for name, module in model.named_modules():
            if isinstance(module, (ConvNeXt1DBlock, ResConvFFN)):
                self.blocks[name] = module

        self.enabled = writer is not None and len(self.blocks) > 0
        if self.enabled:
            self._attach_hooks()

    def _attach_hooks(self) -> None:
        for name, module in self.blocks.items():
            self.handles.append(module.register_forward_hook(self._forward_hook(name)))
            self.handles.append(
                module.register_full_backward_hook(self._backward_hook(name))
            )

    def _forward_hook(self, name: str):
        def hook(module, inputs, output):
            if not module.training:
                return
            if not inputs:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor) or not isinstance(output, torch.Tensor):
                return
            if x.shape != output.shape:
                return

            x_detached = x.detach()
            y_detached = output.detach()
            delta = y_detached - x_detached  # residual branch contribution

            skip_rms = _rms(x_detached)
            delta_rms = _rms(delta)
            out_rms = _rms(y_detached)
            ratio = delta_rms / (skip_rms + 1e-12)

            stats = self.forward_stats[name]
            stats["skip_rms"].append(skip_rms)
            stats["delta_rms"].append(delta_rms)
            stats["out_rms"].append(out_rms)
            stats["delta_to_skip_ratio"].append(ratio)

        return hook

    def _backward_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            if not module.training or not grad_output:
                return

            g_out = grad_output[0]
            g_in = grad_input[0] if grad_input else None

            if not isinstance(g_out, torch.Tensor):
                return

            # 1. Base arriving signal (Scalar)
            out_rms = _rms(g_out.detach())

            if isinstance(g_in, torch.Tensor):
                # 2. Base departing signal (Scalar)
                in_rms = _rms(g_in.detach())

                # 3. Isolate the residual branch (Tensor Subtraction -> Scalar RMS)
                residual_grad = g_in.detach() - g_out.detach()
                residual_rms = _rms(residual_grad)

                # 4. Calculate Ratios (Scalar Division)
                preservation_ratio = in_rms / (out_rms + 1e-12)
                activity_ratio = residual_rms / (out_rms + 1e-12)

            else:
                in_rms = 0.0
                residual_rms = 0.0
                preservation_ratio = 0.0
                activity_ratio = 0.0

            # 5. Log everything
            stats = self.backward_stats[name]
            #stats["grad_in_rms"].append(in_rms)
            stats["grad_out_rms"].append(out_rms)
            stats["grad_in_to_out_ratio"].append(preservation_ratio)
            stats["grad_residual_rms"].append(residual_rms)
            stats["grad_activity_ratio"].append(activity_ratio)

        return hook

    @staticmethod
    def _mean(values: List[float]) -> float | None:
        if not values:
            return None
        return float(sum(values) / len(values))

    def log_step(self, step: int) -> None:
        if not self.enabled:
            return
        if step % self.log_every_steps != 0:
            return

        for block_name, block in self.blocks.items():
            fwd = self.forward_stats.get(block_name, {})
            bwd = self.backward_stats.get(block_name, {})

            for item_name, items in fwd.items():
                ave = self._mean(items)
                if ave is not None:
                    self.writer.add_scalar(f"Forward/{block_name}/{item_name}", ave, step)

            for item_name, items in bwd.items():
                ave = self._mean(items)
                if ave is not None:
                    self.writer.add_scalar(f"Backward/{block_name}/{item_name}", ave, step)

        self._clear_buffers()

    def _clear_buffers(self) -> None:
        self.forward_stats.clear()
        self.backward_stats.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._clear_buffers()



from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
def load_tensorboard_scalar(
    log_dir: str,
    scalar_tag: str,
    align: str = "truncate",
):
    """
    Load a scalar from TensorBoard event files.
    Parameters
    ----------
    log_dir : str
        Directory containing run subfolders.
    scalar_tag : str
        TensorBoard scalar name (e.g. 'train/loss').
    align : str
        'truncate' -> trim all runs to shortest length
        'pad' -> pad with NaN to longest length
    Returns
    -------
    values : np.ndarray
        Shape (n_runs, n_steps)
    steps : np.ndarray
        Steps used (based on first run)
    run_names : list[str]
    """
    log_dir = Path(log_dir)
    run_dirs = [d for d in log_dir.iterdir() if d.is_dir()]
    runs = []
    steps_list = []
    run_names = []
    for run in sorted(run_dirs):
        try:
            acc = EventAccumulator(str(run))
            acc.Reload()
            if scalar_tag not in acc.Tags()["scalars"]:
                continue
            events = acc.Scalars(scalar_tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            runs.append(values)
            steps_list.append(steps)
            run_names.append(run.name)
        except Exception:
            continue
    if len(runs) == 0:
        raise ValueError(f"No runs contained scalar '{scalar_tag}'")
    lengths = [len(r) for r in runs]
    if align == "truncate":
        min_len = min(lengths)
        runs = [r[:min_len] for r in runs]
        steps = steps_list[0][:min_len]
    elif align == "pad":
        max_len = max(lengths)
        padded = []
        for r in runs:
            arr = np.full(max_len, np.nan)
            arr[: len(r)] = r
            padded.append(arr)
        runs = padded
        steps = steps_list[0]
    else:
        raise ValueError("align must be 'truncate' or 'pad'")
    values = np.vstack(runs)
    return values, steps, run_names

