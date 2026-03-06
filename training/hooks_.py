from typing import Dict, List, Optional, Any, Callable
import torch
import torch.nn as nn
from dataclasses import dataclass
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass
class HookConfig:
    forward: bool = True
    backward: bool = False
    log_input: bool = False
    log_output: bool = True
    log_grad_input: bool = False
    log_grad_output: bool = True
    log_shape: bool = True
    log_stats: bool = True  # mean/std/min/max
    prefix: str = "layer"


class LayerTracker:
    """
    Attaches hooks to selected layers and collects/logs information.

    Usage examples:

    # 1. With named modules (recommended)
    tracker = LayerTracker({
        "conv1": model.conv1,
        "block2.1.attn": model.block2[1].attn,
        "fc": model.fc
    })

    # 2. With just a list (uses index as name)
    tracker = LayerTracker([model.conv1, model.conv2, model.fc])

    # 3. More control
    tracker = LayerTracker(model.named_modules(), only_layers=["conv", "linear"])
    """

    def __init__(
            self,
            modules: Dict[str, nn.Module] | List[nn.Module] | Any,  # dict, list, or named_modules()
            config: Optional[HookConfig] = None,
            only_layers: Optional[List[str]] = None,  # substring match when using named_modules
            log_level: int = logging.INFO
    ):
        self.hooks: Dict[str, list] = {}  # name → list of hook handles
        self.activations: Dict[str, Any] = {}  # forward activations
        self.gradients: Dict[str, Any] = {}  # backward gradients
        self.config = config or HookConfig()
        self.log_level = log_level

        # Normalize input to dict[str, module]
        if isinstance(modules, dict):
            self.tracked_modules = modules
        elif isinstance(modules, list):
            self.tracked_modules = {f"layer_{i}": m for i, m in enumerate(modules)}
        elif hasattr(modules, 'named_modules'):  # nn.Module
            self.tracked_modules = {}
            for name, mod in modules.named_modules():
                if only_layers is None or any(pat in name for pat in only_layers):
                    if mod is not modules:  # skip the root module
                        self.tracked_modules[name] = mod
        else:
            raise ValueError("modules should be dict, list or nn.Module with named_modules()")

        logger.setLevel(log_level)
        self._attach_hooks()

    def _attach_hooks(self):
        for name, module in self.tracked_modules.items():
            handles = []

            if self.config.forward:
                handles.append(
                    module.register_forward_hook(
                        self._make_forward_hook(name, self.config)
                    )
                )

            if self.config.backward:
                handles.append(
                    module.register_full_backward_hook(
                        self._make_backward_hook(name, self.config)
                    )
                )

            self.hooks[name] = handles

    def _make_forward_hook(self, name: str, cfg: HookConfig):
        def hook_fn(module, input, output):
            if cfg.log_input:
                self._log_input(name, input)
            if cfg.log_output:
                self._log_output(name, output)

            # Store last value
            if cfg.log_output:
                self.activations[name] = output.detach() if isinstance(output, torch.Tensor) else output
            elif cfg.log_input and input:
                self.activations[name] = input[0].detach() if isinstance(input[0], torch.Tensor) else input

        return hook_fn

    def _make_backward_hook(self, name: str, cfg: HookConfig):
        def hook_fn(module, grad_input, grad_output):
            if cfg.log_grad_input and grad_input:
                self._log_grad_input(name, grad_input)
            if cfg.log_grad_output and grad_output:
                self._log_grad_output(name, grad_output)

            # Store last gradients
            if cfg.log_grad_output and grad_output:
                self.gradients[name] = grad_output[0].detach() if isinstance(grad_output[0],
                                                                             torch.Tensor) else grad_output

        return hook_fn

    def _log_tensor_stats(self, name: str, tensor: torch.Tensor, tag: str):
        if not isinstance(tensor, torch.Tensor):
            return
        stats = f"shape={list(tensor.shape):<20} "
        if tensor.numel() > 0:
            stats += f"mean={tensor.mean().item():.4e} std={tensor.std().item():.4e} "
            stats += f"min={tensor.min().item():.4e} max={tensor.max().item():.4e}"
        logger.log(self.log_level, f"{tag} {name:30} {stats}")

    def _log_input(self, name, input_tuple):
        if not input_tuple:
            return
        for i, inp in enumerate(input_tuple):
            if isinstance(inp, torch.Tensor):
                self._log_tensor_stats(name, inp, f"INPUT[{i}] ")

    def _log_output(self, name, output):
        if isinstance(output, tuple):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    self._log_tensor_stats(name, out, f"OUT[{i}] ")
        elif isinstance(output, torch.Tensor):
            self._log_tensor_stats(name, output, "OUT ")

    def _log_grad_input(self, name, grad_input):
        for i, g in enumerate(grad_input):
            if isinstance(g, torch.Tensor):
                self._log_tensor_stats(name, g, f"GRAD_IN[{i}] ")

    def _log_grad_output(self, name, grad_output):
        for i, g in enumerate(grad_output):
            if isinstance(g, torch.Tensor):
                self._log_tensor_stats(name, g, f"GRAD_OUT[{i}] ")

    def clear(self):
        """Remove all hooks"""
        for handles in self.hooks.values():
            for h in handles:
                h.remove()
        self.hooks.clear()
        self.activations.clear()
        self.gradients.clear()

    @contextmanager
    def enable(self):
        """Context manager to temporarily enable logging"""
        prev_level = logger.level
        logger.setLevel(self.log_level)
        try:
            yield
        finally:
            logger.setLevel(prev_level)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()

    def summary(self):
        print("Tracked layers:")
        for name, mod in self.tracked_modules.items():
            print(f"  {name:30} {mod.__class__.__name__}")


class SimpleHook:

    def __init__(self, total=20):
        self.total = total
        self.res = []
        self.x = []
        self.count=0

    def store(self, res, x, mask):
        while self.count <self.total:
            self.res.append(torch.masked_select(res, mask).detach().cpu().numpy())
            self.x.append(torch.masked_select(x, mask).detach().cpu().numpy())
            self.count += 1

    def log(self, logger, name):

        self.res = []
        self.x = []
        self.count = 0
