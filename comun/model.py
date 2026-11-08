from __future__ import annotations

import torch
import torchxrayvision as xrv


def flush_denormals(model: torch.nn.Module) -> int:
    tiny = torch.finfo(torch.float32).tiny
    flushed = 0
    with torch.no_grad():
        for tensor in (*model.parameters(), *model.buffers()):
            if tensor.is_floating_point():
                mask = (tensor != 0) & (tensor.abs() < tiny)
                flushed += int(mask.sum())
                tensor[mask] = 0
    return flushed


def load_model(weights: str, keep_denormals: bool = False) -> tuple[torch.nn.Module, int]:
    model = xrv.models.DenseNet(weights=weights).eval()
    flushed = 0 if keep_denormals else flush_denormals(model)
    return model, flushed
