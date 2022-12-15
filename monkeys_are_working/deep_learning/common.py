from typing import Dict, Any, Union

import torch


def parse_device(device: str):
    import torch

    assert device in {"cpu", "cuda", "gpu", "auto"}
    if device == "gpu":
        device = "cuda"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def transfer_batch_to_device(batch: Union[dict, list, torch.Tensor], device: str):
    device = parse_device(device)
    if isinstance(batch, dict):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
    elif isinstance(batch, list):
        for i_element in range(len(batch)):
            if isinstance(batch[i_element], torch.Tensor):
                batch[i_element] = batch[i_element].to(device)
    elif isinstance(batch, torch.Tensor):
        batch = batch.to(device)
    return batch


def parse_activation_fn(name: str) -> Dict[str, Any]:
    from torch import nn
    import torch.nn.functional as F

    assert name in {"relu", "gelu"}
    if name == "relu":
        module, functional = nn.ReLU(), F.relu
    elif name == "gelu":
        module, functional = nn.GELU(), F.gelu
    else:
        raise NotImplementedError
    return {
        "module": module,
        "functional": functional,
    }
