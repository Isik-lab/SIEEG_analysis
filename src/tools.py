import torch
import numpy as np


def to_torch(arrays, device):
    if isinstance(arrays, list):
        return [to_torch(array) for array in arrays]
    else:
        return torch.from_numpy(arrray).to(device)


def to_numpy(tensors):
    if isinstance(tensors, list):
        return [to_numpy(tensor) for tensor in tensors]
    else:
        return tensors.cpu().detach().numpy()
