import torch
import numpy as np


def to_torch(arrays, device):
    if isinstance(arrays, list):
        return [to_torch(array, device) for array in arrays]
    else:
        return torch.from_numpy(arrays).type(torch.DoubleTensor).to(device)


def to_numpy(tensors):
    if isinstance(tensors, list):
        return [to_numpy(tensor) for tensor in tensors]
    else:
        return tensors.cpu().detach().numpy()
    

def camera_switcher(hemi, view):
    if view == 'lateral':
        if hemi == 'lh':
            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1.5, y=0, z=0)
            )
        else:
            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=0, z=0)
            )
    elif view == 'medial':
        if hemi == 'lh':
            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.75, y=0, z=0)
            )
        else:
            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1.75, y=0, z=0)
            )
    elif view == 'ventral':
        if hemi == 'lh':
            camera = dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=0, z=-2.5)
            )
        else:
            camera = dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=0, z=-2.5)
            )
    else:
        raise 'invalid view'
    return camera
