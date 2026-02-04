import torch
import numpy as np


def dict_to_tensor(train_dict, test_dict, keys, 
                   dtype=torch.float32,
                   device='cpu'):
    def list_to_tensor(l):
        tensors = []
        for item in l:
            if torch.is_tensor(item):
                tensors.append(item)
            elif isinstance(item, np.ndarray):
                tensors.append(torch.from_numpy(item).type(dtype))
            else:
                raise TypeError(f"Unsupported type in list_to_tensor: {type(item)}")
        return torch.hstack(tuple(tensors))

    train_out, test_out, groups = [], [], []
    for i_group, key in enumerate(keys):
        train_val = train_dict[key]
        test_val = test_dict[key]
        # Convert numpy arrays to torch tensors if needed
        if isinstance(train_val, np.ndarray):
            train_val = torch.from_numpy(train_val).type(dtype).to(device)
        if isinstance(test_val, np.ndarray):
            test_val = torch.from_numpy(test_val).type(dtype).to(device)

        if train_val.ndim > 1:
            train_out.append(train_val)
            test_out.append(test_val)
            if torch.is_tensor(test_val):
                n_cols = test_val.size(1)
            else:
                n_cols = test_val.shape[1]
            group_vec = torch.ones(n_cols, dtype=torch.long) * i_group
        else:
            train_out.append(torch.unsqueeze(train_val, 1).to(device))
            test_out.append(torch.unsqueeze(test_val, 1).to(device))
            group_vec = torch.tensor([i_group], dtype=torch.long).to(device)
        groups.append(group_vec)
    return list_to_tensor(train_out), list_to_tensor(test_out), list_to_tensor(groups)


def to_torch(arrays, device, dtype=torch.float32):
    if isinstance(arrays, list):
        return [to_torch(array, device, dtype=dtype) for array in arrays]
    else:
        return torch.from_numpy(arrays).type(dtype).to(device)

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
