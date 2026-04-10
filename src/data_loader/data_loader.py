"""
Data loader module for trajectory prediction and diffusion models.

This module provides functionality to create PyTorch DataLoaders for training and evaluation
of models that work with trajectory data, including masked trajectory datasets.
"""

import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate

# NEW: import your rewritten dataset
from src.data_loader.dataset import TrajMaskDataset   # <- update to your actual path


def reset_seed_worker_init_fn(worker_id):
    """
    Initialize random seeds for DataLoader workers to ensure reproducible results.

    This function is used as the worker_init_fn parameter in DataLoader to ensure
    that each worker process has a different but deterministic random seed based
    on the initial seed and worker ID.

    Args:
        worker_id (int): The ID of the worker process.
    """
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def safe_collate(batch):
    """
    Custom collate function that safely handles mixed data types in batches.

    This function processes batches where items may contain a mix of PyTorch tensors,
    NumPy arrays, and other data types. It converts NumPy arrays to tensors while
    leaving existing tensors unchanged.

    Args:
        batch (list): List of dictionaries containing data samples.

    Returns:
        dict: Collated batch with tensors properly converted.
    """
    def to_tensor_if_needed(x):
        import numpy as _np
        import torch as _torch
        if isinstance(x, _torch.Tensor):
            return x
        if isinstance(x, _np.ndarray):
            return _torch.from_numpy(x)
        return x

    mapped = []
    for item in batch:
        mapped.append({k: to_tensor_if_needed(v) for k, v in item.items()})
    return default_collate(mapped)


def get_data_loader(cfg, train=True):
    """
    Create a DataLoader for trajectory data with configurable options.

    This function creates a PyTorch DataLoader configured for trajectory prediction
    tasks, supporting both training and evaluation modes with various dataset
    options for trajectory masking and processing.

    Args:
        cfg: Configuration object containing dataset and loader parameters.
        train (bool): Whether to create a training (True) or evaluation (False) loader.

    Returns:
        DataLoader: Configured PyTorch DataLoader instance.
    """
    # choose root folder
    if hasattr(cfg, "train_root") and hasattr(cfg, "val_root"):
        root = cfg.train_root if train else cfg.val_root
    else:
        root = cfg.root

    # --------- NEW: dataset options with safe defaults ---------
    img_size = getattr(cfg, "img_size", 64)  # Image size for trajectory visualization
    n_points = getattr(cfg, "n_points", 128)  # Number of points in trajectory sequences

    # mask style
    traj_mask_mode = getattr(cfg, "traj_mask_mode", "soft")  # "soft" or "hard" masking
    soft_sigma = float(getattr(cfg, "soft_sigma", 0.75))  # Sigma for soft masking
    line_thickness = int(getattr(cfg, "line_thickness", 1))  # Thickness of trajectory lines

    # your trajectory storage convention
    traj_order = getattr(cfg, "traj_order", "yx")            # Coordinate order: "yx" or "xy"
    traj_flip_y = bool(getattr(cfg, "traj_flip_y", False))   # Whether to flip Y coordinates

    use_occ = bool(getattr(cfg, "use_occ", False))  # Include occupancy information
    use_trav = bool(getattr(cfg, "use_trav", False))  # Include traversability information
    # ----------------------------------------------------------

    dataset = TrajMaskDataset(
        root_dir=root,
        img_size=img_size,
        n_points=n_points,
        traj_mask_mode=traj_mask_mode,
        line_thickness=line_thickness,
        soft_sigma=soft_sigma,
        traj_order=traj_order,
        traj_flip_y=traj_flip_y,
        use_occ=use_occ,
        use_trav=use_trav,
    )

    # Configure sampler for distributed training or regular training
    if getattr(cfg, "distributed", False):
        sampler = DistributedSampler(dataset, shuffle=train)
    else:
        sampler = None

    # If sampler is provided, DataLoader.shuffle must be False
    shuffle_flag = (sampler is None and getattr(cfg, "shuffle", train))

    # Create DataLoader with optimized settings
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=getattr(cfg, "batch_size", 32),
        num_workers=getattr(cfg, "num_workers", 4),
        shuffle=shuffle_flag,
        sampler=sampler,
        collate_fn=safe_collate,
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=True,  # Pin memory for faster GPU transfers
        persistent_workers=(getattr(cfg, "num_workers", 0) > 0),  # Keep workers alive between epochs
        drop_last=False,  # Don't drop the last incomplete batch
    )
    return data_loader


def train_data_loader(cfg):
    """
    Create a DataLoader for training data.

    Args:
        cfg: Configuration object containing training parameters.

    Returns:
        DataLoader: DataLoader configured for training.
    """
    cfg_local = copy.deepcopy(cfg)
    return get_data_loader(cfg=cfg_local, train=True)


def evaluation_data_loader(cfg):
    """
    Create a DataLoader for evaluation/validation data.

    Args:
        cfg: Configuration object containing evaluation parameters.

    Returns:
        DataLoader: DataLoader configured for evaluation.
    """
    cfg_local = copy.deepcopy(cfg)
    return get_data_loader(cfg=cfg_local, train=False)


# =============================================================================
# LEGACY CODE - Previous implementation (commented out for reference)
# =============================================================================

# import copy
# import random
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, DistributedSampler
# from torch.utils.data._utils.collate import default_collate

# # swap to your RGB+traj dataset
# from src.data_loader.dataset import TrajDataset   # <-- was TrainData


# def reset_seed_worker_init_fn(worker_id):
#     seed = torch.initial_seed() % (2 ** 32)
#     np.random.seed(seed)
#     random.seed(seed)


# def safe_collate(batch):
#     """
#     Leave torch.Tensors as-is (fast path).
#     Convert numpy arrays to tensors.
#     Everything else: let default collate handle (lists/dicts of tensors).
#     """
    

#     def to_tensor_if_needed(x):
#         import numpy as _np
#         import torch as _torch
#         if isinstance(x, _torch.Tensor):
#             return x
#         if isinstance(x, _np.ndarray):
#             # keep dtype if float/bool/int; model expects float32 generally
#             return _torch.from_numpy(x)
#         return x

#     # map each dict’s values
#     mapped = []
#     for item in batch:
#         mapped.append({k: to_tensor_if_needed(v) for k, v in item.items()})
#     return default_collate(mapped)


# def get_data_loader(cfg, train=True):
#     """
#     If cfg has train_root / val_root, use them.
#     Otherwise fall back to cfg.root.
#     """
#     # choose root folder
#     if hasattr(cfg, "train_root") and hasattr(cfg, "val_root"):
#         root = cfg.train_root if train else cfg.val_root
#     else:
#         root = cfg.root

#     dataset = TrajDataset(root=root, n_points=getattr(cfg, "n_points", 128))

#     # sampler: shuffle only for training
#     if cfg.distributed:
#         sampler = DistributedSampler(dataset, shuffle=train)
#     else:
#         sampler = None

#     # If sampler is provided, DataLoader.shuffle must be False
#     shuffle_flag = (sampler is None and getattr(cfg, "shuffle", train))

#     data_loader = DataLoader(
#         dataset=dataset,
#         batch_size=cfg.batch_size,
#         num_workers=cfg.num_workers,
#         shuffle=shuffle_flag,
#         sampler=sampler,
#         collate_fn=safe_collate,
#         worker_init_fn=reset_seed_worker_init_fn,
#         pin_memory=True,
#         persistent_workers=(cfg.num_workers > 0),
#         drop_last=False,
#     )
#     return data_loader


# def train_data_loader(cfg):
#     cfg_local = copy.deepcopy(cfg)
#     return get_data_loader(cfg=cfg_local, train=True)


# def evaluation_data_loader(cfg):
#     cfg_local = copy.deepcopy(cfg)
#     return get_data_loader(cfg=cfg_local, train=False)

# import copy
# import os
# import pickle
# import random
# import warnings
# import cv2
# from functools import partial
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader, DistributedSampler

# from src.data_loader.dataset import TrainData


# def reset_seed_worker_init_fn(worker_id):
#     r"""Reset seed for data loader worker."""
#     seed = torch.initial_seed() % (2 ** 32)
#     # print(worker_id, seed)
#     np.random.seed(seed)
#     random.seed(seed)


# def registration_collate_fn_stack_mode(data_dicts):
#     r"""Collate function for registration in stack mode.
#     Args:
#         data_dicts (List[Dict])
#     Returns:
#         collated_dict (Dict)
#     """
#     # merge data with the same key from different samples into a list
#     collated_dict = {}
#     for data_dict in data_dicts:
#         for key, value in data_dict.items():
#             value = torch.from_numpy(np.asarray(value)).to(torch.float)
#             if key not in collated_dict:
#                 collated_dict[key] = []
#             collated_dict[key].append(value)
#     for key, value in collated_dict.items():
#         collated_dict[key] = torch.stack(value, dim=0)
#     return collated_dict


# def get_data_loader(cfg, train=True):
#     dataset = TrainData(cfg=cfg, train=train)
#     sampler = DistributedSampler(dataset) if cfg.distributed else None
#     data_loader = DataLoader(
#         dataset=dataset,
#         batch_size=cfg.batch_size,
#         num_workers=cfg.num_workers,
#         shuffle=cfg.shuffle,
#         sampler=sampler,
#         collate_fn=partial(registration_collate_fn_stack_mode),
#         worker_init_fn=reset_seed_worker_init_fn,
#         pin_memory=False,
#         drop_last=False,
#     )
#     return data_loader


# def train_data_loader(cfg):
#     """
#     This function is to create a training dataloader with pytorch interface
#     Args:
#         cfg: The configuration of the dataset
#     Returns:
#         a dataloader in pytorch format
#     """
#     cfgs = copy.deepcopy(cfg)
#     return get_data_loader(cfg=cfgs, train=True)


# def evaluation_data_loader(cfg):
#     """
#     This function is to create a evaluation dataloader with pytorch interface
#     Args:
#         cfg: The configuration of the dataset
#     Returns:
#         a dataloader in pytorch format
#     """
#     cfgs = copy.deepcopy(cfg)
#     return get_data_loader(cfg=cfgs, train=False)