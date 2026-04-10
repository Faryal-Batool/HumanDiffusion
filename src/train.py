"""
Training module for trajectory generation models.

This module provides the main training infrastructure for diffusion-based
trajectory generation using Denoising Diffusion Probabilistic Models (DDPM).

Diffusion models in this project:
- Generate trajectories as 3-channel pixel masks (start, goal, path)
- Use mask denoising and reconstruction objectives for training

The Trainer class handles:
- Multi-GPU training (single GPU and distributed)
- Mixed precision training
- TensorBoard logging
- Model checkpointing
- Learning rate scheduling
- Periodic evaluation
"""

import time
import os
from warnings import warn
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import os.path as osp
from datetime import timedelta

from src.utils.configs import (
    TrainingConfig,
    ScheduleMethods,
    LossNames,
    LogNames,
    LogTypes,
    DataDict,
    GeneratorType,
)
from src.loss import Loss
from src.models.model import get_model
from src.utils.functions import to_device, get_device, release_cuda
from src.data_loader.data_loader import train_data_loader, evaluation_data_loader


class Trainer:
    """
    Main training driver for trajectory generation models.

    Supports diffusion-based training with unified distributed, logging,
    checkpointing, and evaluation support.

    For diffusion models (GeneratorType.diffusion):
    - Dataset provides traversability maps, ground-truth masks, start/goal coordinates
    - Model performs DDPM over pixel-space masks
    - Loss is mask-based reconstruction
    """

    def __init__(self, cfgs: TrainingConfig):
        """
        Initialize the trainer with configuration.

        Sets up model, optimizer, scheduler, data loaders, and distributed training
        infrastructure based on the provided configuration.

        Args:
            cfgs: Training configuration containing all hyperparameters and settings
        """
        # Training configuration
        self.name = cfgs.name
        self.max_epoch = cfgs.max_epoch
        self.evaluation_freq = cfgs.evaluation_freq
        self.train_time_steps = cfgs.train_time_steps  # Multiple updates per batch (diffusion)

        # Training state
        self.iteration = 0
        self.epoch = 0
        self.training = False
        self.global_step = 0

        # Device and distributed training setup
        self._setup_device_and_distribution(cfgs)

        # Model initialization
        self._setup_model(cfgs)

        # Logging setup (TensorBoard)
        self._setup_logging(cfgs)

        # Optimizer and learning rate scheduler
        self._setup_optimizer_and_scheduler(cfgs)

        # Loss function
        self._setup_loss(cfgs)

        # Data loaders
        self._setup_data_loaders(cfgs)

        # Diffusion-specific parameters
        self.use_traversability = cfgs.loss.use_traversability
        self.generator_type = cfgs.model.generator_type
        self.time_step_number = cfgs.model.diffusion.traversable_steps

    def _setup_device_and_distribution(self, cfgs):
        """Setup device and distributed training configuration."""
        if cfgs.gpus.device == "cuda":
            self.device = "cuda"
        else:
            self.device = get_device(device=cfgs.gpus.device)

        # Check for distributed training environment
        if 'WORLD_SIZE' in os.environ and cfgs.gpus.device == "cuda":
            world_size = int(os.environ['WORLD_SIZE'])
            print(f"World size: {world_size}")
            self.distributed = cfgs.data.distributed = world_size >= 1
        else:
            print("World size: 0")
            self.distributed = cfgs.data.distributed = False

    def _setup_model(self, cfgs):
        """Initialize model and load snapshot if provided."""
        self.model = get_model(config=cfgs.model, device=self.device)
        self.snapshot = cfgs.snapshot

        if self.snapshot:
            state_dict = self.load_snapshot(self.snapshot)

        self.current_rank = 0
        if self.device != torch.device("cpu"):
            self._set_model_gpus(cfgs.gpus)

    def _setup_logging(self, cfgs):
        """Setup TensorBoard logging."""
        self.output_dir = cfgs.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        log_dir = os.path.join(self.output_dir, "tensorboard")
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        print(f"[INFO] TensorBoard logging to: {log_dir}")

    def _setup_optimizer_and_scheduler(self, cfgs):
        """Setup optimizer and learning rate scheduler."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfgs.lr,
            weight_decay=cfgs.weight_decay
        )

        self.scheduler_type = cfgs.scheduler
        if self.scheduler_type == ScheduleMethods.step:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, cfgs.lr_decay_steps, gamma=cfgs.lr_decay
            )
        elif self.scheduler_type == ScheduleMethods.cosine:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                eta_min=cfgs.lr_min,
                T_0=cfgs.lr_t0,
                T_mult=cfgs.lr_tm
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")

        if self.snapshot and not cfgs.only_model:
            self.load_learning_parameters(self.snapshot)

    def _setup_loss(self, cfgs):
        """Setup loss function."""
        if self.device == "cuda":
            self.loss_func = Loss(cfg=cfgs.loss).cuda()
        else:
            self.loss_func = Loss(cfg=cfgs.loss).to(self.device)

    def _setup_data_loaders(self, cfgs):
        """Setup training and evaluation data loaders."""
        self.training_data_loader = train_data_loader(cfg=cfgs.data)
        self.evaluation_data_loader = evaluation_data_loader(cfg=cfgs.data)

    def _set_model_gpus(self, cfg):
        """
        Setup GPU configuration for single or distributed training.

        Handles:
        - Distributed training initialization (NCCL backend)
        - Device placement and memory format optimization
        - Batch normalization synchronization
        - DistributedDataParallel wrapping
        """
        if self.distributed:
            # Initialize distributed training
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
            print(f"OS world size: {world_size}, local_rank: {local_rank}, rank: {rank}")

            torch.cuda.set_device(cfg.local_rank)
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                timeout=timedelta(seconds=5000)
            )
            world_size = dist.get_world_size()
            self.current_rank = dist.get_rank()
            print(
                f'Training in distributed mode with multiple processes, 1 GPU per process. '
                f'Process {self.current_rank}, total {world_size}.'
            )
            dist.barrier()
        else:
            print('Training with a single process on 1 GPU.')

        assert self.current_rank >= 0, "Rank must be >= 0"

        # Move model to GPU(s)
        if self.distributed:
            self.model.cuda()
        else:
            self.model.to(self.device)

        # Memory format optimization for convolutional networks
        if cfg.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        # Setup synchronized batch normalization for distributed training
        if self.distributed and cfg.sync_bn:
            assert not cfg.split_bn
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if cfg.local_rank == 0:
                print(
                    'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                    'zero initialized BN layers while sync-bn enabled.'
                )

        # Wrap model with DistributedDataParallel
        if self.distributed:
            if cfg.local_rank == 0:
                print("Using native Torch DistributedDataParallel.")
            self.model = DDP(
                self.model,
                device_ids=[cfg.local_rank],
                broadcast_buffers=not cfg.no_ddp_bb,
                find_unused_parameters=True  # Allow different parameter usage across ranks
            )

    def load_snapshot(self, snapshot):
        print('Loading from "{}".'.format(snapshot))
        state_dict = torch.load(snapshot, map_location=torch.device(self.device))

        # Load model
        model_dict = state_dict['state_dict']
        self.model.load_state_dict(model_dict, strict=False)

        snapshot_keys = set(model_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        missing_keys = model_keys - snapshot_keys
        unexpected_keys = snapshot_keys - model_keys
        if len(missing_keys) > 0:
            warn('Missing keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            warn('Unexpected keys: {}'.format(unexpected_keys))
        print('Model has been loaded.')
        return state_dict

    def load_learning_parameters(self, state_dict):
        """
        Load training state from checkpoint (epoch, iteration, optimizer, scheduler).

        Args:
            state_dict: Checkpoint dictionary containing training state
        """
        if 'epoch' in state_dict:
            self.epoch = state_dict['epoch']
            print(f'Epoch has been loaded: {self.epoch}.')
        if 'iteration' in state_dict:
            self.iteration = state_dict['iteration']
            print(f'Iteration has been loaded: {self.iteration}.')
        if 'optimizer' in state_dict and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state_dict['optimizer'])
                print('Optimizer has been loaded.')
            except Exception as e:
                print(f"Failed to load optimizer: {e}")
        if 'scheduler' in state_dict and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(state_dict['scheduler'])
                print('Scheduler has been loaded.')
            except Exception as e:
                print(f"Failed to load scheduler: {e}")

    def save_snapshot(self, filename):
        """
        Save model checkpoint to file.

        Saves both model weights and training state (epoch, iteration, optimizer, scheduler).
        Only rank 0 saves in distributed training to avoid conflicts.

        Args:
            filename: Path to save checkpoint
        """
        if self.distributed:
            model_state_dict = self.model.module.state_dict()  # Unwrap DDP
        else:
            model_state_dict = self.model.state_dict()

        # Basic checkpoint with model weights
        state_dict = {'state_dict': model_state_dict}
        torch.save(state_dict, filename)

        # Extended checkpoint with training state (only on rank 0)
        if not self.distributed or (self.distributed and self.current_rank == 0):
            state_dict['epoch'] = self.epoch
            state_dict['iteration'] = self.iteration
            snapshot_filename = osp.join(self.output_dir, f"{self.name}_snapshot.pth.tar")
            state_dict['optimizer'] = self.optimizer.state_dict()
            if self.scheduler is not None:
                state_dict['scheduler'] = self.scheduler.state_dict()
            torch.save(state_dict, snapshot_filename)

    def cleanup(self):
        """Cleanup training resources (TensorBoard writer, distributed processes)."""
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.distributed:
            dist.destroy_process_group()

    def set_train_mode(self):
        """Set model to training mode with gradients enabled."""
        self.training = True
        self.model.train()
        torch.set_grad_enabled(True)

    def set_eval_mode(self):
        """Set model to evaluation mode with gradients disabled."""
        self.training = False
        self.model.eval()
        torch.set_grad_enabled(False)

    def optimizer_step(self):
        """Perform optimizer step and reset gradients."""
        self.optimizer.step()
        self.optimizer.zero_grad()

    def step(self, data_dict, train=True) -> dict:
        """
        Execute one training or evaluation step.

        Handles diffusion training with appropriate data preprocessing and loss
        computation.

        For diffusion models:
        - Input: traversability maps, ground-truth masks, start/goal coordinates
        - Process: DDPM training on pixel-space masks
        - Loss: mask denoising reconstruction

        Args:
            data_dict: Batch of training/evaluation data
            train: Whether to perform training (True) or evaluation (False)

        Returns:
            Dictionary containing model outputs and computed losses
        """
        # Move data to appropriate device
        data_dict = to_device(data_dict, device=self.device)

        # Legacy compatibility: map "rgb" → DataDict.camera if present
        if "rgb" in data_dict:
            data_dict[DataDict.camera] = data_dict["rgb"]

        # For backwards compatibility: use "trav" also as DataDict.local_map
        if "trav" in data_dict:
            trav = data_dict["trav"]  # (B,1,H,W)
            data_dict[DataDict.local_map] = trav  # continuous [0,1] traversability

        # Extract ground-truth trajectory (may be None for diffusion)
        gt_path = data_dict.get(DataDict.path, None)

        if train:
            # ==================== TRAINING MODE ====================
            output_dict = self.model(data_dict, sample=False)

            # Diffusion branch: mask-based DDPM training
            # Pass ground-truth mask and occupancy map to loss function
            if "mask_gt" in data_dict:
                output_dict["mask_gt"] = data_dict["mask_gt"]  # (B,3,H,W)
            if "occ_map" in data_dict:
                output_dict["occ_map"] = data_dict["occ_map"]  # (B,1,H,W)

            # Compute loss using diffusion mask loss
            loss_dict = self.loss_func(output_dict)
            output_dict.update(loss_dict)

        else:
            # ====================== EVALUATION MODE ======================
            output_dict = self.model(data_dict, sample=True)

            # Diffusion evaluation: attach ground-truth for evaluation metrics
            if "mask_gt" in data_dict:
                output_dict["mask_gt"] = data_dict["mask_gt"]
            if "occ_map" in data_dict:
                output_dict["occ_map"] = data_dict["occ_map"]

            # Evaluate without gradients (mask-based for diffusion)
            eval_dict = self.loss_func.evaluate(output_dict)
            if eval_dict:
                output_dict.update(eval_dict)

        # For diffusion, gt_path is not essential but kept for compatibility/visualization
        if gt_path is not None:
            output_dict["gt_path"] = gt_path

        return output_dict

    def update_log(self, results, timestep=None, log_name=None):
        """
        Log scalar metrics to TensorBoard.

        Args:
            results: Dictionary of metric names -> values
            timestep: Optional step time for performance monitoring
            log_name: Log category ("train", "evaluation", etc.)
        """
        if self.tb_writer is None:
            return

        step = self.iteration

        if timestep is not None:
            self.tb_writer.add_scalar(f"{log_name}/step_time", timestep, step)

        prefix = "" if log_name is None else f"{log_name}/"

        for key, value in results.items():
            try:
                scalar = value.item() if hasattr(value, "item") else float(value)
            except Exception:
                continue  # Skip non-scalar values

            self.tb_writer.add_scalar(prefix + key, scalar, step)

    def run_epoch(self):
        """
        Run one complete training epoch over the training dataset.

        For diffusion models, performs multiple gradient updates per batch
        (train_time_steps) to increase training stability and sample efficiency.

        Process:
        1. Iterate through training data loader
        2. For each batch, perform train_time_steps forward/backward passes
        3. Log losses and timing to TensorBoard
        4. Step learning rate scheduler at epoch end
        """
        self.optimizer.zero_grad()

        last_time = time.time()
        for iteration, data_dict in enumerate(
                tqdm(self.training_data_loader, desc=f"Training Epoch {self.epoch}")):
            self.iteration += 1

            # Set timestep limit for diffusion traversability sampling
            # This bounds the noise levels used during training
            data_dict[DataDict.traversable_step] = self.time_step_number

            # Multiple updates per batch (diffusion-specific)
            for step_iteration in range(self.train_time_steps):
                output_dict = self.step(data_dict=data_dict, train=True)
                torch.cuda.empty_cache()

                # Backward pass and optimization
                output_dict[LossNames.loss].backward()
                self.optimizer_step()

                optimize_time = time.time()

                # Log training metrics to TensorBoard
                output_dict = release_cuda(output_dict)
                self.update_log(
                    results=output_dict,
                    timestep=optimize_time - last_time,
                    log_name=LogTypes.train
                )
                last_time = time.time()

        # Step learning rate scheduler at end of epoch
        self.scheduler.step()

        if not self.distributed or (self.distributed and self.current_rank == 0):
            os.makedirs('{}/models'.format(self.output_dir), exist_ok=True)
            self.save_snapshot('{}/models/{}_{}.pth'.format(self.output_dir, self.name, self.epoch))

    def inference_epoch(self):
        """
        Run periodic evaluation epoch over the evaluation dataset.

        Only executed every evaluation_freq epochs. Computes evaluation metrics
        without gradient computation. For diffusion models, evaluates mask-based
        trajectory generation quality.

        Logs per-batch metrics and computes mean epoch loss for monitoring.
        """
        if (self.evaluation_freq > 0) and (self.epoch % self.evaluation_freq == 0) and (self.epoch != 0):
            for iteration, data_dict in enumerate(
                    tqdm(self.evaluation_data_loader,
                         desc=f"Evaluation Losses Epoch {self.epoch}")):
                sum_loss = 0.0
                count = 0

                start_time = time.time()
                output_dict = self.step(data_dict, train=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_time = time.time()

                # Accumulate loss for mean computation
                if "loss" in output_dict:
                    sum_loss += output_dict["loss"].item()
                    count += 1

                # Log evaluation metrics
                output_dict = release_cuda(output_dict)
                torch.cuda.empty_cache()
                self.update_log(
                    results=output_dict,
                    timestep=step_time - start_time,
                    log_name=LogTypes.others
                )

            # Log mean evaluation loss for the epoch
            if count > 0:
                mean_eval_loss = sum_loss / count
                self.tb_writer.add_scalar("evaluation/mean_loss", mean_eval_loss, self.epoch)

    def run(self):
        """
        Execute the complete training loop.

        Training process:
        1. For each epoch from current_epoch to max_epoch:
           a. Run evaluation (if enabled and at correct frequency)
           b. Run training epoch
           c. Save model checkpoint
        2. Cleanup resources

        Enables anomaly detection for debugging and handles distributed training
        sampler epoch setting.
        """
        # Enable anomaly detection for debugging NaN/inf issues
        torch.autograd.set_detect_anomaly(True)

        for self.epoch in range(self.epoch, self.max_epoch, 1):
            # ---- Evaluate BEFORE training each epoch ----
            self.set_eval_mode()
            self.inference_epoch()

            # ---- Training epoch ----
            self.set_train_mode()

            # Set epoch for distributed samplers (important for shuffling)
            if self.distributed:
                self.training_data_loader.sampler.set_epoch(self.epoch)
                if self.evaluation_freq > 0:
                    self.evaluation_data_loader.sampler.set_epoch(self.epoch)

            self.run_epoch()

            # ---- Save checkpoint at end of epoch ----
            if not self.distributed or (self.distributed and self.current_rank == 0):
                os.makedirs(f'{self.output_dir}/models', exist_ok=True)
                self.save_snapshot(f'{self.output_dir}/models/{self.name}_{self.epoch}.pth')

        # Cleanup resources
        self.cleanup()
