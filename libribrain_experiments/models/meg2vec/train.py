"""
Meg2Vec Pre-training Script

This script implements the pre-training pipeline for Meg2Vec using contrastive learning,
similar to Wav2Vec2's self-supervised training approach.

The training uses:
- Masking strategy: Random time spans are masked in the feature sequence
- Contrastive learning: Model learns to predict masked features using context
- Vector quantization: Discrete latent representations for target features
- Negative sampling: Distractors for contrastive loss computation

Usage:
    python train.py --config_path configs/meg2vec_pretrain.yaml
"""

import os
import sys
import argparse
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

import wandb
from tqdm import tqdm

# Import our model
from model import (
    Meg2VecForPreTraining, 
    _compute_mask_indices, 
    _sample_negative_indices,
    Meg2VecForPreTrainingOutput
)

# Import LibriBrain dataset
try:
    from pnpl.datasets import LibriBrainSpeech
    from pnpl.datasets.libribrain2025.constants import RUN_KEYS
    HAS_LIBRIBRAIN = True
except ImportError:
    HAS_LIBRIBRAIN = False
    logging.warning("LibriBrain dataset not available. Install with: pip install pnpl")


@dataclass
class PreTrainingConfig:
    """Configuration for pre-training Meg2Vec"""
    
    # Model architecture
    sequence_length: int = 500
    in_channels: int = 306
    encoder_out_channels: int = 256
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    feature_projection_dropout: float = 0.1
    
    # Quantization parameters
    num_codevector_groups: int = 2
    num_codevectors_per_group: int = 320
    codevector_dim: int = 256
    proj_codevector_dim: int = 256
    feat_quantizer_dropout: float = 0.1
    
    # Contrastive learning parameters
    contrastive_logits_temperature: float = 0.1
    num_negatives: int = 100
    diversity_loss_weight: float = 0.1
    
    # Masking parameters
    mask_time_prob: float = 0.065
    mask_time_length: int = 10
    mask_time_min_masks: int = 2
    
    # Training parameters
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 32000
    max_steps: int = 400000
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Gumbel temperature scheduling
    gumbel_temperature_decay: float = 0.999995
    min_gumbel_temperature: float = 0.1
    max_gumbel_temperature: float = 2.0
    
    # Logging and saving
    log_interval: int = 100
    save_interval: int = 5000
    eval_interval: int = 1000
    num_workers: int = 4
    
    # Data source configuration
    use_libribrain: bool = True  # Use LibriBrain dataset vs custom numpy files
    data_dir: str = "./data"  # For LibriBrain this is the base path
    output_dir: str = "outputs/meg2vec_pretrain"
    resume_from_checkpoint: Optional[str] = None
    
    # LibriBrain-specific parameters
    libribrain_partition_train: str = "train"
    libribrain_partition_eval: str = "validation"
    libribrain_tmin: float = 0.0
    libribrain_tmax: float = 2.0
    libribrain_standardize: bool = True
    libribrain_download: bool = True
    libribrain_preload_files: bool = True
    libribrain_include_run_keys_train: Optional[List[List[str]]] = None
    libribrain_include_run_keys_eval: Optional[List[List[str]]] = None
    libribrain_exclude_run_keys_train: Optional[List[List[str]]] = None
    libribrain_exclude_run_keys_eval: Optional[List[List[str]]] = None
    
    # Fallback for numpy files
    numpy_data_dir: str = "data/meg_pretraining"
    
    # Experiment tracking
    use_wandb: bool = True
    wandb_project: str = "meg2vec-pretraining"
    wandb_name: Optional[str] = None


class LibriBrainPreTrainingDataset(Dataset):
    """
    Dataset adapter for LibriBrain data for Meg2Vec pre-training.
    
    This class wraps the LibriBrainSpeech dataset and adapts it for self-supervised
    pre-training by providing only the MEG data without labels (since we don't need
    supervised labels for pre-training).
    """
    
    def __init__(
        self, 
        data_path: str,
        sequence_length: int = 500,
        partition: str = "train",
        tmin: float = 0.0,
        tmax: float = 2.0,  # 2 seconds at 250Hz = 500 samples
        include_run_keys: Optional[List[Tuple[str, str, str, str]]] = None,
        exclude_run_keys: Optional[List[Tuple[str, str, str, str]]] = None,
        standardize: bool = True,
        download: bool = True,
        preload_files: bool = True,
    ):
        if not HAS_LIBRIBRAIN:
            raise ImportError("LibriBrain dataset not available. Install with: pip install pnpl")
        
        self.sequence_length = sequence_length
        self.data_path = data_path
        
        # Calculate tmax based on desired sequence length and sampling rate (250 Hz)
        # sequence_length samples at 250 Hz = sequence_length / 250 seconds
        self.tmax = sequence_length / 250.0
        self.tmin = tmin
        
        # Initialize LibriBrain dataset
        if include_run_keys is None and partition:
            # Use partition shortcuts
            self.dataset = LibriBrainSpeech(
                data_path=data_path,
                partition=partition,
                tmin=self.tmin,
                tmax=self.tmax,
                standardize=standardize,
                download=download,
                preload_files=preload_files,
            )
        else:
            # Use custom run keys
            self.dataset = LibriBrainSpeech(
                data_path=data_path,
                tmin=self.tmin,
                tmax=self.tmax,
                include_run_keys=include_run_keys or [],
                exclude_run_keys=exclude_run_keys or [],
                standardize=standardize,
                download=download,
                preload_files=preload_files,
            )
        
        logging.info(f"Loaded LibriBrain dataset with {len(self.dataset)} samples")
        logging.info(f"Time window: {self.tmin:.2f}s to {self.tmax:.2f}s ({self.sequence_length} samples)")
        
        # Store channel statistics for potential use
        self.channel_means = self.dataset.channel_means
        self.channel_stds = self.dataset.channel_stds
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get MEG data and labels from LibriBrain dataset
        meg_data, labels = self.dataset[idx]  # meg_data: (306, seq_len), labels: (seq_len,)
        
        # For pre-training, we only need the MEG data
        # meg_data is already in the format (channels, time) which is what we expect
        # Shape: (306, sequence_length)
        
        # Ensure correct sequence length
        if meg_data.shape[1] != self.sequence_length:
            if meg_data.shape[1] > self.sequence_length:
                # Random crop
                start_idx = np.random.randint(0, meg_data.shape[1] - self.sequence_length + 1)
                meg_data = meg_data[:, start_idx:start_idx + self.sequence_length]
            else:
                # Pad with zeros
                padding = self.sequence_length - meg_data.shape[1]
                meg_data = np.pad(meg_data, ((0, 0), (0, padding)), mode='constant', constant_values=0)
        
        # Convert to tensor if it's not already
        if not isinstance(meg_data, torch.Tensor):
            meg_data = torch.from_numpy(meg_data).float()
        
        # Create attention mask (1 for real data, 0 for padding)
        attention_mask = torch.ones(self.sequence_length, dtype=torch.long)
        if meg_data.shape[1] < self.sequence_length:
            attention_mask[meg_data.shape[1]:] = 0
            
        return {
            'input_values': meg_data,
            'attention_mask': attention_mask,
        }


class MEGPreTrainingDataset(Dataset):
    """
    Fallback dataset for MEG pre-training data (numpy files).
    
    Expected data format:
    - Each sample is a numpy array of shape (306, sequence_length)
    - Data should be preprocessed (filtered, normalized, etc.)
    """
    
    def __init__(self, data_dir: str, sequence_length: int = 500, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.split = split
        
        # Load file list
        split_file = self.data_dir / f"{split}.txt"
        if split_file.exists():
            with open(split_file, 'r') as f:
                self.file_list = [line.strip() for line in f.readlines()]
        else:
            # Fallback: use all .npy files in the directory
            pattern = "*.npy"
            self.file_list = list(self.data_dir.glob(pattern))
            
        logging.info(f"Found {len(self.file_list)} files for {split} split")
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Load MEG data
        if isinstance(self.file_list[idx], str):
            file_path = self.data_dir / self.file_list[idx]
        else:
            file_path = self.file_list[idx]
            
        try:
            data = np.load(file_path)  # Shape: (306, time_steps)
        except Exception as e:
            logging.warning(f"Error loading {file_path}: {e}")
            # Return zeros as fallback
            data = np.zeros((306, self.sequence_length), dtype=np.float32)
        
        # Ensure correct shape
        if data.shape[0] != 306:
            logging.warning(f"Unexpected channel count in {file_path}: {data.shape[0]}")
            data = np.zeros((306, self.sequence_length), dtype=np.float32)
            
        # Trim or pad to target sequence length
        if data.shape[1] > self.sequence_length:
            # Random crop
            start_idx = np.random.randint(0, data.shape[1] - self.sequence_length + 1)
            data = data[:, start_idx:start_idx + self.sequence_length]
        elif data.shape[1] < self.sequence_length:
            # Pad with zeros
            padding = self.sequence_length - data.shape[1]
            data = np.pad(data, ((0, 0), (0, padding)), mode='constant', constant_values=0)
        
        # Convert to tensor
        data = torch.from_numpy(data).float()
        
        # Create attention mask (1 for real data, 0 for padding)
        attention_mask = torch.ones(self.sequence_length, dtype=torch.long)
        if data.shape[1] < self.sequence_length:
            attention_mask[data.shape[1]:] = 0
            
        return {
            'input_values': data,
            'attention_mask': attention_mask,
        }


class Trainer:
    """Training loop for Meg2Vec pre-training"""
    
    def __init__(self, config: PreTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self._setup_logging()
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Setup data
        self.train_dataloader = self._create_dataloader('train')
        # For LibriBrain, validation split is always available
        # For numpy files, check if eval.txt exists
        if self.config.use_libribrain:
            self.eval_dataloader = self._create_dataloader('eval')
        else:
            self.eval_dataloader = self._create_dataloader('eval') if os.path.exists(
                os.path.join(self.config.numpy_data_dir, 'eval.txt')
            ) else None
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Resume from checkpoint if specified
        if config.resume_from_checkpoint:
            self._load_checkpoint(config.resume_from_checkpoint)
            
        # Setup experiment tracking
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_name,
                config=config.__dict__,
            )
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create output directory first
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.output_dir, 'train.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _create_model(self) -> Meg2VecForPreTraining:
        """Create Meg2Vec model for pre-training"""
        model = Meg2VecForPreTraining(
            sequence_length=self.config.sequence_length,
            in_channels=self.config.in_channels,
            encoder_out_channels=self.config.encoder_out_channels,
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            num_hidden_layers=self.config.num_hidden_layers,
            intermediate_size=self.config.intermediate_size,
            hidden_dropout=self.config.hidden_dropout,
            attention_dropout=self.config.attention_dropout,
            activation_dropout=self.config.activation_dropout,
            feature_projection_dropout=self.config.feature_projection_dropout,
            num_codevector_groups=self.config.num_codevector_groups,
            num_codevectors_per_group=self.config.num_codevectors_per_group,
            codevector_dim=self.config.codevector_dim,
            proj_codevector_dim=self.config.proj_codevector_dim,
            feat_quantizer_dropout=self.config.feat_quantizer_dropout,
            contrastive_logits_temperature=self.config.contrastive_logits_temperature,
            num_negatives=self.config.num_negatives,
            diversity_loss_weight=self.config.diversity_loss_weight,
        )
        
        logging.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        return model
    
    def _create_dataloader(self, split: str) -> DataLoader:
        """Create data loader for given split"""
        if self.config.use_libribrain:
            # Use LibriBrain dataset
            partition = (
                self.config.libribrain_partition_train 
                if split == 'train' 
                else self.config.libribrain_partition_eval
            )
            
            include_run_keys = None
            exclude_run_keys = None
            
            if split == 'train':
                include_run_keys = self.config.libribrain_include_run_keys_train
                exclude_run_keys = self.config.libribrain_exclude_run_keys_train
            else:
                include_run_keys = self.config.libribrain_include_run_keys_eval
                exclude_run_keys = self.config.libribrain_exclude_run_keys_eval
            
            # Convert string lists to tuples if provided
            if include_run_keys:
                include_run_keys = [tuple(key) if isinstance(key, list) else key for key in include_run_keys]
            if exclude_run_keys:
                exclude_run_keys = [tuple(key) if isinstance(key, list) else key for key in exclude_run_keys]
            
            dataset = LibriBrainPreTrainingDataset(
                data_path=self.config.data_dir,
                sequence_length=self.config.sequence_length,
                partition=partition,
                tmin=self.config.libribrain_tmin,
                tmax=self.config.libribrain_tmax,
                include_run_keys=include_run_keys,
                exclude_run_keys=exclude_run_keys,
                standardize=self.config.libribrain_standardize,
                download=self.config.libribrain_download,
                preload_files=self.config.libribrain_preload_files,
            )
        else:
            # Use numpy files dataset
            dataset = MEGPreTrainingDataset(
                data_dir=self.config.numpy_data_dir,
                sequence_length=self.config.sequence_length,
                split=split
            )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(split == 'train'),
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with weight decay"""
        # Separate parameters for weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight', 'norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        return optim.AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                return max(0.1, (self.config.max_steps - step) / (self.config.max_steps - self.config.warmup_steps))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _compute_masks_and_negatives(self, batch_size: int, sequence_length: int, attention_mask: torch.Tensor):
        """Compute masking indices and negative samples for contrastive learning"""
        # Compute mask indices
        mask_time_indices = _compute_mask_indices(
            shape=(batch_size, sequence_length),
            mask_prob=self.config.mask_time_prob,
            mask_length=self.config.mask_time_length,
            attention_mask=attention_mask.cpu().numpy() if attention_mask is not None else None,
            min_masks=self.config.mask_time_min_masks,
        )
        
        # Sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape=(batch_size, sequence_length),
            num_negatives=self.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        
        # Convert to tensors
        mask_time_indices = torch.from_numpy(mask_time_indices).to(self.device)
        sampled_negative_indices = torch.from_numpy(sampled_negative_indices).to(self.device)
        
        return mask_time_indices, sampled_negative_indices
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step"""
        input_values = batch['input_values'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        batch_size, _, sequence_length = input_values.shape
        
        # Get feature sequence length after conv layers
        feature_sequence_length = self.model._get_feat_extract_output_lengths(
            torch.tensor(sequence_length)
        ).item()
        
        # Compute masks and negative samples
        mask_time_indices, sampled_negative_indices = self._compute_masks_and_negatives(
            batch_size, feature_sequence_length, attention_mask
        )
        
        # Update Gumbel temperature
        current_temp = max(
            self.config.min_gumbel_temperature,
            self.config.max_gumbel_temperature * (self.config.gumbel_temperature_decay ** self.global_step)
        )
        self.model.set_gumbel_temperature(current_temp)
        
        # Forward pass
        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
            sampled_negative_indices=sampled_negative_indices,
            return_dict=True,
        )
        
        loss = outputs.loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Log metrics
        metrics = {
            'train/loss': outputs.loss.item(),
            'train/contrastive_loss': outputs.contrastive_loss.item() if outputs.contrastive_loss is not None else 0.0,
            'train/diversity_loss': outputs.diversity_loss.item() if outputs.diversity_loss is not None else 0.0,
            'train/codevector_perplexity': outputs.codevector_perplexity.item(),
            'train/gumbel_temperature': current_temp,
            'train/learning_rate': self.scheduler.get_last_lr()[0],
        }
        
        return metrics
    
    def _evaluation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one evaluation step"""
        input_values = batch['input_values'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        batch_size, _, sequence_length = input_values.shape
        
        # Get feature sequence length after conv layers
        feature_sequence_length = self.model._get_feat_extract_output_lengths(
            torch.tensor(sequence_length)
        ).item()
        
        # Compute masks and negative samples
        mask_time_indices, sampled_negative_indices = self._compute_masks_and_negatives(
            batch_size, feature_sequence_length, attention_mask
        )
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_values=input_values,
                attention_mask=attention_mask,
                mask_time_indices=mask_time_indices,
                sampled_negative_indices=sampled_negative_indices,
                return_dict=True,
            )
        
        metrics = {
            'eval/loss': outputs.loss.item(),
            'eval/contrastive_loss': outputs.contrastive_loss.item() if outputs.contrastive_loss is not None else 0.0,
            'eval/diversity_loss': outputs.diversity_loss.item() if outputs.diversity_loss is not None else 0.0,
            'eval/codevector_perplexity': outputs.codevector_perplexity.item(),
        }
        
        return metrics
    
    def _save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': step,
            'epoch': self.epoch,
            'config': self.config.__dict__,
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.output_dir, f'checkpoint-{step}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
        
        logging.info(f"Saved checkpoint at step {step}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        logging.info(f"Loaded checkpoint from {checkpoint_path}, step {self.global_step}")
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on validation set"""
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        eval_metrics = []
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            metrics = self._evaluation_step(batch)
            eval_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        for key in eval_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in eval_metrics])
        
        self.model.train()
        return avg_metrics
    
    def train(self):
        """Main training loop"""
        logging.info("Starting training...")
        self.model.train()
        
        # Calculate steps per epoch for better progress tracking
        steps_per_epoch = len(self.train_dataloader)
        total_epochs = (self.config.max_steps + steps_per_epoch - 1) // steps_per_epoch
        
        # Create main progress bar for overall training
        pbar = tqdm(
            total=self.config.max_steps,
            desc="Training",
            unit="step",
            ncols=120,
            position=0,
            leave=True
        )
        
        # Set initial progress if resuming from checkpoint
        if self.global_step > 0:
            pbar.update(self.global_step)
        
        data_iter = iter(self.train_dataloader)
        accumulated_loss = 0.0
        current_epoch_step = 0
        
        while self.global_step < self.config.max_steps:
            try:
                batch = next(data_iter)
                current_epoch_step += 1
            except StopIteration:
                # Reset data iterator
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
                self.epoch += 1
                current_epoch_step = 1
                
                # Update progress bar description with epoch info
                pbar.set_description(f"Training (Epoch {self.epoch}/{total_epochs})")
            
            # Training step
            step_metrics = self._training_step(batch)
            accumulated_loss += step_metrics['train/loss']
            
            # Gradient update
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Update parameters
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update progress bar
                pbar.update(1)
                
                # Log training metrics
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = accumulated_loss / self.config.log_interval
                    step_metrics['train/loss'] = avg_loss
                    
                    # Update progress bar with current metrics
                    pbar.set_postfix({
                        'Loss': f"{avg_loss:.4f}",
                        'LR': f"{step_metrics['train/learning_rate']:.1e}",
                        'Temp': f"{step_metrics['train/gumbel_temperature']:.3f}",
                        'Epoch_Step': f"{current_epoch_step}/{steps_per_epoch}"
                    })
                    
                    if self.config.use_wandb:
                        wandb.log(step_metrics, step=self.global_step)
                    
                    logging.info(
                        f"Step {self.global_step}: "
                        f"Loss={avg_loss:.4f}, "
                        f"LR={step_metrics['train/learning_rate']:.2e}, "
                        f"Temp={step_metrics['train/gumbel_temperature']:.3f}"
                    )
                    
                    accumulated_loss = 0.0
                
                # Evaluation
                if self.global_step % self.config.eval_interval == 0 and self.eval_dataloader is not None:
                    # Temporarily pause main progress bar for evaluation
                    pbar.write(f"Running evaluation at step {self.global_step}...")
                    eval_metrics = self.evaluate()
                    
                    if self.config.use_wandb:
                        wandb.log(eval_metrics, step=self.global_step)
                    
                    eval_loss = eval_metrics.get('eval/loss', float('inf'))
                    is_best = eval_loss < self.best_eval_loss
                    if is_best:
                        self.best_eval_loss = eval_loss
                    
                    # Log evaluation results
                    eval_msg = f"Eval - Loss: {eval_loss:.4f}, Best: {self.best_eval_loss:.4f}"
                    pbar.write(eval_msg)
                    logging.info(f"Eval metrics: {eval_metrics}")
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_interval == 0:
                        pbar.write(f"Saving checkpoint at step {self.global_step}...")
                        self._save_checkpoint(self.global_step, is_best)
                
                # Save checkpoint
                elif self.global_step % self.config.save_interval == 0:
                    pbar.write(f"Saving checkpoint at step {self.global_step}...")
                    self._save_checkpoint(self.global_step)
            
            self.global_step += 1
        
        # Close progress bar
        pbar.close()
        
        # Final checkpoint
        self._save_checkpoint(self.global_step, is_best=False)
        logging.info("Training completed!")


def load_config(config_path: str) -> PreTrainingConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return PreTrainingConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser(description="Meg2Vec Pre-training")
    parser.add_argument("--config", type=str, default="configs/meg2vec_pretrain.yaml",
                        help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, help="Override data directory")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        logging.warning(f"Config file {args.config} not found, using default config")
        config = PreTrainingConfig()
    
    # Override with command line arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.resume:
        config.resume_from_checkpoint = args.resume
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
