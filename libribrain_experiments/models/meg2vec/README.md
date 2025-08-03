# Meg2Vec: Multi-channel MEG adaptation of Wav2Vec2

This directory contains the implementation of Meg2Vec, a self-supervised pre-training model for MEG (magnetoencephalography) data, adapted from the Wav2Vec2 architecture.

## Overview

Meg2Vec adapts the proven Wav2Vec2 self-supervised learning approach for multi-channel MEG data:

- **Multi-channel input**: Processes 306 MEG channels simultaneously (vs. single audio channel)
- **Contrastive learning**: Uses masked prediction with negative sampling
- **Vector quantization**: Learns discrete representations of MEG patterns
- **Transformer architecture**: Captures long-range dependencies in neural signals

## Key Differences from Wav2Vec2

1. **Input shape**: `(batch, 306, seq_len)` instead of `(batch, 1, seq_len)`
2. **Feature extraction**: Adapted convolutional layers for multi-channel MEG
3. **GroupNorm**: Better normalization for spatial-temporal MEG patterns
4. **MEG-specific preprocessing**: Handles neurological time-series data characteristics

## Files

- `model.py`: Core model implementation
  - `Meg2VecModel`: Base model for feature extraction
  - `Meg2VecForPreTraining`: Pre-training model with contrastive learning
  - Utility functions for masking and negative sampling

- `train.py`: Pre-training script
  - Implements the full training loop
  - Supports resuming from checkpoints
  - Includes evaluation and logging

- `prepare_data.py`: Data preparation utilities
  - Converts raw MEG files to training format
  - Applies preprocessing (filtering, normalization)
  - Creates train/eval splits

- `configs/`: Configuration files
  - `pretrain_base.yaml`: Full-scale pre-training configuration
  - `pretrain_small.yaml`: Smaller model for development/testing

## Quick Start

### Option 1: Using LibriBrain Dataset (Recommended)

The easiest way to get started is with the LibriBrain competition dataset:

```bash
# For development (small model, subset of data)
python train.py --config ../../../configs/meg2vec/pretrain_small.yaml

# For full pre-training (large model, full dataset)
python train.py --config ../../../configs/meg2vec/pretrain_base.yaml
```

The LibriBrain dataset will be automatically downloaded (requires internet connection). The small config uses only a subset of runs for faster development.

### Option 2: Using Custom MEG Data

If you have your own MEG data, first prepare it:

### Option 2: Using Custom MEG Data

If you have your own MEG data, first prepare it:

```bash
# Convert MEG files to training format
python prepare_data.py \\
    --input_dir /path/to/your/meg/files \\
    --output_dir data/meg_pretraining \\
    --file_pattern "*.fif" \\
    --segment_length 5000 \\
    --split_ratio 0.9
```

This will:
- Load MEG files (supports .fif format via MNE-Python)
- Ensure 306 channels (pad or truncate as needed)
- Apply bandpass filtering (1-40 Hz) and normalization
- Split long recordings into segments
- Create train/eval splits

Then update your config to use custom data:

```yaml
# Set use_libribrain: false in your config
use_libribrain: false
numpy_data_dir: "data/meg_pretraining"
```

### 2. Start Pre-training

```bash
# Run training with your chosen configuration
python train.py --config path/to/your/config.yaml
```

### 3. Monitor Training

The script supports Weights & Biases logging:
- Set `use_wandb: true` in config
- Training metrics, evaluation results, and model checkpoints are logged
- View progress at wandb.ai

## Data Sources

Meg2Vec supports two data sources:

### LibriBrain Dataset (Default)
- **Automatic download**: Dataset downloaded automatically from HuggingFace
- **Rich annotations**: Speech vs silence labels (not used during pre-training)
- **Preprocessed**: Already filtered, normalized, and downsampled to 250Hz
- **Standardized**: 306 MEG channels, consistent preprocessing
- **Multiple sessions**: Access to different sessions/runs of the Sherlock Holmes audiobook

### Custom MEG Data
- **Flexible format**: Support for .fif files via MNE-Python
- **Preprocessing pipeline**: Automatic filtering and normalization
- **Custom segmentation**: Configurable segment lengths
- **Data organization**: Automatic train/eval splits

## Configuration Options

### LibriBrain-specific Settings

```yaml
# Data source
use_libribrain: true
data_dir: "./data"  # Base path for LibriBrain data

# LibriBrain parameters
libribrain_partition_train: "train"      # Use predefined train split
libribrain_partition_eval: "validation"  # Use predefined validation split
libribrain_standardize: true             # Z-score normalization
libribrain_download: true                # Auto-download missing files

# Custom run selection (overrides partitions)
libribrain_include_run_keys_train:
  - ["0", "1", "Sherlock1", "1"]  # Subject, Session, Task, Run
  - ["0", "1", "Sherlock1", "2"]
libribrain_include_run_keys_eval:
  - ["0", "11", "Sherlock1", "2"]
```

### Custom Data Settings

```yaml
# Data source
use_libribrain: false
numpy_data_dir: "data/meg_pretraining"  # Path to numpy files
```

## Model Architecture

### Meg2VecModel (Base)
```
Input: (batch, 306, 500) MEG data
    ↓
Meg2VecFeatureExtractor:
    - Conv1d(306→256, k=10, s=5) + GroupNorm
    - Conv1d(256→256, k=3, s=2) + GELU  
    - Conv1d(256→256, k=3, s=2) + GELU
    - Conv1d(256→256, k=2, s=2) + GELU
    ↓ (batch, 256, ~31)
Meg2VecFeatureProjection:
    - LayerNorm + Linear(256→768) + Dropout
    ↓ (batch, ~31, 768)
Meg2VecEncoder:
    - Positional conv embeddings
    - 12-layer Transformer encoder
    ↓
Output: (batch, ~31, 768) contextualized features
```

### Meg2VecForPreTraining
Adds quantization and contrastive learning components:
- **Vector Quantizer**: Learns discrete MEG representations
- **Projection heads**: Project features to contrastive learning space
- **Masking**: Random spans of features are masked during training
- **Contrastive loss**: Model learns to predict masked features using context

## Training Methodology

### Self-Supervised Pre-training
1. **Masking**: Random time spans in feature sequence are masked
2. **Context encoding**: Transformer processes unmasked features  
3. **Quantization**: Masked features are quantized to discrete codes
4. **Contrastive learning**: Model predicts correct quantized code vs. distractors
5. **Joint optimization**: Contrastive loss + diversity loss encourage rich representations

### Hyperparameters (Base Config)
- **Model**: 768 hidden size, 12 layers, 12 attention heads
- **Training**: 400k steps, 5e-4 learning rate, warmup + decay
- **Masking**: 6.5% of spans masked, 10 timesteps per span
- **Contrastive**: 100 negative samples, temperature 0.1
- **Quantization**: 2 groups × 320 codes = 640 total codes

## Usage Examples

### Basic Model Usage
```python
from model import Meg2VecModel

# Initialize model
model = Meg2VecModel(
    sequence_length=500,
    in_channels=306, 
    hidden_size=768,
    num_hidden_layers=12
)

# Forward pass
meg_data = torch.randn(batch_size, 306, 500)
output = model(meg_data)
features = output['last_hidden_state']  # (batch, ~31, 768)
```

### Pre-training Model
```python
from model import Meg2VecForPreTraining, _compute_mask_indices, _sample_negative_indices

# Initialize pre-training model
model = Meg2VecForPreTraining(
    sequence_length=500,
    in_channels=306,
    hidden_size=768
)

# Prepare contrastive learning inputs
batch_size, seq_len = 8, 31  # After conv layers
mask_indices = _compute_mask_indices(
    shape=(batch_size, seq_len),
    mask_prob=0.065,
    mask_length=10
)
negative_indices = _sample_negative_indices(
    features_shape=(batch_size, seq_len),
    num_negatives=100,
    mask_time_indices=mask_indices
)

# Training forward pass
outputs = model(
    input_values=meg_data,
    mask_time_indices=torch.from_numpy(mask_indices),
    sampled_negative_indices=torch.from_numpy(negative_indices)
)

loss = outputs.loss  # Contrastive + diversity loss
```

## Data Requirements

### Input Format
- **Shape**: `(306, time_steps)` where 306 is the standard MEG sensor count
- **Type**: Float32 numpy arrays
- **Sampling rate**: Typically 1000 Hz (configurable)
- **Length**: Variable (will be segmented during preprocessing)

### Preprocessing Recommendations
- **Filtering**: 1-40 Hz bandpass (removes slow drifts and high-freq noise)
- **Normalization**: Z-score per channel (handles different sensor sensitivities)
- **Segmentation**: 5-second segments work well (5000 samples @ 1kHz)
- **Artifacts**: Remove obvious artifacts before training

### Data Organization
```
data/meg_pretraining/
├── meg_000000_000.npy    # Processed MEG segments
├── meg_000000_001.npy
├── ...
├── train.txt             # List of training files
└── eval.txt              # List of evaluation files
```

## Dependencies

### Core Requirements
```bash
pip install torch numpy pyyaml tqdm wandb tensorboard
```

### Data Processing (Optional)
```bash
pip install mne scipy  # For .fif file support and filtering
```

### Full Environment
```bash
# Create conda environment
conda create -n meg2vec python=3.9
conda activate meg2vec

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio

# Install other dependencies
pip install numpy scipy pyyaml tqdm wandb tensorboard mne
```

## Configuration

### Key Parameters to Tune

**Model Size**:
- `hidden_size`: 384 (small), 768 (base), 1024 (large)
- `num_hidden_layers`: 6 (small), 12 (base), 24 (large)
- `num_attention_heads`: Should divide `hidden_size`

**Training**:
- `batch_size`: Limited by GPU memory (4-16 typical)
- `learning_rate`: 1e-3 (small), 5e-4 (base), 3e-4 (large)
- `mask_time_prob`: 0.065 works well, tune for your data

**Hardware**:
- `num_workers`: Match CPU cores for data loading
- `gradient_accumulation_steps`: Increase if batch size is memory-limited

## Known Issues & Tips

1. **Memory usage**: Large models need significant GPU memory. Use gradient accumulation if needed.

2. **Convergence**: Monitor codevector perplexity - should be > 50 for good quantization.

3. **Data quality**: Clean MEG data is crucial. Remove sessions with excessive artifacts.

4. **Sequence length**: Longer sequences help but increase memory. 500 samples (0.5s) is a good starting point.

5. **Channel count**: If you have < 306 channels, the model will pad with zeros. If > 306, it takes the first 306.

## Citation

Based on the Wav2Vec2 paper:
```
@article{baevski2020wav2vec,
  title={wav2vec 2.0: A framework for self-supervised learning of speech representations},
  author={Baevski, Alexei and Zhou, Hao and Mohamed, Abdelrahman and Auli, Michael},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={12449--12460},
  year={2020}
}
```
