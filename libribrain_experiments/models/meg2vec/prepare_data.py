"""
MEG Data Preparation Script for Meg2Vec Pre-training

This script helps prepare MEG data for pre-training by:
1. Converting MEG data to the expected format (306 channels, variable length)
2. Applying basic preprocessing (filtering, normalization)
3. Creating train/eval splits
4. Saving data in numpy format

Expected input format:
- Raw MEG files (e.g., .fif format from MNE)
- Each file should contain MEG sensor data

Output format:
- Numpy arrays of shape (306, time_steps)
- Saved as .npy files in the specified output directory
- train.txt and eval.txt files listing the training and evaluation files

Usage:
    python prepare_data.py --input_dir /path/to/raw/meg --output_dir data/meg_pretraining --split_ratio 0.9
"""

import os
import argparse
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    logging.warning("MNE not available. Install with: pip install mne")

try:
    import scipy.signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logging.warning("SciPy not available. Install with: pip install scipy")


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def preprocess_meg_data(
    data: np.ndarray, 
    sfreq: float = 1000.0,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    normalize: bool = True
) -> np.ndarray:
    """
    Apply basic preprocessing to MEG data
    
    Args:
        data: MEG data of shape (n_channels, n_times)
        sfreq: Sampling frequency
        l_freq: Low-pass filter frequency
        h_freq: High-pass filter frequency
        normalize: Whether to normalize the data
    
    Returns:
        Preprocessed MEG data
    """
    if not HAS_SCIPY:
        logging.warning("SciPy not available, skipping filtering")
        processed_data = data.copy()
    else:
        # Apply bandpass filter
        nyquist = sfreq / 2
        low = l_freq / nyquist
        high = h_freq / nyquist
        
        if high >= 1.0:
            # Only apply high-pass filter
            sos = scipy.signal.butter(4, low, btype='high', output='sos')
        else:
            # Apply bandpass filter
            sos = scipy.signal.butter(4, [low, high], btype='band', output='sos')
        
        processed_data = scipy.signal.sosfiltfilt(sos, data, axis=1)
    
    if normalize:
        # Normalize each channel independently
        processed_data = (processed_data - processed_data.mean(axis=1, keepdims=True)) / (
            processed_data.std(axis=1, keepdims=True) + 1e-8
        )
    
    return processed_data


def load_meg_file(file_path: str) -> Tuple[np.ndarray, float]:
    """
    Load MEG data from file
    
    Args:
        file_path: Path to MEG file
    
    Returns:
        Tuple of (data, sampling_frequency)
        data shape: (n_channels, n_times)
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.npy':
        # Already preprocessed numpy file
        data = np.load(file_path)
        sfreq = 1000.0  # Default sampling frequency
        return data, sfreq
    
    elif file_path.suffix == '.fif' and HAS_MNE:
        # MNE-Python format
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        
        # Get MEG channels only
        meg_picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, ecg=False)
        data = raw.get_data(picks=meg_picks)
        sfreq = raw.info['sfreq']
        
        return data, sfreq
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def ensure_306_channels(data: np.ndarray) -> np.ndarray:
    """
    Ensure data has exactly 306 channels (standard MEG sensor count)
    
    Args:
        data: Input data of shape (n_channels, n_times)
    
    Returns:
        Data with 306 channels
    """
    n_channels, n_times = data.shape
    
    if n_channels == 306:
        return data
    elif n_channels > 306:
        # Take first 306 channels
        logging.warning(f"Taking first 306 channels from {n_channels} channels")
        return data[:306]
    else:
        # Pad with zeros
        logging.warning(f"Padding from {n_channels} to 306 channels with zeros")
        padded_data = np.zeros((306, n_times), dtype=data.dtype)
        padded_data[:n_channels] = data
        return padded_data


def split_long_recording(
    data: np.ndarray, 
    segment_length: int = 5000,  # 5 seconds at 1000 Hz
    overlap: int = 1000,  # 1 second overlap
) -> List[np.ndarray]:
    """
    Split long recording into shorter segments for training
    
    Args:
        data: MEG data of shape (306, n_times)
        segment_length: Length of each segment in samples
        overlap: Overlap between segments in samples
    
    Returns:
        List of segments, each of shape (306, segment_length)
    """
    n_channels, n_times = data.shape
    
    if n_times <= segment_length:
        return [data]
    
    segments = []
    start = 0
    
    while start + segment_length <= n_times:
        segment = data[:, start:start + segment_length]
        segments.append(segment)
        start += (segment_length - overlap)
    
    return segments


def process_meg_files(
    input_files: List[str],
    output_dir: str,
    segment_length: int = 5000,
    preprocess: bool = True,
    filter_params: Optional[dict] = None
) -> List[str]:
    """
    Process MEG files and save as numpy arrays
    
    Args:
        input_files: List of input MEG files
        output_dir: Output directory for processed files
        segment_length: Length of segments to create
        preprocess: Whether to apply preprocessing
        filter_params: Parameters for filtering (if preprocessing enabled)
    
    Returns:
        List of output file names
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filter_params is None:
        filter_params = {'l_freq': 1.0, 'h_freq': 40.0, 'normalize': True}
    
    output_files = []
    
    for i, input_file in enumerate(tqdm(input_files, desc="Processing MEG files")):
        try:
            # Load data
            data, sfreq = load_meg_file(input_file)
            
            # Ensure 306 channels
            data = ensure_306_channels(data)
            
            # Preprocessing
            if preprocess:
                data = preprocess_meg_data(data, sfreq=sfreq, **filter_params)
            
            # Split into segments
            segments = split_long_recording(data, segment_length=segment_length)
            
            # Save each segment
            for j, segment in enumerate(segments):
                output_name = f"meg_{i:06d}_{j:03d}.npy"
                output_path = output_dir / output_name
                np.save(output_path, segment.astype(np.float32))
                output_files.append(output_name)
            
        except Exception as e:
            logging.error(f"Error processing {input_file}: {e}")
            continue
    
    return output_files


def create_splits(
    file_list: List[str], 
    split_ratio: float = 0.9,
    output_dir: str = "."
) -> Tuple[List[str], List[str]]:
    """
    Create train/eval splits and save to text files
    
    Args:
        file_list: List of processed file names
        split_ratio: Ratio of files to use for training
        output_dir: Directory to save split files
    
    Returns:
        Tuple of (train_files, eval_files)
    """
    # Shuffle files
    np.random.shuffle(file_list)
    
    # Split
    n_train = int(len(file_list) * split_ratio)
    train_files = file_list[:n_train]
    eval_files = file_list[n_train:]
    
    output_dir = Path(output_dir)
    
    # Save train split
    with open(output_dir / "train.txt", 'w') as f:
        for file_name in train_files:
            f.write(f"{file_name}\\n")
    
    # Save eval split
    with open(output_dir / "eval.txt", 'w') as f:
        for file_name in eval_files:
            f.write(f"{file_name}\\n")
    
    logging.info(f"Created splits: {len(train_files)} train, {len(eval_files)} eval")
    return train_files, eval_files


def main():
    parser = argparse.ArgumentParser(description="Prepare MEG data for pre-training")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing raw MEG files")
    parser.add_argument("--output_dir", type=str, default="data/meg_pretraining",
                        help="Output directory for processed data")
    parser.add_argument("--file_pattern", type=str, default="*.fif",
                        help="File pattern for MEG files (e.g., '*.fif', '*.npy')")
    parser.add_argument("--segment_length", type=int, default=5000,
                        help="Length of segments in samples (default: 5000 = 5s at 1kHz)")
    parser.add_argument("--split_ratio", type=float, default=0.9,
                        help="Ratio of data to use for training")
    parser.add_argument("--no_preprocess", action="store_true",
                        help="Skip preprocessing (filtering, normalization)")
    parser.add_argument("--l_freq", type=float, default=1.0,
                        help="Low-pass filter frequency")
    parser.add_argument("--h_freq", type=float, default=40.0,
                        help="High-pass filter frequency")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    setup_logging()
    np.random.seed(args.seed)
    
    # Find input files
    input_dir = Path(args.input_dir)
    input_files = list(input_dir.glob(args.file_pattern))
    
    if len(input_files) == 0:
        logging.error(f"No files found matching pattern '{args.file_pattern}' in {input_dir}")
        return
    
    logging.info(f"Found {len(input_files)} MEG files to process")
    
    # Setup filter parameters
    filter_params = {
        'l_freq': args.l_freq,
        'h_freq': args.h_freq,
        'normalize': True
    }
    
    # Process files
    output_files = process_meg_files(
        input_files=[str(f) for f in input_files],
        output_dir=args.output_dir,
        segment_length=args.segment_length,
        preprocess=not args.no_preprocess,
        filter_params=filter_params
    )
    
    if len(output_files) == 0:
        logging.error("No files were successfully processed")
        return
    
    # Create splits
    train_files, eval_files = create_splits(
        output_files, 
        split_ratio=args.split_ratio,
        output_dir=args.output_dir
    )
    
    logging.info(f"Data preparation completed!")
    logging.info(f"Total segments: {len(output_files)}")
    logging.info(f"Training segments: {len(train_files)}")
    logging.info(f"Evaluation segments: {len(eval_files)}")
    logging.info(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
