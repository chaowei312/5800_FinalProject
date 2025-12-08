"""
Data loading utilities for SST-2 dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from typing import Dict, List, Optional, Tuple
from transformers import BertTokenizer, AutoTokenizer
import numpy as np


class SST2Dataset(Dataset):
    """SST-2 sentiment analysis dataset."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 128,
        truncation: bool = True,
        padding: str = 'max_length'
    ):
        """
        Initialize SST-2 dataset.
        
        Args:
            data_path: Path to the pickle file
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            truncation: Whether to truncate sequences
            padding: Padding strategy
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        
        # Load data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle different data formats
        if isinstance(data, dict):
            self.texts = data.get('texts', data.get('sentences', []))
            self.labels = data.get('labels', [])
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                self.texts = [item['text'] for item in data]
                self.labels = [item['label'] for item in data]
            elif isinstance(data[0], tuple):
                self.texts = [item[0] for item in data]
                self.labels = [item[1] for item in data]
            else:
                raise ValueError("Unsupported data format")
        else:
            raise ValueError("Unsupported data format")
        
        # Ensure labels are integers
        self.labels = [int(label) for label in self.labels]
        
        print(f"Loaded {len(self.texts)} samples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=self.truncation,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DynamicBatchSampler:
    """Dynamic batch sampler for efficient batching."""
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        if batch and not self.drop_last:
            yield batch


def create_data_loader(
    data_path: str,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    Create a DataLoader for SST-2 dataset.
    
    Args:
        data_path: Path to the data file
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        DataLoader instance
    """
    dataset = SST2Dataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def load_tokenizer(model_type: str = 'bert-base-uncased') -> AutoTokenizer:
    """
    Load tokenizer based on model type.
    
    Args:
        model_type: Type of model/tokenizer to load
        
    Returns:
        Tokenizer instance
    """
    # Map model types to tokenizer paths
    tokenizer_map = {
        'bert-tiny': 'prajjwal1/bert-tiny',
        'bert-small': 'prajjwal1/bert-small',
        'bert-mini': 'prajjwal1/bert-mini',
        'bert-base': 'bert-base-uncased',
        'bert-large': 'bert-large-uncased'
    }
    
    tokenizer_name = tokenizer_map.get(model_type, model_type)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except:
        # Fallback to BERT base tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print(f"Failed to load {tokenizer_name}, using bert-base-uncased")
    
    return tokenizer


def prepare_yelp_data(
    data_dir: str = 'data/processed',
    tokenizer = None,
    batch_size: int = 32,
    max_length: int = 128
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare yelp data loaders.
    
    Args:
        data_dir: Directory containing processed data
        tokenizer: Tokenizer instance (if None, will load default)
        batch_size: Batch size
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if tokenizer is None:
        tokenizer = load_tokenizer('bert-base-uncased')
    
    # Create data loaders
    train_loader = create_data_loader(
        os.path.join(data_dir, 'yelp_train.pkl'),
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=True
    )
    
    val_loader = create_data_loader(
        os.path.join(data_dir, 'yelp_internal_val.pkl'),
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False
    )
    
    test_loader = create_data_loader(
        os.path.join(data_dir, 'yelp_val.pkl'),
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader

def prepare_sst2_data(
    data_dir: str = 'data/processed',
    tokenizer = None,
    batch_size: int = 32,
    max_length: int = 128
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare SST-2 data loaders.
    
    Args:
        data_dir: Directory containing processed data
        tokenizer: Tokenizer instance (if None, will load default)
        batch_size: Batch size
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if tokenizer is None:
        tokenizer = load_tokenizer('bert-base-uncased')
    
    # Create data loaders
    train_loader = create_data_loader(
        os.path.join(data_dir, 'sst2_train.pkl'),
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=True
    )
    
    val_loader = create_data_loader(
        os.path.join(data_dir, 'sst2_val.pkl'),
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False
    )
    
    test_loader = create_data_loader(
        os.path.join(data_dir, 'sst2_test.pkl'),
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched dictionary
    """
    keys = batch[0].keys()
    batched = {}
    
    for key in keys:
        if key == 'labels':
            batched[key] = torch.stack([sample[key] for sample in batch])
        else:
            batched[key] = torch.stack([sample[key] for sample in batch])
    
    return batched
