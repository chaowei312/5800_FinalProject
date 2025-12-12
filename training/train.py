"""
Simplified training function for baseline and recurrent models.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from tqdm import tqdm
import numpy as np
import copy
import os


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    name: str = "Model",
    num_epochs: int = 5,
    learning_rate: float = 3e-5,
    patience: int = 3,
    min_delta: float = 0.001,
    save_dir: str = "checkpoints",
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Train a model with early stopping and learning rate scheduling.
    
    Args:
        model: The model to train (should already be on device)
        train_loader: Training data loader
        val_loader: Validation data loader
        name: Model name for logging and saving (e.g., 'baseline', 'recurrent')
        num_epochs: Maximum number of epochs
        learning_rate: Initial learning rate
        patience: Early stopping patience (epochs without improvement)
        min_delta: Minimum improvement to reset patience
        save_dir: Directory to save best model
        device: Device to use (defaults to model's device)
    
    Returns:
        Dictionary with training history
    """
    # Get device from model if not specified
    if device is None:
        device = next(model.parameters()).device
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # Early stopping state (use accuracy for model selection)
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"Training {name} on {device}")
    print(f"  Epochs: {num_epochs}, LR: {learning_rate}, Patience: {patience}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        # === Training Phase ===
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"{name} Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # === Validation Phase ===
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                val_losses.append(outputs['loss'].item())
                predictions = outputs['logits'].argmax(dim=-1)
                val_correct += (predictions == batch['labels']).sum().item()
                val_total += batch['labels'].size(0)
        
        # Calculate metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_acc = val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # === Early Stopping Check (based on accuracy) ===
        improved = val_acc > best_val_acc + min_delta
        
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            # Save best model
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{name}.pt")
            torch.save({
                'model_state_dict': best_model_state,
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'history': history
            }, save_path)
            
            print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}, acc={val_acc:.4f} * saved → {save_path}")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}, acc={val_acc:.4f}")
        
        # Check early stopping
        if patience_counter >= patience:
            print(f"{name}: Early stopping at epoch {epoch+1}")
            break
    
    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"{name}: Loaded best model (val_acc={best_val_acc:.4f})")
    
    return history


def load_trained_model(model: nn.Module, checkpoint_path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load a trained model from checkpoint.
    
    Args:
        model: Model architecture (uninitialized weights OK)
        checkpoint_path: Path to saved checkpoint
        device: Device to load to
    
    Returns:
        Checkpoint dictionary with history and metadata
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}, Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    return checkpoint


