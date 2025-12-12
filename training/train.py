import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import os
import copy


def train_model(
    model,
    train_loader,
    val_loader,
    name="Model",
    num_epochs=5,
    learning_rate=3e-5,
    patience=3,
    min_delta=1e-4,
    use_fp16=False,
    save_dir="checkpoints",
    device=None
):
    """
    Train model with optional FP16 (AMP) and early stopping based on val_loss.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        name: Model name for logging
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate for optimizer
        patience: Early stopping patience (epochs)
        min_delta: Minimum improvement to reset patience
        use_fp16: Whether to use mixed precision training
        save_dir: Directory to save checkpoints
        device: Device to train on (auto-detected if None)
    
    Returns:
        history: Dictionary with training metrics
    """
    
    # Setup device
    if device is None:
        device = next(model.parameters()).device
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Setup mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    
    # Initialize history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_epoch = -1
    best_model_state = None
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"Training {name} on {device}")
    print(f"FP16: {use_fp16}, Epochs: {num_epochs}, LR: {learning_rate}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        
        # ========================
        # TRAINING PHASE
        # ========================
        model.train()
        train_losses = []
        
        pbar = tqdm(
            train_loader,
            desc=f"{name} Epoch {epoch+1}/{num_epochs}",
            leave=False
        )
        
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward pass with optional FP16
            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(**batch)
                loss = outputs["loss"]
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
            pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})
        
        # ========================
        # VALIDATION PHASE
        # ========================
        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass with optional FP16
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    outputs = model(**batch)
                
                val_losses.append(outputs["loss"].item())
                
                # Get predictions
                logits = outputs["logits"]
                preds = logits.argmax(dim=-1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # ========================
        # COMPUTE METRICS
        # ========================
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="weighted")
        lr = optimizer.param_groups[0]["lr"]
        
        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(lr)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # ========================
        # EARLY STOPPING CHECK
        # ========================
        if val_loss < best_val_loss - min_delta:
            # Improvement found
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            # Save checkpoint
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{name}_best.pt")
            torch.save({
                "model_state_dict": best_model_state,
                "epoch": best_epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "history": history
            }, save_path)
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"train={train_loss:.4f}, val={val_loss:.4f}, "
                  f"acc={val_acc:.4f}, f1={val_f1:.4f}, lr={lr:.2e} "
                  f"→ ✓ NEW BEST (saved to {save_path})")
        else:
            # No improvement
            patience_counter += 1
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"train={train_loss:.4f}, val={val_loss:.4f}, "
                  f"acc={val_acc:.4f}, f1={val_f1:.4f}, lr={lr:.2e} "
                  f"(no improvement {patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\n{name}: Early stopping triggered at epoch {epoch+1}")
                break
    
    # ========================
    # RESTORE BEST MODEL
    # ========================
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n{'='*60}")
        print(f"Loaded best model from epoch {best_epoch}")
        print(f"Best val_loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")
    
    return history


def evaluate_model(model, test_loader, device=None):
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        device: Device to evaluate on (auto-detected if None)
    
    Returns:
        metrics: Dictionary with test metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    test_losses = []
    all_preds = []
    all_labels = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            test_losses.append(outputs["loss"].item())
            
            preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    test_loss = np.mean(test_losses)
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="weighted")
    
    metrics = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1": test_f1
    }
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}\n")
    
    return metrics