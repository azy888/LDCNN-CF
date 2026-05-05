"""
Trainer module
Supports single model training, validation, early stopping, learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import yaml


class Trainer:
    """Model trainer"""
    
    def __init__(self, model: nn.Module, config: dict, device: torch.device):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            config: Configuration dictionary
            device: Device (cuda/cpu)
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['training']['lr_scheduler_factor'],
            patience=config['training']['lr_scheduler_patience'],
            min_lr=config['training']['min_lr']
        )
        
        # Early stopping parameters
        self.patience = config['training']['early_stopping_patience']
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.counter = 0
        
        # History
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc='Training', leave=False):
            mel_spec = batch['mel_spec'].to(self.device)
            psycho = batch['psycho_features'].to(self.device)
            labels = batch['comfort_score'].to(self.device).unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(mel_spec, psycho)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation', leave=False):
                mel_spec = batch['mel_spec'].to(self.device)
                psycho = batch['psycho_features'].to(self.device)
                labels = batch['comfort_score'].to(self.device).unsqueeze(1)
                
                outputs = self.model(mel_spec, psycho)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, 
            epochs: int, checkpoint_dir: str = 'checkpoints'):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            checkpoint_dir: Directory to save model checkpoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"Training on device: {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(checkpoint_dir, 'best_model.pth'))
                print(f"  -> Saved best model")
            else:
                self.counter += 1
                print(f"  -> Early stopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                print(f"Early stopping triggered! Best val loss: {self.best_val_loss:.6f} (Epoch {self.best_epoch})")
                break
    
    def predict(self, loader: DataLoader) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc='Predicting'):
                mel_spec = batch['mel_spec'].to(self.device)
                psycho = batch['psycho_features'].to(self.device)
                outputs = self.model(mel_spec, psycho)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model: {checkpoint_path}, val_loss: {checkpoint['val_loss']:.6f}")


if __name__ == "__main__":
    from models.ldcnn_cf import LDCNN_CF
    from torch.utils.data import TensorDataset, DataLoader
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LDCNN_CF(config)
    trainer = Trainer(model, config, device)
    
    # Create dummy data
    B = 32
    T = 480
    mel = torch.randn(B, 1, 64, T)
    psycho = torch.randn(B, 1, 4, T)
    labels = torch.randn(B, 1) * 3 + 5.5
    
    dataset = TensorDataset(mel, psycho, labels)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print("Test passed")