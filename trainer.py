import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
import wandb
from torch.utils.tensorboard import SummaryWriter
from data_utils import (calculate_psnr, calculate_ssim, CharbonnierLoss, preprocess_batch, 
                       calculate_metrics, MixedLoss, FrequencyLoss, EdgeLoss)

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
        if config['loss_function'] == 'mse':
            self.criterion = nn.MSELoss()
        elif config['loss_function'] == 'l1':
            self.criterion = nn.L1Loss()
        elif config['loss_function'] == 'charbonnier':
            self.criterion = CharbonnierLoss()
        else:
            self.criterion = nn.MSELoss()
            
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
            
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['epochs'])
        
        self.train_losses = []
        self.val_losses = []
        self.val_psnrs = []
        self.val_ssims = []
        
    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            noisy, clean, _ = preprocess_batch(batch, self.device)
            
            self.optimizer.zero_grad()
            output = self.model(noisy)
            loss = self.criterion(output, clean)
            loss.backward()
            
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix(loss=f"{loss.item():.6f}")
            
        return epoch_loss / num_batches
    
    def validate_epoch(self):
        self.model.eval()
        val_loss = 0
        val_psnr = 0
        val_ssim = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                noisy, clean, _ = preprocess_batch(batch, self.device)
                
                output = self.model(noisy)
                loss = self.criterion(output, clean)
                
                val_loss += loss.item()
                val_psnr += calculate_psnr(output, clean).item()
                val_ssim += calculate_ssim(output, clean).item()
                num_batches += 1
                
        return val_loss / num_batches, val_psnr / num_batches, val_ssim / num_batches
    
    def train(self):
        best_psnr = 0
        start_time = time.time()
        
        for epoch in range(1, self.config['epochs'] + 1):
            print(f"\nEpoch {epoch}/{self.config['epochs']}")
            
            train_loss = self.train_epoch()
            val_loss, val_psnr, val_ssim = self.validate_epoch()
            
            self.scheduler.step()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_psnrs.append(val_psnr)
            self.val_ssims.append(val_ssim)
            
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}, PSNR: {val_psnr:.2f}dB, SSIM: {val_ssim:.4f}")
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                self.save_checkpoint(epoch, best=True)
                
            if epoch % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch)
                
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Best validation PSNR: {best_psnr:.2f}dB")
        
    def test(self):
        self.model.eval()
        test_loss = 0
        test_psnr = 0
        test_ssim = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                noisy, clean, _ = preprocess_batch(batch, self.device)
                
                output = self.model(noisy)
                loss = self.criterion(output, clean)
                
                test_loss += loss.item()
                test_psnr += calculate_psnr(output, clean).item()
                test_ssim += calculate_ssim(output, clean).item()
                num_batches += 1
                
        test_loss /= num_batches
        test_psnr /= num_batches
        test_ssim /= num_batches
        
        print(f"Test Loss: {test_loss:.6f}")
        print(f"Test PSNR: {test_psnr:.2f}dB")
        print(f"Test SSIM: {test_ssim:.4f}")
        
        return test_loss, test_psnr, test_ssim
    
    def save_checkpoint(self, epoch, best=False):
        os.makedirs('checkpoints', exist_ok=True)
        suffix = '_best' if best else f'_epoch_{epoch}'
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_psnrs': self.val_psnrs,
            'val_ssims': self.val_ssims,
            'config': self.config
        }
        
        filename = f"checkpoints/{self.config['model_name']}{suffix}.pth"
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_psnrs = checkpoint.get('val_psnrs', [])
        self.val_ssims = checkpoint.get('val_ssims', [])
        
        return checkpoint['epoch']
    
    def plot_metrics(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        axes[0, 0].plot(epochs, self.train_losses, label='Train Loss')
        axes[0, 0].plot(epochs, self.val_losses, label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(epochs, self.val_psnrs)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].set_title('Validation PSNR')
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(epochs, self.val_ssims)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].set_title('Validation SSIM')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(epochs, [lr_scheduler.get_last_lr()[0] for lr_scheduler in [self.scheduler] * len(epochs)])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

class AdvancedTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
        # Initialize loss function
        self.criterion = self._get_loss_function(config['loss_function'])
            
        # Initialize optimizer
        self.optimizer = self._get_optimizer(config)
            
        # Initialize scheduler
        self.scheduler = self._get_scheduler(config)
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' and config.get('mixed_precision', True) else None
        
        # Logging
        self.use_wandb = config.get('use_wandb', False)
        self.use_tensorboard = config.get('use_tensorboard', False)
        
        if self.use_wandb:
            try:
                import wandb
                wandb.init(project="image-denoising", config=config)
            except ImportError:
                self.use_wandb = False
                
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(f"runs/{config.get('model_name', 'model')}_{time.strftime('%Y%m%d_%H%M%S')}")
            except ImportError:
                self.use_tensorboard = False
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_psnrs = []
        self.val_ssims = []
        self.best_psnr = 0
        self.best_model_path = f"best_{config.get('model_name', 'model')}.pth"
        
    def _get_loss_function(self, loss_type):
        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'l1':
            return nn.L1Loss()
        elif loss_type == 'charbonnier':
            return CharbonnierLoss()
        elif loss_type == 'mixed':
            return MixedLoss()
        elif loss_type == 'frequency':
            return FrequencyLoss()
        elif loss_type == 'edge':
            return EdgeLoss()
        else:
            return nn.MSELoss()
    
    def _get_optimizer(self, config):
        if config['optimizer'] == 'adam':
            return optim.Adam(self.model.parameters(), lr=config['learning_rate'], 
                            weight_decay=config.get('weight_decay', 1e-4))
        elif config['optimizer'] == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=config['learning_rate'], 
                             weight_decay=config.get('weight_decay', 1e-4))
        elif config['optimizer'] == 'sgd':
            return optim.SGD(self.model.parameters(), lr=config['learning_rate'], 
                           momentum=config.get('momentum', 0.9), 
                           weight_decay=config.get('weight_decay', 1e-4))
        else:
            return optim.Adam(self.model.parameters(), lr=config['learning_rate'])
    
    def _get_scheduler(self, config):
        scheduler_type = config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['epochs'])
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=config.get('step_size', 30), gamma=0.1)
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=10)
        else:
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['epochs'])
    
    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        for batch_idx, batch in enumerate(pbar):
            noisy, clean, _ = preprocess_batch(batch, self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    output = self.model(noisy)
                    loss = self.criterion(output, clean)
                
                self.scaler.scale(loss).backward()
                
                if self.config.get('grad_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(noisy)
                loss = self.criterion(output, clean)
                loss.backward()
                
                if self.config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                
                self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log batch metrics
            if self.use_wandb and batch_idx % 100 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch": epoch,
                    "batch": batch_idx
                })
            
            if self.use_tensorboard and batch_idx % 100 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Loss/Batch', loss.item(), global_step)
            
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        
        return epoch_loss / num_batches
    
    def validate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0
        val_psnr = 0
        val_ssim = 0
        val_mse = 0
        val_mae = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                noisy, clean, _ = preprocess_batch(batch, self.device)
                
                output = self.model(noisy)
                loss = self.criterion(output, clean)
                
                val_loss += loss.item()
                
                # Calculate comprehensive metrics
                metrics = calculate_metrics(output, clean)
                val_psnr += metrics['PSNR']
                val_ssim += metrics['SSIM']
                val_mse += metrics['MSE']
                val_mae += metrics['MAE']
                num_batches += 1
        
        # Average metrics
        avg_metrics = {
            'loss': val_loss / num_batches,
            'psnr': val_psnr / num_batches,
            'ssim': val_ssim / num_batches,
            'mse': val_mse / num_batches,
            'mae': val_mae / num_batches
        }
        
        # Log validation metrics
        if self.use_wandb:
            wandb.log({
                "val_loss": avg_metrics['loss'],
                "val_psnr": avg_metrics['psnr'],
                "val_ssim": avg_metrics['ssim'],
                "val_mse": avg_metrics['mse'],
                "val_mae": avg_metrics['mae'],
                "epoch": epoch
            })
        
        if self.use_tensorboard:
            self.writer.add_scalar('Loss/Validation', avg_metrics['loss'], epoch)
            self.writer.add_scalar('PSNR/Validation', avg_metrics['psnr'], epoch)
            self.writer.add_scalar('SSIM/Validation', avg_metrics['ssim'], epoch)
        
        return avg_metrics
    
    def train(self):
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            
            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.val_psnrs.append(val_metrics['psnr'])
            self.val_ssims.append(val_metrics['ssim'])
            
            # Save best model
            if val_metrics['psnr'] > self.best_psnr:
                self.best_psnr = val_metrics['psnr']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_psnr': self.best_psnr,
                    'config': self.config
                }, self.best_model_path)
                print(f"💾 New best model saved! PSNR: {self.best_psnr:.2f}dB")
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_metrics['loss']:.6f}")
            print(f"Val PSNR: {val_metrics['psnr']:.2f}dB, Val SSIM: {val_metrics['ssim']:.4f}")
            print(f"Val MSE: {val_metrics['mse']:.6f}, Val MAE: {val_metrics['mae']:.6f}")
            
            # Early stopping
            if self.config.get('early_stopping', False):
                if self._early_stopping_check(epoch):
                    print("Early stopping triggered!")
                    break
        
        training_time = time.time() - start_time
        print(f"\n🎉 Training completed in {training_time:.1f}s")
        print(f"🏆 Best PSNR: {self.best_psnr:.2f}dB")
        
        # Close logging
        if self.use_wandb:
            wandb.finish()
        if self.use_tensorboard:
            self.writer.close()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_psnrs': self.val_psnrs,
            'val_ssims': self.val_ssims,
            'best_psnr': self.best_psnr,
            'training_time': training_time
        }
    
    def _early_stopping_check(self, current_epoch):
        patience = self.config.get('early_stopping_patience', 15)
        if current_epoch < patience:
            return False
        
        recent_psnrs = self.val_psnrs[-patience:]
        return all(psnr <= self.best_psnr for psnr in recent_psnrs)
    
    def test(self):
        """Comprehensive testing on test set"""
        checkpoint = torch.load(self.best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        test_metrics = {'loss': 0, 'psnr': 0, 'ssim': 0, 'mse': 0, 'mae': 0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                noisy, clean, _ = preprocess_batch(batch, self.device)
                output = self.model(noisy)
                
                loss = self.criterion(output, clean)
                test_metrics['loss'] += loss.item()
                
                metrics = calculate_metrics(output, clean)
                test_metrics['psnr'] += metrics['PSNR']
                test_metrics['ssim'] += metrics['SSIM']
                test_metrics['mse'] += metrics['MSE']
                test_metrics['mae'] += metrics['MAE']
                num_batches += 1
        
        # Average metrics
        for key in test_metrics:
            test_metrics[key] /= num_batches
        
        print(f"\n📊 TEST RESULTS:")
        print(f"Test PSNR: {test_metrics['psnr']:.2f}dB")
        print(f"Test SSIM: {test_metrics['ssim']:.4f}")
        print(f"Test MSE: {test_metrics['mse']:.6f}")
        print(f"Test MAE: {test_metrics['mae']:.6f}")
        
        return test_metrics
    
    def visualize_results(self, num_samples=6):
        """Visualize denoising results"""
        checkpoint = torch.load(self.best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        batch = next(iter(self.test_loader))
        noisy, clean, _ = preprocess_batch(batch, self.device)
        
        with torch.no_grad():
            denoised = self.model(noisy[:num_samples])
        
        fig, axes = plt.subplots(3, num_samples, figsize=(15, 8))
        
        for i in range(num_samples):
            # Noisy
            if noisy.shape[1] == 1:  # Grayscale
                axes[0, i].imshow(noisy[i].cpu().squeeze(), cmap='gray')
                axes[1, i].imshow(clean[i].cpu().squeeze(), cmap='gray')
                axes[2, i].imshow(denoised[i].cpu().squeeze(), cmap='gray')
            else:  # RGB
                axes[0, i].imshow(noisy[i].cpu().permute(1, 2, 0))
                axes[1, i].imshow(clean[i].cpu().permute(1, 2, 0))
                axes[2, i].imshow(denoised[i].cpu().permute(1, 2, 0))
            
            # Calculate PSNR for this sample
            psnr = calculate_metrics(denoised[i:i+1], clean[i:i+1])['PSNR']
            
            axes[0, i].set_title(f'Noisy {i+1}')
            axes[1, i].set_title(f'Clean {i+1}')
            axes[2, i].set_title(f'Denoised {i+1}\nPSNR: {psnr:.2f}dB')
            
            for j in range(3):
                axes[j, i].axis('off')
        
        plt.tight_layout()
        plt.show()


def get_default_config(model_name):
    return {
        'model_name': model_name,
        'epochs': 50,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'optimizer': 'adamw',
        'loss_function': 'charbonnier',
        'grad_clip': 1.0,
        'save_interval': 10
    }

def get_advanced_config(model_name):
    """Get advanced configuration for different models"""
    base_config = {
        'epochs': 50,
        'learning_rate': 2e-4,
        'batch_size': 32,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'mixed_precision': True,
        'early_stopping': True,
        'early_stopping_patience': 15,
        'use_wandb': False,
        'use_tensorboard': True,
        'model_name': model_name
    }
    
    # Model-specific configurations
    model_configs = {
        'DnCNN': {
            'learning_rate': 1e-3,
            'loss_function': 'mse',
            'epochs': 40
        },
        'FFDNet': {
            'learning_rate': 1e-3,
            'loss_function': 'charbonnier',
            'epochs': 50
        },
        'RIDNet': {
            'learning_rate': 2e-4,
            'loss_function': 'mixed',
            'epochs': 60
        },
        'NAFNet': {
            'learning_rate': 2e-4,
            'loss_function': 'charbonnier',
            'epochs': 80
        },
        'RCAN': {
            'learning_rate': 2e-4,
            'loss_function': 'mixed',
            'epochs': 70
        },
        'DRUNet': {
            'learning_rate': 2e-4,
            'loss_function': 'charbonnier',
            'epochs': 60
        },
        'BRDNet': {
            'learning_rate': 1e-3,
            'loss_function': 'mse',
            'epochs': 50
        },
        'HINet': {
            'learning_rate': 2e-4,
            'loss_function': 'mixed',
            'epochs': 70
        }
    }
    
    if model_name in model_configs:
        base_config.update(model_configs[model_name])
    else:
        base_config['loss_function'] = 'mse'
    
    return base_config
