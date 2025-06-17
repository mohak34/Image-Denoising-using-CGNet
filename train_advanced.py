#!/usr/bin/env python3
"""
Advanced Image Denoising Training Script

This script provides comprehensive training capabilities for state-of-the-art image denoising models.
Features:
- Multiple model architectures (DnCNN, FFDNet, RIDNet, NAFNet, RCAN, DRUNet, BRDNet, HINet)
- Advanced loss functions (MSE, L1, Charbonnier, Mixed, Frequency, Edge)
- Mixed precision training
- Advanced data augmentation
- Comprehensive evaluation metrics
- Tensorboard and WandB logging support
- Model comparison and analysis

Usage:
    python train_advanced.py --model NAFNet --dataset mnist --epochs 50 --noise_type gaussian --noise_level 0.25
"""

import argparse
import torch
import torch.nn as nn
import time
import json
from pathlib import Path

from models import (DnCNN, UNet, RCAN, NAFNet, DRUNet, FFDNet, RIDNet, BRDNet, HINet)
from data_utils import load_advanced_dataset
from trainer import AdvancedTrainer, get_advanced_config

def parse_arguments():
    parser = argparse.ArgumentParser(description='Advanced Image Denoising Training')
    
    # Model selection
    parser.add_argument('--model', type=str, default='NAFNet',
                       choices=['DnCNN', 'UNet', 'RCAN', 'NAFNet', 'DRUNet', 
                               'FFDNet', 'RIDNet', 'BRDNet', 'HINet'],
                       help='Model architecture to use')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10'],
                       help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    
    # Noise configuration
    parser.add_argument('--noise_type', type=str, default='gaussian',
                       choices=['gaussian', 'poisson', 'speckle', 'salt_pepper', 
                               'impulse', 'uniform', 'mixed'],
                       help='Type of noise to add')
    parser.add_argument('--noise_level', type=float, default=0.25,
                       help='Noise level (0.0-1.0)')
    parser.add_argument('--use_real_noise', action='store_true',
                       help='Use realistic camera noise model')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer to use')
    parser.add_argument('--loss_function', type=str, default='charbonnier',
                       choices=['mse', 'l1', 'charbonnier', 'mixed', 'frequency', 'edge'],
                       help='Loss function to use')
    
    # Advanced features
    parser.add_argument('--use_augmentation', action='store_true', default=True,
                       help='Use data augmentation')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping threshold')
    
    # Logging and evaluation
    parser.add_argument('--use_tensorboard', action='store_true', default=True,
                       help='Use tensorboard logging')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--save_samples', action='store_true', default=True,
                       help='Save sample denoising results')
    
    # Hardware configuration
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment')
    
    return parser.parse_args()

def get_model(model_name, channels):
    """Initialize model based on name and number of channels"""
    
    if model_name == 'DnCNN':
        return DnCNN(channels=channels, num_layers=17, features=64)
    elif model_name == 'UNet':
        return UNet(n_channels=channels, n_classes=channels)
    elif model_name == 'RCAN':
        return RCAN(n_channels=channels, n_feats=64, n_blocks=10, reduction=16)
    elif model_name == 'NAFNet':
        return NAFNet(img_channel=channels, width=32, middle_blk_num=12)
    elif model_name == 'DRUNet':
        return DRUNet(in_nc=channels, out_nc=channels, nc=[64, 128, 256, 512], nb=4)
    elif model_name == 'FFDNet':
        return FFDNet(num_input_channels=channels, num_feature_maps=64, num_layers=15)
    elif model_name == 'RIDNet':
        return RIDNet(in_channels=channels, out_channels=channels, feature_channels=64, num_blocks=4)
    elif model_name == 'BRDNet':
        return BRDNet(in_channels=channels, out_channels=channels, num_features=64, num_blocks=20)
    elif model_name == 'HINet':
        return HINet(in_chn=channels, wf=64, depth=5)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def setup_experiment(args):
    """Setup experiment directory and configuration"""
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"{args.model}_{args.dataset}_{args.noise_type}_{timestamp}"
    
    # Create experiment directory
    exp_dir = output_dir / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = exp_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    return exp_dir

def main():
    args = parse_arguments()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 Starting Advanced Image Denoising Training")
    print(f"📱 Device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Setup experiment
    exp_dir = setup_experiment(args)
    print(f"📁 Experiment directory: {exp_dir}")
    
    # Load dataset
    print(f"\\n📊 Loading dataset: {args.dataset}")
    print(f"🎯 Noise type: {args.noise_type}, Level: {args.noise_level}")
    
    train_loader, val_loader, test_loader, channels = load_advanced_dataset(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
        use_augmentation=args.use_augmentation,
        use_real_noise=args.use_real_noise
    )
    
    print(f"📈 Training batches: {len(train_loader)}")
    print(f"📉 Validation batches: {len(val_loader)}")
    print(f"🧪 Test batches: {len(test_loader)}")
    print(f"🎨 Image channels: {channels}")
    
    # Initialize model
    print(f"\\n🧠 Initializing model: {args.model}")
    model = get_model(args.model, channels)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {num_params:,}")
    print(f"💾 Model size: {num_params * 4 / 1e6:.2f}MB")\n    
    # Get configuration
    config = get_advanced_config(args.model)
    
    # Override with command line arguments
    config.update({
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'optimizer': args.optimizer,
        'loss_function': args.loss_function,
        'mixed_precision': args.mixed_precision,
        'grad_clip': args.grad_clip,
        'use_tensorboard': args.use_tensorboard,
        'use_wandb': args.use_wandb,
        'model_name': args.model,
        'dataset': args.dataset,
        'noise_type': args.noise_type,
        'noise_level': args.noise_level
    })
    
    print(f"⚙️ Training configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Initialize trainer
    print(f"\\n🎓 Initializing advanced trainer...")
    trainer = AdvancedTrainer(model, train_loader, val_loader, test_loader, device, config)
    
    # Start training
    print(f"\\n🏋️ Starting training for {config['epochs']} epochs...")
    start_time = time.time()
    
    training_results = trainer.train()
    
    total_time = time.time() - start_time
    print(f"\\n✅ Training completed in {total_time:.1f}s")
    
    # Test evaluation
    print(f"\\n🧪 Evaluating on test set...")
    test_results = trainer.test()
    
    # Save results
    results = {
        'training_results': training_results,
        'test_results': test_results,
        'config': config,
        'model_info': {
            'name': args.model,
            'parameters': num_params,
            'size_mb': num_params * 4 / 1e6
        }
    }
    
    results_path = exp_dir / 'results.json'
    with open(results_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            else:
                return obj
        
        json.dump(convert_numpy(results), f, indent=2)
    
    # Visualize results
    if args.save_samples:
        print(f"\\n🖼️ Generating sample visualizations...")
        trainer.visualize_results(num_samples=6)
    
    # Print final summary
    print(f"\\n🏆 FINAL RESULTS:")
    print(f"📈 Best Validation PSNR: {training_results['best_psnr']:.2f}dB")
    print(f"🧪 Test PSNR: {test_results['psnr']:.2f}dB")
    print(f"🧪 Test SSIM: {test_results['ssim']:.4f}")
    print(f"⏱️ Total Training Time: {total_time:.1f}s")
    print(f"📁 Results saved to: {exp_dir}")
    
    print(f"\\n🎉 Experiment completed successfully!")

if __name__ == '__main__':
    main()
