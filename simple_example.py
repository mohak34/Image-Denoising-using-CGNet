#!/usr/bin/env python3
"""
Simple Example: Advanced Image Denoising

This script demonstrates how to use the advanced image denoising models
with minimal setup. Perfect for getting started quickly.

Usage:
    python simple_example.py
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Import our models and utilities
from models import DnCNN, NAFNet, RIDNet
from data_utils import load_dataset, calculate_psnr, calculate_ssim
from trainer import Trainer, get_default_config

def simple_training_example():
    """Simple training example with DnCNN"""
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading MNIST dataset...")
    train_loader, val_loader, test_loader, channels = load_dataset(
        dataset_name='mnist',
        batch_size=32,
        noise_type='gaussian',
        noise_level=0.2
    )
    
    # Initialize model
    print("Initializing DnCNN model...")
    model = DnCNN(channels=channels, num_layers=17, features=64)
    
    # Get training configuration
    config = get_default_config('DnCNN')
    config.update({
        'epochs': 10,  # Quick training for demo
        'learning_rate': 1e-3,
        'batch_size': 32
    })
    
    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader, device, config)
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Test the model
    print("Testing model...")
    test_loss, test_psnr, test_ssim = trainer.test()
    
    print(f"Test Results:")
    print(f"  PSNR: {test_psnr:.2f}dB")
    print(f"  SSIM: {test_ssim:.4f}")
    
    return trainer

def compare_models_example():
    """Compare different models on the same task"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    train_loader, val_loader, test_loader, channels = load_dataset(
        dataset_name='mnist',
        batch_size=32,
        noise_type='gaussian',
        noise_level=0.25
    )
    
    # Define models to compare
    models_to_test = {
        'DnCNN': DnCNN(channels=channels, num_layers=17, features=64),
        'NAFNet': NAFNet(img_channel=channels, width=32, middle_blk_num=6),  # Smaller for demo
        'RIDNet': RIDNet(in_channels=channels, out_channels=channels, feature_channels=32, num_blocks=2)  # Smaller for demo
    }
    
    results = {}
    
    for model_name, model in models_to_test.items():
        print(f"\nTraining {model_name}...")
        
        # Get model-specific config
        config = get_default_config(model_name)
        config.update({
            'epochs': 5,  # Quick training for demo
            'batch_size': 32
        })
        
        # Train model
        trainer = Trainer(model, train_loader, val_loader, test_loader, device, config)
        trainer.train()
        
        # Test model
        test_loss, test_psnr, test_ssim = trainer.test()
        
        results[model_name] = {
            'psnr': test_psnr,
            'ssim': test_ssim,
            'parameters': sum(p.numel() for p in model.parameters())
        }
        
        print(f"{model_name} Results:")
        print(f"  PSNR: {test_psnr:.2f}dB")
        print(f"  SSIM: {test_ssim:.4f}")
        print(f"  Parameters: {results[model_name]['parameters']:,}")
    
    # Print comparison
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    for model_name, result in results.items():
        print(f"{model_name:<10}: PSNR = {result['psnr']:.2f}dB, "
              f"SSIM = {result['ssim']:.4f}, "
              f"Params = {result['parameters']:,}")
    
    return results

def visualize_denoising_example():
    """Visualize denoising results"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load a small dataset for visualization
    train_loader, val_loader, test_loader, channels = load_dataset(
        dataset_name='mnist',
        batch_size=8,
        noise_type='gaussian',
        noise_level=0.3
    )
    
    # Train a quick model
    model = DnCNN(channels=channels, num_layers=10, features=32)  # Smaller for speed
    config = get_default_config('DnCNN')
    config.update({'epochs': 5, 'batch_size': 8})
    
    trainer = Trainer(model, train_loader, val_loader, test_loader, device, config)
    print("Training model for visualization...")
    trainer.train()
    
    # Get a test batch
    model.eval()
    batch = next(iter(test_loader))
    noisy, clean, _ = batch
    noisy, clean = noisy.to(device), clean.to(device)
    
    # Denoise images
    with torch.no_grad():
        denoised = model(noisy)
    
    # Visualize results
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    
    for i in range(4):
        # Noisy image
        axes[0, i].imshow(noisy[i].cpu().squeeze(), cmap='gray')
        axes[0, i].set_title(f'Noisy {i+1}')
        axes[0, i].axis('off')
        
        # Clean image
        axes[1, i].imshow(clean[i].cpu().squeeze(), cmap='gray')
        axes[1, i].set_title(f'Clean {i+1}')
        axes[1, i].axis('off')
        
        # Denoised image
        axes[2, i].imshow(denoised[i].cpu().squeeze(), cmap='gray')
        
        # Calculate PSNR for this image
        psnr = calculate_psnr(denoised[i:i+1], clean[i:i+1]).item()
        ssim = calculate_ssim(denoised[i:i+1], clean[i:i+1]).item()
        
        axes[2, i].set_title(f'Denoised {i+1}\nPSNR: {psnr:.1f}dB')
        axes[2, i].axis('off')
    
    plt.suptitle('Image Denoising Results', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print(f"Average PSNR: {calculate_psnr(denoised, clean).item():.2f}dB")
    print(f"Average SSIM: {calculate_ssim(denoised, clean).item():.4f}")

def noise_robustness_example():
    """Test model robustness across different noise levels"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Testing noise robustness...")
    
    # Train a model on moderate noise
    train_loader, val_loader, test_loader, channels = load_dataset(
        dataset_name='mnist',
        batch_size=32,
        noise_type='gaussian',
        noise_level=0.2  # Training noise level
    )
    
    model = DnCNN(channels=channels, num_layers=15, features=48)
    config = get_default_config('DnCNN')
    config.update({'epochs': 8, 'batch_size': 32})
    
    trainer = Trainer(model, train_loader, val_loader, test_loader, device, config)
    print("Training model...")
    trainer.train()
    
    # Test on different noise levels
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = []
    
    model.eval()
    for noise_level in noise_levels:
        print(f"Testing on noise level {noise_level}...")
        
        # Create test set with this noise level
        _, _, test_loader_noise, _ = load_dataset(
            dataset_name='mnist',
            batch_size=32,
            noise_type='gaussian',
            noise_level=noise_level
        )
        
        total_psnr = 0
        total_ssim = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_loader_noise:
                noisy, clean, _ = batch
                noisy, clean = noisy.to(device), clean.to(device)
                
                output = model(noisy)
                
                total_psnr += calculate_psnr(output, clean).item()
                total_ssim += calculate_ssim(output, clean).item()
                num_batches += 1
        
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches
        results.append((noise_level, avg_psnr, avg_ssim))
        
        print(f"  PSNR: {avg_psnr:.2f}dB, SSIM: {avg_ssim:.4f}")
    
    # Plot results
    noise_levels_plot = [r[0] for r in results]
    psnrs = [r[1] for r in results]
    ssims = [r[2] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(noise_levels_plot, psnrs, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Noise Level')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('PSNR vs Noise Level')
    ax1.grid(True)
    
    ax2.plot(noise_levels_plot, ssims, 's-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Noise Level')
    ax2.set_ylabel('SSIM')
    ax2.set_title('SSIM vs Noise Level')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results

def main():
    """Run all examples"""
    
    print("🎉 Welcome to Advanced Image Denoising Examples!")
    print("="*50)
    
    examples = {
        '1': ('Simple Training Example', simple_training_example),
        '2': ('Model Comparison Example', compare_models_example),
        '3': ('Visualization Example', visualize_denoising_example),
        '4': ('Noise Robustness Example', noise_robustness_example),
        '5': ('Run All Examples', None)
    }
    
    print("Available examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '5':
        # Run all examples
        for key, (name, func) in examples.items():
            if func is not None:
                print(f"\n{'='*60}")
                print(f"Running: {name}")
                print('='*60)
                try:
                    func()
                    print(f"✅ {name} completed successfully!")
                except Exception as e:
                    print(f"❌ {name} failed: {e}")
    elif choice in examples and examples[choice][1] is not None:
        name, func = examples[choice]
        print(f"\nRunning: {name}")
        print('-'*50)
        try:
            func()
            print(f"\n✅ {name} completed successfully!")
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
    else:
        print("Invalid choice. Please run the script again.")
    
    print("\n🎉 Examples completed! Check out the advanced notebooks for more features.")

if __name__ == '__main__':
    main()
