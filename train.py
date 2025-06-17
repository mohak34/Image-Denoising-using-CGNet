#!/usr/bin/env python3

import torch
import argparse
from models import DnCNN, UNet, RCAN, NAFNet, DRUNet
from data_utils import load_dataset
from trainer import Trainer, get_default_config

def main():
    parser = argparse.ArgumentParser(description='Image Denoising with State-of-the-Art Models')
    parser.add_argument('--model', type=str, default='DnCNN', 
                       choices=['DnCNN', 'UNet', 'RCAN', 'NAFNet', 'DRUNet'],
                       help='Model architecture to use')
    parser.add_argument('--dataset', type=str, default='mnist', 
                       choices=['mnist', 'cifar10'],
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, 
                       help='Learning rate')
    parser.add_argument('--noise_level', type=float, default=0.2, 
                       help='Noise level for training')
    parser.add_argument('--noise_type', type=str, default='gaussian', 
                       choices=['gaussian', 'poisson', 'speckle', 'salt_pepper'],
                       help='Type of noise to add')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Training {args.model} on {args.dataset} dataset")
    print(f"Noise: {args.noise_type} (level: {args.noise_level})")
    
    train_loader, val_loader, test_loader, channels = load_dataset(
        args.dataset, args.batch_size, args.noise_type, args.noise_level
    )
    
    if args.model == 'DnCNN':
        model = DnCNN(channels=channels, num_layers=17, features=64)
    elif args.model == 'UNet':
        model = UNet(n_channels=channels, n_classes=channels)
    elif args.model == 'RCAN':
        model = RCAN(n_channels=channels, n_feats=64, n_blocks=10, reduction=16)
    elif args.model == 'NAFNet':
        model = NAFNet(img_channel=channels, width=32, middle_blk_num=12)
    elif args.model == 'DRUNet':
        model = DRUNet(in_nc=channels, out_nc=channels)
    
    config = get_default_config(args.model)
    config.update({
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'dataset': args.dataset,
        'noise_type': args.noise_type,
        'noise_level': args.noise_level
    })
    
    trainer = Trainer(model, train_loader, val_loader, test_loader, device, config)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer.train()
    test_loss, test_psnr, test_ssim = trainer.test()
    
    print(f"\nFinal Results:")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test PSNR: {test_psnr:.2f} dB")
    print(f"Test SSIM: {test_ssim:.4f}")

if __name__ == "__main__":
    main()
