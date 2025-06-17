import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
import numpy as np
from PIL import Image
import os
import kornia
import cv2
from skimage import metrics
import torchvision

class DenoisingDataset(Dataset):
    def __init__(self, dataset, noise_type='gaussian', noise_level=0.2):
        self.dataset = dataset
        self.noise_type = noise_type
        self.noise_level = noise_level
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        clean = img.clone()
        
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(img) * self.noise_level
            noisy = img + noise
        elif self.noise_type == 'poisson':
            noisy = torch.poisson(img * 255) / 255.0
        elif self.noise_type == 'speckle':
            noise = torch.randn_like(img) * self.noise_level
            noisy = img + img * noise
        elif self.noise_type == 'salt_pepper':
            noisy = img.clone()
            salt = torch.rand_like(img) < self.noise_level / 2
            pepper = torch.rand_like(img) < self.noise_level / 2
            noisy[salt] = 1.0
            noisy[pepper] = 0.0
        else:
            raise ValueError(f"Noise type {self.noise_type} not supported")
            
        noisy = torch.clamp(noisy, 0, 1)
        return noisy, clean, label

class AdvancedDenoisingDataset(Dataset):
    def __init__(self, dataset, noise_type='gaussian', noise_level=0.2, use_augmentation=True):
        self.dataset = dataset
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.use_augmentation = use_augmentation
        
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ]) if use_augmentation else None
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        if self.augmentation and torch.rand(1) < 0.5:
            img = self.augmentation(img)
            
        clean = img.clone()
        
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(img) * self.noise_level
            noisy = img + noise
        elif self.noise_type == 'poisson':
            img_scaled = torch.clamp(img * 255, 0, 255)
            noisy = torch.poisson(img_scaled) / 255.0
        elif self.noise_type == 'speckle':
            noise = torch.randn_like(img) * self.noise_level
            noisy = img + img * noise
        elif self.noise_type == 'salt_pepper':
            noisy = img.clone()
            prob = torch.rand_like(img)
            noisy[prob < self.noise_level / 2] = 0.0
            noisy[prob > 1 - self.noise_level / 2] = 1.0
        elif self.noise_type == 'impulse':
            noisy = img.clone()
            mask = torch.rand_like(img) < self.noise_level
            noisy[mask] = torch.rand_like(noisy[mask])
        elif self.noise_type == 'uniform':
            noise = (torch.rand_like(img) - 0.5) * 2 * self.noise_level
            noisy = img + noise
        elif self.noise_type == 'mixed':
            if torch.rand(1) < 0.5:
                noise = torch.randn_like(img) * self.noise_level
                noisy = img + noise
            else:
                prob = torch.rand_like(img)
                noisy = img.clone()
                noisy[prob < self.noise_level / 4] = 0.0
                noisy[prob > 1 - self.noise_level / 4] = 1.0
        else:
            raise ValueError(f"Noise type {self.noise_type} not supported")
            
        noisy = torch.clamp(noisy, 0, 1)
        return noisy, clean, label


class RealNoiseDataset(Dataset):
    def __init__(self, dataset, noise_level=0.2):
        self.dataset = dataset
        self.noise_level = noise_level
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        clean = img.clone()
        
        # Simulate real camera noise
        img_np = img.permute(1, 2, 0).numpy()
        
        # Add shot noise (Poisson)
        img_scaled = np.clip(img_np * 255, 0, 255)
        shot_noise = np.random.poisson(img_scaled) / 255.0
        
        # Add read noise (Gaussian)
        read_noise = np.random.normal(0, self.noise_level * 0.1, img_np.shape)
        
        # Add dark current noise
        dark_noise = np.random.exponential(self.noise_level * 0.05, img_np.shape)
        
        # Combine noises
        noisy_np = shot_noise + read_noise + dark_noise
        noisy_np = np.clip(noisy_np, 0, 1)
        
        noisy = torch.from_numpy(noisy_np).permute(2, 0, 1).float()
        return noisy, clean, label


def load_dataset(dataset_name='mnist', batch_size=64, noise_type='gaussian', noise_level=0.2, train_split=0.8):
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        full_dataset = MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = MNIST('./data', train=False, download=True, transform=transform)
        channels = 1
        
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        full_dataset = CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = CIFAR10('./data', train=False, download=True, transform=transform)
        channels = 3
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_dataset = DenoisingDataset(train_dataset, noise_type, noise_level)
    val_dataset = DenoisingDataset(val_dataset, noise_type, noise_level)
    test_dataset = DenoisingDataset(test_dataset, noise_type, noise_level)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader, channels

def add_noise_to_image(image, noise_type='gaussian', noise_level=0.2):
    if noise_type == 'gaussian':
        noise = torch.randn_like(image) * noise_level
        noisy = image + noise
    elif noise_type == 'poisson':
        noisy = torch.poisson(image * 255) / 255.0
    elif noise_type == 'speckle':
        noise = torch.randn_like(image) * noise_level
        noisy = image + image * noise
    elif noise_type == 'salt_pepper':
        noisy = image.clone()
        salt = torch.rand_like(image) < noise_level / 2
        pepper = torch.rand_like(image) < noise_level / 2
        noisy[salt] = 1.0
        noisy[pepper] = 0.0
    else:
        raise ValueError(f"Noise type {noise_type} not supported")
    
    return torch.clamp(noisy, 0, 1)

def calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)

def calculate_ssim(img1, img2, window_size=11, size_average=True):
    def gaussian_window(size, sigma):
        gauss = torch.Tensor([torch.exp(torch.tensor(-(x - size//2)**2/float(2*sigma**2))) for x in range(size)])
        return gauss/gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return torch.mean(loss)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True).features
        self.features = nn.Sequential(*list(vgg.children())[:16])
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        pred_features = self.features(pred.repeat(1, 3, 1, 1) if pred.size(1) == 1 else pred)
        target_features = self.features(target.repeat(1, 3, 1, 1) if target.size(1) == 1 else target)
        return F.mse_loss(pred_features, target_features)

def preprocess_batch(batch, device):
    noisy, clean, labels = batch
    noisy = noisy.to(device, non_blocking=True)
    clean = clean.to(device, non_blocking=True)
    return noisy, clean, labels

def create_degradation_model(degradation_type='blur_noise'):
    """Create realistic image degradation models"""
    degradations = []
    
    if 'blur' in degradation_type:
        # Gaussian blur
        blur_kernel = kornia.filters.get_gaussian_kernel2d((5, 5), (1.0, 1.0))
        degradations.append(lambda x: kornia.filters.filter2d(x, blur_kernel))
    
    if 'noise' in degradation_type:
        # Additive noise
        degradations.append(lambda x: x + torch.randn_like(x) * 0.1)
    
    if 'jpeg' in degradation_type:
        # JPEG compression artifacts simulation
        degradations.append(lambda x: simulate_jpeg_compression(x, quality=70))
    
    return degradations


def simulate_jpeg_compression(img, quality=80):
    """Simulate JPEG compression artifacts"""
    # Convert to numpy for OpenCV processing
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # Encode and decode with JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img_np, encode_param)
    decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    
    # Convert back to tensor
    result = torch.from_numpy(decimg.astype(np.float32) / 255.0).permute(2, 0, 1)
    return result


def calculate_metrics(pred, target):
    """Calculate comprehensive image quality metrics"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Ensure proper shape for metrics calculation
    if pred_np.ndim == 4:  # Batch dimension
        pred_np = pred_np[0]
        target_np = target_np[0]
    
    if pred_np.shape[0] in [1, 3]:  # Channel first
        pred_np = pred_np.transpose(1, 2, 0)
        target_np = target_np.transpose(1, 2, 0)
    
    if pred_np.shape[-1] == 1:  # Grayscale
        pred_np = pred_np.squeeze(-1)
        target_np = target_np.squeeze(-1)
    
    # Calculate metrics
    psnr = metrics.peak_signal_noise_ratio(target_np, pred_np, data_range=1.0)
    ssim = metrics.structural_similarity(target_np, pred_np, data_range=1.0, multichannel=len(pred_np.shape)==3)
    
    # Mean Squared Error
    mse = np.mean((pred_np - target_np) ** 2)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(pred_np - target_np))
    
    return {
        'PSNR': psnr,
        'SSIM': ssim,
        'MSE': mse,
        'MAE': mae
    }


class MixedLoss(nn.Module):
    def __init__(self, l1_weight=1.0, l2_weight=1.0, perceptual_weight=0.1, ssim_weight=0.1):
        super(MixedLoss, self).__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        l2 = self.l2_loss(pred, target)
        
        # SSIM loss (1 - SSIM for minimization)
        ssim = 1 - calculate_ssim(pred, target)
        
        # Perceptual loss
        perceptual = self.perceptual_loss(pred, target)
        
        total_loss = (self.l1_weight * l1 + 
                     self.l2_weight * l2 + 
                     self.perceptual_weight * perceptual + 
                     self.ssim_weight * ssim)
        
        return total_loss


class FrequencyLoss(nn.Module):
    def __init__(self):
        super(FrequencyLoss, self).__init__()
        
    def forward(self, pred, target):
        # Convert to frequency domain
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        # Calculate loss in frequency domain
        freq_loss = F.mse_loss(pred_fft.real, target_fft.real) + F.mse_loss(pred_fft.imag, target_fft.imag)
        
        return freq_loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
    def forward(self, pred, target):
        if pred.is_cuda:
            self.sobel_x = self.sobel_x.cuda()
            self.sobel_y = self.sobel_y.cuda()
        
        # Calculate gradients
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)
        
        # Edge magnitude
        pred_edges = torch.sqrt(pred_grad_x**2 + pred_grad_y**2)
        target_edges = torch.sqrt(target_grad_x**2 + target_grad_y**2)
        
        return F.mse_loss(pred_edges, target_edges)


def load_advanced_dataset(dataset_name='mnist', batch_size=64, noise_type='gaussian', 
                         noise_level=0.2, train_split=0.8, use_augmentation=True, 
                         use_real_noise=False):
    """Load dataset with advanced preprocessing and augmentation"""
    
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        full_dataset = MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = MNIST('./data', train=False, download=True, transform=transform)
        channels = 1
        
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        full_dataset = CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = CIFAR10('./data', train=False, download=True, transform=transform)
        channels = 3
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    # Split train/validation
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Apply noise and augmentation
    if use_real_noise:
        train_dataset = RealNoiseDataset(train_dataset, noise_level)
        val_dataset = RealNoiseDataset(val_dataset, noise_level)
        test_dataset = RealNoiseDataset(test_dataset, noise_level)
    else:
        train_dataset = AdvancedDenoisingDataset(train_dataset, noise_type, noise_level, use_augmentation)
        val_dataset = AdvancedDenoisingDataset(val_dataset, noise_type, noise_level, False)
        test_dataset = AdvancedDenoisingDataset(test_dataset, noise_type, noise_level, False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True, persistent_workers=True)
    
    return train_loader, val_loader, test_loader, channels


def create_multi_scale_patches(image, scales=[1.0, 0.8, 0.6]):
    """Create multi-scale patches for training"""
    patches = []
    for scale in scales:
        if scale != 1.0:
            size = (int(image.shape[-2] * scale), int(image.shape[-1] * scale))
            scaled = F.interpolate(image, size=size, mode='bilinear', align_corners=False)
            patches.append(scaled)
        else:
            patches.append(image)
    return patches
