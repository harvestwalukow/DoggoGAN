"""
Progressive Growing of GANs (ProGAN) for Dog Image Generation
================================================================
Implementation of ProGAN (Karras et al., 2018) with progressive resolution growing.
Starts training at 4x4 resolution and progressively doubles resolution up to 64x64.

Features:
- Progressive resolution growing (4→8→16→32→64)
- Equalized learning rate
- Pixel normalization
- Smooth transition between resolutions
- WGAN-GP loss for stable training
- Multi-GPU support with DataParallel

Usage on Kaggle:
    1. Create a new notebook and enable GPU
    2. Add the competition dataset: generative-dog-images
    3. Upload this script and run
"""

import os
import glob
import random
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass, field, asdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ProGANConfig:

    """Configuration for ProGAN training."""
    # Data
    data_path: str = "/kaggle/working/all-dogs"
    output_dir: str = "/kaggle/working/progan_output"
    target_image_size: int = 1024
    num_channels: int = 3
    
    # Model architecture
    latent_dim: int = 512
    # Base channels for each resolution: 4, 8, 16, 32, 64, 128, 256, 512, 1024
    # From specifications: 512, 512, 512, 512, 256, 128, 64, 32, 16
    channel_progression: List[int] = field(default_factory=lambda: [512, 512, 512, 512, 256, 128, 64, 32, 16])
    
    # Training Schedule (Paper strictness)
    # Map resolution (int) -> batch_size (int)
    batch_sizes: Dict[int, int] = field(default_factory=lambda: {
        4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4, 1024: 4
    })
    
    # Map resolution (int) -> kimg to train (int). 600k images as per typical reproduction, Paper uses more.
    # Paper: "we train the network for 600k images for each resolution"
    train_schedule: Dict[int, int] = field(default_factory=lambda: {
        4: 600, 8: 600, 16: 600, 32: 600, 64: 600, 128: 600, 256: 600, 512: 600, 1024: 600
    })
    
    learning_rate: float = 0.001
    beta1: float = 0.0
    beta2: float = 0.99
    
    # Progressive growing
    fade_in_kimg: int = 600  # Number of kimg for fade-in. Usually matches train schedule for simplicity or is half.
    start_resolution: int = 4
    
    # WGAN-GP + Drift
    gp_lambda: float = 10.0
    drift_epsilon: float = 0.001
    n_critic: int = 1
    
    # Logging and checkpoints
    save_interval_kimg: int = 200
    sample_interval_kimg: int = 100
    num_sample_images: int = 16
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Create output directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/samples", exist_ok=True)
        os.makedirs(f"{self.output_dir}/checkpoints", exist_ok=True)
        
        # Calculate number of progressive steps
        self.num_progressive_steps = len(self.channel_progression)




# =============================================================================
# Dataset
# =============================================================================

class ProgressiveDogDataset(Dataset):
    """Dataset for loading dog images with progressive resolution support."""
    
    def __init__(self, root_dir: str, image_size: int = 64, augment: bool = True):
        self.root_dir = root_dir
        self.image_size = image_size
        
        # Find all image files
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        print(f"Found {len(self.image_paths)} images")
        
        # Define transforms for target resolution
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))
    
    def get_resized_dataset(self, new_size: int):
        """Get a new dataset instance with different resolution."""
        return ProgressiveDogDataset(self.root_dir, new_size, augment=True)


# =============================================================================
# ProGAN Building Blocks
# =============================================================================

class MinibatchStdDev(nn.Module):
    """Minibatch standard deviation layer for Discriminator."""
    
    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.size()
        G = min(self.group_size, N)
        
        # If batch size not divisible by group size, fall back to full batch as group
        if N % G != 0:
            G = N
            
        # [G, N/G, C, H, W] -> Split batch into groups
        y = x.view(G, -1, C, H, W)
        
        # Subtract mean over group
        y = y - y.mean(dim=0, keepdim=True)
        # Calc variance over group
        y = y.mean(dim=0) # [N/G, C, H, W] 
        # Wait, StdDev logic: sqrt(mean((x - mean)^2))
        # Original: y = y - mean(y, 0) -> y = mean(y**2, 0) -> sqrt
        # If I reshape to [G, M, ...], mean(dim=0) averages over the G images in a group?
        # Standard implementation: 
        # 1. Reshape [N//G, G, C, H, W] (groups of G images) -> This seems more natural for "group size 4"
        # 2. Calc std over dim 1 (the group dim)
        # Let's verify Karras implementation logic.
        # "We compute stddev for each feature ... over a minibatch of N=4 images"
        # If x is [16, ...], we want 4 groups of 4.
        # So we want to average over the "4" dimension.
        
        # Let's stick to the previous shape logic which was: x.view(group_size, -1, ...).
        # if group_size=4. [4, 4, ...]. dim 0 is 4. averaging over dim 0 reduces it to [4, ...].
        # That means we have 4 resulting stats? No.
        # We want to append a features map. The scalar is usually average over all pixels too.
        
        # Correct logic from paper:
        # 1. Compute stddev over group.
        # 2. Average over [C, H, W] to get one single scalar per group.
        # 3. Replicate this scalar to input shape.
        
        y = x.view(G, -1, C, H, W) # [G, M, C, H, W]
        y = y - y.mean(dim=0, keepdim=True) # [G, M, C, H, W]
        y = (y ** 2).mean(dim=0) # [M, C, H, W] - Variance over group
        y = torch.sqrt(y + 1e-8) # [M, C, H, W] - Std over group
        
        # Average over feature and spatial locations
        y = y.mean(dim=[1, 2, 3], keepdim=True) # [M, 1, 1, 1]
        
        # Replicate feature map
        y = y.view(-1, 1, 1, 1) # [M, 1, 1, 1]
        y = y.repeat(G, 1, H, W) # [N, 1, H, W]
        
        return torch.cat([x, y], dim=1)



class PixelNorm(nn.Module):
    """Pixel normalization layer."""
    
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)



class WSLinear(nn.Module):
    """Linear layer with equalized learning rate."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.scale = (2 / in_features) ** 0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.scale, self.bias)


class WSConv2d(nn.Module):
    """Convolution layer with equalized learning rate."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.scale = (2 / (in_channels * kernel_size * kernel_size)) ** 0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)



class WSConvTranspose2d(nn.Module):
    """Weight-scaled transposed convolution with equalized learning rate."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, 
                 stride: int = 2, padding: int = 1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        nn.init.kaiming_normal_(self.conv.weight)
        self.scale = (2 / (in_channels * kernel_size ** 2)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x * self.scale) + self.bias.view(1, -1, 1, 1) if self.bias is not None else self.conv(x * self.scale)






class FromRGB(nn.Module):
    """Initial layer to convert RGB input to feature maps."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = WSConv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.conv(x), 0.2)


class ToRGB(nn.Module):
    """Final layer to convert feature maps to RGB output."""
    
    def __init__(self, in_channels: int, out_channels: int = 3):
        super().__init__()
        self.conv = WSConv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.conv(x))


# =============================================================================
# Progressive Generator
# =============================================================================

class ProGANGenerator(nn.Module):
    """
    Progressive Generator with strict Karras et al. 2017 architecture from CelebA-HQ table.
    """
    def __init__(self, config: ProGANConfig):
        super().__init__()
        self.config = config
        self.current_step = 0
        self.num_steps = len(config.channel_progression)
        
        # Explicitly store blocks for each resolution
        self.blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        
        # 4x4 Block (Step 0)
        # 512, 4x4
        self.blocks.append(nn.Sequential(
            # Dense -> Reshape (handled in forward)
            WSLinear(config.latent_dim, config.channel_progression[0] * 4 * 4), 
            nn.LeakyReLU(0.2), # Activation
            PixelNorm(),       # Norm
            WSConv2d(config.channel_progression[0], config.channel_progression[0], 3, 1, 1), # One 3x3 Conv
            nn.LeakyReLU(0.2), # Activation
            PixelNorm()        # Norm
        ))

        self.to_rgb.append(ToRGB(config.channel_progression[0]))
        
        # Growth Blocks (Step 1 to 8: 8x8 to 1024x1024)
        for i in range(1, self.num_steps):
            in_ch = config.channel_progression[i-1]
            out_ch = config.channel_progression[i]
            
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                WSConv2d(in_ch, out_ch, 3, 1, 1),
                nn.LeakyReLU(0.2),
                PixelNorm(),
                WSConv2d(out_ch, out_ch, 3, 1, 1),
                nn.LeakyReLU(0.2),
                PixelNorm()
            )
            self.blocks.append(block)
            self.to_rgb.append(ToRGB(out_ch))

    def forward(self, z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        # 4x4 Block Logic
        out = self.blocks[0][0](z) # Linear
        out = out.view(-1, self.config.channel_progression[0], 4, 4)
        # Apply rest of 4x4 block: LeakyReLU -> PixelNorm -> Conv -> LeakyReLU -> PixelNorm
        for layer in self.blocks[0][1:]:
            out = layer(out)
            
        if self.current_step == 0:
            return self.to_rgb[0](out)
            
        # Growth Blocks up to current step
        for i in range(1, self.current_step + 1):
            prev_out = out # Save for fade-in
            out = self.blocks[i](out)
            
        # Fade-in Logic (RGB Space)
        rgb_curr = self.to_rgb[self.current_step](out)
        
        if alpha < 1.0:
            # Upsample previous resolution ToRGB output
            rgb_prev = self.to_rgb[self.current_step - 1](prev_out)
            rgb_prev = F.interpolate(rgb_prev, scale_factor=2, mode='nearest')
            return alpha * rgb_curr + (1 - alpha) * rgb_prev
            
        return rgb_curr

    def grow_network(self):
        if self.current_step < self.num_steps - 1:
            self.current_step += 1
            res = 4 * (2 ** self.current_step)
            print(f"Growing Generator to {res}x{res} resolution (Step {self.current_step})")


class ProGANDiscriminator(nn.Module):
    """
    Progressive Discriminator with strict Karras et al. 2017 architecture from CelebA-HQ table.
    """
    def __init__(self, config: ProGANConfig):
        super().__init__()
        self.config = config
        self.current_step = 0
        self.num_steps = len(config.channel_progression)
        
        self.blocks = nn.ModuleList()
        self.from_rgb = nn.ModuleList()
        
        # Build blocks from low res (4x4) to high res (1024x1024)
        # But we need to index them carefully. Let's store them in resolution order [4x4, 8x8, ..., 1024x1024]
        # And select the subset we need during forward pass.
        
        # Final 4x4 Block
        self.final_block = nn.Sequential(
            MinibatchStdDev(),
            WSConv2d(config.channel_progression[0] + 1, config.channel_progression[0], 3, 1, 1),
            nn.LeakyReLU(0.2),
            WSConv2d(config.channel_progression[0], config.channel_progression[0], 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            WSLinear(config.channel_progression[0], 1)
        )

        # The `blocks` list stores the growth blocks (from 8x8 up to 1024x1024)
        # The `final_block` (4x4) is handled separately in the forward pass.
        self.from_rgb.append(FromRGB(3, config.channel_progression[0]))
        
        # Growth Blocks (Step 1 to 8: 8x8 to 1024x1024)
        for i in range(1, self.num_steps):
            in_ch = config.channel_progression[i] # Current res channel (e.g. 16 for 1024)
            out_ch = config.channel_progression[i-1] # Next lower res channel (e.g. 32 for 512)
            
            block = nn.Sequential(
                WSConv2d(in_ch, in_ch, 3, 1, 1),
                nn.LeakyReLU(0.2),
                WSConv2d(in_ch, out_ch, 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2)
            )
            self.blocks.append(block)
            self.from_rgb.append(FromRGB(3, in_ch))

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        # current_step defines the resolution we are training at.
        # 0 -> 4x4, 8 -> 1024x1024
        
        # Step 0 (4x4): Just FromRGB -> FinalBlock
        if self.current_step == 0:
            out = self.from_rgb[0](x)
            out = self.final_block(out)
            return out
            
        # Steps > 0
        # Entry point with Fade-in check
        if alpha < 1.0:
            # Path A (New High Res): FromRGB -> Block -> Downsample
            # blocks[current_step - 1] corresponds to the block for current_step
            y_new = self.from_rgb[self.current_step](x)
            y_new = self.blocks[self.current_step - 1](y_new)
            
            # Path B (Old Low Res): Downsample -> FromRGB_Prev
            y_old = F.avg_pool2d(x, 2)
            y_old = self.from_rgb[self.current_step - 1](y_old)
            
            # Blend
            out = alpha * y_new + (1 - alpha) * y_old
        else:
            # Standard path: FromRGB -> Block
            out = self.from_rgb[self.current_step](x)
            out = self.blocks[self.current_step - 1](out)
        
        # Continue through remaining growth blocks (High Res -> Low Res)
        # We just processed 'current_step'. Now we need to process from 'current_step - 1' down to 1.
        # blocks index for step 's' is 's-1'.
        for s in range(self.current_step - 1, 0, -1):
            out = self.blocks[s - 1](out)
            
        # Always end with Final Block (4x4)
        out = self.final_block(out)
        return out



    def grow_network(self):
        if self.current_step < self.num_steps - 1:
            self.current_step += 1
            res = 4 * (2 ** self.current_step)
            print(f"Growing Discriminator to {res}x{res} resolution (Step {self.current_step})")





# =============================================================================
# Gradient Penalty for WGAN-GP
# =============================================================================

def gradient_penalty(discriminator: ProGANDiscriminator, real_images: torch.Tensor, 
                    fake_images: torch.Tensor, alpha: float, device: str) -> torch.Tensor:
    """Compute gradient penalty for WGAN-GP."""
    batch_size = real_images.size(0)
    
    # Random interpolation
    interp_alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = interp_alpha * real_images + (1 - interp_alpha) * fake_images
    interpolated.requires_grad_(True)
    
    # Discriminator output
    d_interpolated = discriminator(interpolated, alpha=alpha)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Gradient penalty
    gradients = gradients.view(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty



# =============================================================================
# Training Utilities
# =============================================================================

class ProGANLogger:
    """Logs training metrics for ProGAN."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.log_path = os.path.join(output_dir, "progan_training_log.json")
        self.history = {
            "resolution": [],
            "kimg": [],
            "g_loss": [],
            "d_loss": [],
            "gp_loss": [],
            "drift_loss": [],
            "alpha": [],
            "timestamp": []
        }
    
    def log(self, resolution: int, kimg: float, g_loss: float, d_loss: float, 
            gp_loss: float, drift_loss: float, alpha: float):
        self.history["resolution"].append(resolution)
        self.history["kimg"].append(kimg)
        self.history["g_loss"].append(g_loss)
        self.history["d_loss"].append(d_loss)
        self.history["gp_loss"].append(gp_loss)
        self.history["drift_loss"].append(drift_loss)
        self.history["alpha"].append(alpha)
        self.history["timestamp"].append(datetime.now().isoformat())
        
        with open(self.log_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def show_samples(generator: ProGANGenerator, fixed_noise: torch.Tensor, 
                resolution: int, kimg: float, alpha: float, nrow: int = 4):
    """Generate and display sample images."""
    import matplotlib.pyplot as plt
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise, alpha=alpha).detach().cpu()
    fake_images = (fake_images + 1) / 2  # Denormalize to [0, 1]
    grid = vutils.make_grid(fake_images, nrow=nrow, normalize=False)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f"Resolution: {resolution}x{resolution}, Kimg: {kimg:.1f}, Alpha: {alpha:.2f}")
    plt.axis("off")
    plt.savefig(f"sample_{resolution}_{int(kimg)}.png")
    plt.close() # Close to avoid memory leaks
    generator.train()


def save_progan_checkpoint(generator: ProGANGenerator, discriminator: ProGANDiscriminator,
                          optimizer_g: optim.Optimizer, optimizer_d: optim.Optimizer,
                          resolution: int, kimg: float, alpha: float, config: ProGANConfig):
    """Save ProGAN checkpoint."""
    checkpoint = {
        "resolution": resolution,
        "kimg": kimg,
        "alpha": alpha,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "config": asdict(config)
    }
    
    path = os.path.join(config.output_dir, "checkpoints", 
                        f"progan_res_{resolution}_kimg_{int(kimg)}.pth")
    torch.save(checkpoint, path)
    
    latest_path = os.path.join(config.output_dir, "checkpoints", "progan_latest.pth")
    torch.save(checkpoint, latest_path)


# =============================================================================
# ProGAN Training Loop
# =============================================================================

def train_progan(config: ProGANConfig) -> Dict[str, List[float]]:
    """Train ProGAN with progressive resolution growing."""
    print(f"Training ProGAN on device: {config.device}")
    print(f"Target resolution: {config.target_image_size}x{config.target_image_size}")
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # Initialize models
    generator = ProGANGenerator(config).to(config.device)
    discriminator = ProGANDiscriminator(config).to(config.device)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs with DataParallel!")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
    
    # Optimizers
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2)
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2)
    )
    
    # Fixed noise for consistent sampling
    fixed_noise = torch.randn(config.num_sample_images, config.latent_dim, device=config.device)
    
    # Logger
    logger = ProGANLogger(config.output_dir)
    
    # Global training state
    current_resolution = config.start_resolution
    total_imgs = 0 # Track global images seen
    
    for step in range(config.num_progressive_steps):
        batch_size = config.batch_sizes.get(current_resolution, 4)
        target_kimg = config.train_schedule.get(current_resolution, 600)
        
        print(f"\n{'='*60}")
        print(f"Progressive Step {step + 1}/{config.num_progressive_steps}")
        print(f"Resolution: {current_resolution}x{current_resolution}")
        print(f"Batch Size: {batch_size}")
        print(f"Target Kimg: {target_kimg}")
        print(f"{'='*60}")
        
        # Create dataset for current resolution
        dataset = ProgressiveDogDataset(config.data_path, current_resolution)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        # Training state for this resolution
        res_imgs = 0
        pbar = tqdm(total=target_kimg * 1000, desc=f"Res {current_resolution} Training", unit="img")
        
        while res_imgs < target_kimg * 1000:
            for i, real_images in enumerate(dataloader):
                if res_imgs >= target_kimg * 1000:
                    break
                    
                batch_size = real_images.size(0)
                real_images = real_images.to(config.device)
                
                # Calculate alpha
                # Fade in for the first half of the schedule (or specific kimg count)
                fade_point = config.fade_in_kimg * 1000
                if step > 0 and res_imgs < fade_point:
                    alpha = res_imgs / fade_point
                else:
                    alpha = 1.0
                
                # ========================
                # Train Discriminator
                # ========================
                optimizer_d.zero_grad()
                
                # Real images
                d_real = discriminator(real_images, alpha=alpha)
                
                # Fake images
                z = torch.randn(batch_size, config.latent_dim, device=config.device)
                fake_images = generator(z, alpha=alpha).detach()
                d_fake = discriminator(fake_images, alpha=alpha)
                
                # WGAN-GP Loss + Drift Penalty
                loss_d = d_fake.mean() - d_real.mean()
                gp = gradient_penalty(discriminator, real_images, fake_images, alpha, config.device)
                drift = (d_real ** 2).mean() * config.drift_epsilon
                
                loss_d += config.gp_lambda * gp + drift
                
                loss_d.backward()
                optimizer_d.step()
                
                # ========================
                # Train Generator
                # ========================
                if res_imgs % config.n_critic == 0: # Technically n_critic=1 so always
                    optimizer_g.zero_grad()
                    z = torch.randn(batch_size, config.latent_dim, device=config.device)
                    fake_images = generator(z, alpha=alpha)
                    d_fake = discriminator(fake_images, alpha=alpha)
                    loss_g = -d_fake.mean()
                    loss_g.backward()
                    optimizer_g.step()
                
                # Update counters
                res_imgs += batch_size
                total_imgs += batch_size
                pbar.update(batch_size)
                
                # Logging
                if res_imgs % (config.sample_interval_kimg * 1000) < batch_size: # approx logic
                     # Log metrics
                     logger.log(current_resolution, res_imgs/1000, loss_g.item(), loss_d.item(), gp.item(), drift.item(), alpha)
                     pbar.set_postfix(G=f"{loss_g.item():.3f}", D=f"{loss_d.item():.3f}", A=f"{alpha:.2f}")

            # Checkpoints and Samples (Logic simplified for readability in loop)
            current_kimg = res_imgs / 1000
            if int(current_kimg) % config.sample_interval_kimg == 0:
                 show_samples(generator, fixed_noise, current_resolution, current_kimg, alpha)
            
            if int(current_kimg) % config.save_interval_kimg == 0:
                 save_progan_checkpoint(generator, discriminator, optimizer_g, optimizer_d,
                                      current_resolution, current_kimg, alpha, config)

        pbar.close()
        
        # Grow network for next resolution
        if step < config.num_progressive_steps - 1:
            generator.module.grow_network() if isinstance(generator, nn.DataParallel) else generator.grow_network()
            discriminator.module.grow_network() if isinstance(discriminator, nn.DataParallel) else discriminator.grow_network()
            current_resolution *= 2
    
    print("\n" + "=" * 60)
    print("ProGAN Training Completed!")
    return logger.history



# =============================================================================
# Main Entry Point
# =============================================================================

def is_notebook() -> bool:
    """Detect if running in a Jupyter/Kaggle notebook environment."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        return False
    except (NameError, AttributeError):
        return False


def parse_args():
    if is_notebook():
        return None
    
    parser = argparse.ArgumentParser(description="ProGAN for Dog Image Generation")
    parser.add_argument("--data_path", type=str,
                        default="/kaggle/input/generative-dog-images/all-dogs/all-dogs")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/progan_output")
    parser.add_argument("--target_size", type=int, default=64)
    parser.add_argument("--epochs_per_res", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--fade_epochs", type=int, default=10)
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if args is None:
        print("📓 Running in notebook mode with default configuration")
        config = ProGANConfig(device=device)
        train_progan(config)
        return
    
    config = ProGANConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        target_image_size=args.target_size,
        epochs_per_resolution=args.epochs_per_res,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        latent_dim=args.latent_dim,
        fade_in_epochs=args.fade_epochs,
        device=device
    )
    train_progan(config)


if __name__ == "__main__":
    main()