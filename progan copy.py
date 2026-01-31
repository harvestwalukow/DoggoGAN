"""
Strict ProGAN Implementation (Karras et al. 2018)
Table 2 & Appendix A Compliance
"""

import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm
import math

# ==============================================================================
# 1. CONSTANTS & CONFIGURATION (Strict Appendix A)
# ==============================================================================

RESOLUTION_CHANNELS = {
    4: 512,
    8: 512,
    16: 512,
    32: 512,
    64: 256,
    128: 128,
    256: 64,
    512: 32,
    1024: 16
}

# Training Schedule (Appendix A.1)
# 800k images per phase.
IMAGES_PER_PHASE = 800_000
# Batch sizes to fit memory (Appendix A.1)
BATCH_SIZES = {
    4: 16, 8: 16, 16: 16, 32: 16, 64: 16, 128: 16,
    256: 14,
    512: 6,
    1024: 3
}

LEARNING_RATE = 0.001
ADAM_BETA1 = 0.0
ADAM_BETA2 = 0.99
ADAM_EPS = 1e-8
WGAN_LAMBDA = 10.0
DRIFT_EPSILON = 0.001

# ==============================================================================
# 2. FUNCTIONAL LAYERS & HELPERS
# ==============================================================================

def pixel_norm(x, epsilon=1e-8):
    """Pixelwise feature vector normalization (Generator only)."""
    return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + epsilon)


def minibatch_stddev(x, group_size=None):
    """
    Minibatch standard deviation (Discriminator only).
    Strict ProGAN: Use full minibatch (group_size=None/Batch).
    """
    B, C, H, W = x.shape
    # Strict: Across entire minibatch
    G = B 

    # [G, B/G, C, H, W] -> [B, 1, C, H, W] since G=B
    y = x.view(G, -1, C, H, W)
    y = y - y.mean(dim=0, keepdim=True)
    y = torch.sqrt(y.square().mean(dim=0) + 1e-8)
    y = y.mean(dim=[1, 2, 3], keepdim=True)
    y = y.repeat(G, 1, H, W)
    return torch.cat([x, y], dim=1)


class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate (runtime scaling)."""
    def __init__(self, in_features, out_features, bias=True, gain=2**0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features)) # N(0,1) init
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scale = gain * (in_features ** -0.5)

    def forward(self, x):
        w = self.weight * self.scale
        return F.linear(x, w, self.bias)

class EqualizedConv2d(nn.Module):
    """Conv2d layer with equalized learning rate (runtime scaling)."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, gain=2**0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size)) # N(0,1) init
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = gain * (fan_in ** -0.5)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        w = self.weight * self.scale
        return F.conv2d(x, w, self.bias, stride=self.stride, padding=self.padding)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleDict()
        self.to_rgb_layers = nn.ModuleDict()

        # 4x4 Block (Strict Table 2)
        # "Conv 4x4" on 1x1 input
        # Input: 512x1x1
        # Target: 512x4x4
        # Kernel=4, Stride=1, Padding=3 -> (1 + 2*3 - 4) + 1 = 4.
        self.block_4x4 = nn.ModuleList([
            EqualizedConv2d(512, 512, kernel_size=4, stride=1, padding=3), 
            # LReLU
            # NO PixelNorm (Strict A.1: "after each Conv 3x3")
            EqualizedConv2d(512, 512, 3, 1, 1),
            # LReLU
            # PixelNorm
        ])
        self.blocks['4'] = self.block_4x4
        self.to_rgb_layers['4'] = EqualizedConv2d(512, 3, 1, 1, 0, gain=1.0)

        # Progressive Blocks
        resolutions = [8, 16, 32, 64, 128, 256, 512, 1024]
        for res in resolutions:
            in_ch = RESOLUTION_CHANNELS[res // 2]
            out_ch = RESOLUTION_CHANNELS[res]
            
            block = nn.ModuleList([
                EqualizedConv2d(in_ch, out_ch, 3, 1, 1),
                EqualizedConv2d(out_ch, out_ch, 3, 1, 1),
            ])
            self.blocks[str(res)] = block
            self.to_rgb_layers[str(res)] = EqualizedConv2d(out_ch, 3, 1, 1, 0, gain=1.0)

    def forward(self, z, step, alpha):
        # z: (B, 512) or (B, 512, 1, 1)
        if z.ndim == 2:
            z = z.view(-1, 512, 1, 1)
        
        # 4x4 Block (Base)
        x = self.blocks['4'][0](z) # Conv 4x4 (1x1->4x4)
        x = F.leaky_relu(x, 0.2)
        # NO PixelNorm here!
        
        x = self.blocks['4'][1](x) # Conv 3x3
        x = F.leaky_relu(x, 0.2)
        x = pixel_norm(x) # Yes PixelNorm here

        
        if step == 0:
            return self.to_rgb_layers['4'](x)

        # Progressive Blocks
        for i in range(1, step + 1):
            res = 2 ** (i + 2)
            
            # Upsample (Nearest Neighbor)
            # Strict logic: Upsample -> Branch for Fade (Old RGB) AND Branch for Next Block
            upsampled_x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = upsampled_x # Main branch continues
            
            # Block processing
            block = self.blocks[str(res)]
            x = block[0](x)
            x = F.leaky_relu(x, 0.2)
            x = pixel_norm(x)
            x = block[1](x)
            x = F.leaky_relu(x, 0.2)
            x = pixel_norm(x)
            
            # Fade-in check at the OUTPUT of this block
            if i == step:
                # We are at the final resolution for this step
                to_rgb = self.to_rgb_layers[str(res)]
                out = to_rgb(x)
                
                if alpha < 1.0:
                    # Previous RGB branch: ToRGB_prev(upsampled_prev_features)
                    # "upsampled_x" is exactly "prev_features_upsampled"
                    res_prev = 2 ** (i + 2 - 1)
                    to_rgb_prev = self.to_rgb_layers[str(res_prev)]
                    out_prev = to_rgb_prev(upsampled_x)
                    
                    out = alpha * out + (1 - alpha) * out_prev
                
                return out
                
        # Fallback (should be caught by loop return)
        return self.to_rgb_layers[str(2**(step+2))](x)


# ==============================================================================
# 4. DISCRIMINATOR (STRICT TABLE 2)
# ==============================================================================

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleDict()
        self.from_rgb_layers = nn.ModuleDict()
        
        resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        
        for res in resolutions:
            ch = RESOLUTION_CHANNELS[res]
            self.from_rgb_layers[str(res)] = EqualizedConv2d(3, ch, 1, 1, 0)
            
            if res == 4:
                # 4x4 Block (Last one in flow)
                # Input 513 (512+1 stddev) -> 512
                # Conv 3x3, LReLU
                # Conv 4x4 (Dense), LReLU
                # Dense (1)
                self.blocks['4'] = nn.ModuleList([
                    EqualizedConv2d(513, 512, 3, 1, 1),
                    EqualizedConv2d(512, 512, 4, 1, 0), # Strict Conv 4x4 (512x4x4 -> 512x1x1)
                    EqualizedLinear(512, 1), # Linear
                ])

                # NOTE: The "Conv 4x4" here outputting 1x1 is essentially a dense layer 
                # but with spatial awareness if size > 4x4 (which it isn't).
            else:
                # Standard Block
                # Conv 3x3 (Cin->Cin)
                # Conv 3x3 (Cin->Cout)
                # Downsample (AvgPool)
                # Table 2: 
                # 8x8: Conv 512->512, Conv 512->512, Downsample
                # 16x16: Conv 512->512, Conv 512->512, Downsample
                # ...
                # Wait, Table 2 Channel Progression for Disc:
                # 1024 input: Conv 1x1 (FromRGB) -> 16
                # Conv 3x3 (16->16)
                # Conv 3x3 (16->32)
                # Down.
                # So Block input CH is current res CH, output is next smaller res CH.
                ch_in = RESOLUTION_CHANNELS[res]
                ch_out = RESOLUTION_CHANNELS[res // 2]
                
                self.blocks[str(res)] = nn.ModuleList([
                    EqualizedConv2d(ch_in, ch_in, 3, 1, 1),
                    EqualizedConv2d(ch_in, ch_out, 3, 1, 1),
                ])

    def forward(self, x, step, alpha):
        # x is Image Batch
        
        # Decide starting resolution
        current_res = 2 ** (step + 2)
        
        # 1. FROM RGB (Entry)
        # Handle Fade-in Logic
        # Path A: FromRGB(Current)
        
        # We need to process the block handling logic.
        # Flow: res -> res/2 -> ... -> 4
        
        # Start at current_res
        out = self.from_rgb_layers[str(current_res)](x)
        # NO LReLU after FromRGB (Strict Table 2)
        
        if alpha < 1.0 and step > 0:
            # Fade-in Logic
            # Path B: Downsample Input -> FromRGB(Prev)
            prev_res = current_res // 2
            x_down = F.avg_pool2d(x, 2)
            out_prev = self.from_rgb_layers[str(prev_res)](x_down)
            # NO LReLU after FromRGB
            
            # Now run 'out' (Path A) through the first block of 'current_res'

            # to align it to 'prev_res' spatial/channel size?
            # Table 2/Fig 2b: 
            # Path A: Input -> FromRGB -> Conv -> Conv -> Downsample.
            # Blend happens AFTER Downsample of Path A.
            
            # Execute Block 'current_res' for Path A
            block = self.blocks[str(current_res)]
            out = block[0](out) # Conv 3x3
            out = F.leaky_relu(out, 0.2)
            out = block[1](out) # Conv 3x3 (changes channels)
            out = F.leaky_relu(out, 0.2)
            out = F.avg_pool2d(out, 2) # Downsample
            
            # Now Blend
            out = alpha * out + (1 - alpha) * out_prev
            
            # Continue from next block (step - 1)
            start_idx = step - 1
        else:
            # No fade, just run the first block normally if step > 0
            if step > 0:
                block = self.blocks[str(current_res)]
                out = block[0](out)
                out = F.leaky_relu(out, 0.2)
                out = block[1](out)
                out = F.leaky_relu(out, 0.2)
                out = F.avg_pool2d(out, 2)
                start_idx = step - 1
            else:
                # 4x4 case
                start_idx = -1 # Skip loop

        # Loop down to 4x4 (exclusive)
        for i in range(start_idx, -1, -1): # step-1 down to 0
            res = 2 ** (i + 2)
            # If res is 4, break (handled separately)
            if res == 4: break 
            
            block = self.blocks[str(res)]
            out = block[0](out)
            out = F.leaky_relu(out, 0.2)
            out = block[1](out) # Changes channels to next lower
            out = F.leaky_relu(out, 0.2)
            out = F.avg_pool2d(out, 2)
            
        # Final 4x4 Block
        # Minibatch StdDev
        out = minibatch_stddev(out)
        
        # Conv 3x3 (513->512)
        block4 = self.blocks['4']
        out = block4[0](out)
        out = F.leaky_relu(out, 0.2)
        
        # Conv 4x4 (512->512) -> Output is 1x1
        out = block4[1](out) # Conv 4x4
        out = F.leaky_relu(out, 0.2)
        
        # Dense (512->1)
        out = out.view(out.size(0), -1)
        out = block4[2](out)

        
        return out

# ==============================================================================
# 5. DATASET & TRAINING
# ==============================================================================

class DogDataset(Dataset):
    def __init__(self, root_dir, image_size):
        self.files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
            self.files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        
        print(f"Dataset: Found {len(self.files)} images in {root_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx % len(self.files)]).convert('RGB')
            return self.transform(img)
        except:
            return self.__getitem__(idx + 1)

def train_progan(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Ensure output directory exists immediately
    os.makedirs('progan_output', exist_ok=True)
    
    gen = Generator().to(device)
    disc = Discriminator().to(device)
    
    # Dual GPU Support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        gen = nn.DataParallel(gen)
        disc = nn.DataParallel(disc)

    
    g_optimizer = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPS)
    d_optimizer = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPS)
    
    import matplotlib.pyplot as plt
    fixed_noise = torch.randn(4, 512, device=device)
    
    # State tracking
    total_images_shown = 0
    
    # Schedule logic
    def get_scheduler_state(n_img):
        if n_img < IMAGES_PER_PHASE:
            return 0, 1.0, 4 
        img = n_img - IMAGES_PER_PHASE
        phase_idx = img // IMAGES_PER_PHASE
        step = 1 + (phase_idx // 2)
        if step > 8: step = 8
        res = 2 ** (step + 2)
        is_fade = (phase_idx % 2 == 0)
        if is_fade:
            alpha = (img % IMAGES_PER_PHASE) / IMAGES_PER_PHASE
        else:
            alpha = 1.0
        return step, alpha, res

    # STRICT: Load at 1024, downsample recursively
    dataset_full = DogDataset(args.data_path, 1024) 
    total_limit = 100 * len(dataset_full) 
    
    pbar = tqdm(total=total_limit)
    
    current_res = 4
    dataloader = DataLoader(dataset_full, batch_size=BATCH_SIZES[4], shuffle=True, num_workers=4, drop_last=True)
    
    def recursive_downsample(img, current_res):
        if current_res == 1024: return img
        cur = img
        steps_down = int(math.log2(1024 // current_res))
        for _ in range(steps_down):
            cur = F.avg_pool2d(cur, 2)
        return cur
    
    while total_images_shown < total_limit:
        step, alpha, res = get_scheduler_state(total_images_shown)
        
        if res != current_res or total_images_shown == 0:
            current_res = res
            batch_size = BATCH_SIZES[res]
            dataloader = DataLoader(dataset_full, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        
        for real_img_1024 in dataloader:            
            real_img_1024 = real_img_1024.to(device)
            B = real_img_1024.size(0)
            
            step, alpha, res = get_scheduler_state(total_images_shown)
            if res != current_res:
                break 
            
            real_img = recursive_downsample(real_img_1024, res)
            
            # TRAIN DISCRIMINATOR
            d_optimizer.zero_grad()
            d_real = disc(real_img, step, alpha).view(-1)
            
            z = torch.randn(B, 512, device=device)
            fake_img = gen(z, step, alpha).detach()
            d_fake = disc(fake_img, step, alpha).view(-1)
            
            loss_wgan = d_fake.mean() - d_real.mean()
            
            eps = torch.rand(B, 1, 1, 1, device=device)
            x_hat = (eps * real_img + (1 - eps) * fake_img).requires_grad_(True)
            d_hat = disc(x_hat, step, alpha).view(-1)
            grads = torch.autograd.grad(d_hat.sum(), x_hat, create_graph=True, retain_graph=True)[0]
            loss_gp = ((grads.reshape(B, -1).norm(2, dim=1) - 1) ** 2).mean() * WGAN_LAMBDA
            
            loss_drift = (d_real ** 2).mean() * DRIFT_EPSILON
            loss_d = loss_wgan + loss_gp + loss_drift
            loss_d.backward()
            d_optimizer.step()
            
            # TRAIN GENERATOR
            g_optimizer.zero_grad()
            z = torch.randn(B, 512, device=device)
            fake_img = gen(z, step, alpha)
            d_fake_g = disc(fake_img, step, alpha).view(-1)
            loss_g = -d_fake_g.mean()
            loss_g.backward()
            g_optimizer.step()
            
            total_images_shown += B
            pbar.update(B)
            
            if total_images_shown % 20000 == 0:
                pbar.set_description(f"Res:{res} A:{alpha:.2f} D:{loss_d.item():.2f} G:{loss_g.item():.2f}")
                
            if total_images_shown >= total_limit:
                break
                
            # Sample every 2 epochs (~40k images)
            if (total_images_shown // 40000) > ((total_images_shown - B) // 40000):
                     g_module = gen.module if isinstance(gen, nn.DataParallel) else gen
                     sample = g_module(fixed_noise, step, alpha)
                     # Match user traceback: save directly with vutils
                     # sample is [-1, 1], normalize to [0, 1] for saving
                     vutils.save_image(sample * 0.5 + 0.5, f"progan_output/img_{total_images_shown // 1000}k.png", nrow=4)





if __name__ == "__main__":
    # Param verification
    g = Generator()
    d = Discriminator()
    g_params = sum(p.numel() for p in g.parameters())
    d_params = sum(p.numel() for p in d.parameters())
    print(f"Generator Params: {g_params/1e6:.2f}M")
    print(f"Discriminator Params: {d_params/1e6:.2f}M")
    
    import argparse
    import zipfile
    
    # Kaggle Setup: Unzip if needed
    def setup_data(data_path_zip, data_path_dir):
        if os.path.exists(data_path_zip) and not os.path.exists(data_path_dir):
            print(f"Extracting {data_path_zip}...")
            with zipfile.ZipFile(data_path_zip, 'r') as z:
                z.extractall("/kaggle/working")
            print("Extraction complete.")
            return "/kaggle/working/all-dogs"
        elif os.path.exists(data_path_dir):
            return data_path_dir
        return "/kaggle/input/generative-dog-images/all-dogs/all-dogs" # Default fallback

    # Seeding
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    set_seed(42)

    # Notebook Check
    def is_notebook():
        try:
            from IPython import get_ipython
            if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
                return True
            return False
        except:
            return False

    if is_notebook():
        # Kaggle Notebook Mode
        # Auto-setup data
        data_path = setup_data("/kaggle/input/generative-dog-images/all-dogs.zip", "/kaggle/working/all-dogs")
        
        class Args:
            data_path = data_path
        
        args = Args()
        if os.path.exists(args.data_path):
            train_progan(args)
        else:
            print("Dataset not found. Please add 'generative-dog-images'.")
            
    else:
        # Script Mode
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path", type=str, default="/kaggle/input/generative-dog-images/all-dogs/all-dogs")
        args = parser.parse_args()
        
        # Check extraction too if script run in environment with zip
        if args.data_path.endswith(".zip"):
             # Handle zip input?
             pass 
        
        if os.path.exists(args.data_path):
            train_progan(args)
        else:
            print("Data path not found, running param check only.")

