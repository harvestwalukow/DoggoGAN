import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import io
import base64
from PIL import Image
import sys
import os

from train_code import Generator

import argparse
import json

def generate_image(class_id=None, num_images=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint_path = "dog_gan_64.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}", file=sys.stderr)
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}", file=sys.stderr)
        return

    num_classes = 120
    latent_size = 256
    depth = 5
    
    gen = Generator(depth=depth, latent_size=latent_size, use_eql=True, use_spec_norm=False).to(device)
    
    # We load the shadow EMA weights for higher quality generations
    state_dict = checkpoint.get('gen_shadow_state_dict', checkpoint.get('gen_state_dict'))
    
    # Kaggle used DataParallel (2 GPUs), which prefixes all keys with "module."
    # We must strip this prefix to load it into a standard single-GPU/CPU model
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    gen.load_state_dict(new_state_dict)
    gen.eval()

    # The training finished at resolution 64x64, which is depth scale 4.
    current_depth = 4 
    alpha = 1.0
    
    if class_id is None:
        # Generate random dog class from 0 to 119
        class_id = torch.randint(0, num_classes, (1,)).item()
    
    results = []
    with torch.no_grad():
        for i in range(num_images):
            # Generator takes: noise, depth, alpha, labels=None
            z = torch.randn(1, latent_size - num_classes, device=device)
            label_info = torch.nn.functional.one_hot(torch.tensor([class_id]).to(device), num_classes).float()
            gan_input = torch.cat((label_info, z), dim=-1)
            
            # Generate image
            fake_img = gen(gan_input, current_depth, alpha)
            
            # Normalize [-1, 1] to [0, 1] (as it uses Tanh)
            fake_img = fake_img * 0.5 + 0.5
            
            # Convert to PIL Image
            grid = vutils.make_grid(fake_img, padding=0)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            
            # Save to buffer
            buffer = io.BytesIO()
            im.save(buffer, format="PNG")
            buffer.seek(0)
            
            # Encode to base64
            img_str = base64.b64encode(buffer.read()).decode('utf-8')
            results.append(f"data:image/png;base64,{img_str}")
        
    # Output JSON
    print(json.dumps({
        "variations": results,
        "class_id": class_id
    }))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_id", type=int, default=None)
    parser.add_argument("--num_images", type=int, default=1)
    args = parser.parse_args()
    
    generate_image(class_id=args.class_id, num_images=args.num_images)
