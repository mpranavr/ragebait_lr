import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Import your architectures
from diffusion import SpatialPriorDiffusion, SpectralPriorDiffusion
from inference import self_supervised_inference  # Reuses your fixed inference loop

def process_large_image_by_patches(image_path, spatial_model, spectral_model, scale_factor=4, patch_size=64, overlap=16):
    """
    Chops a massive input image into clean patches, processes them through the 
    diffusion model, and stitches them back together seamlessly.
    """
    device = next(spatial_model.parameters()).device
    
    # 1. Load the true input image at its full authentic size
    img = Image.open(image_path).convert("RGB")
    to_tensor = transforms.ToTensor()
    lr_tensor = to_tensor(img).unsqueeze(0).to(device)  # Shape: (1, 3, H, W)
    
    B, C, H, W = lr_tensor.shape
    
    # Calculate output dimensions
    hr_H, hr_W = H * scale_factor, W * scale_factor
    hr_patch_size = patch_size * scale_factor
    hr_overlap = overlap * scale_factor
    
    # Create empty canvases to accumulate the reconstructed patches
    output_canvas = torch.zeros((B, C, hr_H, hr_W), device=device)
    weight_canvas = torch.zeros((B, C, hr_H, hr_W), device=device)
    
    # Create a 2D Gaussian or Linear blending window to eliminate visible seam borders
    window = torch.ones((hr_patch_size, hr_patch_size), device=device)
    # Fade the edges out slightly to blend smoothly
    for i in range(hr_overlap):
        fade = i / hr_overlap
        window[i, :] *= fade
        window[-i-1, :] *= fade
        window[:, i] *= fade
        window[:, -i-1] *= fade
    window = window.view(1, 1, hr_patch_size, hr_patch_size).expand(B, C, -1, -1)

    # 2. Slide the window across the image coordinates
    stride = patch_size - overlap
    
    print("✂️ Chopping image and running patch-based reverse sampling loops...")
    y_steps = range(0, H - overlap, stride)
    x_steps = range(0, W - overlap, stride)
    
    # Ensure we cover the far edges if stride doesn't align perfectly
    y_indices = list(y_steps)
    if y_indices[-1] + patch_size < H: y_indices.append(H - patch_size)
    x_indices = list(x_steps)
    if x_indices[-1] + patch_size < W: x_indices.append(W - patch_size)

    for y in y_indices:
        for x in x_indices:
            # Slice out the local 64x64 low-res crop
            lr_patch = lr_tensor[:, :, y:y+patch_size, x:x+patch_size]
            
            # If patch hits structural edge boundaries, handle pad sizing adjustments
            if lr_patch.shape[-2] != patch_size or lr_patch.shape[-1] != patch_size:
                continue
                
            # Generate the 256x256 high-res patch via your U-Net priors
            hr_patch_pred = self_supervised_inference(
                raw_lr_observation=lr_patch,
                spatial_model=spatial_model,
                spectral_model=spectral_model,
                scale_factor=scale_factor
            )
            
            # Map coordinates to the output grid
            out_y, out_x = y * scale_factor, x * scale_factor
            
            # Blend the patch onto the canvases using the smoothing window weight maps
            output_canvas[:, :, out_y:out_y+hr_patch_size, out_x:out_x+hr_patch_size] += hr_patch_pred * window
            weight_canvas[:, :, out_y:out_y+hr_patch_size, out_x:out_x+hr_patch_size] += window
            
    # 3. Normalize the final stitched canvas to avoid lighting mismatches
    final_hr = output_canvas / (weight_canvas + 1e-8)
    return torch.clamp(final_hr, 0.0, 1.0)

# ==========================================
# Script Runner Entry Block
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Patch-Inference Engine on: {device}")

    spatial_model = SpatialPriorDiffusion().to(device)
    spectral_model = SpectralPriorDiffusion().to(device)

    # Load master checkpoint dictionary bundle
    checkpoint_path = "checkpoints/checkpoint_latest.pt"
    if os.path.exists(checkpoint_path):
        print(f"🔄 Loading trained weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        spatial_model.load_state_dict(checkpoint['spatial_model_state_dict'])
        spectral_model.load_state_dict(checkpoint['spectral_model_state_dict'])
    else:
        print("⚠️ No checkpoint found! Running with randomized states for pipeline validation.")

    # Paths setup
    input_image_path = "datasets/DIV2K/train/0001.png"
    output_image_path = "stitched_clean_highres.png"

    if os.path.exists(input_image_path):
        # Process the entire image using overlapping 64x64 low-res patches
        final_output = process_large_image_by_patches(
            image_path=input_image_path,
            spatial_model=spatial_model,
            spectral_model=spectral_model,
            scale_factor=4,
            patch_size=64,   # Native low-res patch size matching training architecture
            overlap=16       # Pixels of overlap to smoothly blend stitching seams
        )
        
        # Save output image to disk
        final_tensor = final_output.squeeze(0).cpu()
        to_pil = transforms.ToPILImage()
        output_image = to_pil(final_tensor)
        output_image.save(output_image_path)
        print(f"🎉 Success! Full-size blended high-resolution image saved to: {output_image_path}")
    else:
        print(f"Error: Target image file {input_image_path} not found.")
