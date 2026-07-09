import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os

# Import your model architectures from diffusion.py
from diffusion import SpatialPriorDiffusion, SpectralPriorDiffusion


@torch.no_grad()
def self_supervised_inference(raw_lr_observation, spatial_model, spectral_model, scale_factor=4):
    spatial_model.eval()
    spectral_model.eval()

    B, C, lr_H, lr_W = raw_lr_observation.shape
    target_hr_shape = (B, C, lr_H * scale_factor, lr_W * scale_factor)
    x = torch.randn(target_hr_shape, device=raw_lr_observation.device)

    beta = torch.linspace(1e-4, 0.02, 1000).to(raw_lr_observation.device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    for t in reversed(range(1000)):
        t_tensor = torch.full((B,), t, device=raw_lr_observation.device, dtype=torch.long)

        pred_noise_spatial = spatial_model(x, t_tensor)

        # Early step diagnostic tracking print to trace dead weight freezes
        if t == 999:
            print("\n--- INFERENCE START MONITOR ---")
            print("Initial Spatial Noise Prediction STD Var:", pred_noise_spatial.std().item())

        spectral_x = spectral_model.img_to_spectral_tensor(x)
        spec_scale = spectral_x.std() + 1e-8

        pred_noise_spec_norm = spectral_model(spectral_x / spec_scale, t_tensor)
        pred_noise_spec = pred_noise_spec_norm * spec_scale
        pred_noise_spectral = spectral_model.spectral_tensor_to_img(pred_noise_spec)

        pred_noise = 0.5 * pred_noise_spatial + 0.5 * pred_noise_spectral

        a_bar = alpha_bar[t]
        a = alpha[t]

        # Shifted structural update vector (Internal inner loop hard clipping removed)
        x0_hat = (x - torch.sqrt(1.0 - a_bar) * pred_noise) / torch.sqrt(a_bar)

        if t > 0:
            z = torch.randn_like(x)
            sigma = torch.sqrt((1.0 - alpha_bar[t - 1]) / (1.0 - a_bar) * beta[t])
        else:
            z, sigma = 0, 0

        x = (torch.sqrt(a) * (1.0 - alpha_bar[t - 1]) / (1.0 - a_bar)) * x + \
            (torch.sqrt(alpha_bar[t - 1]) * beta[t] / (1.0 - a_bar)) * x0_hat + sigma * z

    return torch.clamp(x, 0.0, 1.0)


# ==========================================
# Script Entry Point for Direct Execution
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # 1. Instantiate the model architectures as usual
    spatial_model = SpatialPriorDiffusion().to(device)
    spectral_model = SpectralPriorDiffusion().to(device)

    # 2. Extract weights directly from the combined epoch checkpoint
    checkpoint_path = "checkpoints/checkpoint_latest.pt"

    if os.path.exists(checkpoint_path):
        print(f"🔄 Loading models directly from combined checkpoint: {checkpoint_path}")

        # Load the unified dictionary package
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Pull out the exact model states using their internal dictionary keys
        spatial_model.load_state_dict(checkpoint['spatial_model_state_dict'])
        spectral_model.load_state_dict(checkpoint['spectral_model_state_dict'])

        print(f"🚀 Dual-priors successfully loaded! (Saved at the end of Epoch {checkpoint['epoch']})")
    else:
        print("⚠️ Warning: Checkpoint file not found. Running with random initialization.")

    """if os.path.exists(spatial_weights_path) and os.path.exists(spectral_weights_path):
        spatial_model.load_state_dict(torch.load(spatial_weights_path, map_location=device))
        spectral_model.load_state_dict(torch.load(spectral_weights_path, map_location=device))
        print("Successfully loaded pre-trained dual-prior weights!")
    else:
        print(
            "⚠️ Warning: Pre-trained weights not found. Running generation with random initialization for structural verification.")
            """

    # 3. Load an actual test Low-Resolution image (e.g., from your DIV2K test folder)
    # Make sure this path points to a valid image file on your disk
    input_path = "datasets/DIV2K/train/test_img.png"
    output_path = "reconstructed_hr_output.png"

    if os.path.exists(input_path):
        lr_image = Image.open(input_path).convert("RGB")

        # Crop or resize down slightly if the input file is massive, to act as a test LR observation
        # For a standard test, let's select a 64x64 input patch
        crop_transform = transforms.Compose([
            transforms.CenterCrop((64, 64)),
            transforms.ToTensor()
        ])
        raw_lr_observation = crop_transform(lr_image).unsqueeze(0).to(device)  # Shape: (1, 3, 64, 64)
    else:
        print(f"Could not locate {input_path}. Falling back to a synthetic random low-resolution observation matrix.")
        raw_lr_observation = torch.rand(1, 3, 64, 64).to(device)

    # 4. Run the manifold reconstruction pipeline loop
    print("Beginning dual-prior reverse sampling manifold generation iterations...")
    reconstructed_tensor = self_supervised_inference(
        raw_lr_observation=raw_lr_observation,
        spatial_model=spatial_model,
        spectral_model=spectral_model,
        scale_factor=4
    )

    # ========================================================
    # 💥 PLACE TEST A HERE (Right before stripping batch/clamping)
    # ========================================================
    print("\n--- TEST A: DIAGNOSTIC RAW VALUE CHECK ---")
    print("Raw Tensor Minimum Value:", reconstructed_tensor.min().item())
    print("Raw Tensor Maximum Value:", reconstructed_tensor.max().item())
    print("------------------------------------------\n")

    # 5. Convert back to image space and save to disk
    reconstructed_tensor = reconstructed_tensor.squeeze(0).cpu()  # Remove batch dim
    to_pil = transforms.ToPILImage()
    output_image = to_pil(reconstructed_tensor)
    output_image.save(output_path)
    print(f"🎉 Reconstruction finished! High-Resolution image saved successfully to: {output_path}")
