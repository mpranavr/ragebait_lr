import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Import the sampling logic from your other files
from diffusion import SpatialPriorDiffusion, SpectralPriorDiffusion
from posterior_sampling import DualPriorPosteriorSampler
from inference import self_supervised_inference



def load_and_preprocess_img(img_path, target_lr_size=(64, 64)):
    """
    Loads a local image using OpenCV, converts it to an RGB PyTorch tensor,
    normalizes it to [0, 1], and resizes it to act as our test LR observation.
    """
    # 1. Read image with OpenCV
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Convert to PyTorch float tensor scaled between [0.0, 1.0]
    # Shape transforms from (H, W, C) to (C, H, W)
    transform = transforms.ToTensor()
    img_tensor = transform(img)

    # 3. Create a mock Low-Resolution observation by downsampling it to 64x64
    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    lr_observation = F.interpolate(
        img_tensor.unsqueeze(0),
        size=target_lr_size,
        mode='bicubic',
        align_corners=False
    )

    return lr_observation


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    # --- 1. Load the low-resolution observation ---
    # Replace this with an actual valid image path inside your DIV2K dataset folder
    test_image_path = "datasets/DIV2K/train/0001.png"
    try:
        raw_lr_observation = load_and_preprocess_img(test_image_path).to(device)
    except FileNotFoundError:
        print(f"Warning: Defaulting to random noise tensor because {test_image_path} was not found.")
        raw_lr_observation = torch.rand(1, 3, 64, 64).to(device)

    # --- 2. Initialize your trained Prior Models ---
    spatial_model = SpatialPriorDiffusion().to(device)
    spectral_model = SpectralPriorDiffusion().to(device)

    # Load your saved weights from train.py (uncomment when weights are trained!)
    # spatial_model.load_state_dict(torch.load("spatial_prior_unet.pth", map_location=device))
    # spectral_model.load_state_dict(torch.load("spectral_prior_unet.pth", map_location=device))

    # --- 3. Execute Posterior Sampling (Method A) ---
    print("Running Dual-Prior Guided Posterior Sampling Reconstructions...")
    # Using a placeholder value for pipeline parameter matching your file structure
    sampler = DualPriorPosteriorSampler(spatial_model, spectral_model, pipeline=None,
                                        num_steps=20)  # Low steps for quick testing

    hr_shape = (1, 3, 256, 256)
    reconstructed_hr_posterior = sampler.reconstruct(
        lr_img=raw_lr_observation,
        hr_shape=hr_shape,
        alpha_spatial=0.5,
        lambda_dc=0.5
    )

    # --- 4. Execute Self-Supervised Inference (Method B) ---
    print("Running Self-Supervised Manifold Denoising Inference...")
    reconstructed_hr_inference = self_supervised_inference(
        raw_lr_observation=raw_lr_observation,
        spatial_model=spatial_model,
        spectral_model=spectral_model,
        scale_factor=4
    )

    # --- 5. Convert back to OpenCV format to display the output ---
    # Pick the posterior sampling output to display, remove batch dimension, change to (H, W, C)
    output_tensor = reconstructed_hr_posterior.squeeze(0).cpu()

    # Rescale back from [0.0, 1.0] float to [0, 255] uint8 arrays
    output_numpy = output_tensor.permute(1, 2, 0).numpy()
    output_numpy = (output_numpy * 255.0).astype('uint8')

    # Convert back to BGR color space for OpenCV window compatibility
    output_img_bgr = cv2.cvtColor(output_numpy, cv2.COLOR_RGB2BGR)

    print(f"Reconstruction Complete. Final Output Shape: {output_img_bgr.shape}")

    # Display Result
    cv2.imshow("Reconstructed HR Output", output_img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()