import os
import glob
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from diffusion import SpatialPriorDiffusion, SpectralPriorDiffusion


class DIV2KUnsupervisedDataset(Dataset):
    def __init__(self, lr_dir, target_patch_size=(256, 256), scale_factor=4):
        self.image_paths = sorted(glob.glob(os.path.join(lr_dir, "*.png")))
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No PNG images found in directory: {lr_dir}")
        self.scale_factor = scale_factor
        self.target_hr_size = target_patch_size
        self.lr_patch_size = (target_patch_size[0] // scale_factor, target_patch_size[1] // scale_factor)
        self.crop_transform = transforms.RandomCrop(self.lr_patch_size)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        lr_img = Image.open(self.image_paths[idx]).convert('RGB')
        lr_patch = self.crop_transform(lr_img)
        raw_lr_tensor = self.to_tensor(lr_patch)

        pseudo_hr_tensor = F.interpolate(
            raw_lr_tensor.unsqueeze(0),
            size=self.target_hr_size,
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
        return raw_lr_tensor, pseudo_hr_tensor


class SelfSupervisedLoop:
    def __init__(self, spatial_model, spectral_model, scale_factor=4):
        self.spatial_model = spatial_model
        self.spectral_model = spectral_model
        self.scale_factor = scale_factor
        self.num_steps = 1000
        self.device = next(spatial_model.parameters()).device
        self.beta = torch.linspace(1e-4, 0.02, self.num_steps).to(self.device)
        self.alpha_bar = torch.cumprod(1.0 - self.beta, dim=0)

    def forward_spectral_degradation(self, img, radius_pct=0.4):
        B, C, H, W = img.shape
        fft_shifted = torch.fft.fftshift(torch.fft.fft2(img, dim=(-2, -1)), dim=(-2, -1))
        center_h, center_w = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H) - center_h, torch.arange(W) - center_w, indexing='ij')
        distance = torch.sqrt(y ** 2 + x ** 2).to(img.device)
        max_radius = min(center_h, center_w) * radius_pct
        mask = (distance <= max_radius).float().view(1, 1, H, W)
        filtered = fft_shifted * mask
        return torch.real(torch.fft.ifft2(torch.fft.ifftshift(filtered, dim=(-2, -1)), dim=(-2, -1)))

    def forward_spatial_degradation(self, img):
        H, W = img.shape[-2], img.shape[-1]
        return F.interpolate(img, size=(H // self.scale_factor, W // self.scale_factor), mode='bicubic',
                             align_corners=False)

    def train_step(self, raw_lr_img, pseudo_hr, optimizer):
        B = raw_lr_img.shape[0]
        t = torch.randint(0, self.num_steps, (B,), device=self.device).long()
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).view(B, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar[t]).view(B, 1, 1, 1)

        # 1. Spatial Prior Denoising
        spatial_noise = torch.randn_like(pseudo_hr)
        noisy_spatial = sqrt_alpha_bar * pseudo_hr + sqrt_one_minus_alpha_bar * spatial_noise
        pred_noise_spatial = self.spatial_model(noisy_spatial, t)

        # 2. Corrected Spectral Prior Denoising (Real Floats Only)
        spectral_target = self.spectral_model.img_to_spectral_tensor(pseudo_hr)
        spec_scale = spectral_target.std() + 1e-8
        spectral_target_norm = spectral_target / spec_scale

        spectral_noise = torch.randn_like(spectral_target_norm)
        noisy_spectral = sqrt_alpha_bar * spectral_target_norm + sqrt_one_minus_alpha_bar * spectral_noise

        pred_noise_spec_norm = self.spectral_model(noisy_spectral, t)
        pred_noise_spec = pred_noise_spec_norm * spec_scale
        pred_noise_spectral = self.spectral_model.spectral_tensor_to_img(pred_noise_spec)

        # 3. Combine & Predict HR Manifold Guess
        combined_pred_noise = 0.5 * pred_noise_spatial + 0.5 * pred_noise_spectral
        reconstructed_hr = (noisy_spatial - sqrt_one_minus_alpha_bar * combined_pred_noise) / sqrt_alpha_bar
        reconstructed_hr = torch.clamp(reconstructed_hr, 0.0, 1.0)

        # 4. Re-degradation Loss Checking
        recon_spectrally_degraded = self.forward_spectral_degradation(reconstructed_hr)
        recon_spatially_downsampled = self.forward_spatial_degradation(reconstructed_hr)
        true_spectrally_degraded = self.forward_spectral_degradation(pseudo_hr)
        true_spatially_downsampled = raw_lr_img

        loss_spatial_cycle = F.mse_loss(recon_spatially_downsampled, true_spatially_downsampled)
        loss_spectral_cycle = F.mse_loss(recon_spectrally_degraded, true_spectrally_degraded)
        total_cycle_loss = loss_spatial_cycle + loss_spectral_cycle

        optimizer.zero_grad()
        total_cycle_loss.backward()
        optimizer.step()
        return total_cycle_loss.item()


def save_checkpoint(epoch, spatial_model, spectral_model, optimizer, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_state = {
        'epoch': epoch,
        'spatial_model_state_dict': spatial_model.state_dict(),
        'spectral_model_state_dict': spectral_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint_state, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))
    torch.save(checkpoint_state, os.path.join(checkpoint_dir, "checkpoint_latest.pt"))
    print(f"\n💾 Checkpoint saved at Epoch {epoch}")


def load_checkpoint(spatial_model, spectral_model, optimizer, checkpoint_dir="checkpoints"):
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    if not os.path.exists(latest_path):
        print("✨ Starting fresh training.")
        return 0
    device = next(spatial_model.parameters()).device
    checkpoint = torch.load(latest_path, map_location=device)
    spatial_model.load_state_dict(checkpoint['spatial_model_state_dict'])
    spectral_model.load_state_dict(checkpoint['spectral_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"🔄 Resuming from Epoch {checkpoint['epoch']}")
    return checkpoint['epoch']


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spatial_model = SpatialPriorDiffusion().to(device)
    spectral_model = SpectralPriorDiffusion().to(device)

    # RTX 5050 Optimization Flag (Optional)
    spatial_model.unet.enable_gradient_checkpointing()
    spectral_model.unet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(list(spatial_model.parameters()) + list(spectral_model.parameters()), lr=1e-6)
    trainer = SelfSupervisedLoop(spatial_model, spectral_model, scale_factor=4)

    dataset = DIV2KUnsupervisedDataset(lr_dir="datasets/DIV2K/train/", target_patch_size=(256, 256), scale_factor=4)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    start_epoch = load_checkpoint(spatial_model, spectral_model, optimizer)
    epochs = 100

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        spatial_model.train()
        spectral_model.train()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        for batch_idx, (raw_lr_imgs, pseudo_hrs) in enumerate(progress_bar):
            raw_lr_imgs = raw_lr_imgs.to(device)
            pseudo_hrs = pseudo_hrs.to(device)

            loss_val = trainer.train_step(raw_lr_imgs, pseudo_hrs, optimizer)
            epoch_loss += loss_val
            progress_bar.set_postfix({"batch_loss": f"{loss_val:.4f}"})

        print(f"   ↳ Finished Epoch {epoch + 1} | Avg Loss: {epoch_loss / len(dataloader):.6f}")
        if (epoch + 1) % 5 == 0:
            save_checkpoint(epoch + 1, spatial_model, spectral_model, optimizer)
