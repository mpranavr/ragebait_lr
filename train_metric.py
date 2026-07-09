import os
import glob
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from diffusion import SpatialPriorDiffusion, SpectralPriorDiffusion

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import csv
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from diffusers.models import UNet2DModel
from tqdm import tqdm

# ------------------------------
# Evaluation Metrics
# ------------------------------
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure
)



class MetricTracker:
    """
    Computes all evaluation metrics for Super Resolution.
    """

    def __init__(self, device):

        self.device = device

        self.psnr_metric = PeakSignalNoiseRatio(
            data_range=1.0
        ).to(device)

        self.ssim_metric = StructuralSimilarityIndexMeasure(
            data_range=1.0
        ).to(device)



        self.reset()

    def reset(self):

        self.loss = 0.0
        self.psnr = 0.0
        self.ssim = 0.0
        self.mse = 0.0
        self.rmse = 0.0
        self.mae = 0.0

        self.count = 0

    @torch.no_grad()
    def update(self, prediction, target, loss):

        prediction = prediction.clamp(0, 1)
        target = target.clamp(0, 1)

        mse = F.mse_loss(prediction, target)
        mae = F.l1_loss(prediction, target)

        rmse = torch.sqrt(mse)

        psnr = self.psnr_metric(
            prediction,
            target
        )

        ssim = self.ssim_metric(
            prediction,
            target
        )



        self.loss += loss
        self.psnr += psnr.item()
        self.ssim += ssim.item()
        self.mse += mse.item()
        self.rmse += rmse.item()
        self.mae += mae.item()

        self.count += 1

    def average(self):

        if self.count == 0:
            return {}

        return {

            "Loss": self.loss / self.count,

            "PSNR": self.psnr / self.count,

            "SSIM": self.ssim / self.count,

            "MSE": self.mse / self.count,

            "RMSE": self.rmse / self.count,

            "MAE": self.mae / self.count

        }

class CSVLogger:

    def __init__(self, filename="training_metrics.csv"):

        self.filename = filename

        if not os.path.exists(filename):

            with open(filename, "w", newline="") as f:

                writer = csv.writer(f)

                writer.writerow([

                    "Epoch",

                    "Loss",

                    "PSNR",

                    "SSIM",

                    "MSE",

                    "RMSE",

                    "MAE",

                    "LearningRate",

                    "EpochTime"

                ])

    def log(

        self,

        epoch,

        metrics,

        lr,

        epoch_time

    ):

        with open(self.filename, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([

                epoch,

                metrics["Loss"],

                metrics["PSNR"],

                metrics["SSIM"],


                metrics["MSE"],

                metrics["RMSE"],

                metrics["MAE"],

                lr,

                epoch_time

            ])


class BestMetrics:

    def __init__(self):

        self.best_psnr = 0

        self.best_ssim = 0

        self.best_epoch = 0

    def update(
        self,
        epoch,
        metrics,
        spatial_model,
        spectral_model
    ):

        improved = False

        if metrics["PSNR"] > self.best_psnr:
            self.best_psnr = metrics["PSNR"]
            self.best_ssim = metrics["SSIM"]

            improved = True

        if improved:

            self.best_epoch = epoch

            torch.save(
                spatial_model.state_dict(),
                "best_spatial_prior_unet.pth"
            )

            torch.save(
                spectral_model.state_dict(),
                "best_spectral_prior_unet.pth"
            )

            print()

            print("★★★★★ NEW BEST MODEL SAVED ★★★★★")

            print(f"Epoch : {epoch}")

            print(f"Best PSNR : {self.best_psnr:.4f}")

            print(f"Best SSIM : {self.best_ssim:.6f}")

            print(f"Best Epoch    : {best_metrics.best_epoch}")

            print()


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

        # ------------------------------------------
        # Return tensors for evaluation metrics
        # (No change to training logic)
        # ------------------------------------------

        return (
            total_cycle_loss.item(),
            reconstructed_hr.detach(),
            pseudo_hr.detach()
        )


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

    metric_tracker = MetricTracker(device)

    csv_logger = CSVLogger()

    best_metrics = BestMetrics()

    dataset = DIV2KUnsupervisedDataset(lr_dir="datasets/DIV2K/train/", target_patch_size=(256, 256), scale_factor=4)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)



    start_epoch = load_checkpoint(spatial_model, spectral_model, optimizer)
    epochs = 100

    print("=" * 80)

    print("Training Configuration")

    print(f"Device        : {device}")

    print(f"Epochs        : {epochs}")

    print(f"Batch Size    : {dataloader.batch_size}")

    print(f"Learning Rate : {optimizer.param_groups[0]['lr']}")

    print(f"Dataset Size  : {len(dataset)}")

    print("=" * 80)

    for epoch in range(start_epoch, epochs):

        epoch_start_time = time.time()

        spatial_model.train()

        spectral_model.train()

        metric_tracker.reset()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")



        for batch_idx, (raw_lr_imgs, pseudo_hrs) in enumerate(progress_bar):
            raw_lr_imgs = raw_lr_imgs.to(device)
            pseudo_hrs = pseudo_hrs.to(device)

            loss_val, reconstructed_hr, target_hr = trainer.train_step(
                raw_lr_imgs,
                pseudo_hrs,
                optimizer
            )

            metric_tracker.update(
                reconstructed_hr,
                target_hr,
                loss_val
            )

            avg = metric_tracker.average()

            progress_bar.set_postfix({
                "Loss": f"{loss_val:.4f}",
                "PSNR": f"{avg['PSNR']:.2f}",
                "SSIM": f"{avg['SSIM']:.4f}"
            })

        metrics = metric_tracker.average()

        epoch_time = time.time() - epoch_start_time

        best_metrics.update(

            epoch + 1,

            metrics,

            spatial_model,

            spectral_model

        )

        current_lr = optimizer.param_groups[0]["lr"]

        csv_logger.log(
            epoch + 1,
            metrics,
            current_lr,
            epoch_time
        )

        print("\n")

        print("=" * 80)

        print(f"Epoch {epoch + 1}/{epochs}")

        print("-" * 80)

        print(f"Loss          : {metrics['Loss']:.6f}")

        print(f"PSNR          : {metrics['PSNR']:.4f} dB")

        print(f"SSIM          : {metrics['SSIM']:.6f}")


        print(f"MSE           : {metrics['MSE']:.8f}")

        print(f"RMSE          : {metrics['RMSE']:.8f}")

        print(f"MAE           : {metrics['MAE']:.8f}")

        print(f"Learning Rate : {current_lr:.8f}")

        print(f"Epoch Time    : {epoch_time:.2f} sec")

        print()

        print(f"Best PSNR     : {best_metrics.best_psnr:.4f}")

        print(f"Best SSIM     : {best_metrics.best_ssim:.6f}")

        print("=" * 80)

        print()


        if (epoch + 1) % 5 == 0:
            save_checkpoint(epoch + 1, spatial_model, spectral_model, optimizer)

    print()

    print("Converting CSV to Excel...")

    df = pd.read_csv("training_metrics.csv")

    plt.figure(figsize=(8, 5))

    plt.plot(df["Epoch"], df["Loss"])

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.grid(True)

    plt.savefig("loss_curve.png")

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(df["Epoch"], df["PSNR"])

    plt.xlabel("Epoch")

    plt.ylabel("PSNR")

    plt.title("PSNR")

    plt.grid(True)

    plt.savefig("psnr_curve.png")

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(df["Epoch"], df["SSIM"])

    plt.xlabel("Epoch")

    plt.ylabel("SSIM")

    plt.title("SSIM")

    plt.grid(True)

    plt.savefig("ssim_curve.png")

    plt.close()



    plt.figure(figsize=(8, 5))

    plt.plot(df["Epoch"], df["MSE"])

    plt.xlabel("Epoch")

    plt.ylabel("MSE")

    plt.title("MSE")

    plt.grid(True)

    plt.savefig("mse_curve.png")

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(df["Epoch"], df["RMSE"])

    plt.xlabel("Epoch")

    plt.ylabel("RMSE")

    plt.title("RMSE")

    plt.grid(True)

    plt.savefig("rmse_curve.png")

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(df["Epoch"], df["MAE"])

    plt.xlabel("Epoch")

    plt.ylabel("MAE")

    plt.title("MAE")

    plt.grid(True)

    plt.savefig("mae_curve.png")

    plt.close()

    print()

    print("Graphs saved successfully.")

    df.to_excel(
        "training_metrics.xlsx",
        index=False
    )

    print("Excel saved.")

    torch.save(
        spatial_model.state_dict(),
        "spatial_prior_unet.pth"
    )

    torch.save(
        spectral_model.state_dict(),
        "spectral_prior_unet.pth"
    )

    print("Final weights saved.")
