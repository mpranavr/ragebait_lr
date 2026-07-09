import torch
import torch.nn.functional as F
import math


class SpectralDegradation:
    """Handles frequency-domain modifications (Blurring / High-Frequency Suppression)"""

    def __init__(self, img: torch.Tensor):
        self.img = img

    def _prepare_tensor(self):
        if len(self.img.shape) == 2:
            return self.img.unsqueeze(0), True
        return self.img, False

    def low_pass_filter(self, radius_percentage: float) -> torch.Tensor:
        """Cuts off high frequencies past a certain radial threshold."""
        img_3d, is_2d = self._prepare_tensor()
        C, H, W = img_3d.shape

        fft_shifted = torch.fft.fftshift(torch.fft.fft2(img_3d), dim=(-2, -1))

        center_h, center_w = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H) - center_h, torch.arange(W) - center_w, indexing='ij')
        distance = torch.sqrt(y ** 2 + x ** 2).to(img_3d.device)

        max_radius = min(center_h, center_w) * radius_percentage
        mask = (distance <= max_radius).float().unsqueeze(0)

        filtered_shifted = fft_shifted * mask
        degraded_img = torch.real(torch.fft.ifft2(torch.fft.ifftshift(filtered_shifted, dim=(-2, -1))))

        return degraded_img.squeeze(0) if is_2d else degraded_img


class SpatialDegradation:
    """Handles geometric adjustments (Resolution reduction / Scaling)"""

    def __init__(self, img: torch.Tensor):
        self.img = img

    def _prepare_tensor(self):
        if len(self.img.shape) == 2:
            return self.img.unsqueeze(0).unsqueeze(0), True
        return self.img.unsqueeze(0), False

    def downsample(self, scale_factor: int, mode: str = 'bicubic') -> torch.Tensor:
        """Downsamples the image by a specific scale factor (e.g., 2x, 4x)."""
        img_4d, is_2d = self._prepare_tensor()
        H, W = img_4d.shape[-2], img_4d.shape[-1]

        target_size = (H // scale_factor, W // scale_factor)
        align_corners = False if mode in ['bilinear', 'bicubic'] else None

        lr_tensor = F.interpolate(
            img_4d,
            size=target_size,
            mode=mode,
            align_corners=align_corners
        )

        lr_tensor = lr_tensor.squeeze(0)
        return lr_tensor.squeeze(0) if is_2d else lr_tensor


class SRDegradationPipeline:
    """
    A unified pipeline to transform High-Resolution (HR) images
    into realistic Low-Resolution (LR) pairs for Super-Resolution training.
    """

    def __init__(self, scale_factor: int = 4):
        self.scale_factor = scale_factor

    def add_gaussian_noise(self, img: torch.Tensor, std: float = 0.05) -> torch.Tensor:
        """Simulates camera sensor noise."""
        noise = torch.randn_like(img) * std
        noisy_img = img + noise
        return torch.clamp(noisy_img, 0.0, 1.0)  # Keeps pixel values valid between 0 and 1

    def generate_lr_pair(self, hr_img: torch.Tensor, radius_pct: float = 0.5,
                         downsample_mode: str = 'bicubic', noise_std: float = 0.02) -> torch.Tensor:
        """
        Executes the classic SR degradation chain:
        1. Spectral low-pass filter (Anti-aliasing blur)
        2. Spatial downsampling (HR to LR grid transformation)
        3. Sensor noise injection
        """
        # Step 1: Spectral Degradation (Blur)
        spectral = SpectralDegradation(hr_img)
        blurred_hr = spectral.low_pass_filter(radius_percentage=radius_pct)

        # Step 2: Spatial Degradation (Downsample)
        spatial = SpatialDegradation(blurred_hr)
        clean_lr = spatial.downsample(scale_factor=self.scale_factor, mode=downsample_mode)

        # Step 3: Noise Injection
        final_lr = self.add_gaussian_noise(clean_lr, std=noise_std)

        return final_lr