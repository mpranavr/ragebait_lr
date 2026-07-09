import torch
import torch.nn as nn
from diffusers import UNet2DModel

class SpatialPriorDiffusion(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=256,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        )

    def forward(self, sample, timestep):
        return self.unet(sample, timestep).sample

class SpectralPriorDiffusion(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.fft_channels = in_channels * 2
        self.unet = UNet2DModel(
            sample_size=256,
            in_channels=self.fft_channels,
            out_channels=self.fft_channels,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        )

    def img_to_spectral_tensor(self, img):
        """Converts spatial image to real-valued 6-channel frequency representation (B, 2*C, H, W)"""
        fft_map = torch.fft.fft2(img, dim=(-2, -1))
        fft_map = torch.fft.fftshift(fft_map, dim=(-2, -1))
        return torch.cat([torch.real(fft_map), torch.imag(fft_map)], dim=1)

    def spectral_tensor_to_img(self, spectral_tensor):
        """Converts the Real/Imaginary frequency tensor back to standard pixel space"""
        real_part, imag_part = torch.chunk(spectral_tensor, 2, dim=1)
        fft_map = torch.complex(real_part, imag_part)
        fft_map = torch.fft.ifftshift(fft_map, dim=(-2, -1))
        return torch.real(torch.fft.ifft2(fft_map, dim=(-2, -1)))

    def forward(self, spectral_sample, timestep):
        return self.unet(spectral_sample, timestep).sample