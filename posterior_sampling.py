import torch
import torch.nn.functional as F


class DualPriorPosteriorSampler:
    def __init__(self, spatial_model, spectral_model, num_steps=1000):
        self.spatial_model = spatial_model.eval()
        self.spectral_model = spectral_model.eval()
        self.num_steps = num_steps
        self.beta = torch.linspace(1e-4, 0.02, num_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    @torch.enable_grad()
    def reconstruct(self, lr_img: torch.Tensor, hr_shape: tuple,
                    alpha_spatial: float = 0.5, lambda_dc: float = 0.5) -> torch.Tensor:
        device = lr_img.device
        B, C, H, W = hr_shape
        x = torch.randn(hr_shape, device=device)

        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)

        for t in reversed(range(self.num_steps)):
            timestep_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            x = x.detach().requires_grad_(True)

            # Grad calculation stays enabled through model calls to preserve tracking graph
            noise_pred_spatial = self.spatial_model(x, timestep_tensor)

            spectral_x = self.spectral_model.img_to_spectral_tensor(x)
            spec_scale = spectral_x.std() + 1e-8
            noise_pred_spectral_domain = self.spectral_model(spectral_x / spec_scale, timestep_tensor) * spec_scale
            noise_pred_spectral = self.spectral_model.spectral_tensor_to_img(noise_pred_spectral_domain)

            noise_pred = alpha_spatial * noise_pred_spatial + (1 - alpha_spatial) * noise_pred_spectral

            a_bar = self.alpha_bar[t]
            a = self.alpha[t]
            if t > 0:
                z = torch.randn_like(x)
                sigma = torch.sqrt((1 - self.alpha_bar[t - 1]) / (1 - a_bar) * self.beta[t])
            else:
                z, sigma = 0, 0

            # Dynamic inner clamping removed to avoid step-1 optimization flattening
            x0_pred = (x - torch.sqrt(1 - a_bar) * noise_pred) / torch.sqrt(a_bar)

            x_next_mean = (torch.sqrt(a) * (1 - self.alpha_bar[t - 1]) / (1 - a_bar)) * x + \
                          (torch.sqrt(self.alpha_bar[t - 1]) * self.beta[t] / (1 - a_bar)) * x0_pred
            x_next = x_next_mean + sigma * z

            # Data Consistency Guidance Update Block
            lr_simulated = F.interpolate(x0_pred, size=(lr_img.shape[-2], lr_img.shape[-1]), mode='bicubic',
                                         align_corners=False)
            dc_loss = F.mse_loss(lr_simulated, lr_img)

            dc_grad = torch.autograd.grad(outputs=dc_loss, inputs=x)[0]
            norm_factor = torch.norm(dc_grad) + 1e-8

            x = x_next - (lambda_dc / norm_factor) * dc_grad
            x = torch.clamp(x, -0.5, 1.5)

        return torch.clamp(x, 0.0, 1.0)