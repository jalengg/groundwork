import torch
import torch.nn.functional as F


class DDPM:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)

    def _to_device(self, device):
        self.alpha_bar = self.alpha_bar.to(device)

    def forward_diffusion(self, x0, t):
        """Add noise to x0 at timestep t. Returns (x_t, eps)."""
        self._to_device(x0.device)
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        eps = torch.randn_like(x0)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        return x_t, eps

    def training_loss(self, model, x0, cond, cfg_prob=0.5):
        """DDPM epsilon-prediction MSE loss with classifier-free guidance dropout."""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t, eps = self.forward_diffusion(x0, t)
        mask = (torch.rand(B, device=x0.device) > cfg_prob).float()
        cond_masked = cond * mask[:, None, None, None]
        eps_pred = model(x_t, t, cond_masked)
        return F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def sample_ddim(self, model, cond, n_steps=50, guidance_scale=3.0,
                    latent_shape=(1, 4, 64, 64)):
        """DDIM sampling with classifier-free guidance."""
        self._to_device(cond.device)
        device = cond.device
        x = torch.randn(latent_shape, device=device)
        step_size = self.T // n_steps
        timesteps = list(range(self.T - 1, -1, -step_size))

        for t_val in timesteps:
            t = torch.full((latent_shape[0],), t_val, device=device, dtype=torch.long)
            eps_cond = model(x, t, cond)
            eps_uncond = model(x, t, torch.zeros_like(cond))
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            ab = self.alpha_bar[t_val]
            ab_prev = self.alpha_bar[max(t_val - step_size, 0)]
            x0_pred = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            x0_pred = x0_pred.clamp(-3, 3)
            x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps
        return x
