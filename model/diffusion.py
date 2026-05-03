import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_class_weight_latent(vae, road, class_weights):
    """DRoLaS Eq. 9: W = Σ_i w_i · E(m_i).

    For each road class i, build a 5-channel input where only channel i is
    populated (a one-hot 'as if only class i existed'), encode through the
    frozen VAE encoder, weighted-sum across classes. Returns a per-sample
    latent-space weight tensor matching x0's shape.

    Earlier impl approximated E(m_i) with adaptive_avg_pool2d, which
    collapses to a 1-channel scalar mass map and discards the encoder's
    learned spatial structure. This is paper-literal.

    Args:
        vae: frozen RoadVAE (eval mode).
        road: (B, C_cls, H, W) one-hot road raster.
        class_weights: (C_cls,) tensor of per-class weights.
    Returns:
        (B, latent_channels, latent_H, latent_W) weight tensor.
    """
    B, C_cls, H, W = road.shape
    weights = class_weights.to(road.device)
    weight_latent = None
    for i in range(C_cls):
        m_i_5ch = torch.zeros_like(road)
        m_i_5ch[:, i:i + 1] = road[:, i:i + 1]
        mu, _ = vae.encode(m_i_5ch)
        contrib = weights[i] * mu
        weight_latent = contrib if weight_latent is None else weight_latent + contrib
    return weight_latent


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

    def training_loss(self, model, x0, cond, cfg_prob=0.5, weight_latent=None):
        """DDPM epsilon-prediction loss with classifier-free guidance dropout.

        If `weight_latent` is provided (shape matching x0), applies the DRoLaS
        Eq. 8 class-weighted denoising loss `L_w = E[||W ⊙ (ε − ε_θ)||²]`.
        Otherwise falls back to uniform MSE.

        Build `weight_latent` via `compute_class_weight_latent(vae, road, w)`.
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t, eps = self.forward_diffusion(x0, t)
        mask = (torch.rand(B, device=x0.device) > cfg_prob).float()
        cond_masked = cond * mask[:, None, None, None]
        eps_pred = model(x_t, t, cond_masked)

        if weight_latent is None:
            return F.mse_loss(eps_pred, eps)
        return ((eps - eps_pred) * weight_latent).pow(2).mean()

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
