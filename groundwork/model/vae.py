import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels, stride=1, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.stride = stride
        self.conv1 = (
            nn.ConvTranspose2d(
                channels, channels, 3, stride=stride, padding=1, output_padding=stride - 1
            )
            if upsample
            else nn.Conv2d(channels, channels, 3, stride=stride, padding=1)
        )
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()  # Swish

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        if self.upsample:
            shortcut = F.interpolate(x, scale_factor=2)
        elif self.stride > 1:
            shortcut = F.avg_pool2d(x, self.stride)
        else:
            shortcut = x
        return self.act(h + shortcut)


class VAEEncoder(nn.Module):
    def __init__(self, in_channels=5, latent_channels=4, base_ch=64):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            ResBlock(base_ch, stride=2),           # 512 → 256
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            ResBlock(base_ch * 2, stride=2),       # 256 → 128
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, padding=1),
            ResBlock(base_ch * 4, stride=2),       # 128 → 64
        )
        self.to_mu = nn.Conv2d(base_ch * 4, latent_channels, 1)
        self.to_logvar = nn.Conv2d(base_ch * 4, latent_channels, 1)

    def forward(self, x):
        h = self.blocks(x)
        return self.to_mu(h), self.to_logvar(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_channels=4, out_channels=5, base_ch=64):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(latent_channels, base_ch * 4, 3, padding=1),
            ResBlock(base_ch * 4, stride=2, upsample=True),  # 64 → 128
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),
            ResBlock(base_ch * 2, stride=2, upsample=True),  # 128 → 256
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            ResBlock(base_ch, stride=2, upsample=True),      # 256 → 512
            nn.Conv2d(base_ch, out_channels, 3, padding=1),
        )

    def forward(self, z):
        return self.blocks(z)


class RoadVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VAEEncoder()
        self.decoder = VAEDecoder()

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
