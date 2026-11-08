"""Symmetric pixel actor-critic: one learned conv encoder (shared by actor and critic) on the two camera images,
spatial-softmax keypoints, random-shift augmentation while training; heads take [latent | proprioception].
No privileged information anywhere: neither head sees the part pose or point clouds.

Drop-in for assembly_policy.rl.nets.ActorCritic: encode(img) -> latent; dist(latent, prop); value(latent, prop).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from assembly_policy.rl.nets import ActorCritic, ResBlock

LATENT = 128


class RandomShift(nn.Module):
    """DrQ-v2 random shift: replicate-pad by `pad` pixels, then crop at a random integer offset per image."""

    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, _, h, w = x.shape
        x = F.pad(x, (self.pad,) * 4, mode="replicate")
        eps = 1.0 / (h + 2 * self.pad)
        ar = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        ar = ar.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base = torch.cat([ar, ar.transpose(1, 0)], dim=2).unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype) * (2.0 / (h + 2 * self.pad))
        return F.grid_sample(x, base + shift, padding_mode="zeros", align_corners=False)


class SpatialSoftmax(nn.Module):
    """Per-channel softmax over the feature map -> expected (x, y) in [-1, 1]: C keypoints = 2C numbers."""

    def __init__(self, h, w):
        super().__init__()
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        self.register_buffer("px", xs.reshape(1, 1, -1)); self.register_buffer("py", ys.reshape(1, 1, -1))
        self.temp = nn.Parameter(torch.ones(1))

    def forward(self, x):
        p = F.softmax(x.flatten(2) / self.temp, dim=-1)                       # [B, C, HW]
        return torch.cat([(p * self.px).sum(-1), (p * self.py).sum(-1)], -1)   # [B, 2C]


class ConvEncoder(nn.Module):
    """Two cameras, shared weights: 4 conv layers -> 64 x (res/8)^2 -> [keypoints 128 | channel means 64] per camera
    -> linear -> LayerNorm -> tanh latent."""

    def __init__(self, res=96, n_cam=2, out=LATENT, pad=4):
        super().__init__()
        self.aug = RandomShift(pad)
        self.conv = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU())
        hw = (res + 7) // 8
        self.ss = SpatialSoftmax(hw, hw)
        self.proj = nn.Linear(n_cam * (2 * 64 + 64), out)
        self.ln = nn.LayerNorm(out)
        self.n_cam, self.res = n_cam, res

    def forward(self, img):
        """img uint8 [B, n_cam, H, W, 3] -> latent [B, out]."""
        B = img.shape[0]
        x = img.permute(0, 1, 4, 2, 3).reshape(B * self.n_cam, 3, img.shape[2], img.shape[3]).float() / 255.0 - 0.5
        if self.training:
            x = self.aug(x)
        h = self.conv(x)
        f = torch.cat([self.ss(h), h.mean((2, 3))], -1).reshape(B, -1)
        return torch.tanh(self.ln(self.proj(f)))


class Head(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, expansion=1024, blocks=2):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden, expansion) for _ in range(blocks)])
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.head(self.ln(self.blocks(self.inp(x))))


class PixelActorCritic(ActorCritic):
    def __init__(self, res=96, proprio_dim=150, act_dim=8, latent=LATENT, **kw):
        super().__init__(16, proprio_dim, act_dim, **kw)          # base builds point-cloud nets; replaced below
        self.encoder = ConvEncoder(res, out=latent)
        self.actor = Head(latent + proprio_dim, act_dim)
        self.critic = Head(latent + proprio_dim, 1)
        nn.init.orthogonal_(self.actor.head.weight, 0.01); nn.init.zeros_(self.actor.head.bias)
        nn.init.orthogonal_(self.critic.head.weight, 1.0); nn.init.zeros_(self.critic.head.bias)

    def encode(self, img):
        return self.encoder(img)

    def dist(self, z, prop):
        mean = self.actor(torch.cat([z, self.norm(prop)], -1))
        return torch.distributions.Normal(mean, self.std().expand_as(mean))

    def value(self, z, prop):
        return self.critic(torch.cat([z, self.norm(prop)], -1)).squeeze(-1)
