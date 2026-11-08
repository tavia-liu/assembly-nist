"""Symmetric actor-critic on a fine-tuned ImageNet ResNet-18 trunk (conv1 .. layer3) over the two camera images.
Spatial-softmax keypoints + channel means on the 8x8 (res/16) feature map -> linear -> LayerNorm -> tanh latent,
shared by actor and critic. No privileged information anywhere: neither head sees the part pose or point clouds.

Same interface as cap/pixel_nets.PixelActorCritic: encode(img) -> latent; dist(latent, prop); value(latent, prop).
BatchNorm layers are frozen (ImageNet statistics): PPO minibatches are on-policy and small, batch statistics would drift.
"""
from __future__ import annotations

import contextlib
import os

import torch
import torch.nn as nn
import torchvision
from torchvision.ops.misc import FrozenBatchNorm2d

from assembly_policy.rl.nets import ActorCritic
from pixel_nets import LATENT, Head, SpatialSoftmax

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
TRUNK_CH = 256                       # layer3 output channels


def _freeze_bn(m: nn.Module):
    for name, c in m.named_children():
        if isinstance(c, nn.BatchNorm2d):
            f = FrozenBatchNorm2d(c.num_features, eps=c.eps)
            f.weight.copy_(c.weight.data); f.bias.copy_(c.bias.data)
            f.running_mean.copy_(c.running_mean); f.running_var.copy_(c.running_var)
            setattr(m, name, f)
        else:
            _freeze_bn(c)


class ResNetEncoder(nn.Module):
    def __init__(self, res=128, n_cam=2, out=LATENT, freeze=False, pretrained=True):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=w)
        self.trunk = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3)
        _freeze_bn(self.trunk)
        if freeze:
            for p in self.trunk.parameters():
                p.requires_grad_(False)
        hw = (res + 15) // 16
        self.ss = SpatialSoftmax(hw, hw)
        self.proj = nn.Linear(n_cam * (2 * TRUNK_CH + TRUNK_CH), out)
        self.ln = nn.LayerNorm(out)
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1))
        self.n_cam, self.res, self.freeze = n_cam, res, freeze
        self.amp = bool(int(os.environ.get("CAP_AMP", "1")))     # bf16 autocast for the trunk (A40); heads stay fp32

    def forward(self, img):
        """img uint8 [B, n_cam, H, W, 3] -> latent [B, out]."""
        B = img.shape[0]
        x = img.permute(0, 1, 4, 2, 3).reshape(B * self.n_cam, 3, img.shape[2], img.shape[3]).float() / 255.0
        x = (x - self.mean) / self.std
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (self.amp and x.is_cuda) else contextlib.nullcontext()
        with ctx:
            if self.freeze:
                with torch.no_grad():
                    h = self.trunk(x)
            else:
                h = self.trunk(x)
        h = h.float()
        f = torch.cat([self.ss(h), h.mean((2, 3))], -1).reshape(B, -1)
        return torch.tanh(self.ln(self.proj(f)))


class ResNetActorCritic(ActorCritic):
    def __init__(self, res=128, proprio_dim=150, act_dim=8, latent=LATENT, freeze=False, pretrained=True, **kw):
        super().__init__(16, proprio_dim, act_dim, **kw)          # base builds point-cloud nets; replaced below
        self.encoder = ResNetEncoder(res, out=latent, freeze=freeze, pretrained=pretrained)
        self.actor = Head(latent + proprio_dim, act_dim)
        self.critic = Head(latent + proprio_dim, 1)
        nn.init.orthogonal_(self.actor.head.weight, 0.01); nn.init.zeros_(self.actor.head.bias)
        nn.init.orthogonal_(self.critic.head.weight, 1.0); nn.init.zeros_(self.critic.head.bias)

    def backbone_parameters(self):
        return [p for p in self.encoder.trunk.parameters() if p.requires_grad]

    def other_parameters(self):
        bb = {id(p) for p in self.encoder.trunk.parameters()}
        return [p for p in self.parameters() if id(p) not in bb and p.requires_grad]

    def encode(self, img):
        return self.encoder(img)

    def dist(self, z, prop):
        mean = self.actor(torch.cat([z, self.norm(prop)], -1))
        return torch.distributions.Normal(mean, self.std().expand_as(mean))

    def value(self, z, prop):
        return self.critic(torch.cat([z, self.norm(prop)], -1)).squeeze(-1)
