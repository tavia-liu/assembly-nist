"""Actor on frozen image features, critic on privileged point clouds (asymmetric actor-critic).

Drop-in for assembly_policy.rl.nets.ActorCritic: dist()/value() take x = (pts, feat) instead of pts, so the trainer
only changes what _points() returns.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from assembly_policy.rl.nets import ActorCritic, ResBlock, RunningNorm, SimBa


class FeatSimBa(nn.Module):
    def __init__(self, feat_dim, proprio_dim, out_dim, hidden=256, expansion=1024, blocks=2, enc=256):
        super().__init__()
        self.fnorm = RunningNorm(feat_dim)
        self.enc = nn.Sequential(nn.Linear(feat_dim, enc), nn.ELU(), nn.Linear(enc, 128), nn.ELU())
        self.inp = nn.Linear(128 + proprio_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden, expansion) for _ in range(blocks)])
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, feat, prop):
        return self.head(self.ln(self.blocks(self.inp(torch.cat([self.enc(self.fnorm(feat)), prop], -1)))))


class FeatActorCritic(ActorCritic):
    def __init__(self, n_pts, feat_dim=1024, proprio_dim=150, act_dim=8, **kw):
        super().__init__(n_pts, proprio_dim, act_dim, **kw)
        self.actor = FeatSimBa(feat_dim, proprio_dim, act_dim)
        nn.init.orthogonal_(self.actor.head.weight, 0.01); nn.init.zeros_(self.actor.head.bias)

    def dist(self, x, prop):
        pts, feat = x
        with torch.no_grad():
            if self.training:
                self.actor.fnorm.update(feat)
        mean = self.actor(feat, self.norm(prop))
        return torch.distributions.Normal(mean, self.std().expand_as(mean))

    def value(self, x, prop):
        pts, _ = x
        return self.critic(pts, self.norm(prop)).squeeze(-1)
