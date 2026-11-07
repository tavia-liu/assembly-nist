"""BatchedAssemblyEnv whose observation carries frozen-encoder features of the two camera images.

Each control step renders both cameras at low resolution (RGB only), applies a random shift (DrQ-style) and passes
the images through a frozen ImageNet ResNet-18; observe() adds `feat` = [global 512 | wrist 512]. The simulator's
part pose stays in the observation for the privileged critic only (the actor never sees it; see feat_nets.py).

    env = EncodedEnv([task], np.zeros(N, int), ecfg)      # drop-in for BatchedAssemblyEnv
"""
from __future__ import annotations

import os
import time

import mujoco
import numpy as np
import torch
import torchvision

from assembly_policy.sim.env import STATE_SPEC, BatchedAssemblyEnv
from assembly_policy.sim.scene import build_xml
from perception import add_cameras

RES = int(os.environ.get("CAP_FEAT_RES", "128"))
FEAT_DIM = 1024
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Encoder:
    """Frozen ResNet-18 (ImageNet) -> 512-d per image, batched on the GPU when there is one."""

    def __init__(self, device=None):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        self.net = net.eval().to(self.dev)
        self.mean, self.std = _MEAN.to(self.dev), _STD.to(self.dev)

    @torch.no_grad()
    def __call__(self, imgs, shift=4):
        """imgs uint8 [B,H,W,3] -> float [B,512]; random shift augmentation (pad + crop) per image."""
        x = torch.as_tensor(np.ascontiguousarray(imgs), device=self.dev).permute(0, 3, 1, 2).float() / 255.0
        if shift:
            B, _, H, W = x.shape
            x = torch.nn.functional.pad(x, (shift,) * 4, mode="replicate")
            dx = torch.randint(0, 2 * shift + 1, (B,), device=self.dev); dy = torch.randint(0, 2 * shift + 1, (B,), device=self.dev)
            idx_y = (dy[:, None] + torch.arange(H, device=self.dev)[None]).view(B, 1, H, 1).expand(B, 3, H, W)
            x = torch.gather(x, 2, idx_y.expand(B, 3, H, W + 2 * shift)[:, :, :, :W + 2 * shift]) if False else \
                torch.stack([x[b, :, dy[b]:dy[b] + H, dx[b]:dx[b] + W] for b in range(B)])
        x = (x - self.mean) / self.std
        return self.net(x).float().cpu().numpy()


class EncodedEnv(BatchedAssemblyEnv):
    def __init__(self, task_names, task_of_env, ecfg=None, pcfg=None, res=RES, encoder=None):
        super().__init__(task_names, task_of_env, ecfg, pcfg)
        assert len(self.task_names) == 1
        self.vm = mujoco.MjModel.from_xml_string(add_cameras(build_xml(self.task_names[0], visual=True)))
        self.vd = mujoco.MjData(self.vm)
        self.r = mujoco.Renderer(self.vm, res, res)
        self.cams = [mujoco.mj_name2id(self.vm, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in ("global", "wrist")]
        self.enc = encoder or Encoder()
        self.feat = np.zeros((self.N, FEAT_DIM), np.float32)
        self.stats = dict(render=0.0, encode=0.0, steps=0)

    def _render_all(self, ids):
        imgs = np.empty((len(ids), 2, self.r.height, self.r.width, 3), np.uint8)
        for j, i in enumerate(ids):
            mujoco.mj_setState(self.vm, self.vd, self.state[i], STATE_SPEC)
            mujoco.mj_forward(self.vm, self.vd)
            for c, cam in enumerate(self.cams):
                self.r.update_scene(self.vd, camera=cam)
                imgs[j, c] = self.r.render()
        return imgs

    def _refresh(self, ids):
        ids = np.atleast_1d(ids)
        if len(ids) == 0:
            return
        t0 = time.perf_counter()
        imgs = self._render_all(ids)
        t1 = time.perf_counter()
        f = self.enc(imgs.reshape(-1, *imgs.shape[2:]))
        self.feat[ids] = f.reshape(len(ids), -1)
        self.stats["render"] += t1 - t0; self.stats["encode"] += time.perf_counter() - t1; self.stats["steps"] += len(ids)

    def set_states(self, env_ids, states, task_ids=None):
        super().set_states(env_ids, states, task_ids)
        self._refresh(env_ids)

    def step(self, action):
        out = super().step(action)
        self._refresh(np.arange(self.N))
        return out

    def observe(self):
        o = super().observe()
        o["feat"] = self.feat.copy()
        return o
