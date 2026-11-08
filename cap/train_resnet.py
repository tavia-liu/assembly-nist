"""PPO from pixels, symmetric, with a fine-tuned ImageNet ResNet-18 trunk as the shared encoder (cap/resnet_nets.py).
No oracle part pose and no point clouds anywhere; same env/reward/PPO as cap/train_pixel.py.

Differences from train_pixel.py:
  * encoder = ResNet-18 conv1..layer3 (ImageNet init, BN frozen), trained with lr * --backbone-lr-mult (default 0.1);
    --freeze-backbone keeps the trunk fixed (only keypoint/projection/heads learn)
  * random-shift augmentation is applied once when a frame is collected and the shifted frame is what is stored, so
    the PPO ratio compares identical inputs (fresh shifts inside the update inflate the KL and pin the adaptive lr at
    its floor, which is what happened in the small-conv pixel run); --no-aug disables it
  * frames are rendered at CAP_PIX_RES (default 128) by cap/pixel_env.py

  PYTHONPATH=.:cap MUJOCO_GL=egl CAP_PIX_RES=128 python cap/train_resnet.py --task P4 --run-dir runs/peg_4mm_resnet_near --envs 128 --iters 30000
"""
from __future__ import annotations

import sys

import torch

import assembly_policy.train as T
from assembly_policy.rl import vecenv
from pixel_env import RES
from pixel_nets import RandomShift
from resnet_nets import ResNetActorCritic
from train_pixel import PixelLocalVecEnv

if "img" not in T.OBS:                         # importing train_pixel already registers it
    vecenv.OBS_KEYS = vecenv.OBS_KEYS + ("img",)
    T.OBS = T.OBS + ("img",)


class _ScaledGroup(dict):
    """Optimizer param group whose lr is scaled by lr_mult whenever the trainer sets it (adaptive-KL schedule)."""

    def __setitem__(self, k, v):
        if k == "lr":
            v = v * self.get("lr_mult", 1.0)
        super().__setitem__(k, v)


class ResNetTrainer(T.Trainer):
    def __init__(self, a):
        self._swapped = False                      # base __init__ must not resume into the point-cloud model
        super().__init__(a)
        self.model = ResNetActorCritic(res=a.pix_res, freeze=a.freeze_backbone, init_std=a.init_std, max_std=a.max_std,
                                       min_std=a.min_std).to(self.dev)
        groups = [dict(params=self.model.other_parameters(), lr=a.lr, lr_mult=1.0)]
        if not a.freeze_backbone:
            groups.append(dict(params=self.model.backbone_parameters(), lr=a.lr * a.backbone_lr_mult, lr_mult=a.backbone_lr_mult))
        self.opt = torch.optim.Adam(groups)
        self._wrap_groups()
        self.aug = RandomShift(a.aug_pad).to(self.dev) if a.aug_pad > 0 else None
        n = sum(p.numel() for p in self.model.parameters())
        nb = sum(p.numel() for p in self.model.encoder.trunk.parameters())
        print(f"[train_resnet] {n / 1e6:.2f}M parameters, backbone {nb / 1e6:.2f}M ({'frozen' if a.freeze_backbone else f'lr x{a.backbone_lr_mult}'}), "
              f"aug pad {a.aug_pad}, {a.pix_res}x{a.pix_res} x 2 cameras", flush=True)
        self._swapped = True
        self._maybe_resume()

    def _wrap_groups(self):
        self.opt.param_groups = [g if isinstance(g, _ScaledGroup) else _ScaledGroup(g) for g in self.opt.param_groups]
        for g in self.opt.param_groups:
            g["lr"] = self.lr

    def _maybe_resume(self):
        if self._swapped:
            super()._maybe_resume()
            self._wrap_groups()                    # load_state_dict rebuilds plain dicts; re-apply the multipliers

    def _obs(self, o, ids=None):
        ob = super()._obs(o, ids)
        sel = (lambda x: x) if ids is None else (lambda x: x[ids])
        img = torch.as_tensor(sel(o["img"]), device=self.dev)            # uint8 [B, 2, H, W, 3]
        if self.aug is not None and self.model.training:
            B, C, H, W, _ = img.shape
            x = img.permute(0, 1, 4, 2, 3).reshape(B * C, 3, H, W).float()
            with torch.no_grad():
                x = self.aug(x)
            img = x.round_().clamp_(0, 255).to(torch.uint8).reshape(B, C, 3, H, W).permute(0, 1, 3, 4, 2).contiguous()
        ob["img"] = img
        return ob

    def _points(self, ob):
        return self.model.encode(ob["img"])


def _pop(argv, flag, default=None, is_flag=False):
    if flag not in argv:
        return default, argv
    i = argv.index(flag)
    if is_flag:
        return True, argv[:i] + argv[i + 1:]
    return argv[i + 1], argv[:i] + argv[i + 2:]


def main():
    argv = sys.argv[1:]
    task, argv = _pop(argv, "--task", "P4")
    bb_mult, argv = _pop(argv, "--backbone-lr-mult", "0.1")
    aug_pad, argv = _pop(argv, "--aug-pad", "4")
    freeze, argv = _pop(argv, "--freeze-backbone", False, is_flag=True)
    no_aug, argv = _pop(argv, "--no-aug", False, is_flag=True)
    argv = ["--tasks", task, "--bank-dir", "data/banks_near", "--sampler", "uniform", "--no-eval"] + argv
    T.LocalVecEnv = PixelLocalVecEnv
    a = T.parse(argv)
    a.pix_res = RES; a.arch = "resnet"
    a.backbone_lr_mult = float(bb_mult); a.freeze_backbone = bool(freeze); a.aug_pad = 0 if no_aug else int(aug_pad)
    print(f"[train_resnet] task {task}, symmetric ResNet-18 pixel PPO, {a.envs} envs, {RES}x{RES} x 2 cameras", flush=True)
    tr = ResNetTrainer(a)
    tr.train()
    for p in tr._children:
        p.wait()


if __name__ == "__main__":
    main()
