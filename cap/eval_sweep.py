"""Held-out eval curve for a PPO run: evaluate every kept checkpoint (sampled + deterministic, 128 near eval starts),
cache <run>/eval_<ckpt>[_det].json, and log eval/* against `it` into the run's wandb run.

  PYTHONPATH=.:cap MUJOCO_GL=egl python cap/eval_sweep.py --run-dir runs/state_P4 --kind state --wandb-id state_P4_near_ppo
  PYTHONPATH=.:cap MUJOCO_GL=egl python cap/eval_sweep.py --run-dir runs/feat_P4 --kind feat --wandb-id feat_P4_near_ppo_full
Re-run any time: checkpoints that already have both json files are skipped (only their numbers are re-logged).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--kind", choices=["state", "feat", "pixel"], required=True)
    ap.add_argument("--task", default="P4")
    ap.add_argument("--bank-dir", default="data/banks_near")
    ap.add_argument("--every", type=int, default=1, help="use every k-th kept checkpoint")
    ap.add_argument("--nthread", type=int, default=8)
    ap.add_argument("--wandb-id", default=None)
    ap.add_argument("--project", default="assemble")
    a = ap.parse_args()
    run = Path(a.run_dir)
    cks = sorted(run.glob("model_*.pt"))[:: a.every]
    script = {"state": "cap/eval_state.py", "feat": "cap/eval_feat.py", "pixel": "cap/eval_pixel.py"}[a.kind]
    rows = []
    for ck in cks:
        rec = {"it": int(ck.stem.split("_")[1])}
        for det in (False, True):
            out = run / f"eval_{ck.stem}{'_det' if det else ''}.json"
            if not out.exists():
                cmd = [sys.executable, script, "--ckpt", str(ck), "--task", a.task, "--bank-dir", a.bank_dir, "--nthread", str(a.nthread)]
                if a.kind in ("state", "pixel"):
                    cmd += ["--video", "0"]
                if det:
                    cmd.append("--deterministic")
                subprocess.run(cmd, check=True)
            r = json.loads(out.read_text())
            tag = "det" if det else "sampled"
            rec[f"eval/success_{tag}"] = r["success_rate"]
            rec[f"eval/safety_{tag}"] = r["safety"]
            rec[f"eval/timeouts_{tag}"] = r["timeouts"]
            if r.get("time_mean_s") is not None:
                rec[f"eval/time_s_{tag}"] = r["time_mean_s"]
            if r.get("transitions") is not None:
                rec["transitions"] = r["transitions"]
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    (run / "eval_curve.json").write_text(json.dumps(rows, indent=1))
    if a.wandb_id:
        import wandb
        w = wandb.init(entity="WASHU-MARL", project=a.project, id=a.wandb_id, resume="allow")
        wandb.define_metric("it"); wandb.define_metric("eval/*", step_metric="it")
        for rec in rows:
            wandb.log(rec)
        w.finish()
        print("logged", len(rows), "eval points to", w.url)


if __name__ == "__main__":
    main()
