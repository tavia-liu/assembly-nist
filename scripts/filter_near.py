"""data/banks_near/<task>_{train,eval}.npz = the single-task banks restricted to the near family (AlphaNIST setting)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from assembly_policy.sim.resets import FAMILIES
from assembly_policy.sim.tasks import BANK_DIR
task = sys.argv[1] if len(sys.argv) > 1 else "P4"
out = Path("data/banks_near"); out.mkdir(exist_ok=True)
for split in ("train", "eval"):
    z = np.load(Path(BANK_DIR) / f"{task}_{split}.npz")
    keep = z["family"] == FAMILIES.index("near")
    np.savez_compressed(out / f"{task}_{split}.npz", states=z["states"][keep], family=z["family"][keep], level=z["level"][keep])
    print(f"{task} {split}: {int(keep.sum())} near states -> {out / f'{task}_{split}.npz'}")
