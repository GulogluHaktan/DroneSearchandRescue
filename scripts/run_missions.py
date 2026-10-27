"""Run search missions for one or more planners on the same E forests and print the metrics.

Example: python -I scripts/run_missions.py --envs 8 --planners hybrid greedy-above --time-limit 900
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forestsar.config import MissionCfg  # noqa: E402
from forestsar.metrics import summarize  # noqa: E402
from forestsar.mission import run  # noqa: E402
from forestsar.planners import PLANNERS  # noqa: E402
from forestsar.scenario import build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--envs", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--planners", nargs="+", default=list(PLANNERS))
ap.add_argument("--time-limit", type=float, default=900.0)
ap.add_argument("--density", type=float, default=None, help="trees per hectare")
ap.add_argument("--size", type=float, default=None, help="side of the search area [m]")
ap.add_argument("--log-every", type=float, default=60.0, help="progress print period [s sim]")
ap.add_argument("--out", type=str, default="")
a = ap.parse_args()

cfg = MissionCfg()
cfg.flight.time_limit = a.time_limit
if a.density is not None:
    cfg.forest.density = a.density
if a.size is not None:
    cfg.terrain.size = a.size
dev = "cuda" if torch.cuda.is_available() else "cpu"
world, maps = build(cfg, a.envs, a.seed, dev)
print(f"forest: {a.envs} envs, trees/env={world.n_trees.float().mean():.0f}, canopy cover={1 - float(maps.gap.mean()):.2f}, "
      f"descent cells={float(maps.descent.float().mean()) * 100:.1f} %", flush=True)
rows = {}
for name in a.planners:
    res = run(cfg, world, maps, name, seed=a.seed, log_every=1.0 / a.log_every if a.log_every else 0.0, record=bool(a.out))
    rows[name] = summarize(res, cfg)
    print(name, json.dumps(rows[name]), flush=True)
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "results": rows}, indent=1))
