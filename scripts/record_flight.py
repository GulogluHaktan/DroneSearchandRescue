"""Record one mission at 10 Hz for the video renderer (isaac/render_video.py).

Picks, among E forests, the mission with a find from under the canopy that has the most time
between the trees (and is shorter than --max-s), and saves its flight + forest seed.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forestsar.config import MissionCfg  # noqa: E402
from forestsar.mission import run  # noqa: E402
from forestsar.scenario import build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--envs", type=int, default=12)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--planner", default="hybrid")
ap.add_argument("--max-s", type=float, default=150.0)
ap.add_argument("--out", default="runs/flight_video.pt")
a = ap.parse_args()

cfg = MissionCfg()
cfg.flight.time_limit = a.max_s
w, m = build(cfg, a.envs, a.seed, "cuda")
r = run(cfg, w, m, a.planner, seed=0, flight_hz=10.0, record=False)
score = (r.found & (r.layer_at_find == 1)).float() * r.t_under  # found from under the canopy, long walk between trees
print("found", r.found.tolist(), "layer", r.layer_at_find.tolist(), "t_found", [round(x, 0) for x in r.t_found.tolist()], "t_under", [round(x, 0) for x in r.t_under.tolist()])
e = int(score.argmax())
if score[e] <= 0:
    e = int(r.found.float().argmax())
print("chosen env", e)
t = torch.tensor([f[0] for f in r.flight])
pos = torch.stack([f[1][e] for f in r.flight])
yaw = torch.stack([f[2][e] for f in r.flight])
layer = torch.stack([f[3][e] for f in r.flight])
pit = torch.stack([f[4][e] for f in r.flight])
act = torch.stack([f[5][e] for f in r.flight])
n = int(act.sum()) + 1
Path(a.out).parent.mkdir(exist_ok=True)
torch.save(dict(seed=a.seed, env=e, planner=a.planner, t=t[:n], pos=pos[:n], yaw=yaw[:n], layer=layer[:n], pitch=pit[:n],
                person=w.person_pos[e, 0].cpu(), found=bool(r.found[e]), t_found=float(r.t_found[e])), a.out)
print(f"saved {a.out}: {n} frames (10 Hz), found={bool(r.found[e])} at {float(r.t_found[e]):.0f}s")
