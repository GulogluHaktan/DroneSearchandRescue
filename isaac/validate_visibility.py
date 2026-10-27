"""Level B check: compare the fast ray-cast visible fraction (torch) with the rendered one (Isaac RTX).

For random camera poses that look at a lost person (above the canopy: oblique; between the trees:
3 m above ground) it renders a semantic mask twice: full forest, and forest hidden. Rendered
visible fraction = person pixels (full) / person pixels (forest hidden). The fast layer gives the
fraction of the 12 body points with a free line of sight. Also saves a few RGB views.

Run: scripts/run_isaaclab.sh isaac/validate_visibility.py --enable_cameras --poses 60
"""
import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--poses", type=int, default=60, help="per layer (above / under)")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--hfov", type=float, default=80.0)
parser.add_argument("--out-dir", type=str, default="/workspace/DroneTubitak1002/runs/isaac_vis")
parser.add_argument("--save-rgb", type=int, default=6)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = True
app = AppLauncher(args).app

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402

sys.path.insert(0, "/workspace/DroneTubitak1002")
from forestsar.config import MissionCfg  # noqa: E402
from forestsar.scenario import build  # noqa: E402
from forestsar.usd_export import export, set_visible  # noqa: E402

os.makedirs(args.out_dir, exist_ok=True)
dev = "cuda:0"
cfg = MissionCfg()
cfg.person.n = 3  # more targets per forest
world, maps = build(cfg, 1, args.seed, dev)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=dev))
stage = omni.usd.get_context().get_stage()
paths = export(stage, world, 0)
focal = 20.955 / (2 * math.tan(math.radians(args.hfov) / 2))
cam = Camera(CameraCfg(
    prim_path="/World/Cam", update_period=0, height=args.height, width=args.width,
    data_types=["rgb", "semantic_segmentation"], colorize_semantic_segmentation=False,
    spawn=sim_utils.PinholeCameraCfg(focal_length=focal, horizontal_aperture=20.955, clipping_range=(0.1, 600.0)),
))
sim.reset()


def render(n=6):
    for _ in range(n):
        sim.step(render=True)
        cam.update(dt=sim.get_physics_dt())


def person_pixels():
    """Works for both outputs of this build: integer ids, or RGBA colours keyed as '(r, g, b, a)'."""
    seg = cam.data.output["semantic_segmentation"][0]
    info = cam.data.info[0]["semantic_segmentation"] if isinstance(cam.data.info, list) else cam.data.info["semantic_segmentation"]
    keys = [k for k, v in info["idToLabels"].items() if "person" in str(v)]
    if os.environ.get("SEG_DEBUG"):
        u = torch.unique(seg.reshape(-1, seg.shape[-1]), dim=0)
        print("SEGDBG", seg.shape, seg.dtype, info, u[:10].tolist(), flush=True)
    if not keys:
        return 0
    if str(keys[0]).startswith("("):  # colour keys; pixels are RGBA packed into one int32 (r | g<<8 | b<<16 | a<<24)
        n = 0
        flat = seg.reshape(-1).to(torch.int64) & 0xFFFFFFFF
        for k in keys:
            r, g, b, a = (int(c) for c in str(k).strip("()").split(","))
            n += int((flat == (r | g << 8 | b << 16 | a << 24)).sum())
        return n
    return int(torch.isin(seg.reshape(-1).long(), torch.tensor([int(k) for k in keys], device=seg.device)).sum())


# ---- sample poses ----------------------------------------------------------------------
rng = np.random.default_rng(args.seed)
P = world.person_pos.shape[1]
poses = []
for layer in ("above", "under"):
    k = 0
    tries = 0
    while k < args.poses and tries < args.poses * 50:
        tries += 1
        j = int(rng.integers(P))
        tgt = world.person_pos[0, j].clone()
        tgt[2] += float(world.person_dims[0, j, 2]) / 2
        az = rng.uniform(0, 2 * np.pi)
        if layer == "above":
            el = math.radians(rng.uniform(35, 85))
            d = rng.uniform(35, 75)
            eye = tgt + torch.tensor((d * math.cos(el) * math.cos(az), d * math.cos(el) * math.sin(az), d * math.sin(el)), device=dev)
            top = float(maps.canopy_top[0].max())
            if float(eye[2]) < top + 3:
                continue
        else:
            d = rng.uniform(4, 30)
            eye = tgt.clone()
            eye[0] += d * math.cos(az)
            eye[1] += d * math.sin(az)
            if not (1 < float(eye[0]) < world.size - 1 and 1 < float(eye[1]) < world.size - 1):
                continue
            eye[2] = float(world.ground(eye[None, None, :2])[0, 0]) + cfg.flight.alt_under
            if bool(world.collide(eye[None], cfg.flight.drone_radius)[0]):
                continue
        pts = world.person_points[0, j][None]
        o = eye[None, None].expand(1, pts.shape[1], 3)
        t = world.cast(o, pts)
        rngd = (pts - o).norm(dim=-1)
        vis_fast = float((t >= 1 - 0.05 / rngd).float().mean())
        poses.append(dict(layer=layer, person=j, eye=eye.tolist(), target=tgt.tolist(), vis_fast=vis_fast))
        k += 1
print(f"poses: {len(poses)}", flush=True)

# ---- render: forest on, then forest off --------------------------------------------------
render(30)
for idx, ps in enumerate(poses):
    cam.set_world_poses_from_view(torch.tensor([ps["eye"]], device=dev), torch.tensor([ps["target"]], device=dev))
    render()
    ps["px_full"] = person_pixels()
    if idx < args.save_rgb or (ps["layer"] == "under" and idx < len(poses) // 2 + args.save_rgb):
        rgb = cam.data.output["rgb"][0].cpu().numpy()[..., :3]
        imageio.imwrite(os.path.join(args.out_dir, f"{ps['layer']}_{idx:03d}_vis{ps['vis_fast']:.2f}.png"), rgb.astype(np.uint8))
set_visible(stage, paths["occluders"], False)
render(10)
for ps in poses:
    cam.set_world_poses_from_view(torch.tensor([ps["eye"]], device=dev), torch.tensor([ps["target"]], device=dev))
    render()
    ps["px_open"] = person_pixels()
    ps["vis_render"] = ps["px_full"] / ps["px_open"] if ps["px_open"] > 0 else float("nan")

# ---- compare ------------------------------------------------------------------------------
summary = {}
for layer in ("above", "under"):
    rows = [p for p in poses if p["layer"] == layer and p["px_open"] >= 30]
    a = np.array([p["vis_fast"] for p in rows])
    b = np.array([p["vis_render"] for p in rows])
    if len(rows) < 3:
        summary[layer] = {"n": len(rows)}
        continue
    summary[layer] = {
        "n": len(rows),
        "mean_fast": round(float(a.mean()), 3), "mean_render": round(float(b.mean()), 3),
        "mae": round(float(np.abs(a - b).mean()), 3), "bias_fast_minus_render": round(float((a - b).mean()), 3),
        "pearson_r": round(float(np.corrcoef(a, b)[0, 1]), 3) if a.std() > 0 and b.std() > 0 else None,
        "agree_visible_or_not(>0.05)": round(float(((a > 0.05) == (b > 0.05)).mean()), 3),
        "median_px_open": int(np.median([p["px_open"] for p in rows])),
    }
print(json.dumps(summary, indent=1), flush=True)
with open(os.path.join(args.out_dir, "visibility_check.json"), "w") as fh:
    json.dump({"args": vars(args), "summary": summary, "poses": poses}, fh, indent=1, default=float)
print("VALIDATE DONE", flush=True)
app.close()
