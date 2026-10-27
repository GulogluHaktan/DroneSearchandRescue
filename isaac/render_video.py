"""Render a recorded mission (scripts/record_flight.py) in Isaac Sim: chase camera + onboard camera side by side.

Run (waits for the shared GPU lock):
  flock /home/haktan/Projects/DroneRacing/runs/.isaac.lock scripts/run_isaaclab.sh isaac/render_video.py --enable_cameras
Frames are written as PNG, then encoded with the ffmpeg binary of imageio_ffmpeg. 10 Hz flight -> 20 fps (2x speed).
"""
import argparse
import math
import os
import subprocess
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--flight", default="/workspace/DroneTubitak1002/runs/flight_video.pt")
parser.add_argument("--out", default="/workspace/DroneTubitak1002/runs/forest_sar_flight.mp4")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=360)
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--hfov", type=float, default=80.0)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = True
app = AppLauncher(args).app

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402

sys.path.insert(0, "/workspace/DroneTubitak1002")
from forestsar.config import MissionCfg  # noqa: E402
from forestsar.scenario import build  # noqa: E402
from forestsar.usd_export import export  # noqa: E402

dev = "cuda:0"
fl = torch.load(args.flight)
cfg = MissionCfg()
world, _ = build(cfg, 1, int(fl["seed"]), dev, first_env=int(fl["env"]))
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=dev))
stage = omni.usd.get_context().get_stage()
export(stage, world, 0)

# drone model: body + 4 arms + 4 rotor discs (visible in the chase view only: the onboard camera sits 0.4 m ahead)
UsdGeom.Xform.Define(stage, "/World/Drone")
drone_xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Drone"))
t_op, r_op = drone_xf.AddTranslateOp(), drone_xf.AddRotateZOp()


def part(path, geom, size, xyz, rgb):
    g = geom.Define(stage, path)
    UsdGeom.Xformable(g).AddTranslateOp().Set(Gf.Vec3d(*xyz))
    UsdGeom.Xformable(g).AddScaleOp().Set(Gf.Vec3f(*size))
    UsdGeom.Gprim(g).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])


part("/World/Drone/Body", UsdGeom.Cube, (0.16, 0.16, 0.06), (0, 0, 0), (0.08, 0.08, 0.1))
for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, -1), (-1, 1))):
    part(f"/World/Drone/Arm{i}", UsdGeom.Cube, (0.30, 0.02, 0.02), (sx * 0.10, sy * 0.10, 0), (0.15, 0.15, 0.15))
    part(f"/World/Drone/Rotor{i}", UsdGeom.Cylinder, (0.20, 0.20, 0.01), (sx * 0.19, sy * 0.19, 0.03), (0.9, 0.9, 0.1))

focal = 20.955 / (2 * math.tan(math.radians(args.hfov) / 2))
spawn = sim_utils.PinholeCameraCfg(focal_length=focal, horizontal_aperture=20.955, clipping_range=(0.1, 800.0))
chase = Camera(CameraCfg(prim_path="/World/CamChase", update_period=0, height=args.height, width=args.width, data_types=["rgb"], spawn=spawn))
fpv = Camera(CameraCfg(prim_path="/World/CamFpv", update_period=0, height=args.height, width=args.width, data_types=["rgb"], spawn=spawn))
sim.reset()


def render(n=3):
    for _ in range(n):
        sim.step(render=True)
        chase.update(dt=sim.get_physics_dt())
        fpv.update(dt=sim.get_physics_dt())


def look(cam, eye, target):
    cam.set_world_poses_from_view(torch.tensor([eye], device=dev, dtype=torch.float32), torch.tensor([target], device=dev, dtype=torch.float32))


render(40)
pos, yaw, layer, pit, t = fl["pos"].numpy(), fl["yaw"].numpy(), fl["layer"].numpy(), fl["pitch"].numpy(), fl["t"].numpy()
person = fl["person"].numpy()
N = len(t)
fdir = os.path.join(os.path.dirname(args.out), "video_frames")
os.makedirs(fdir, exist_ok=True)
cy = yaw[0]  # smoothed chase heading
idx = list(range(0, N, args.stride))
for n, i in enumerate(idx):
    p = pos[i]
    cy = cy + 0.25 * ((yaw[i] - cy + math.pi) % (2 * math.pi) - math.pi)
    t_op.Set(Gf.Vec3d(*map(float, p)))
    r_op.Set(math.degrees(float(yaw[i])))
    under = layer[i] == 1
    back, up = (4.5, 1.6) if under else (9.0, 3.5)
    look(chase, [p[0] - back * math.cos(cy), p[1] - back * math.sin(cy), p[2] + up], [p[0], p[1], p[2]])
    pr = math.radians(float(pit[i]))
    eye = [p[0] + 0.4 * math.cos(yaw[i]), p[1] + 0.4 * math.sin(yaw[i]), p[2] - 0.05]
    look(fpv, eye, [eye[0] + math.cos(pr) * math.cos(yaw[i]), eye[1] + math.cos(pr) * math.sin(yaw[i]), eye[2] - math.sin(pr)])
    render()
    a = chase.data.output["rgb"][0].cpu().numpy()[..., :3]
    b = fpv.data.output["rgb"][0].cpu().numpy()[..., :3]
    img = Image.fromarray(np.concatenate((a, b), 1).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, img.width, 22), fill=(0, 0, 0))
    status = "UNDER THE TREES" if under else "ABOVE THE CANOPY"
    found = fl["found"] and t[i] >= fl["t_found"] - 0.5
    d.text((8, 6), f"{fl['planner']}   t = {t[i]:5.1f} s   {status}   x2 speed", fill=(255, 255, 255))
    d.text((args.width + 8, 6), "onboard camera" + ("   PERSON FOUND" if found else ""), fill=(255, 120, 60) if found else (255, 255, 255))
    img.save(os.path.join(fdir, f"f{n:05d}.png"))
    if n % 50 == 0:
        print(f"frame {n}/{len(idx)}", flush=True)

import imageio_ffmpeg  # noqa: E402

subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", str(args.fps), "-i", os.path.join(fdir, "f%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args.out], check=True)
print("VIDEO DONE", args.out, flush=True)
app.close()
