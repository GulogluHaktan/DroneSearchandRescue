"""Mission runner: E search missions in parallel, one planner, quadrotor dynamics in the loop.

Rates: control + physics 1/dt (100 Hz), depth fan 10 Hz, camera 2 Hz, planner 1 Hz.
A mission ends at the first true detection (found), at a collision (crash) or at the time limit.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from .config import MissionCfg
from .control import VelocityController, avoid, fan_dirs
from .planners import ABOVE, ASCEND, CRUISE, DESCEND, TO_SHAFT, UNDER, make_planner
from .quad import QuadBatch
from .scenario import Maps
from .sensors import Belief, CamPose, project, true_detection
from .world import World


@dataclass
class Result:
    planner: str
    found: torch.Tensor  # [E] bool
    t_found: torch.Tensor  # [E] s (time limit if not found)
    crashed: torch.Tensor  # [E] bool
    path_m: torch.Tensor  # [E]
    energy_wh: torch.Tensor  # [E]
    false_alarms: torch.Tensor  # [E]
    imaged_ha: torch.Tensor  # [E] cumulative imaged ground area (frames x footprint)
    t_under: torch.Tensor  # [E] s between the trees
    pos_planner: torch.Tensor  # [E] planner-believed probability of success at the end
    vis_at_find: torch.Tensor  # [E] visible fraction of the found person in the detecting frame
    layer_at_find: torch.Tensor  # [E] 0 above, 1 under, -1 not found
    curve_t: list = field(default_factory=list)  # time stamps of the planner PoS curve
    curve_pos: list = field(default_factory=list)  # [E] tensors
    traj: list = field(default_factory=list)  # [(t, pos [E,3], layer [E])] at 1 Hz
    flight: list = field(default_factory=list)  # [(t, pos [E,3], yaw [E], layer [E], pitch [E])] at flight_hz (video)
    wall_s: float = 0.0


@torch.no_grad()
def run(cfg: MissionCfg, world: World, maps: Maps, planner_name: str, seed: int = 0, log_every: float = 0.0,
        record: bool = True, flight_hz: float = 0.0) -> Result:
    dev, E = world.device, world.E
    fl, cc, dc = cfg.flight, cfg.camera, cfg.depth
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    quad = QuadBatch(E, dev, cfg.vehicle, gen)
    ctrl = VelocityController(quad, fl)
    start_xy = maps.launch.clone()
    planner = make_planner(planner_name, cfg, maps, start_xy)
    belief = Belief(maps, world)
    top0 = _canopy_top_at(maps, start_xy)
    start = torch.cat((start_xy, (top0 + fl.alt_above)[:, None]), -1)
    ids = torch.arange(E, device=dev)
    quad.reset(ids, start, torch.zeros(E, device=dev))
    ctrl.reset(ids)

    active = torch.ones(E, dtype=torch.bool, device=dev)
    found = torch.zeros(E, dtype=torch.bool, device=dev)
    crashed = torch.zeros(E, dtype=torch.bool, device=dev)
    t_found = torch.full((E,), fl.time_limit, device=dev)
    path = torch.zeros(E, device=dev)
    energy = torch.zeros(E, device=dev)
    fas = torch.zeros(E, device=dev)
    imaged = torch.zeros(E, device=dev)
    t_under = torch.zeros(E, device=dev)
    blocked_s = torch.zeros(E, device=dev)
    vis_find = torch.zeros(E, device=dev)
    layer_find = torch.full((E,), -1, dtype=torch.long, device=dev)
    v_safe = torch.zeros(E, 2, device=dev)
    heading = torch.zeros(E, device=dev)
    res = Result(planner_name, found, t_found, crashed, path, energy, fas, imaged, t_under,
                 torch.zeros(E, device=dev), vis_find, layer_find)

    n_steps = int(round(fl.time_limit / fl.dt))
    every = lambda hz: max(1, int(round(1.0 / (hz * fl.dt))))  # noqa: E731
    k_plan, k_cam, k_depth, k_col = every(1.0), every(cc.rate_hz), every(dc.rate_hz), every(20.0)
    k_trans = every(10.0)
    k_fl = every(flight_hz) if flight_hz else 0
    cell_ha = maps.cell**2 / 1e4
    wall0 = time.time()
    for k in range(n_steps):
        t = k * fl.dt
        pos = quad.pos
        if k % k_plan == 0:
            planner.decide(t, pos, belief.b, blocked_s, active)
            if record:
                res.traj.append((t, pos.clone(), planner.layer.clone()))
                res.curve_t.append(t)
                res.curve_pos.append(belief.pos_planner().clone())
            if log_every and k % every(1.0 / log_every) == 0:
                print(f"[{planner_name}] t={t:6.0f}s active={int(active.sum())} found={int(found.sum())} "
                      f"crash={int(crashed.sum())} PoS_plan={float(belief.pos_planner().mean()):.2f} "
                      f"under={int((planner.layer == UNDER).sum())} wall={time.time() - wall0:.0f}s", flush=True)
            if not bool(active.any()):
                break

        layer, phase = planner.layer, planner.phase
        if k_fl and k % k_fl == 0:
            pit = torch.where(layer == ABOVE, torch.full_like(pos[:, 0], cc.pitch_above_deg), torch.full_like(pos[:, 0], cc.pitch_under_deg))
            pit = torch.where((phase == DESCEND) | (phase == ASCEND), torch.full_like(pit, 85.0), pit)
            res.flight.append((t, pos.clone().cpu(), _yaw(quad.quat).cpu(), layer.clone().cpu(), pit.cpu(), active.clone().cpu()))
        ground = world.ground(pos[:, None, :2])[:, 0]
        top = _canopy_top_at(maps, pos[:, :2])
        z_above = top + fl.alt_above
        z_under = ground + fl.alt_under
        # ---- horizontal reference -------------------------------------------------------
        to_t = planner.target - pos[:, :2]
        dist = to_t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        v_cr = torch.where(layer == ABOVE, torch.full_like(ground, fl.v_above), torch.full_like(ground, fl.v_under))
        vertical = (phase == DESCEND) | (phase == ASCEND)
        slow = torch.sqrt(2 * 1.5 * dist[:, 0])  # arrive without overshoot
        v_h = to_t / dist * torch.minimum(v_cr, slow)[:, None]
        v_h = torch.where(vertical[:, None], 0.8 * to_t, v_h)  # hold the shaft position
        # ---- obstacle avoidance between the trees (depth fan at 10 Hz) ------------------------
        under_now = (layer == UNDER) & ~vertical
        if k % k_depth == 0 and bool(under_now.any()):
            want = torch.atan2(v_h[:, 1], v_h[:, 0])
            heading = torch.where(v_h.norm(dim=-1) > 0.05, want, heading)
            dirs, _ = fan_dirs(dc, heading)
            o = pos[:, None, :].expand_as(dirs)
            rng = world.cast(o, o + dirs * dc.range_max, terrain=True) * dc.range_max
            rel = dirs * rng[..., None]
            valid = (rng < dc.range_max - 1e-3) & (rel[..., 2].abs() < dc.band)
            v_safe, blk = avoid(v_h, rel[..., :2], valid, fl, prev=v_safe)
            blocked_s = torch.where(blk & under_now, blocked_s + dc.rate_hz**-1, torch.zeros_like(blocked_s))
        elif k % k_depth == 0:
            blocked_s.zero_()
        v_xy = torch.where(under_now[:, None], v_safe, v_h)
        # ---- vertical reference ----------------------------------------------------------
        z_ref = torch.where(layer == ABOVE, z_above, z_under)
        vz = (1.2 * (z_ref - pos[:, 2])).clamp(-2.5, 2.5)
        vz = torch.where(phase == DESCEND, torch.full_like(vz, -fl.v_vertical), vz)
        vz = torch.where(phase == ASCEND, torch.full_like(vz, fl.v_vertical), vz)
        v_ref = torch.cat((v_xy, vz[:, None]), -1) * active[:, None]
        yaw_ref = torch.where(v_xy.norm(dim=-1) > 0.3, torch.atan2(v_xy[:, 1], v_xy[:, 0]), _yaw(quad.quat))
        quad.step(ctrl.action(v_ref, yaw_ref), fl.dt)
        # ---- phase transitions -------------------------------------------------------------
        at_shaft = (phase == TO_SHAFT) & ((pos[:, :2] - planner.target).norm(dim=-1) < 1.2) & (quad.vel[:, :2].norm(dim=-1) < 0.6)
        go_down = at_shaft & (layer == ABOVE)
        go_up = at_shaft & (layer == UNDER)
        planner.phase = torch.where(go_down, torch.full_like(phase, DESCEND), planner.phase)
        planner.phase = torch.where(go_up, torch.full_like(phase, ASCEND), planner.phase)
        if k % k_trans == 0:  # host sync: checked at 10 Hz only
            done_down = (planner.phase == DESCEND) & (pos[:, 2] <= z_under + 0.2)
            done_up = (planner.phase == ASCEND) & (pos[:, 2] >= z_above - 1.0)
            for e in torch.nonzero(done_down | done_up).flatten().tolist():
                planner.on_vertical_done(e)
        # ---- bookkeeping -----------------------------------------------------------------
        sp = quad.vel.norm(dim=-1)
        path += sp * fl.dt * active
        energy += fl.hover_power_w * (quad.thr * quad.twr).clamp_min(0).pow(1.5) * fl.dt / 3600 * active
        t_under += ((layer == UNDER) | vertical).float() * fl.dt * active
        if k % k_col == 0:
            hit = world.collide(quad.pos, fl.drone_radius) & active
            crashed |= hit
            active &= ~hit
        # ---- camera ----------------------------------------------------------------------
        if k % k_cam == 0:
            pitch = torch.where(layer == ABOVE, torch.full_like(ground, math.radians(cc.pitch_above_deg)),
                                torch.full_like(ground, math.radians(cc.pitch_under_deg)))
            pitch = torch.where(vertical, torch.full_like(pitch, math.radians(85.0)), pitch)
            cam = CamPose(quad.pos.clone(), _yaw(quad.quat), pitch)
            under_cam = (layer == UNDER) | (phase == ASCEND) | (quad.pos[:, 2] < _canopy_top_at(maps, quad.pos[:, :2]) - 1.0)
            belief.update(cam, cc, under_cam, active.float())
            ok, _, _ = project(cam, belief.centres, cc)
            area_ha = ok.float().sum(-1) * cell_ha
            imaged += area_ha * active
            lam = cc.fa_per_ha_frame * area_ha
            fas += torch.poisson(lam, generator=gen) * active
            p, vis, det = true_detection(world, cam, cc, gen)
            hitp = det.any(-1) & active
            if bool(hitp.any()):
                vbest = (vis * det).amax(-1)
                vis_find[:] = torch.where(hitp, vbest, vis_find)
                layer_find[:] = torch.where(hitp, under_cam.long(), layer_find)
                t_found[:] = torch.where(hitp, torch.full_like(t_found, t), t_found)
                found |= hitp
                active &= ~hitp
    res.pos_planner = belief.pos_planner()
    res.wall_s = time.time() - wall0
    return res


def _yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _canopy_top_at(maps: Maps, xy: torch.Tensor) -> torch.Tensor:
    G = maps.canopy_top.shape[-1]
    ij = (xy / maps.cell).long().clamp(0, G - 1)
    E = xy.shape[0]
    return maps.canopy_top[torch.arange(E, device=xy.device), ij[:, 1], ij[:, 0]]
