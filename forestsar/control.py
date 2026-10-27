"""Velocity controller (velocity + yaw command -> CTBR action of QuadBatch) and reactive avoidance."""
from __future__ import annotations

import math

import torch

from .config import DepthCfg, FlightCfg
from .quad import GRAVITY, QuadBatch, quat_to_rot


class VelocityController:
    def __init__(self, quad: QuadBatch, fl: FlightCfg, kv: float = 2.2, kr: float = 7.0, kyaw: float = 2.5):
        self.q, self.fl = quad, fl
        self.kv, self.kr, self.kyaw = kv, kr, kyaw
        self.ki, self.i_max = 0.3, 3.0
        self.integ = torch.zeros(quad.n, 3, device=quad.device)
        self.gains = torch.tensor((kr, kr, kyaw), device=quad.device)
        c = quad.cfg
        self.scale = torch.tensor((1.0, 1.0, c.yaw_rate_scale), device=quad.device) * c.max_rate

    def reset(self, ids: torch.Tensor) -> None:
        self.integ[ids] = 0.0

    def action(self, v_ref: torch.Tensor, yaw_ref: torch.Tensor) -> torch.Tensor:
        q, dev = self.q, self.q.device
        err = v_ref - q.vel
        small = (err.abs() < 1.0).float()  # anti-windup: integrate only near the set-point
        self.integ = (self.integ + small * err * self.fl.dt).clamp(-self.i_max, self.i_max)
        R = quat_to_rot(q.quat)
        a = self.kv * err + self.ki * self.integ
        ah = a[:, :2]
        n = ah.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        a[:, :2] = ah * (self.fl.a_max / n).clamp(max=1.0)
        a[:, 2] = a[:, 2].clamp(-3.0, 4.0)
        a = a - q.drag_acc_world(R)  # drag feed-forward from the nominal model (outside the a_max limit)
        a[:, 2] += GRAVITY
        zb = R[:, :, 2]
        thrust = q.mass * (a * zb).sum(-1)
        coll = (thrust / q.tmax).clamp(0.0, 1.0) * 2 - 1
        zd = a / a.norm(dim=-1, keepdim=True)
        xc = torch.stack((torch.cos(yaw_ref), torch.sin(yaw_ref), torch.zeros_like(yaw_ref)), -1)
        yd = torch.linalg.cross(zd, xc)
        yd = yd / yd.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        xd = torch.linalg.cross(yd, zd)
        Rd = torch.stack((xd, yd, zd), -1)
        E = Rd.transpose(1, 2) @ R - R.transpose(1, 2) @ Rd
        e_r = 0.5 * torch.stack((E[:, 2, 1], E[:, 0, 2], E[:, 1, 0]), -1)
        w = -self.gains * e_r
        return torch.cat((coll[:, None], (w / self.scale).clamp(-1, 1)), -1)


def fan_dirs(dc: DepthCfg, heading: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit ray directions [E, n_el * n_az, 3] around ``heading`` [E] and their azimuth offsets [n_az].

    A real drone would use a 360 deg LiDAR or 4-6 depth cameras for this."""
    dev = heading.device
    off = torch.linspace(-0.5, 0.5, dc.n_az, device=dev) * math.radians(dc.az_span_deg)
    az = heading[:, None] + off[None]  # [E, A]
    els = torch.tensor([math.radians(e) for e in dc.elevations_deg], device=dev)
    ce, se = torch.cos(els)[:, None, None], torch.sin(els)[:, None, None]
    d = torch.stack((ce * torch.cos(az)[None], ce * torch.sin(az)[None], se.expand(-1, *az.shape)), -1)  # [L, E, A, 3]
    return d.permute(1, 0, 2, 3).reshape(heading.shape[0], -1, 3), off


def avoid(v_des: torch.Tensor, hits: torch.Tensor, valid: torch.Tensor, fl: FlightCfg, prev: torch.Tensor | None = None,
          n_cand: int = 72, lookahead: float = 8.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose a collision-free horizontal velocity from the depth-fan hit points.

    hits [E, N, 2] obstacle points relative to the drone (only those inside the drone's height band),
    valid [E, N]. For each candidate heading the corridor of half-width (radius + margin) is checked
    up to ``lookahead``; the free heading closest to the desired one wins, speed is limited so that
    the drone can stop before the first point in its corridor. Points closer than radius + 0.3 m add
    a repulsive velocity. Returns (velocity [E, 2], blocked [E]).
    """
    dev = v_des.device
    E = v_des.shape[0]
    ang = torch.arange(n_cand, device=dev, dtype=torch.float32) * (2 * math.pi / n_cand)
    u = torch.stack((torch.cos(ang), torch.sin(ang)), -1)  # [K, 2]
    proj = hits @ u.T  # [E, N, K]
    perp = (hits[..., None, 0] * u[None, None, :, 1] - hits[..., None, 1] * u[None, None, :, 0]).abs()
    half = fl.drone_radius + fl.safety_margin
    block = valid[..., None] & (proj > -0.3) & (proj < lookahead) & (perp < half)
    first = torch.where(block, proj, torch.full_like(proj, lookahead)).amin(1)  # [E, K]
    speed = v_des.norm(dim=-1)
    lag = 0.6  # [s] velocity-loop lag: the drone keeps going this long before a new heading takes effect
    need = speed**2 / (2 * fl.a_max) + lag * speed + 1.0
    free = first >= need[:, None]
    des = torch.atan2(v_des[:, 1], v_des[:, 0])
    dev_ang = (torch.remainder(ang[None] - des[:, None] + math.pi, 2 * math.pi) - math.pi).abs()
    cost = dev_ang + (~free) * 100.0 + (dev_ang > 2.0) * 10.0  # do not turn back unless nothing else is free
    if prev is not None:
        moving = prev.norm(dim=-1) > 0.2
        pa = torch.atan2(prev[:, 1], prev[:, 0])
        dprev = (torch.remainder(ang[None] - pa[:, None] + math.pi, 2 * math.pi) - math.pi).abs()
        cost = cost + 0.4 * dprev * moving[:, None]
    k = cost.argmin(-1)
    ok = free.gather(1, k[:, None])[:, 0] & (cost.gather(1, k[:, None])[:, 0] < 100.0)
    room = (first.gather(1, k[:, None])[:, 0] - 1.0).clamp_min(0.0)
    v_lim = torch.minimum(torch.sqrt(2 * fl.a_max * room), room / lag)
    s = torch.where(ok, torch.minimum(speed, v_lim), torch.zeros_like(speed))
    v = u[k] * s[:, None]
    # bounded repulsion from the closest point
    d = torch.where(valid, hits.norm(dim=-1), torch.full_like(hits[..., 0], 1e3))
    j = d.argmin(-1)
    dj = d.gather(1, j[:, None])[:, 0]
    hj = hits[torch.arange(E, device=dev), j]
    push = (fl.drone_radius + 0.5 - dj).clamp(0.0, 0.5) * 2.0  # up to 1 m/s
    v = v - hj / dj.clamp_min(1e-3)[:, None] * push[:, None]
    return v, ~ok
