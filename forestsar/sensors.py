"""Gimballed camera: true detections from geometric lines of sight, and the planner's expected
detection model used to update the belief map.

Two models on purpose (two-level fidelity, gap G2 of the literature review):
  * ground truth  : per body point ray cast through trunks / foliage clumps / bushes / terrain
  * planner model : vertical gap fraction ** (1 / sin(elevation)) above the canopy, and a
                    Boolean-model extinction exp(-lambda * d) under the canopy
Both share the pixel-size detection curve. Funded phase P1 replaces the curve by a real detector
on rendered images (Isaac layer) and fits the planner model to it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import CameraCfg
from .scenario import Maps
from .world import World

MEAN_BODY = (0.95, 0.43, 1.0)  # mean box of the three postures [m]: planner does not know the posture


@dataclass
class CamPose:
    pos: torch.Tensor  # [E, 3]
    yaw: torch.Tensor  # [E]
    pitch: torch.Tensor  # [E] below the horizon [rad]

    def axes(self):
        cy, sy = torch.cos(self.yaw), torch.sin(self.yaw)
        cp, sp = torch.cos(self.pitch), torch.sin(self.pitch)
        fwd = torch.stack((cp * cy, cp * sy, -sp), -1)
        right = torch.stack((sy, -cy, torch.zeros_like(cy)), -1)
        up = torch.linalg.cross(right, fwd)
        return fwd, right, up


def project(cam: CamPose, pts: torch.Tensor, cc: CameraCfg):
    """pts [E, M, 3] -> (in_fov [E, M], range [E, M], unit view dir camera->point [E, M, 3])."""
    fwd, right, up = cam.axes()
    v = pts - cam.pos[:, None]
    rng = v.norm(dim=-1).clamp_min(1e-6)
    zc = (v * fwd[:, None]).sum(-1)
    xc = (v * right[:, None]).sum(-1)
    yc = (v * up[:, None]).sum(-1)
    th = math.tan(math.radians(cc.hfov_deg) / 2)
    tv = th * cc.height_px / cc.width_px
    ok = (zc > 0.5) & (xc.abs() < th * zc) & (yc.abs() < tv * zc) & (rng < cc.range_max)
    return ok, rng, v / rng[..., None]


def focal_px(cc: CameraCfg) -> float:
    return 0.5 * cc.width_px / math.tan(math.radians(cc.hfov_deg) / 2)


def box_area(view: torch.Tensor, dims: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Projected area [m^2] of a yawed box (l, w, h) seen along unit ``view``."""
    lx = torch.stack((torch.cos(yaw), torch.sin(yaw)), -1)
    wx = torch.stack((-torch.sin(yaw), torch.cos(yaw)), -1)
    vl = (view[..., :2] * lx).sum(-1).abs()
    vw = (view[..., :2] * wx).sum(-1).abs()
    vz = view[..., 2].abs()
    l, w, h = dims.unbind(-1)
    return vl * w * h + vw * l * h + vz * l * w


def p_from_pixels(eff_px: torch.Tensor, vis: torch.Tensor, cc: CameraCfg) -> torch.Tensor:
    p = cc.p_max * torch.sigmoid((eff_px - cc.px50) / cc.px_slope)
    return torch.where(vis >= cc.vis_min, p, torch.zeros_like(p))


@torch.no_grad()
def true_detection(world: World, cam: CamPose, cc: CameraCfg, gen: torch.Generator):
    """Returns (p_detect [E, P], visible fraction [E, P], detected [E, P] bool, n false alarms [E])."""
    E, P, K, _ = world.person_points.shape
    pts = world.person_points.reshape(E, P * K, 3)
    ok, rng, view = project(cam, pts, cc)
    o = cam.pos[:, None].expand(-1, P * K, -1)
    t = world.cast(o, pts)
    eps = 0.05 / rng
    vis_pt = ok & (t >= 1 - eps)
    vis = vis_pt.reshape(E, P, K).float().mean(-1)
    r_c = rng.reshape(E, P, K).mean(-1)
    v_c = view.reshape(E, P, K, 3).mean(2)
    v_c = v_c / v_c.norm(dim=-1, keepdim=True)
    area = box_area(v_c, world.person_dims, world.person_yaw)
    px2 = area * (focal_px(cc) / r_c) ** 2
    p = p_from_pixels(torch.sqrt(vis * px2), vis, cc)
    det = torch.rand(E, P, generator=gen, device=p.device) < p
    return p, vis, det


class Belief:
    """Unnormalised posterior mass on the G x G grid: b <- b * (1 - p_expected) after each frame."""

    def __init__(self, maps: Maps, world: World):
        self.maps, self.world = maps, world
        E, G, _ = maps.prior.shape
        self.G, self.cell = G, maps.cell
        self.b = maps.prior.clone()
        c = (torch.arange(G, device=maps.prior.device, dtype=torch.float32) + 0.5) * maps.cell
        yy, xx = torch.meshgrid(c, c, indexing="ij")
        xy = torch.stack((xx, yy), -1).reshape(1, -1, 2).expand(E, -1, -1).contiguous()
        z = world.ground(xy) + 0.5
        self.centres = torch.cat((xy, z[..., None]), -1)  # [E, G*G, 3]
        self.gap = maps.gap.reshape(E, -1)
        self.top = maps.canopy_top.reshape(E, -1)

    def expected_p(self, cam: CamPose, cc: CameraCfg, under: torch.Tensor) -> torch.Tensor:
        """Planner's detection probability for a person in each cell [E, G*G]. ``under`` [E] bool."""
        ok, rng, view = project(cam, self.centres, cc)
        sin_el = (-view[..., 2]).clamp_min(0.12)
        vis_above = self.gap.clamp_min(1e-4) ** (1.0 / sin_el)
        dxy = (self.centres[..., :2] - cam.pos[:, None, :2]).norm(dim=-1)
        vis_under = torch.exp(-self.maps.lam_under[:, None] * dxy)
        vis = torch.where(under[:, None], vis_under, vis_above)
        dims = torch.tensor(MEAN_BODY, device=rng.device).expand(*rng.shape, 3)
        area = box_area(view, dims, torch.zeros_like(rng))
        px2 = area * (focal_px(cc) / rng) ** 2
        p = p_from_pixels(torch.sqrt(vis * px2), vis, cc)
        return torch.where(ok, p, torch.zeros_like(p))

    def update(self, cam: CamPose, cc: CameraCfg, under: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        p = self.expected_p(cam, cc, under) * active[:, None]
        self.b = self.b * (1 - p.reshape(self.b.shape))
        return p

    def pos_planner(self) -> torch.Tensor:
        """Probability of success the planner believes it has reached [E]."""
        return 1.0 - self.b.sum((1, 2))
