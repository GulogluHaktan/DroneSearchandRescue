"""Batched world (E independent missions): terrain + forest + persons as device tensors.

Geometry queries used by every other module:
  cast(origins, ends)    -> first-hit fraction along each segment (1 = free)   [lines of sight, depth]
  collide(pos, radius)   -> sphere vs trunks / foliage / bushes / ground       [crash test]
Trees are padded to the same count per env (padding trees sit far outside and never match).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .forest import ForestNp
from .terrain import Terrain

FAR = 1e6


@dataclass
class PersonsNp:
    pos: np.ndarray  # [P, 3] feet position (absolute z)
    yaw: np.ndarray  # [P]
    posture: np.ndarray  # [P] 0 standing, 1 sitting, 2 lying
    dims: np.ndarray  # [P, 3] body box length (along yaw), width, height
    points: np.ndarray  # [P, K, 3] body sample points (absolute)


POSTURE_DIMS = {0: (0.45, 0.30, 1.75), 1: (0.75, 0.50, 1.00), 2: (1.75, 0.50, 0.30)}
POSTURE_NAMES = ("standing", "sitting", "lying")


def body_points(pos: np.ndarray, yaw: float, posture: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Lattice of ``k`` points inside the body box (3 along length x 2 width x k/6 height layers)."""
    l, w, h = POSTURE_DIMS[int(posture)]
    nz = max(1, k // 6)
    fl = np.array([-1 / 3, 0.0, 1 / 3]) * l
    fw = np.array([-0.25, 0.25]) * w
    fz = (np.arange(nz) + 0.5) / nz * h
    L, W, Z = np.meshgrid(fl, fw, fz, indexing="ij")
    c, s = np.cos(yaw), np.sin(yaw)
    x = pos[0] + c * L - s * W
    y = pos[1] + s * L + c * W
    z = pos[2] + Z
    pts = np.stack((x, y, z), -1).reshape(-1, 3)[:k]
    return pts.astype(np.float32), np.array((l, w, h), dtype=np.float32)


class World:
    def __init__(self, terrain: Terrain, forests: list[ForestNp], persons: list[PersonsNp], size: float, device: str):
        self.terrain, self.size, self.device = terrain, size, device
        self.E = len(forests)
        f32 = dict(device=device, dtype=torch.float32)
        T = max(len(f.tree_xy) for f in forests)
        C = forests[0].clumps.shape[1]
        B = max(len(f.bushes) for f in forests)
        self.tree_xy = torch.full((self.E, T, 2), FAR, **f32)
        self.tree_z0 = torch.zeros(self.E, T, **f32)
        self.tree_h = torch.zeros(self.E, T, **f32)
        self.crown_base = torch.zeros(self.E, T, **f32)
        self.crown_r = torch.zeros(self.E, T, **f32)
        self.trunk_r = torch.zeros(self.E, T, **f32)
        self.clumps = torch.zeros(self.E, T, C, 4, **f32)
        self.clumps[..., :2] = FAR
        self.bushes = torch.zeros(self.E, max(B, 1), 4, **f32)
        self.bushes[..., :2] = FAR
        self.n_trees = torch.zeros(self.E, dtype=torch.long, device=device)
        for e, f in enumerate(forests):
            n = len(f.tree_xy)
            self.n_trees[e] = n
            self.tree_xy[e, :n] = torch.as_tensor(f.tree_xy)
            self.tree_z0[e, :n] = torch.as_tensor(f.tree_z0)
            self.tree_h[e, :n] = torch.as_tensor(f.tree_h)
            self.crown_base[e, :n] = torch.as_tensor(f.crown_base)
            self.crown_r[e, :n] = torch.as_tensor(f.crown_r)
            self.trunk_r[e, :n] = torch.as_tensor(f.trunk_r)
            self.clumps[e, :n] = torch.as_tensor(f.clumps)
            nb = len(f.bushes)
            if nb:
                self.bushes[e, :nb] = torch.as_tensor(f.bushes)
        self.trunk_top = self.tree_z0 + 0.9 * self.tree_h
        cr_max = self.clumps[..., 3].amax(-1)
        self.bound_r = torch.maximum(self.crown_r + cr_max, self.trunk_r)  # xy radius of everything of a tree
        self.top_z = self.tree_z0 + self.tree_h + cr_max
        self.forests_np, self.persons_np = forests, persons
        P = max(len(p.pos) for p in persons)
        K = persons[0].points.shape[1]
        self.person_pos = torch.zeros(self.E, P, 3, **f32)
        self.person_dims = torch.zeros(self.E, P, 3, **f32)
        self.person_yaw = torch.zeros(self.E, P, **f32)
        self.person_posture = torch.zeros(self.E, P, dtype=torch.long, device=device)
        self.person_points = torch.zeros(self.E, P, K, 3, **f32)
        for e, p in enumerate(persons):
            n = len(p.pos)
            self.person_pos[e, :n] = torch.as_tensor(p.pos)
            self.person_dims[e, :n] = torch.as_tensor(p.dims)
            self.person_yaw[e, :n] = torch.as_tensor(p.yaw)
            self.person_posture[e, :n] = torch.as_tensor(p.posture)
            self.person_points[e, :n] = torch.as_tensor(p.points)

    # ------------------------------------------------------------------ ray casting
    def cast(self, o: torch.Tensor, p: torch.Tensor, terrain: bool = True, bushes: bool = True,
             chunk: int = 6_000_000) -> torch.Tensor:
        """Segments o -> p, both [E, R, 3]. Returns first-hit fraction t in [0, 1] [E, R] (1 = no hit)."""
        E, R = o.shape[:2]
        out = torch.ones(E, R, device=o.device)
        T = self.tree_xy.shape[1]
        step = max(1, chunk // max(1, E * T))
        for r0 in range(0, R, step):
            sl = slice(r0, min(R, r0 + step))
            out[:, sl] = self._cast_trees(o[:, sl], p[:, sl])
        if bushes:
            out = torch.minimum(out, self._cast_spheres(o, p, self.bushes, chunk))
        if terrain:
            out = torch.minimum(out, self._cast_terrain(o, p))
        return out

    def _cast_trees(self, o, p):
        E, R = o.shape[:2]
        d = p - o
        o2, d2 = o[:, :, None, :2], d[:, :, None, :2]
        c2 = self.tree_xy[:, None]
        L2 = (d2 * d2).sum(-1).clamp_min(1e-9)
        s = (((c2 - o2) * d2).sum(-1) / L2).clamp(0, 1)
        dist2 = ((o2 + s[..., None] * d2 - c2) ** 2).sum(-1)
        zlo = torch.minimum(o[..., 2], p[..., 2])[:, :, None]
        zhi = torch.maximum(o[..., 2], p[..., 2])[:, :, None]
        cand = (dist2 <= self.bound_r[:, None] ** 2) & (zhi >= self.tree_z0[:, None]) & (zlo <= self.top_z[:, None])
        idx = cand.nonzero(as_tuple=True)
        t_hit = torch.ones(E, R, device=o.device)
        if idx[0].numel() == 0:
            return t_hit
        e, r, t = idx
        oo, dd = o[e, r], d[e, r]  # [P, 3]
        # trunk: vertical cylinder from the ground to 0.9 h
        cx = self.tree_xy[e, t]
        f2 = oo[:, :2] - cx
        a = (dd[:, :2] ** 2).sum(-1).clamp_min(1e-12)
        b = 2 * (f2 * dd[:, :2]).sum(-1)
        c = (f2 ** 2).sum(-1) - self.trunk_r[e, t] ** 2
        disc = b * b - 4 * a * c
        sq = disc.clamp_min(0).sqrt()
        t1, t2 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
        dz = dd[:, 2]
        z0, z1 = self.tree_z0[e, t], self.trunk_top[e, t]
        safe = torch.where(dz.abs() < 1e-9, torch.full_like(dz, 1e-9), dz)
        tz_a, tz_b = (z0 - oo[:, 2]) / safe, (z1 - oo[:, 2]) / safe
        flat = dz.abs() < 1e-9
        inside_z = (oo[:, 2] >= z0) & (oo[:, 2] <= z1)
        tz1 = torch.where(flat, torch.where(inside_z, torch.zeros_like(dz), torch.full_like(dz, 2.0)), torch.minimum(tz_a, tz_b))
        tz2 = torch.where(flat, torch.where(inside_z, torch.ones_like(dz), torch.full_like(dz, -1.0)), torch.maximum(tz_a, tz_b))
        lo = torch.maximum(torch.maximum(t1, tz1), torch.zeros_like(t1))
        hi = torch.minimum(torch.minimum(t2, tz2), torch.ones_like(t2))
        trunk_t = torch.where((disc >= 0) & (lo <= hi), lo, torch.ones_like(lo))
        # foliage clumps of the candidate trees
        cl = self.clumps[e, t]  # [P, C, 4]
        clump_t = _seg_sphere_t(oo[:, None, :], dd[:, None, :], cl).amin(-1)
        tp = torch.minimum(trunk_t, clump_t)
        flat_idx = e * R + r
        t_hit.view(-1).scatter_reduce_(0, flat_idx, tp, reduce="amin")
        return t_hit

    def _cast_spheres(self, o, p, sph, chunk):
        E, R = o.shape[:2]
        S = sph.shape[1]
        out = torch.ones(E, R, device=o.device)
        step = max(1, chunk // max(1, E * S))
        for r0 in range(0, R, step):
            sl = slice(r0, min(R, r0 + step))
            oo, dd = o[:, sl, None, :], (p - o)[:, sl, None, :]
            out[:, sl] = _seg_sphere_t(oo, dd, sph[:, None]).amin(-1)
        return out

    def _cast_terrain(self, o, p, n: int = 32):
        E, R = o.shape[:2]
        ts = torch.linspace(0.0, 1.0, n + 2, device=o.device)[1:-1]
        pts = o[:, :, None, :] + ts[None, None, :, None] * (p - o)[:, :, None, :]
        h = self.terrain.height(pts[..., :2].reshape(E, -1, 2)).reshape(E, R, n)
        below = pts[..., 2] < h
        first = torch.where(below, ts.expand_as(below), torch.ones_like(below, dtype=torch.float32))
        return first.amin(-1)

    # ------------------------------------------------------------------ collision
    def collide(self, pos: torch.Tensor, radius: float) -> torch.Tensor:
        """pos [E, 3] -> bool [E]: the drone sphere touches a trunk, a foliage clump, a bush or the ground."""
        d2 = ((pos[:, None, :2] - self.tree_xy) ** 2).sum(-1)
        near = d2 < (self.bound_r + radius) ** 2  # [E, T]
        trunk = (d2.sqrt() < self.trunk_r + radius) & (pos[:, None, 2] >= self.tree_z0) & (pos[:, None, 2] <= self.trunk_top)
        hit = trunk.any(-1)
        e, t = near.nonzero(as_tuple=True)
        if e.numel():
            cl = self.clumps[e, t]
            dd = ((cl[..., :3] - pos[e][:, None, :]) ** 2).sum(-1).sqrt() < cl[..., 3] + radius
            hit_c = torch.zeros(self.E, dtype=torch.int32, device=pos.device)
            hit_c.index_add_(0, e, dd.any(-1).int())
            hit = hit | (hit_c > 0)
        bd = ((self.bushes[..., :3] - pos[:, None, :]) ** 2).sum(-1).sqrt() < self.bushes[..., 3] + radius
        hit = hit | bd.any(-1)
        ground = self.terrain.height(pos[:, None, :2])[:, 0]
        return hit | (pos[:, 2] - ground < radius)

    def ground(self, xy: torch.Tensor) -> torch.Tensor:
        return self.terrain.height(xy)


def _seg_sphere_t(o, d, sph):
    """Segment o + t d (t in [0,1]) vs spheres (x, y, z, r); broadcast; returns entry t or 1."""
    f = o - sph[..., :3]
    a = (d * d).sum(-1).clamp_min(1e-12)
    b = 2 * (f * d).sum(-1)
    c = (f * f).sum(-1) - sph[..., 3] ** 2
    disc = b * b - 4 * a * c
    sq = disc.clamp_min(0).sqrt()
    t1, t2 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
    hit = (disc >= 0) & (t1 <= 1) & (t2 >= 0)
    return torch.where(hit, t1.clamp_min(0.0), torch.ones_like(t1))
