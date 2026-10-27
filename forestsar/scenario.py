"""Build E missions (terrain, forest, prior, persons) and the maps the planner is allowed to know.

Planner knowledge (what a real team has before take-off):
  * DEM (terrain height)                                   -> terrain-following altitude
  * canopy maps as from airborne LiDAR: vertical gap fraction per belief cell, canopy top height,
    and the 1 m "descent" map (columns that are free from the canopy down to the under-canopy altitude)
  * mean stand statistics (trees / m^2, mean trunk radius, bushes / m^2) for the analytic sight model
  * the lost-person prior
The planner does NOT know the trunk / clump / bush positions or the persons.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .config import MissionCfg
from .forest import make_forest
from .prior import height_np, make_prior, sample_from_prior
from .terrain import Terrain, make_heightmap
from .world import PersonsNp, World, body_points


@dataclass
class Maps:
    prior: torch.Tensor  # [E, G, G] sums to 1 per env
    ipp: torch.Tensor  # [E, 2]
    gap: torch.Tensor  # [E, G, G] vertical gap fraction to 1 m above ground (1 = open)
    canopy_top: torch.Tensor  # [E, G, G] absolute height of the highest foliage near the cell (15 m window)
    descent: torch.Tensor  # [E, N1, N1] bool, 1 m grid: a vertical shaft to the under-canopy altitude is free
    lam_under: torch.Tensor  # [E] line-of-sight extinction under the canopy [1/m] (Boolean model)
    launch: torch.Tensor  # [E, 2] take-off point (area edge closest to the IPP)
    cell: float
    fine: float = 1.0


def _persons(prior, cell, cfg, hmap, hcell, forest, rng) -> PersonsNp:
    pc = cfg.person
    xy = sample_from_prior(prior, cell, pc.n, rng)
    for i in range(pc.n):
        if rng.uniform() < pc.shelter_prob and len(forest.tree_xy):
            d = np.sqrt(((forest.tree_xy - xy[i]) ** 2).sum(-1))
            j = int(np.argmin(d))
            if d[j] < 20.0:
                a = rng.uniform(0, 2 * np.pi)
                rr = forest.trunk_r[j] + rng.uniform(0.6, 1.4)
                xy[i] = forest.tree_xy[j] + rr * np.array((np.cos(a), np.sin(a)), dtype=np.float32)
    xy = np.clip(xy, 1.0, cfg.terrain.size - 1.0)
    z = height_np(hmap, hcell, xy)
    posture = rng.choice(3, size=pc.n, p=np.array(pc.posture_probs) / np.sum(pc.posture_probs))
    yaw = rng.uniform(0, 2 * np.pi, pc.n).astype(np.float32)
    pos = np.concatenate((xy, z[:, None]), -1).astype(np.float32)
    pts, dims = zip(*[body_points(pos[i], yaw[i], posture[i], pc.n_points) for i in range(pc.n)])
    return PersonsNp(pos, yaw, posture.astype(np.int64), np.stack(dims), np.stack(pts))


def build(cfg: MissionCfg, n_envs: int, seed: int, device: str, first_env: int = 0) -> tuple[World, Maps]:
    tc = cfg.terrain
    hmaps, forests, persons, priors, ipps = [], [], [], [], []
    for e in range(n_envs):
        rng = np.random.default_rng(seed * 1000 + first_env + e)  # env index is part of the seed: any env can be rebuilt alone
        h = make_heightmap(tc, rng)
        f = make_forest(cfg.forest, tc.size, lambda xy, h=h: height_np(h, tc.cell, xy), rng)
        pr, ipp = make_prior(cfg.prior, h, tc.cell, tc.size, rng)
        persons.append(_persons(pr, cfg.prior.cell, cfg, h, tc.cell, f, rng))
        hmaps.append(h)
        forests.append(f)
        priors.append(pr)
        ipps.append(ipp)
    terrain = Terrain(torch.as_tensor(np.stack(hmaps), device=device), tc.cell, tc.size)
    world = World(terrain, forests, persons, tc.size, device)
    maps = canopy_maps(world, cfg, torch.as_tensor(np.stack(priors), device=device), torch.as_tensor(np.stack(ipps), device=device))
    return world, maps


@torch.no_grad()
def canopy_maps(world: World, cfg: MissionCfg, prior: torch.Tensor, ipp: torch.Tensor) -> Maps:
    E, size, dev = world.E, world.size, world.device
    n1 = int(round(size))
    c = (torch.arange(n1, device=dev, dtype=torch.float32) + 0.5)
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    xy = torch.stack((xx, yy), -1).reshape(1, -1, 2).expand(E, -1, -1).contiguous()
    g = world.ground(xy)
    top = world.top_z.amax(-1)[:, None] + 2.0
    o = torch.cat((xy, top.expand(-1, xy.shape[1])[..., None]), -1)
    p = torch.cat((xy, (g + 1.0)[..., None]), -1)
    t = world.cast(o, p, terrain=False)
    z_hit = o[..., 2] + t * (p[..., 2] - o[..., 2])
    open_ = (t >= 0.999).float().reshape(E, n1, n1)
    hit_agl = (z_hit - g).reshape(E, n1, n1)  # first hit from above, height above ground (1.0 = reached person height)
    G = int(round(size / cfg.prior.cell))
    k = n1 // G
    gap = F.avg_pool2d(open_[:, None], k)[:, 0]
    foliage_top = torch.where(t < 0.999, z_hit, g).reshape(E, 1, n1, n1)
    top15 = F.max_pool2d(foliage_top, 15, stride=1, padding=7)
    canopy_top = F.max_pool2d(top15, k)[:, 0]
    fl = cfg.flight
    free = (hit_agl <= fl.alt_under - 0.5).float()  # nothing above the under-canopy flight level
    # trunks are thinner than the 1 m ray grid: rasterise them explicitly (conservative, cell half-diagonal)
    for e in range(E):
        n = int(world.n_trees[e])
        txy, tr = world.tree_xy[e, :n], world.trunk_r[e, :n]
        ij = txy.floor().long()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cx, cy = ij[:, 0] + dx, ij[:, 1] + dy
                inside = (cx >= 0) & (cx < n1) & (cy >= 0) & (cy < n1)
                cc = torch.stack((cx, cy), -1).float() + 0.5
                near = ((cc - txy) ** 2).sum(-1).sqrt() < tr + 0.71
                m = inside & near
                free[e, cy[m], cx[m]] = 0.0
    rad = int(np.ceil(fl.drone_radius + fl.safety_margin + 0.5))
    descent = -F.max_pool2d(-free[:, None], 2 * rad + 1, stride=1, padding=rad)[:, 0] > 0.5  # erosion
    area = size * size
    lam = []
    for e in range(E):
        n = int(world.n_trees[e])
        tr = float(world.trunk_r[e, :n].mean()) if n else 0.0
        nb = int((world.bushes[e, :, 0] < 1e5).sum())
        br = float(world.bushes[e, :nb, 3].mean()) if nb else 0.0
        lam.append(n / area * 2 * tr + 0.5 * nb / area * 2 * br)  # bushes block about half of a low sight line
    ins = cfg.prior.launch_inset
    side = torch.stack((ipp[:, 0], size - ipp[:, 0], ipp[:, 1], size - ipp[:, 1]), -1).argmin(-1)
    launch = ipp.clone()
    launch[:, 0] = torch.where(side == 0, torch.full_like(launch[:, 0], ins), launch[:, 0])
    launch[:, 0] = torch.where(side == 1, torch.full_like(launch[:, 0], size - ins), launch[:, 0])
    launch[:, 1] = torch.where(side == 2, torch.full_like(launch[:, 1], ins), launch[:, 1])
    launch[:, 1] = torch.where(side == 3, torch.full_like(launch[:, 1], size - ins), launch[:, 1])
    return Maps(prior, ipp, gap, canopy_top, descent, torch.tensor(lam, device=dev), launch, cfg.prior.cell)
