"""Procedural forest: trees (trunk cylinder + crown of foliage spheres), understory bushes, clearings.

A crown is a cloud of foliage "clumps" (spheres). Light and lines of sight pass through the gaps
between clumps, so canopy occlusion is geometric, not a fixed opacity. The same primitives are
exported to USD for the Isaac renderer (isaac/forest_usd.py), so both layers see one forest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ForestCfg


@dataclass
class ForestNp:
    """One forest on the CPU. Heights are absolute (terrain included)."""

    tree_xy: np.ndarray  # [T, 2]
    tree_z0: np.ndarray  # [T] ground height at the trunk
    tree_h: np.ndarray  # [T] tree height above ground
    crown_base: np.ndarray  # [T] crown base height above ground
    crown_r: np.ndarray  # [T]
    trunk_r: np.ndarray  # [T]
    conifer: np.ndarray  # [T] bool
    clumps: np.ndarray  # [T, C, 4] absolute x, y, z, radius
    bushes: np.ndarray  # [B, 4] absolute x, y, z, radius
    clearings: np.ndarray  # [K, 3] x, y, radius


def _poisson_points(n_target: int, size: float, min_d: float, rng: np.random.Generator, blocked) -> np.ndarray:
    """Dart throwing with a hash grid; ``blocked(p)`` rejects points (clearings)."""
    cell = min_d / np.sqrt(2)
    g = int(np.ceil(size / cell)) + 1
    grid = -np.ones((g, g), dtype=np.int64)
    pts: list[np.ndarray] = []
    tries = 0
    while len(pts) < n_target and tries < n_target * 30:
        tries += 1
        p = rng.uniform(0.0, size, 2)
        if blocked(p):
            continue
        gx, gy = int(p[0] / cell), int(p[1] / cell)
        ok = True
        for yy in range(max(0, gy - 2), min(g, gy + 3)):
            for xx in range(max(0, gx - 2), min(g, gx + 3)):
                j = grid[yy, xx]
                if j >= 0 and np.sum((pts[j] - p) ** 2) < min_d * min_d:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            grid[gy, gx] = len(pts)
            pts.append(p)
    return np.array(pts, dtype=np.float32).reshape(-1, 2)


def _crown_clumps(base: float, top: float, R: float, conifer: bool, n: int, rc: tuple, rng) -> np.ndarray:
    """Clump centres (relative to the trunk axis, z above ground) and radii [n, 4]."""
    out = np.zeros((n, 4), dtype=np.float32)
    k = 0
    while k < n:
        m = 4 * (n - k)
        u = rng.uniform(0, 1, m)
        if conifer:  # cone: radius shrinks linearly to the top; uniform in volume -> more mass low
            z = base + (top - base) * (1 - np.cbrt(u))
            rmax = R * (top - z) / max(top - base, 1e-3)
        else:  # ellipsoid
            zc, hz = 0.5 * (base + top), 0.5 * (top - base)
            z = zc + hz * rng.uniform(-1, 1, m)
            rmax = R * np.sqrt(np.clip(1 - ((z - zc) / hz) ** 2, 0, 1))
        rad = rmax * np.sqrt(rng.uniform(0.15, 1.0, m))  # foliage concentrates towards the shell
        a = rng.uniform(0, 2 * np.pi, m)
        r = rng.uniform(rc[0], rc[1], m)
        keep = rmax > 0.2
        z, rad, a, r = z[keep], rad[keep], a[keep], r[keep]
        take = min(n - k, len(z))
        out[k:k + take, 0] = rad[:take] * np.cos(a[:take])
        out[k:k + take, 1] = rad[:take] * np.sin(a[:take])
        out[k:k + take, 2] = z[:take]
        out[k:k + take, 3] = np.minimum(r[:take], 0.6 * R)
        k += take
    return out


def make_forest(cfg: ForestCfg, size: float, height_fn, rng: np.random.Generator) -> ForestNp:
    """``height_fn(xy [M,2]) -> [M]`` gives the ground height (numpy)."""
    cl = np.stack(
        (rng.uniform(0.1 * size, 0.9 * size, cfg.clearings), rng.uniform(0.1 * size, 0.9 * size, cfg.clearings),
         rng.uniform(*cfg.clearing_radius, cfg.clearings)), -1).astype(np.float32).reshape(-1, 3)

    def blocked(p):
        return bool(np.any(np.sum((cl[:, :2] - p) ** 2, -1) < cl[:, 2] ** 2))

    area_ha = size * size / 1e4
    xy = _poisson_points(int(cfg.density * area_ha), size, cfg.min_spacing, rng, blocked)
    T = len(xy)
    z0 = height_fn(xy).astype(np.float32)
    h = rng.uniform(*cfg.height, T).astype(np.float32)
    cb = (h * rng.uniform(*cfg.crown_base_frac, T)).astype(np.float32)
    cr = rng.uniform(*cfg.crown_radius, T).astype(np.float32)
    tr = rng.uniform(*cfg.trunk_radius, T).astype(np.float32)
    con = rng.uniform(0, 1, T) < cfg.conifer_frac
    C = cfg.clumps_per_crown
    clumps = np.zeros((T, C, 4), dtype=np.float32)
    for i in range(T):
        c = _crown_clumps(cb[i], h[i], cr[i], bool(con[i]), C, cfg.clump_radius, rng)
        c[:, 0] += xy[i, 0]
        c[:, 1] += xy[i, 1]
        c[:, 2] += z0[i]
        clumps[i] = c

    nb = int(cfg.bush_density * area_ha)
    bxy = rng.uniform(0, size, (nb, 2)).astype(np.float32)
    if T:
        d2 = ((bxy[:, None, :] - xy[None, :, :]) ** 2).sum(-1).min(1)
        bxy = bxy[d2 > 1.0]
    bxy = bxy[[not blocked(p) or rng.uniform() < 0.3 for p in bxy]] if len(bxy) else bxy
    br = rng.uniform(*cfg.bush_radius, len(bxy)).astype(np.float32)
    bz = height_fn(bxy).astype(np.float32) + 0.8 * br
    bushes = np.concatenate((bxy, bz[:, None], br[:, None]), -1).astype(np.float32)
    return ForestNp(xy, z0, h, cb, cr, tr, con, clumps, bushes, cl)
