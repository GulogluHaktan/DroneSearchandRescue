"""Lost-person prior: Monte Carlo walkers from the initial planning point (IPP) on the terrain.

This is a placeholder behaviour model (P0). Walkers keep their heading, prefer downhill, reject
steep climbs and stop at random (exhaustion / shelter). The final positions are histogrammed on
the belief grid and smoothed. Funded phase P3 replaces it with profile-based agents
(Ewers et al. 2025, Ge et al. 2024) and real incident statistics.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import PriorCfg


def height_np(hmap: np.ndarray, cell: float, xy: np.ndarray) -> np.ndarray:
    n = hmap.shape[0]
    g = np.clip(xy / cell, 0, n - 1.000001)
    x0, y0 = np.floor(g[:, 0]).astype(int), np.floor(g[:, 1]).astype(int)
    fx, fy = g[:, 0] - x0, g[:, 1] - y0
    h00, h01 = hmap[y0, x0], hmap[y0, x0 + 1]
    h10, h11 = hmap[y0 + 1, x0], hmap[y0 + 1, x0 + 1]
    return (h00 * (1 - fx) + h01 * fx) * (1 - fy) + (h10 * (1 - fx) + h11 * fx) * fy


def make_prior(cfg: PriorCfg, hmap: np.ndarray, hcell: float, size: float, rng: np.random.Generator):
    """Returns (prior [G, G] summing to 1, row = y; ipp [2])."""
    m = cfg.ipp_margin * size
    ipp = rng.uniform(m, size - m, 2)
    n = cfg.n_walkers
    pos = np.tile(ipp, (n, 1))
    head = rng.uniform(0, 2 * np.pi, n)
    moving = np.ones(n, dtype=bool)
    for _ in range(cfg.steps):
        head = head + rng.normal(0, np.sqrt(1 - cfg.persistence**2) * 1.2, n)
        step = cfg.step_len * np.stack((np.cos(head), np.sin(head)), -1)
        new = np.clip(pos + step, 0, size)
        dz = height_np(hmap, hcell, new) - height_np(hmap, hcell, pos)
        p_acc = np.exp(-cfg.slope_aversion * np.maximum(dz, 0) / cfg.step_len)  # climbing is costly
        p_acc *= np.where(dz < 0, 1.0, 0.8)  # mild downhill preference
        acc = (rng.uniform(0, 1, n) < p_acc) & moving
        pos = np.where(acc[:, None], new, pos)
        head = np.where(acc, head, head + rng.uniform(-np.pi / 2, np.pi / 2, n))
        moving &= rng.uniform(0, 1, n) > cfg.stop_prob
    g = int(round(size / cfg.cell))
    H, _, _ = np.histogram2d(pos[:, 1], pos[:, 0], bins=g, range=[[0, size], [0, size]])
    H = gaussian_filter(H, cfg.blur_cells) + 1e-6 * H.sum() / H.size  # small floor: nowhere is impossible
    return (H / H.sum()).astype(np.float32), ipp.astype(np.float32)


def sample_from_prior(prior: np.ndarray, cell: float, k: int, rng: np.random.Generator) -> np.ndarray:
    flat = prior.ravel()
    idx = rng.choice(flat.size, size=k, p=flat / flat.sum())
    gy, gx = np.divmod(idx, prior.shape[1])
    return np.stack(((gx + rng.uniform(0, 1, k)) * cell, (gy + rng.uniform(0, 1, k)) * cell), -1).astype(np.float32)
