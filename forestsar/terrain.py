"""Procedural terrain: fBm height map on a regular grid, batched bilinear height queries."""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import zoom

from .config import TerrainCfg


def make_heightmap(cfg: TerrainCfg, rng: np.random.Generator) -> np.ndarray:
    """Height map [N, N] (row = y, column = x), N = size / cell + 1, values in [0, amp]."""
    n = int(round(cfg.size / cfg.cell)) + 1
    h = np.zeros((n, n))
    amp = 1.0
    for o in range(cfg.octaves):
        k = max(2, int(np.ceil(cfg.size / (cfg.scale / 2**o))) + 2)  # control points per side
        base = rng.standard_normal((k, k))
        up = zoom(base, n / k, order=3)[:n, :n]
        h += amp * up
        amp *= 0.5
    h -= h.min()
    h *= cfg.amp / max(h.max(), 1e-6)
    return h.astype(np.float32)


class Terrain:
    """Batched height maps [E, N, N] on the device."""

    def __init__(self, heights: torch.Tensor, cell: float, size: float):
        self.h = heights  # [E, N, N]
        self.cell, self.size = cell, size
        self.n = heights.shape[-1]

    def height(self, xy: torch.Tensor) -> torch.Tensor:
        """xy [E, M, 2] (world, metres) -> ground height [E, M]. Points outside are clamped to the border."""
        E, M = xy.shape[:2]
        g = (xy / self.cell).clamp(0.0, self.n - 1.000001)
        x0, y0 = g[..., 0].floor().long(), g[..., 1].floor().long()
        fx, fy = g[..., 0] - x0, g[..., 1] - y0
        flat = self.h.reshape(E, -1)

        def at(yy, xx):
            return torch.gather(flat, 1, (yy * self.n + xx).reshape(E, -1)).reshape(E, M)

        h00, h01 = at(y0, x0), at(y0, x0 + 1)
        h10, h11 = at(y0 + 1, x0), at(y0 + 1, x0 + 1)
        return (h00 * (1 - fx) + h01 * fx) * (1 - fy) + (h10 * (1 - fx) + h11 * fx) * fy

    def gradient(self, xy: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
        """Central-difference slope dz/dx, dz/dy [E, M, 2]."""
        ex = torch.tensor((eps, 0.0), device=xy.device)
        ey = torch.tensor((0.0, eps), device=xy.device)
        gx = (self.height(xy + ex) - self.height(xy - ex)) / (2 * eps)
        gy = (self.height(xy + ey) - self.height(xy - ey)) / (2 * eps)
        return torch.stack((gx, gy), -1)
