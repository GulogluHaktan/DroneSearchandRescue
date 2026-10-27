"""Search planners (batched over E missions). Layers: 0 = above the canopy, 1 = between the trees.

All planners share one transition logic: to change layer the drone flies to a canopy gap
("descent shaft" from the canopy map) and climbs / descends vertically there.

Baselines in P0:
  lawnmower-above   boustrophedon over the area, above the canopy
  lawnmower-under   boustrophedon between the trees (local avoidance on)
  greedy-above      receding-horizon: max (belief mass x layer detection weight) / travel time
  greedy-under      same, between the trees
  hybrid            greedy in both layers; switches layer when the other layer's gain rate
                    (incl. climb / descent time) is higher
A learned (RL) planner is a funded-phase item; it plugs into the same ``decide`` interface.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import MissionCfg
from .scenario import Maps

ABOVE, UNDER = 0, 1
CRUISE, DESCEND, ASCEND, TO_SHAFT = 0, 1, 2, 3


def _disk(radius_cells: float, device) -> torch.Tensor:
    r = int(math.ceil(radius_cells))
    c = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    return ((xx**2 + yy**2) <= radius_cells**2).float()[None, None]


class Planner:
    name = "base"

    def __init__(self, cfg: MissionCfg, maps: Maps, start_xy: torch.Tensor):
        self.cfg, self.maps = cfg, maps
        E = start_xy.shape[0]
        dev = start_xy.device
        self.E, self.dev = E, dev
        self.layer = torch.zeros(E, dtype=torch.long, device=dev)
        self.phase = torch.zeros(E, dtype=torch.long, device=dev)
        self.goal_layer = torch.zeros(E, dtype=torch.long, device=dev)
        self.target = start_xy.clone()
        self.final_target = start_xy.clone()
        self.t_set = torch.zeros(E, device=dev)
        # descent shafts: 1 m cells that are free from the canopy down to the flight level
        self.shafts = [torch.nonzero(maps.descent[e]).flip(-1).float() + 0.5 for e in range(E)]  # (x, y)
        G = maps.prior.shape[-1]
        c = (torch.arange(G, device=dev, dtype=torch.float32) + 0.5) * maps.cell
        yy, xx = torch.meshgrid(c, c, indexing="ij")
        self.cell_xy = torch.stack((xx, yy), -1).reshape(-1, 2)  # [G*G, 2]
        self.d_shaft = torch.stack([self._nearest_shaft_dist(e, self.cell_xy) for e in range(E)])  # [E, G*G]

    # ---- helpers ---------------------------------------------------------------------
    def _nearest_shaft_dist(self, e: int, xy: torch.Tensor) -> torch.Tensor:
        s = self.shafts[e]
        if len(s) == 0:
            return torch.full((len(xy),), 1e4, device=self.dev)
        return torch.cdist(xy, s).amin(-1)

    def shaft_between(self, e: int, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Shaft minimising |a - s| + |s - b| (fly to it, change layer, continue to b)."""
        s = self.shafts[e]
        if len(s) == 0:
            return a.clone()
        cost = (s - a).norm(dim=-1) + (s - b).norm(dim=-1)
        return s[cost.argmin()]

    def request(self, e: int, layer: int, xy: torch.Tensor, pos_xy: torch.Tensor, t: float) -> None:
        """Set the goal for env e; inserts a layer change through a shaft if needed."""
        self.final_target[e] = xy
        self.goal_layer[e] = layer
        self.t_set[e] = t
        if layer == int(self.layer[e]):
            self.target[e] = xy
            self.phase[e] = CRUISE
        else:
            self.target[e] = self.shaft_between(e, pos_xy, xy)
            self.phase[e] = TO_SHAFT

    def on_vertical_done(self, e: int) -> None:
        self.layer[e] = self.goal_layer[e]
        self.phase[e] = CRUISE
        self.target[e] = self.final_target[e]

    # ---- interface -------------------------------------------------------------------
    def decide(self, t: float, pos: torch.Tensor, belief: torch.Tensor, blocked_s: torch.Tensor, active: torch.Tensor) -> None:
        raise NotImplementedError


class Lawnmower(Planner):
    def __init__(self, cfg, maps, start_xy, layer: int):
        super().__init__(cfg, maps, start_xy)
        self.fixed_layer = layer
        self.name = "lawnmower-" + ("above" if layer == ABOVE else "under")
        size = cfg.terrain.size
        sp = 40.0 if layer == ABOVE else 20.0
        m = sp / 2
        xs = torch.arange(m, size - m + 1e-3, sp)
        pts = []
        for i, x in enumerate(xs):
            ys = (m, size - m) if i % 2 == 0 else (size - m, m)
            pts += [(float(x), ys[0]), (float(x), ys[1])]
        if layer == UNDER:  # densify legs so a blocked leg does not skip a whole lane
            dense = []
            for a, b in zip(pts[:-1], pts[1:]):
                n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) // 15))
                dense += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(n)]
            pts = dense + [pts[-1]]
        self.wps = torch.tensor(pts, device=self.dev)
        # start at the end of the pattern that is closer to the launch point
        d0 = (self.wps[0] - start_xy).norm(dim=-1)
        d1 = (self.wps[-1] - start_xy).norm(dim=-1)
        self.rev = d1 < d0
        self.idx = torch.zeros(self.E, dtype=torch.long, device=self.dev)

    def _wp(self, e):
        i = int(self.idx[e])
        n = len(self.wps)
        return self.wps[n - 1 - i] if bool(self.rev[e]) else self.wps[i]

    def decide(self, t, pos, belief, blocked_s, active):
        for e in range(self.E):
            if not bool(active[e]) or int(self.phase[e]) in (DESCEND, ASCEND):
                continue
            if int(self.phase[e]) == TO_SHAFT:
                continue
            wp = self._wp(e)
            reached = (pos[e, :2] - wp).norm() < (6.0 if self.fixed_layer == ABOVE else 3.0)
            if reached or blocked_s[e] > 10.0:
                self.idx[e] = min(int(self.idx[e]) + 1, len(self.wps) - 1)
                wp = self._wp(e)
            if not torch.equal(self.final_target[e], wp) or int(self.layer[e]) != self.fixed_layer:
                self.request(e, self.fixed_layer, wp, pos[e, :2], t)


class Greedy(Planner):
    """Receding-horizon greedy on the belief, in one layer or both (hybrid)."""

    def __init__(self, cfg, maps, start_xy, layers=(ABOVE,), dwell: float = 10.0):
        super().__init__(cfg, maps, start_xy)
        self.layers = layers
        self.name = "hybrid" if len(layers) == 2 else "greedy-" + ("above" if layers[0] == ABOVE else "under")
        fl = cfg.flight
        self.dwell = dwell
        cell = maps.cell
        self.k_above = _disk(30.0 / cell, self.dev)
        self.k_under = _disk(14.0 / cell, self.dev)
        G = maps.prior.shape[-1]
        self.w_above = maps.gap.clamp_min(1e-4) ** 1.4 * 0.7  # typical slant + pixel factor
        self.w_under = torch.exp(-maps.lam_under[:, None, None] * 7.0).expand(-1, G, G) * 0.8
        dh = (maps.canopy_top.reshape(self.E, -1) + fl.alt_above) - fl.alt_under  # rough climb height per cell (DEM cancels)
        self.t_vert = dh.clamp_min(0) / fl.v_vertical  # [E, G*G]
        self.hold_s = 15.0

    def _gain(self, belief, layer):
        w = self.w_above if layer == ABOVE else self.w_under
        k = self.k_above if layer == ABOVE else self.k_under
        g = F.conv2d((belief * w)[:, None], k, padding=k.shape[-1] // 2)[:, 0]
        return g.reshape(self.E, -1)

    def decide(self, t, pos, belief, blocked_s, active):
        fl = self.cfg.flight
        d = torch.cdist(pos[:, None, :2], self.cell_xy[None].expand(self.E, -1, -1))[:, 0]  # [E, G*G]
        scores = {}
        for L in self.layers:
            gain = self._gain(belief, L)
            v = fl.v_above if L == ABOVE else fl.v_under
            tt = d / v + self.dwell
            # layer change cost from the current layer
            here = self.layer
            if L == UNDER:
                change = (here == ABOVE)[:, None]
                t_change = self.t_vert + self.d_shaft / fl.v_under
            else:
                change = (here == UNDER)[:, None]
                d_up = torch.stack([self._nearest_shaft_dist(e, pos[e:e + 1, :2])[0] for e in range(self.E)])
                t_change = d_up[:, None] / fl.v_under + self.t_vert
            tt = tt + torch.where(change, t_change, torch.zeros_like(tt))
            near = d < (8.0 if L == ABOVE else 4.0)
            s = torch.where(near, torch.zeros_like(gain), gain / tt)
            scores[L] = s
        for e in range(self.E):
            if not bool(active[e]) or int(self.phase[e]) in (DESCEND, ASCEND):
                continue
            cruising = int(self.phase[e]) == CRUISE
            reached = (pos[e, :2] - self.final_target[e]).norm() < (8.0 if int(self.layer[e]) == ABOVE else 3.0)
            stale = t - float(self.t_set[e]) > self.hold_s
            if cruising and not (reached or stale or blocked_s[e] > 8.0):
                continue
            if int(self.phase[e]) == TO_SHAFT and not blocked_s[e] > 8.0:
                continue
            best = None
            for L in self.layers:
                s = scores[L][e]
                k = int(s.argmax())
                val = float(s[k])
                if L != int(self.layer[e]):
                    val /= 1.25  # hysteresis against flip-flopping between layers
                if best is None or val > best[0]:
                    best = (val, L, k)
            _, L, k = best
            self.request(e, L, self.cell_xy[k], pos[e, :2], t)


def make_planner(name: str, cfg: MissionCfg, maps: Maps, start_xy: torch.Tensor) -> Planner:
    if name == "lawnmower-above":
        return Lawnmower(cfg, maps, start_xy, ABOVE)
    if name == "lawnmower-under":
        return Lawnmower(cfg, maps, start_xy, UNDER)
    if name == "greedy-above":
        return Greedy(cfg, maps, start_xy, (ABOVE,))
    if name == "greedy-under":
        return Greedy(cfg, maps, start_xy, (UNDER,))
    if name == "hybrid":
        return Greedy(cfg, maps, start_xy, (ABOVE, UNDER))
    raise ValueError(name)


PLANNERS = ("lawnmower-above", "greedy-above", "lawnmower-under", "greedy-under", "hybrid")
