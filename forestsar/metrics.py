"""Mission metrics (one shared set, gap G6 of the literature review)."""
from __future__ import annotations

import torch

from .config import MissionCfg


def summarize(res, cfg: MissionCfg) -> dict:
    f = res.found.float()
    tf = res.t_found[res.found]
    E = len(f)
    lim = cfg.flight.time_limit
    curve = {}
    for frac in (0.25, 0.5, 0.75, 1.0):
        tc = frac * lim
        curve[f"P(found<= {tc:.0f}s)"] = round(float((res.found & (res.t_found <= tc)).float().mean()), 3)
    under = res.layer_at_find[res.found]
    return {
        "n": E,
        "success": round(float(f.mean()), 3),
        "crash": round(float(res.crashed.float().mean()), 3),
        "ttf_median_s": round(float(tf.median()), 1) if len(tf) else None,
        "ttf_p90_s": round(float(torch.quantile(tf, 0.9)), 1) if len(tf) > 1 else None,
        **curve,
        "path_km": round(float(res.path_m.mean()) / 1000, 2),
        "energy_wh": round(float(res.energy_wh.mean()), 1),
        "false_alarms_per_km2": round(float(res.false_alarms.sum() / res.imaged_ha.sum().clamp_min(1e-6) * 100), 2),
        "time_under_frac": round(float((res.t_under / res.t_found.clamp_min(1)).mean()), 2),
        "found_from_under_frac": round(float((under == 1).float().mean()), 2) if len(under) else None,
        "vis_at_find_mean": round(float(res.vis_at_find[res.found].mean()), 2) if len(tf) else None,
        "planner_pos_end": round(float(res.pos_planner.mean()), 3),
        "wall_s": round(res.wall_s, 1),
    }
