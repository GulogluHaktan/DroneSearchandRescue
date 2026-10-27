"""Configuration dataclasses. SI units (m, s, kg, rad) unless the name says otherwise."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VehicleRanges:
    """Quadrotor parameter ranges (same fields as DroneRacing). Defaults: a 1.3-1.9 kg SAR quad.

    ``dr_scale`` in [0, 1]: 0 = nominal vehicle (range centres), 1 = sample the full ranges.
    """

    mass: tuple[float, float] = (1.3, 1.9)
    max_thrust: tuple[float, float] = (32.0, 52.0)  # total [N]
    twr_range: tuple[float, float] = (2.0, 3.5)
    arm: tuple[float, float] = (0.18, 0.24)
    drag_cd: tuple[float, float] = (0.8, 1.2)
    tau_rate: tuple[float, float] = (0.04, 0.07)
    tau_thrust: tuple[float, float] = (0.04, 0.08)
    max_rate: float = 4.0  # [rad/s] body rate at |action| = 1
    yaw_rate_scale: float = 0.5
    direct_motor: bool = False
    dr_scale: float = 0.0


@dataclass
class TerrainCfg:
    size: float = 300.0  # side of the square search area [m]; origin at the south-west corner
    cell: float = 2.0  # height-map resolution [m]
    amp: float = 12.0  # peak-to-peak relief scale [m]
    scale: float = 120.0  # largest feature size of the fBm noise [m]
    octaves: int = 4


@dataclass
class ForestCfg:
    density: float = 350.0  # trees per hectare
    min_spacing: float = 2.5  # [m] between trunk axes
    height: tuple[float, float] = (14.0, 26.0)  # tree height [m]
    crown_base_frac: tuple[float, float] = (0.40, 0.60)  # crown base height / tree height (lowest live branches)
    crown_radius: tuple[float, float] = (1.8, 3.8)  # [m]
    trunk_radius: tuple[float, float] = (0.12, 0.30)  # [m]
    conifer_frac: float = 0.5  # cone crowns; the rest are ellipsoids (broadleaf)
    clumps_per_crown: int = 36  # foliage spheres per crown (gaps between them are see-through)
    clump_radius: tuple[float, float] = (0.45, 0.85)  # [m]
    bush_density: float = 250.0  # understory bushes per hectare
    bush_radius: tuple[float, float] = (0.4, 0.9)  # bush sphere radius [m]; centre at radius height -> top at 2 r
    clearings: int = 4  # number of open patches (no trees) in the area
    clearing_radius: tuple[float, float] = (8.0, 18.0)


@dataclass
class PersonCfg:
    n: int = 1  # lost persons per mission
    posture_probs: tuple[float, float, float] = (0.3, 0.3, 0.4)  # standing, sitting, lying
    shelter_prob: float = 0.3  # chance to be placed close to a trunk (sheltering under a tree)
    n_points: int = 12  # body sample points for the visibility test


@dataclass
class PriorCfg:
    """Lost-person prior: Monte Carlo walkers from the initial planning point (IPP).

    Placeholder for an agent-based behaviour model (funded phase P3); distances are parameters,
    not real incident statistics.
    """

    n_walkers: int = 4000
    steps: int = 1500
    step_len: float = 1.5  # [m]
    persistence: float = 0.85  # heading memory in [0, 1)
    slope_aversion: float = 3.0  # downhill bias weight; uphill steps are rejected with this weight
    stop_prob: float = 0.0015  # per-step chance that the walker stops (exhausted / sheltering)
    ipp_margin: float = 0.25  # IPP is placed this fraction of the side away from the area edge
    launch_inset: float = 10.0  # the team launches from the area edge closest to the IPP, this far inside [m]
    cell: float = 5.0  # grid of the probability map and the belief map [m]
    blur_cells: float = 1.5  # Gaussian smoothing of the walker histogram [cells]


@dataclass
class CameraCfg:
    width_px: int = 1920
    height_px: int = 1080
    hfov_deg: float = 80.0
    range_max: float = 90.0  # [m] beyond this nothing is evaluated
    pitch_above_deg: float = 60.0  # gimbal pitch below the horizon when above the canopy
    pitch_under_deg: float = 12.0  # gimbal pitch below the horizon when under the canopy
    rate_hz: float = 2.0  # detector frames per second (frames are treated as independent)
    p_max: float = 0.85  # per-frame detection probability of a fully visible, large target
    px50: float = 12.0  # effective linear size [px] (sqrt of visible area) for 50 % of p_max
    px_slope: float = 3.0  # logistic slope [px]
    vis_min: float = 0.05  # visible fraction below this -> no detection
    fa_per_ha_frame: float = 0.0015  # false alarms per hectare of imaged ground per frame


@dataclass
class DepthCfg:
    """Forward obstacle sensor used by the local avoidance (simulated by ray casting)."""

    n_az: int = 72
    az_span_deg: float = 355.0
    elevations_deg: tuple[float, ...] = (-8.0, 0.0, 8.0)
    band: float = 1.0  # hits within +-band of the drone height count as obstacles [m]
    range_max: float = 14.0
    rate_hz: float = 10.0


@dataclass
class FlightCfg:
    dt: float = 0.02  # physics / controller step [s] (50 Hz; tau_rate >= 0.04 s keeps the Euler rate loop stable)
    alt_above: float = 12.0  # clearance above the local canopy top [m]
    alt_under: float = 3.0  # AGL when flying between the trees [m]
    v_above: float = 8.0  # cruise speed above the canopy [m/s]
    v_under: float = 2.0  # cruise speed between the trees [m/s]
    v_vertical: float = 1.5  # climb / descent speed in a canopy gap [m/s]
    drone_radius: float = 0.35  # collision radius incl. propellers [m]
    safety_margin: float = 0.6  # extra clearance demanded by the avoidance [m]
    a_max: float = 4.0  # horizontal acceleration limit of the velocity controller [m/s^2]
    time_limit: float = 900.0  # mission time budget [s]
    hover_power_w: float = 260.0  # electrical power at hover (energy proxy)


@dataclass
class MissionCfg:
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    forest: ForestCfg = field(default_factory=ForestCfg)
    person: PersonCfg = field(default_factory=PersonCfg)
    prior: PriorCfg = field(default_factory=PriorCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    depth: DepthCfg = field(default_factory=DepthCfg)
    flight: FlightCfg = field(default_factory=FlightCfg)
    vehicle: VehicleRanges = field(default_factory=VehicleRanges)
