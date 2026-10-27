"""Batched quadrotor dynamics in PyTorch (no Isaac dependency).

Copied from DroneRacing (droneracing/quad.py, 2026-10-26). Changes: the X-frame sign tensors are
cached (one host->device copy per step cost ~40 % of the run time); VehicleRanges defaults
differ (forestsar/config.py: a 1.3-1.9 kg SAR quad).

Frames: world z-up, body z = thrust axis, quaternion (w, x, y, z).
Action (4, in [-1, 1]): collective thrust command and body-rate commands (CTBR).
A first-order thrust lag and a rate loop with a physical angular-acceleration
limit stand in for motors + flight controller. Mass, max thrust and arm length
are independent parameters, so thrust-to-weight and agility emerge from them.
"""
from __future__ import annotations

import math

import torch

from .config import VehicleRanges

GRAVITY = 9.81


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_exp(rotvec: torch.Tensor) -> torch.Tensor:
    """Rotation vector [rad] -> unit quaternion."""
    theta = rotvec.norm(dim=-1, keepdim=True)
    half = 0.5 * theta
    k = torch.where(theta < 1e-6, 0.5 - theta * theta / 48.0, torch.sin(half) / theta.clamp_min(1e-9))
    return torch.cat((torch.cos(half), k * rotvec), dim=-1)


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (w,x,y,z) -> rotation matrix body->world, shape [..., 3, 3]."""
    w, x, y, z = q.unbind(-1)
    r = torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    )
    return r.reshape(*q.shape[:-1], 3, 3)


def yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(yaw)
    return torch.stack((torch.cos(0.5 * yaw), z, z, torch.sin(0.5 * yaw)), dim=-1)


def _sample(lo_hi: tuple[float, float], n: int, scale: float, gen: torch.Generator, device, log: bool) -> torch.Tensor:
    """Sample around the range centre; scale=1 covers the full range."""
    lo, hi = lo_hi
    u = (torch.rand(n, generator=gen, device=device) * 2 - 1) * scale
    if log:
        mid, half = 0.5 * (math.log(lo) + math.log(hi)), 0.5 * (math.log(hi) - math.log(lo))
        return torch.exp(mid + u * half)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    return mid + u * half


class QuadBatch:
    """N independent quadrotors with per-vehicle randomized parameters."""

    def __init__(self, n: int, device: str, cfg: VehicleRanges, gen: torch.Generator):
        self.n, self.device, self.cfg, self.gen = n, device, cfg, gen
        f = dict(device=device, dtype=torch.float32)
        self.pos = torch.zeros(n, 3, **f)
        self.vel = torch.zeros(n, 3, **f)
        self.quat = torch.zeros(n, 4, **f)
        self.quat[:, 0] = 1.0
        self.omega = torch.zeros(n, 3, **f)
        self.thr = torch.zeros(n, **f)  # mean normalized motor thrust in [0, 1]
        self.motor = torch.zeros(n, 4, **f)  # per-motor normalized thrust state [0, 1] (X frame)
        self.gyro_f = torch.zeros(n, 3, **f)  # low-pass filtered gyro used by the flight-controller rate loop
        self.ext_acc = torch.zeros(n, 3, **f)  # external disturbance acceleration [m/s^2] (wind / gust)
        self.spec_b = torch.zeros(n, 3, **f)  # specific force in body frame [m/s^2] (IMU accelerometer truth)
        self.motor_gain = torch.ones(n, 4, **f)  # per-motor thrust mismatch
        self.kd_axis = torch.ones(n, 3, **f)  # per-body-axis quadratic drag multiplier (anisotropy)
        self.kl = torch.zeros(n, **f)  # linear drag rate [1/s]
        for name in ("mass", "tmax", "arm", "cd", "tau_rate", "tau_thr", "alpha_rp", "kd", "radius", "twr", "cq", "tau_gyro", "inertia_rp"):
            setattr(self, name, torch.ones(n, **f))
        self.sample_params(torch.arange(n, device=device))

    # ---- parameters -------------------------------------------------------------------
    def sample_params(self, ids: torch.Tensor) -> None:
        c, k, g, d = self.cfg, len(ids), self.gen, self.device
        s = c.dr_scale
        mass = _sample(c.mass, k, s, g, d, log=True)
        tmax = _sample(c.max_thrust, k, s, g, d, log=True)
        w = mass * GRAVITY
        tmax = torch.minimum(torch.maximum(tmax, c.twr_range[0] * w), c.twr_range[1] * w)
        arm = _sample(c.arm, k, s, g, d, log=False)
        cd = _sample(c.drag_cd, k, s, g, d, log=False)
        self.mass[ids], self.tmax[ids], self.arm[ids], self.cd[ids] = mass, tmax, arm, cd
        self.twr[ids] = tmax / w
        self.tau_rate[ids] = _sample(c.tau_rate, k, s, g, d, log=False)
        self.tau_thr[ids] = _sample(c.tau_thrust, k, s, g, d, log=False)
        inertia = 0.6 * mass * arm * arm  # roll/pitch inertia, point masses at the arms
        self.inertia_rp[ids] = inertia
        # X frame: each motor sits at lateral lever arm/sqrt(2); max roll/pitch torque ~ 0.5 tmax * arm / sqrt(2)
        self.alpha_rp[ids] = 0.5 * tmax * arm * 0.7071 / inertia  # [rad/s^2] max roll/pitch angular acceleration
        self.cq[ids] = _sample((0.010, 0.022), k, s, g, d, log=False)  # [m] yaw reaction torque / thrust
        self.tau_gyro[ids] = _sample((0.002, 0.006), k, s, g, d, log=False)  # [s] gyro filter of the FC rate loop
        self.motor_gain[ids] = 1.0 + (torch.rand(k, 4, generator=g, device=d) * 2 - 1) * 0.06 * s  # +-6 % thrust mismatch
        axis = torch.exp((torch.rand(k, 3, generator=g, device=d) * 2 - 1) * 0.45 * s)  # x0.64 .. x1.57 per body axis
        axis[:, 2] = axis[:, 2] * (1.0 + 0.35 * s)  # thrust axis: propeller inflow adds drag when climbing/descending
        self.kd_axis[ids] = axis
        self.kl[ids] = torch.rand(k, generator=g, device=d) * 0.25 * s  # linear (parasitic) drag
        area = math.pi * (arm + 0.03) ** 2
        self.kd[ids] = 0.5 * 1.2 * cd * area  # quadratic drag: F = -kd |v| v
        self.radius[ids] = arm + 0.05

    def capability(self, ids: torch.Tensor) -> torch.Tensor:
        """Agility score in [0, 1] from thrust-to-weight and size (used by the track generator)."""
        c = self.cfg
        twr_n = (self.twr[ids] - c.twr_range[0]) / (c.twr_range[1] - c.twr_range[0])
        arm_n = (self.arm[ids] - c.arm[0]) / (c.arm[1] - c.arm[0]) if c.arm[1] - c.arm[0] > 0.03 else torch.full_like(self.arm[ids], 0.5)  # own_drone: arm range is only +-5 mm, not a capability signal
        return (0.5 * twr_n + 0.5 * (1 - arm_n)).clamp(0, 1)

    def param_obs(self) -> torch.Tensor:
        """Normalized vehicle parameters [N, 4]: log-mass, twr, arm, cd."""
        return torch.stack(
            (
                (torch.log(self.mass) - math.log(0.55)) / 0.7,
                (self.twr - 4.0) / 2.0,
                (self.arm - 0.11) / 0.05,
                (self.cd - 0.85) / 0.35,
            ),
            dim=-1,
        )

    # ---- state ------------------------------------------------------------------------
    def reset(self, ids: torch.Tensor, pos: torch.Tensor, yaw: torch.Tensor, vel: torch.Tensor | None = None) -> None:
        self.pos[ids] = pos
        self.vel[ids] = 0.0 if vel is None else vel
        self.quat[ids] = yaw_to_quat(yaw)
        self.omega[ids] = 0.0
        self.thr[ids] = 1.0 / self.twr[ids]  # hover thrust
        self.motor[ids] = (1.0 / self.twr[ids])[:, None]
        self.gyro_f[ids] = 0.0

    # X-quad motor layout (FL, FR, RR, RL): roll sign, pitch sign, yaw sign
    _SX = (1.0, -1.0, -1.0, 1.0)
    _SY = (1.0, 1.0, -1.0, -1.0)
    _SZ = (1.0, -1.0, 1.0, -1.0)

    def actuate(self, action: torch.Tensor, dt: float, omega: torch.Tensor):
        """Flight-controller rate loop -> mixer (airmode, yaw sacrificed first) -> motor lag.

        Returns body-frame (total thrust [N], torque [N m]) and updates the motor states.
        ``omega`` is the true body rate; the controller sees a low-pass filtered copy.
        """
        a = action.clamp(-1.0, 1.0)
        cfg = self.cfg
        dev = self.device
        if not hasattr(self, "_signs"):
            self._signs = tuple(torch.tensor(v, device=dev) for v in (self._SX, self._SY, self._SZ))
            self._rate_scale = torch.tensor((1.0, 1.0, cfg.yaw_rate_scale), device=dev) * cfg.max_rate
        sx, sy, sz = self._signs
        self.gyro_f += (omega - self.gyro_f) * (1.0 - torch.exp(-dt / self.tau_gyro))[:, None]
        fm = self.tmax / 4.0  # per-motor max thrust
        lever = self.arm * 0.7071
        if cfg.direct_motor:  # MonoRace-style: the 4 actions are motor commands (thrust ~ command^2), no inner loop
            u = (0.5 * (a + 1.0)).pow(2)
            return self._lag(u, fm, lever, sx, sy, sz, dt)
        c_cmd = 0.5 * (a[:, 0] + 1.0)
        w_cmd = a[:, 1:4] * self._rate_scale

        # rate loop: P on the filtered gyro, with the tau_rate time constant (desired angular accel)
        alpha_cmd = (w_cmd - self.gyro_f) / self.tau_rate[:, None]
        i_rp = self.inertia_rp
        torque_cmd = alpha_cmd * torch.stack((i_rp, i_rp, 1.6 * i_rp), -1)

        a_x = torque_cmd[:, 0] / (4.0 * lever * fm)
        a_y = torque_cmd[:, 1] / (4.0 * lever * fm)
        a_z = torque_cmd[:, 2] / (4.0 * self.cq * fm)
        d_rp = a_x[:, None] * sx + a_y[:, None] * sy  # roll/pitch part of the motor mix
        z = a_z[:, None] * sz
        rng = lambda x: x.max(-1).values - x.min(-1).values  # noqa: E731
        r_d, r_z = rng(d_rp), rng(z)
        over = rng(d_rp + z) > 1.0
        z_scale = torch.where(over, ((1.0 - r_d) / r_z.clamp_min(1e-6)).clamp(0.0, 1.0), torch.ones_like(r_d))
        mix = d_rp + z * z_scale[:, None]
        mix = mix / rng(mix).clamp_min(1.0)[:, None]  # roll/pitch beyond authority: scale, keep direction
        lo, hi = mix.min(-1).values, mix.max(-1).values
        c_eff = torch.minimum(torch.maximum(c_cmd, -lo), 1.0 - hi)  # airmode: shift collective to avoid clipping
        u = (c_eff[:, None] + mix).clamp(0.0, 1.0)

        return self._lag(u, fm, lever, sx, sy, sz, dt)

    def _lag(self, u, fm, lever, sx, sy, sz, dt):
        """Motor thrust lag (ESC + rotor inertia) -> total body thrust and torque."""
        self.motor += (u * self.motor_gain - self.motor) * (1.0 - torch.exp(-dt / self.tau_thr))[:, None]
        self.thr = self.motor.mean(-1)
        f = self.motor * fm[:, None]
        thrust = f.sum(-1)
        tau = torch.stack(((f * sx).sum(-1) * lever, (f * sy).sum(-1) * lever, (f * sz).sum(-1) * self.cq), -1)
        return thrust, tau

    def drag_acc_world(self, R: torch.Tensor) -> torch.Tensor:
        """Aerodynamic deceleration [m/s^2] in the world frame: anisotropic quadratic + linear body-frame drag."""
        v_b = (R.transpose(1, 2) @ self.vel.unsqueeze(-1)).squeeze(-1)
        speed = self.vel.norm(dim=-1, keepdim=True)
        a_b = -((self.kd_axis * (self.kd / self.mass)[:, None]) * speed + self.kl[:, None]) * v_b
        return (R @ a_b.unsqueeze(-1)).squeeze(-1)

    def inertia_diag(self) -> torch.Tensor:
        i = self.inertia_rp
        return torch.stack((i, i, 1.6 * i), -1)

    def step(self, action: torch.Tensor, dt: float) -> None:
        thrust, tau = self.actuate(action, dt, self.omega)
        inertia = self.inertia_diag()
        gyro = torch.linalg.cross(self.omega, inertia * self.omega)
        self.omega = self.omega + ((tau - gyro) / inertia) * dt
        self.quat = quat_mul(self.quat, quat_exp(self.omega * dt))
        self.quat = self.quat / self.quat.norm(dim=-1, keepdim=True)

        Rm = quat_to_rot(self.quat)
        thrust_acc = (thrust / self.mass)[:, None] * Rm[:, :, 2]
        acc = thrust_acc + self.drag_acc_world(Rm) + self.ext_acc
        acc[:, 2] -= GRAVITY
        acc_g = acc.clone()
        acc_g[:, 2] += GRAVITY
        self.spec_b = (Rm.transpose(1, 2) @ acc_g.unsqueeze(-1)).squeeze(-1)
        self.vel += acc * dt
        self.pos += self.vel * dt
