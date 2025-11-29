"""
caseА/caseA.py

Case A: hand-crafted guidance protocols for simple 2D corridors
(V-neck and S-tunnel) using the GuidedPWCSchedule API.

This module provides small, well-documented helpers to:
  - build piecewise-constant guidance schedules Γ = {β_k, ν_k},
  - simulate guided paths while recording snapshots,
  - generate simple corridor boundaries for plotting.
"""

# CaseA/CaseA.py

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from guided_torch import GuidedPWCSchedule, ustar_guided

Tensor = torch.Tensor
Device = torch.device

# ---------------------------------------------------------------------
# Geometry helper: centerline definitions
# ---------------------------------------------------------------------

@dataclass
class CenterlineConfig:
    """
    Configuration of a 2D centerline for Case A.

    Attributes
    ----------
    start : np.ndarray
        2D coordinates of the entry point (usually near the origin).
    end : np.ndarray
        2D coordinates of the nominal target / corridor exit.
    kind : str
        'vneck'   -> straight corridor (narrowing encoded via β schedule).
        'stunnel' -> S-shaped corridor (curved centerline).
    amplitude : float
        Amplitude of the lateral deviation for the S-tunnel.
    """
    start: np.ndarray
    end: np.ndarray
    kind: str = "vneck"
    amplitude: float = 0.4


def make_centerline_fn(cfg: CenterlineConfig) -> Callable[[float], np.ndarray]:
    """
    Return a callable c(t) : [0,1] -> R^2 representing the corridor centerline.

    For 'vneck' we use a straight line from start to end.
    For 'stunnel' we superimpose a smooth lateral S-shaped deviation.
    """
    start = np.asarray(cfg.start, dtype=float).reshape(2)
    end   = np.asarray(cfg.end,   dtype=float).reshape(2)
    delta = end - start
    L = np.linalg.norm(delta)
    if L == 0.0:
        # degenerate; fall back to constant centerline
        def c_const(t: float) -> np.ndarray:
            return start.copy()
        return c_const

    # unit tangent vector
    tang = delta / L
    # unit normal (rotate by +90 degrees)
    normal = np.array([-tang[1], tang[0]], dtype=float)

    if cfg.kind.lower() == "stunnel":
        A = float(cfg.amplitude)

        def c_s(t: float) -> np.ndarray:
            """
            S-tunnel centerline:
              - follows the straight line from start to end,
              - adds a lateral S-shaped bump along the normal direction.
            """
            t = float(t)
            base = (1.0 - t) * start + t * end
            # smooth S-shaped perturbation: zero at t=0 and t=1, max around t=0.5
            s = math.sin(math.pi * t)
            return base + A * s * normal

        return c_s

    # default: straight centerline (V-neck geometry is encoded via β, not ν)
    def c_lin(t: float) -> np.ndarray:
        t = float(t)
        return (1.0 - t) * start + t * end

    return c_lin


def sample_centerline_on_splits(
    centerline_fn: Callable[[float], np.ndarray],
    splits: np.ndarray,
) -> np.ndarray:
    """
    Approximate a continuous centerline c(t) by piecewise-constant values ν_k.

    For each interval [t_k, t_{k+1}), we take ν_k = c((t_k + t_{k+1})/2).
    """
    splits = np.asarray(splits, dtype=float)
    S = len(splits) - 1
    nu_vals = np.zeros((S, 2), dtype=float)
    for k in range(S):
        t_mid = 0.5 * (splits[k] + splits[k+1])
        nu_vals[k, :] = centerline_fn(t_mid)
    return nu_vals


# ---------------------------------------------------------------------
# Schedule builder for Case A
# ---------------------------------------------------------------------

@dataclass
class CaseAScheduleConfig:
    """
    Configuration for building a GuidedPWCSchedule for Case A.

    Attributes
    ----------
    splits : np.ndarray
        Monotone array of S+1 time splits in [0,1].
    betas : np.ndarray
        Length-S array of stiffness values β_k on each interval.
    centerline : CenterlineConfig
        Geometric description of the corridor centerline.
    """
    splits: np.ndarray
    betas: np.ndarray
    centerline: CenterlineConfig


def build_caseA_schedule(
    cfg: CaseAScheduleConfig,
    device: Device = torch.device("cpu"),
) -> GuidedPWCSchedule:
    """
    Build a GuidedPWCSchedule for Case A from (β_k, splits) and a 2D centerline.

    This is a thin wrapper around GuidedPWCSchedule.build that:
      - samples ν_k from the continuous centerline,
      - constructs the schedule with d=2.
    """
    splits = np.asarray(cfg.splits, dtype=float)
    betas  = np.asarray(cfg.betas,  dtype=float)
    assert splits.ndim == 1 and betas.ndim == 1
    assert len(splits) == len(betas) + 1, "splits must have length S+1 if betas has length S"

    c_fn = make_centerline_fn(cfg.centerline)
    nu_vals = sample_centerline_on_splits(c_fn, splits)

    # GuidedPWCSchedule expects numpy arrays for betas/splits and a (S,d) array for ν_k.
    sched = GuidedPWCSchedule.build(
        betas=betas,
        splits=splits,
        nu_values=nu_vals,
        d=2,
        device=device,
    )
    return sched


# ---------------------------------------------------------------------
# Simulation with snapshots
# ---------------------------------------------------------------------

@dataclass
class SnapshotConfig:
    """
    Configuration for recording trajectory snapshots during simulation.

    Attributes
    ----------
    T : int
        Total number of Euler--Maruyama steps.
    n_save : int
        Number of snapshot times (including final); we use uniform spacing.
    """
    T: int = 1200
    n_save: int = 6


def simulate_caseA_with_snapshots(
    sched: GuidedPWCSchedule,
    gmm,
    M: int = 4000,
    snap_cfg: SnapshotConfig = SnapshotConfig(),
    seed: int = 0,
    device: Device = torch.device("cpu"),
) -> List[Dict[str, np.ndarray]]:
    """
    Simulate GH-PID paths for Case A and record a small number of snapshots.

    Parameters
    ----------
    sched : GuidedPWCSchedule
        Guidance schedule Γ for Case A (V-neck or S-tunnel).
    gmm : GMMTorch-like object
        Target GMM (only used to get dimension and device via .d and .to()).
    M : int
        Number of independent trajectories.
    snap_cfg : SnapshotConfig
        Configuration of number of steps and snapshot times.
    seed : int
        Random seed.
    device : torch.device
        Device for simulation.

    Returns
    -------
    snapshots : list of dict
        Each dict has keys:
          - 't'  : float time in [0,1],
          - 'X'  : (M,2) numpy array of particle positions,
          - 'nu' : (2,) numpy array with ν_t (centerline point at time t).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gmm = gmm.to(device)
    d = int(gmm.d)

    X = torch.zeros((M, d), device=device)
    T = int(snap_cfg.T)
    dt = 1.0 / T
    sdt = math.sqrt(dt)

    # times at which to store snapshots (rounded to nearest step)
    save_times = np.linspace(0.0, 1.0, snap_cfg.n_save)
    save_steps = np.clip(
        np.round(save_times * T).astype(int),
        0,
        T,
    )
    # ensure uniqueness and sorted
    save_steps = np.unique(save_steps)

    snapshots: List[Dict[str, np.ndarray]] = []

    for n in range(T):
        t_mid = (n + 0.5) / T
        u = ustar_guided(X, t_mid, sched, gmm)
        if not torch.isfinite(u).all():
            raise RuntimeError(f"Non-finite control at step {n}, t={t_mid:.6f}")
        X = X + u * dt + torch.randn_like(X) * sdt
        if not torch.isfinite(X).all():
            raise RuntimeError(f"Non-finite state at step {n}, t={t_mid:.6f}")

        # record snapshot after step n, corresponding to time t = (n+1)/T
        step_idx = n + 1
        if step_idx in save_steps:
            t_snap = step_idx / T
            # centerline point at this time
            nu_t = sched.nu_sched.value(float(t_snap)).detach().cpu().numpy()
            snapshots.append(
                {
                    "t": float(t_snap),
                    "X": X.detach().cpu().numpy(),
                    "nu": nu_t,
                }
            )

    return snapshots


# ---------------------------------------------------------------------
# Corridor walls for plotting (optional)
# ---------------------------------------------------------------------

def build_corridor_walls(
    centerline_pts: np.ndarray,
    width_profile: Callable[[float], float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given discrete centerline points c_k and a width profile w(t),
    construct left/right wall polylines for visualization.

    Parameters
    ----------
    centerline_pts : np.ndarray
        Array of shape (S,2) with centerline samples at times t_k.
    width_profile : callable
        w(t) giving half-width of corridor at time t in [0,1].

    Returns
    -------
    left, right : np.ndarray
        Arrays of shape (S,2) with left/right wall coordinates.
    """
    centerline_pts = np.asarray(centerline_pts, dtype=float)
    S = centerline_pts.shape[0]
    left = np.zeros_like(centerline_pts)
    right = np.zeros_like(centerline_pts)

    # approximate tangent by finite differences, then normal
    for k in range(S):
        if k == 0:
            tang = centerline_pts[1] - centerline_pts[0]
            t_val = 0.0
        elif k == S - 1:
            tang = centerline_pts[S - 1] - centerline_pts[S - 2]
            t_val = 1.0
        else:
            tang = centerline_pts[k + 1] - centerline_pts[k - 1]
            t_val = k / (S - 1)

        L = np.linalg.norm(tang)
        if L == 0.0:
            normal = np.array([0.0, 1.0])
        else:
            tang = tang / L
            normal = np.array([-tang[1], tang[0]])

        w = float(width_profile(t_val))
        left[k, :] = centerline_pts[k] + w * normal
        right[k, :] = centerline_pts[k] - w * normal

    return left, right
