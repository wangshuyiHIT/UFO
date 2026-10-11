"""Morphology contract shared by motion FK, online observations and physics.

Morphology is a property of a retargeted motion, not independent observation
noise. A reset selects the motion first and applies exactly its morphology.
The implementation keeps the controlled hinge topology unchanged.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch


def validate_morphology_config(raw: dict | None) -> dict | None:
    if not raw or not raw.get("enabled", False):
        return None
    cfg = copy.deepcopy(dict(raw))
    names = list(cfg.get("family_names", []))
    if not names or len(set(names)) != len(names):
        raise ValueError("morphology.family_names must contain unique names")
    bounds = np.asarray(cfg.get("family_bounds_m"), dtype=float)
    if bounds.shape != (len(names), 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("morphology.family_bounds_m must be finite [families, 2] bounds")
    scale = float(cfg.get("observation_scale_m", 0.05))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("morphology.observation_scale_m must be finite and positive")
    cfg["observation_scale_m"] = scale
    for entry in cfg.get("body_offsets", []):
        index = int(entry["family_index"])
        axis = np.asarray(entry["axis"], dtype=float)
        if not 0 <= index < len(names) or axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1):
            raise ValueError("morphology.body_offsets needs a valid family index and unit axis")
    if not cfg.get("body_offsets") or not cfg.get("profiles"):
        raise ValueError("morphology requires physical body_offsets and profile geometry/inertia metadata")
    return cfg


def motion_morphology_delta(motion: dict, cfg: dict) -> np.ndarray:
    metadata = motion.get("metadata", {})
    delta = np.asarray(metadata.get("morphology_delta_m"), dtype=np.float32)
    bounds = np.asarray(cfg["family_bounds_m"], dtype=np.float32)
    if delta.shape != (len(cfg["family_names"]),) or not np.isfinite(delta).all():
        raise ValueError("Every morphology motion requires finite metadata.morphology_delta_m")
    if np.any(delta < bounds[:, 0] - 1e-7) or np.any(delta > bounds[:, 1] + 1e-7):
        raise ValueError(f"Motion morphology exceeds physical family bounds: {delta.tolist()}")
    if metadata.get("morphology_reference") != "retargeted":
        raise ValueError("Morphology motion requires metadata.morphology_reference=retargeted; a token alone is not retargeting")
    return delta


def morphology_offsets(offsets: torch.Tensor, body_names: list[str], cfg: dict, delta: np.ndarray) -> torch.Tensor:
    result = offsets.clone()
    for entry in cfg["body_offsets"]:
        body = body_names.index(entry["body_name"])
        result[..., body, :] += torch.as_tensor(entry["axis"], dtype=result.dtype, device=result.device) * float(delta[int(entry["family_index"])])
    return result


def append_morphology_state(env: Any, state: torch.Tensor, delta: torch.Tensor | None = None) -> torch.Tensor:
    cfg = getattr(env, "morphology_config", None)
    if not cfg:
        return state
    if delta is None:
        delta = env.morphology_delta_m
    delta = torch.as_tensor(delta, device=state.device, dtype=state.dtype)
    if delta.ndim == 1:
        delta = delta.expand(state.shape[0], -1)
    if delta.shape != (state.shape[0], len(cfg["family_names"])):
        raise ValueError(f"Morphology/state batch mismatch: {delta.shape} vs {state.shape}")
    return torch.cat((state, delta / float(cfg["observation_scale_m"])), dim=-1)


def profile_properties(profile: dict, delta: float) -> dict[str, np.ndarray | float]:
    """Recompute carrier + variable box inertia in the carrier body frame."""
    from scipy.spatial.transform import Rotation

    length = float(profile["nominal_length_m"]) + delta
    if length <= 0 or not np.isfinite(length):
        raise ValueError("Morphology creates a nonpositive profile length")
    sx, sy = np.asarray(profile["section_xy_m"], dtype=float)
    axis = np.asarray(profile["axis"], dtype=float)
    center = np.asarray(profile["nominal_center_m"], dtype=float) - 0.5 * delta * axis
    box_mass = float(profile["linear_density_kg_m"]) * length
    carrier_mass = float(profile["carrier_mass_kg"])
    mass = box_mass + carrier_mass
    carrier_com = np.asarray(profile["carrier_com_m"], dtype=float)
    com = (box_mass * center + carrier_mass * carrier_com) / mass
    rotation = np.asarray(profile["rotation_matrix"], dtype=float)
    box_i = rotation @ np.diag(box_mass / 12 * np.array([sy**2 + length**2, sx**2 + length**2, sx**2 + sy**2])) @ rotation.T

    def pa(v):
        return np.dot(v, v) * np.eye(3) - np.outer(v, v)

    inertia = np.asarray(profile["carrier_inertia_kg_m2"], dtype=float) + box_i + carrier_mass * pa(carrier_com - com) + box_mass * pa(center - com)
    eig, axes = np.linalg.eigh(inertia)
    if np.min(eig) <= 0 or not np.isfinite(eig).all():
        raise ValueError("Morphology creates invalid inertia")
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    quat = Rotation.from_matrix(axes).as_quat()[[3, 0, 1, 2]]
    return {"mass": mass, "com": com, "inertia": eig, "iquat": quat, "size": np.array([sx, sy, length]) * 0.5, "center": center}



def fk_motion_with_morphology(parser, motion: dict, cfg: dict):
    """CPU-compatible FK preview for one already-retargeted motion.

    Restore the parser even on failure so a subsequent nominal preview never
    inherits a previous shape. This does not initialize online training physics.
    """
    cfg = validate_morphology_config(cfg)
    if cfg is None:
        raise ValueError("Morphology FK requires an enabled configuration")
    delta = motion_morphology_delta(motion, cfg)
    nominal_offsets = parser._offsets
    try:
        parser._offsets = morphology_offsets(nominal_offsets, list(parser.body_names), cfg, delta)
        pose = torch.as_tensor(motion["pose_aa"], dtype=torch.float32)
        trans = torch.as_tensor(motion["root_trans_offset"], dtype=torch.float32)
        result = parser.fk_batch(pose[None], trans[None], return_full=True, dt=1.0 / float(motion["fps"]))
    finally:
        parser._offsets = nominal_offsets
    # Humanoid_Batch's legacy fallback sums axis-angle coordinates in body
    # order. A new robot may have negative axes and a different policy order.
    # Preserve the explicit RobotState control-joint order, as MotionLib does.
    if "dof_pos" in motion:
        dof = torch.as_tensor(motion["dof_pos"], dtype=torch.float32)[None]
        velocity = torch.diff(dof, dim=1) * float(motion["fps"])
        result.dof_pos = dof
        result.dof_vels = torch.cat((velocity, velocity[:, -1:]), dim=1)
    return result
