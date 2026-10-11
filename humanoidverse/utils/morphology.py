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
    if cfg.get("motion_purpose", "training") not in ("training", "evaluation"):
        raise ValueError("morphology.motion_purpose must be training or evaluation")
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
    reference = metadata.get("morphology_reference")
    if reference == "synthetic_static":
        # This declares how the pose was constructed, not training eligibility.
        # FK is also used while qualifying a candidate; the formal launcher
        # separately requires completed geometry/FK/temporal train qualification.
        if (metadata.get("source_kind") != "synthetic_constant" or
                metadata.get("reference_kind") != "synthetic_static" or metadata.get("synthetic") is not True):
            raise ValueError("Synthetic morphology requires explicit constant/static/synthetic provenance")
    elif reference != "retargeted":
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


MORPHOLOGY_MODEL_FIELDS = (
    "body_pos", "body_mass", "body_ipos", "body_inertia", "body_iquat",
    "geom_pos", "geom_size", "geom_aabb", "geom_rbound",
)


def morphology_model_parameters(model, cfg: dict, delta: np.ndarray, *, prefix: str = "") -> dict[str, np.ndarray]:
    """Compute physical parameters from a nominal compiled MuJoCo model.

    Pure CPU function used both by the online parameter bank and physics tests.
    Includes broadphase bounds; changing just geom_size leaves stale collisions.
    """
    import mujoco

    delta = motion_morphology_delta({"metadata": {
        "morphology_delta_m": delta, "morphology_reference": "retargeted"}}, cfg)
    result = {name: np.array(getattr(model, name), copy=True) for name in MORPHOLOGY_MODEL_FIELDS}
    # C MuJoCo exposes [ngeom, 6]; Warp exposes [world, ngeom, 2, 3].
    result["geom_aabb"] = result["geom_aabb"].reshape(model.ngeom, 2, 3)

    def index(kind, name):
        idx = mujoco.mj_name2id(model, kind, prefix + name)
        if idx < 0:
            raise ValueError(f"Nominal morphology model is missing {prefix + name!r}")
        return idx

    for entry in cfg["body_offsets"]:
        bid = index(mujoco.mjtObj.mjOBJ_BODY, entry["body_name"])
        result["body_pos"][bid] += float(delta[entry["family_index"]]) * np.asarray(entry["axis"])
    for profile in cfg["profiles"]:
        bid = index(mujoco.mjtObj.mjOBJ_BODY, profile["body_name"])
        gid = index(mujoco.mjtObj.mjOBJ_GEOM, profile["geom_name"])
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX or model.geom_bodyid[gid] != bid:
            raise ValueError("Morphology profiles must be physical boxes attached to the declared body")
        prop = profile_properties(profile, float(delta[profile["family_index"]]))
        for field, key in (("body_mass", "mass"), ("body_ipos", "com"),
                           ("body_inertia", "inertia"), ("body_iquat", "iquat")):
            result[field][bid] = prop[key]
        result["geom_size"][gid] = prop["size"]
        result["geom_pos"][gid] = prop["center"]
        result["geom_aabb"][gid, 0] = 0.
        result["geom_aabb"][gid, 1] = prop["size"]
        result["geom_rbound"][gid] = np.linalg.norm(prop["size"])
    return result


class MjlabMorphologyRuntime:
    """Apply only shapes actually present in the retargeted motion library.

    The bank stores continuous dimension values, not categorical tokens. Reset
    chooses a motion and applies its exact shape. Density/COM randomization is
    sampled on that physical baseline and never accumulated across resets.
    """

    def __init__(self, sim, cfg: dict, motion_deltas: torch.Tensor, *, robot_body_ids,
                 torso_body_id: int, domain_rand, prefix: str = "robot/"):
        from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

        self.sim, self.cfg, self.domain_rand = sim, cfg, domain_rand
        self.bank_deltas = torch.unique(motion_deltas, dim=0)
        self.robot_body_ids = torch.as_tensor(robot_body_ids, device=sim.device, dtype=torch.long)
        self.torso_body_id = int(torso_body_id)
        self.recompute_level = RecomputeLevel.set_const
        nominal = morphology_model_parameters(sim.mj_model, cfg, np.zeros(len(cfg["family_names"])), prefix=prefix)
        for name in ("body_mass", "body_ipos", "geom_pos", "geom_size"):
            if not np.allclose(nominal[name], getattr(sim.mj_model, name), atol=2e-6, rtol=1e-5):
                raise ValueError(f"Morphology requires the nominal carrier model; {name} differs from profile metadata")
        # Use MJLab's public declaration helper to include all derived arrays.
        declaration = requires_model_fields(*MORPHOLOGY_MODEL_FIELDS, recompute=self.recompute_level)(lambda: None)
        sim.expand_model_fields(tuple(declaration.model_fields))
        values = [morphology_model_parameters(sim.mj_model, cfg, d, prefix=prefix)
                  for d in self.bank_deltas.detach().cpu().numpy()]
        self.parameters = {
            field: torch.as_tensor(np.stack([v[field] for v in values]),
                                   dtype=getattr(sim.model, field).dtype, device=sim.device)
            for field in MORPHOLOGY_MODEL_FIELDS
        }
        self.delta_m = torch.zeros((sim.num_envs, len(cfg["family_names"])), device=sim.device)
        self.bank_indices = torch.full((sim.num_envs,), -1, dtype=torch.long, device=sim.device)

    def apply(self, env_ids: torch.Tensor, delta: torch.Tensor, *, randomize: bool = True) -> None:
        delta = torch.as_tensor(delta, dtype=self.bank_deltas.dtype, device=self.bank_deltas.device)
        if delta.shape != (len(env_ids), self.bank_deltas.shape[1]) or not bool(torch.isfinite(delta).all()):
            raise ValueError("Reset requires one finite morphology vector per selected environment")
        matches = torch.isclose(delta[:, None], self.bank_deltas[None], atol=1e-7, rtol=0).all(-1)
        if not bool((matches.sum(-1) == 1).all()):
            raise ValueError("Reset morphology has no unique retargeted motion-bank shape; refusing an unmatched token")
        indices = matches.to(torch.int32).argmax(-1)
        for field, bank in self.parameters.items():
            getattr(self.sim.model, field)[env_ids] = bank[indices]
        dr = self.domain_rand
        if randomize and bool(dr.get("randomize_link_mass", False)):
            low, high = map(float, dr.link_mass_range)
            if not 0 < low <= high:
                raise ValueError("link_mass_range must be positive ordered density multipliers")
            scale = torch.empty((len(env_ids), len(self.robot_body_ids)), device=delta.device).uniform_(low, high)
            world, body = env_ids[:, None], self.robot_body_ids[None]
            self.sim.model.body_mass[world, body] *= scale
            self.sim.model.body_inertia[world, body] *= scale[..., None]
        if randomize and bool(dr.get("randomize_base_com", False)):
            ranges = dr.base_com_range
            offset = torch.stack([torch.empty(len(env_ids), device=delta.device).uniform_(*map(float, ranges[axis]))
                                  for axis in ("x", "y", "z")], dim=-1)
            self.sim.model.body_ipos[env_ids, self.torso_body_id] += offset
        self.delta_m[env_ids] = self.bank_deltas[indices]
        self.bank_indices[env_ids] = indices
        # Recomputes mass/kinematics constants; array addresses stay unchanged.
        self.sim.recompute_constants(self.recompute_level)



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
