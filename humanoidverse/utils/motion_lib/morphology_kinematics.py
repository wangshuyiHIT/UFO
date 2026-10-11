"""Batched pose and velocity FK for morphology-conditioned motion sampling.

Inputs use policy joint order and xyzw quaternions. Velocities are world-frame
body-origin velocities, not COM velocities. This path is intentionally opt-in;
legacy robots retain the original independently interpolated motion features.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch import Tensor


@torch.jit.script
def _mul(a: Tensor, b: Tensor) -> Tensor:
    av, bv = a[..., :3], b[..., :3]
    return torch.cat((a[..., 3:] * bv + b[..., 3:] * av + torch.cross(av, bv, dim=-1),
                      a[..., 3:] * b[..., 3:] - (av * bv).sum(-1, keepdim=True)), -1)


@torch.jit.script
def _rotate(q: Tensor, v: Tensor) -> Tensor:
    t = 2 * torch.cross(q[..., :3], v, dim=-1)
    return v + q[..., 3:] * t + torch.cross(q[..., :3], t, dim=-1)


@torch.jit.script
def interpolate_root(q0: Tensor, q1: Tensor, blend: Tensor, inv_dt: Tensor):
    """The existing shortest-arc slerp/nlerp, plus its exact world angular derivative."""
    q0 = torch.nn.functional.normalize(q0, dim=-1)
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    near = dot > .9995
    theta = torch.acos(dot.clamp(max=1.0 - 1e-7))
    sin_theta = torch.sin(theta)
    spherical = (torch.sin((1-blend)*theta)*q0 + torch.sin(blend*theta)*q1)/sin_theta
    spherical_deriv = theta*(-torch.cos((1-blend)*theta)*q0 + torch.cos(blend*theta)*q1)/sin_theta
    raw = torch.where(near, (1-blend)*q0 + blend*q1, spherical)
    deriv = torch.where(near, q1-q0, spherical_deriv)
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    quat = raw / norm
    quat_dot = (deriv - quat*(quat*deriv).sum(-1, keepdim=True)) / norm * inv_dt
    omega = 2*(quat[..., 3:]*quat_dot[..., :3] - quat_dot[..., 3:]*quat[..., :3]
               + torch.cross(quat[..., :3], quat_dot[..., :3], dim=-1))
    return quat, omega


@torch.jit.script
def _tree_fk(root_pos: Tensor, root_rot: Tensor, root_vel: Tensor, root_ang_vel: Tensor,
             dof_pos: Tensor, dof_vel: Tensor, delta: Tensor, offsets: Tensor,
             morphology_basis: Tensor, static_rot: Tensor, axes: Tensor, anchors: Tensor,
             references: Tensor, joint_indices: Tensor, parents: Tensor, levels: List[Tensor]):
    batch = root_pos.shape[0]
    count = offsets.shape[0]
    q = torch.cat((dof_pos, torch.zeros_like(dof_pos[:, :1])), -1)[:, joint_indices] - references
    dq = torch.cat((dof_vel, torch.zeros_like(dof_vel[:, :1])), -1)[:, joint_indices]
    half = q.unsqueeze(-1) * .5
    joint_rot = torch.cat((axes*torch.sin(half), torch.cos(half)), -1)
    local_rot = _mul(static_rot.expand(batch, -1, -1), joint_rot)
    body_offsets = offsets + torch.einsum('bf,fjc->bjc', delta, morphology_basis)
    # Rotation about a nonzero hinge anchor also translates the body origin.
    local_pos = body_offsets + _rotate(static_rot.expand(batch, -1, -1),
                                       anchors - _rotate(joint_rot, anchors.expand(batch, -1, -1)))
    positions = torch.zeros((batch, count, 3), dtype=root_pos.dtype, device=root_pos.device)
    rotations = torch.zeros((batch, count, 4), dtype=root_pos.dtype, device=root_pos.device)
    velocities = torch.zeros_like(positions)
    angular = torch.zeros_like(positions)
    positions[:, 0] = root_pos
    rotations[:, 0] = root_rot
    velocities[:, 0] = root_vel
    angular[:, 0] = root_ang_vel
    for children in levels:
        parent = parents[children]
        parent_rot = rotations[:, parent]
        displacement = _rotate(parent_rot, local_pos[:, children])
        rotation = _mul(parent_rot, local_rot[:, children])
        rotation = torch.nn.functional.normalize(rotation, dim=-1)
        joint_omega = _rotate(rotation, axes[children].expand(batch, -1, -1)) * dq[:, children, None]
        positions[:, children] = positions[:, parent] + displacement
        rotations[:, children] = rotation
        angular[:, children] = angular[:, parent] + joint_omega
        velocities[:, children] = (velocities[:, parent]
            + torch.cross(angular[:, parent], displacement, dim=-1)
            - torch.cross(joint_omega, _rotate(rotation, anchors[children].expand(batch, -1, -1)), dim=-1))
    return positions, rotations, velocities, angular


class MorphologyKinematics:
    """Compile immutable kinematic tensors once, then batch queries by tree depth."""

    def __init__(self, xml_path, body_names, joint_names, morphology, extend_config, device):
        # Legacy motion libraries do not acquire a new MuJoCo dependency.
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.body_names = list(body_names)
        self.num_bodies = len(body_names)
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names]
        if any(i <= 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Morphology FK body names must uniquely identify physical robot bodies')
        all_hinges = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        if joint_names is None:
            # A single-joint fixture has no order ambiguity. Real robots must
            # declare their raw dof order; actuator/XML order is not a substitute.
            if len(all_hinges) != 1:
                raise ValueError('Morphology motions require explicit joint_names in policy dof order')
            joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, all_hinges[0])]
        self.joint_names = list(joint_names)
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
        if len(set(joint_ids)) != len(joint_ids) or set(joint_ids) != set(all_hinges):
            raise ValueError('Morphology joint_names must name every controlled hinge exactly once')
        parents, offsets, rotations, axes, anchors, references, indices = [], [], [], [], [], [], []
        for k, body in enumerate(ids):
            js = list(range(int(model.body_jntadr[body]), int(model.body_jntadr[body]+model.body_jntnum[body])))
            if k == 0:
                if model.body_parentid[body] != 0 or len(js) != 1 or model.jnt_type[js[0]] != mujoco.mjtJoint.mjJNT_FREE:
                    raise ValueError('Morphology FK requires one free root as first body')
                parents.append(-1)
                js = []
            else:
                if model.body_parentid[body] not in ids[:k] or len(js) > 1:
                    raise ValueError('Morphology FK requires a topological tree with at most one hinge per body')
                parents.append(ids.index(model.body_parentid[body]))
            offsets.append(model.body_pos[body].copy())
            rotations.append(model.body_quat[body][[1,2,3,0]].copy())
            if js:
                j = js[0]
                if j not in joint_ids:
                    raise ValueError('Morphology FK supports only fixed bodies and declared hinges')
                indices.append(joint_ids.index(j)); axes.append(model.jnt_axis[j].copy())
                anchors.append(model.jnt_pos[j].copy()); references.append(model.qpos0[model.jnt_qposadr[j]])
            else:
                indices.append(len(joint_names)); axes.append(np.array([1.,0,0])); anchors.append(np.zeros(3)); references.append(0.)
        for extension in extend_config:
            parents.append(self.body_names.index(extension['parent_name']))
            self.body_names.append(extension['joint_name'])
            offsets.append(extension['pos']); rotations.append(np.asarray(extension['rot'])[[1,2,3,0]])
            indices.append(len(joint_names)); axes.append(np.array([1.,0,0])); anchors.append(np.zeros(3)); references.append(0.)
        basis = np.zeros((len(morphology['family_names']), len(parents), 3))
        for entry in morphology['body_offsets']:
            body = self.body_names.index(entry['body_name'])
            if body == 0:
                raise ValueError('Morphology cannot displace the free root attachment')
            basis[entry['family_index'], body] += entry['axis']
        depths = [0]
        for parent in parents[1:]:
            depths.append(depths[parent]+1)
        tensor = lambda x, dtype=torch.float32: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
        self.offsets, self.morphology_basis = tensor(offsets), tensor(basis)
        self.static_rot, self.axes, self.anchors = tensor(rotations), tensor(axes), tensor(anchors)
        self.references = tensor(references)
        self.joint_indices, self.parents = tensor(indices, torch.long), tensor(parents, torch.long)
        self.levels = [tensor(np.where(np.asarray(depths)==depth)[0], torch.long) for depth in range(1,max(depths)+1)]

    def forward(self, root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, delta):
        if dof_pos.shape[-1] != len(self.joint_names):
            raise ValueError('Morphology dof count does not match declared joint_names')
        return _tree_fk(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, delta,
            self.offsets, self.morphology_basis, self.static_rot, self.axes, self.anchors,
            self.references, self.joint_indices, self.parents, self.levels)
