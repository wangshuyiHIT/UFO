"""Explicit PD randomization and physics-substep command delay for MJLab.

Used only by opt-in morphology environments. Delay is quantized to physics
steps; feed-forward torque noise passes through DcMotorActuator's real limits.
"""
from dataclasses import dataclass, field
import math

import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg


class PhysicsStepDelay:
    def __init__(self, num_envs, num_joints, max_delay, device):
        self.values = torch.zeros(max_delay + 1, num_envs, num_joints, device=device)
        self.delay = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.envs = torch.arange(num_envs, device=device)
        self.cursor = 0

    def reset(self, env_ids, holding_target, delay):
        self.values[:, env_ids] = holding_target[None]
        self.delay[env_ids] = delay

    def push(self, target):
        self.values[self.cursor] = target
        selected = (self.cursor - self.delay) % len(self.values)
        result = self.values[selected, self.envs].clone()
        self.cursor = (self.cursor + 1) % len(self.values)
        return result


@dataclass(kw_only=True)
class MorphologyPositionActionCfg(JointPositionActionCfg):
    domain_rand: dict = field(default_factory=dict)
    effort_limits: tuple[float, ...] = ()

    def build(self, env):
        return MorphologyPositionAction(self, env)


class MorphologyPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.dr = cfg.domain_rand
        self.randomization_enabled = True
        self.delay_bounds = (0, 0)
        if self.dr.get("randomize_ctrl_delay", False):
            if "ctrl_delay_ms_range" not in self.dr:
                raise ValueError("Morphology control delay requires ctrl_delay_ms_range; legacy policy-step delay is ambiguous")
            lo, hi = map(float, self.dr["ctrl_delay_ms_range"])
            if not 0 <= lo <= hi or not math.isfinite(hi):
                raise ValueError("ctrl_delay_ms_range must be finite ordered nonnegative milliseconds")
            tick_ms = 1000. * env.physics_dt
            self.delay_bounds = (math.ceil(lo / tick_ms - 1e-9), math.floor(hi / tick_ms + 1e-9))
            if self.delay_bounds[0] > self.delay_bounds[1]:
                raise ValueError("No physics-step delay lies inside ctrl_delay_ms_range")
        self.delay_line = PhysicsStepDelay(env.num_envs, self.action_dim, self.delay_bounds[1], env.device)
        if len(cfg.effort_limits) != self.action_dim:
            raise ValueError("Action-order effort limits are required for torque randomization")
        self.effort_limits = torch.tensor(cfg.effort_limits, device=env.device)
        self.rfi_scale = torch.zeros(env.num_envs, 1, device=env.device)
        self.last_applied_target = torch.zeros_like(self._raw_actions)
        self.last_rfi_torque = torch.zeros_like(self._raw_actions)

    def reset_dynamics(self, env_ids, holding_target, *, randomize=True):
        self.randomization_enabled = bool(randomize)
        dr = self.dr
        for actuator in self._entity.actuators:
            kp, kd = actuator.default_stiffness[env_ids].clone(), actuator.default_damping[env_ids].clone()
            if randomize and dr.get("randomize_pd_gain", False):
                for value, key in ((kp, "kp_range"), (kd, "kd_range")):
                    low, high = map(float, dr[key])
                    if not 0 < low <= high:
                        raise ValueError(f"{key} must be positive ordered multipliers")
                    value *= torch.empty_like(value).uniform_(low, high)
            actuator.set_gains(env_ids, kp=kp, kd=kd)
        lo, hi = self.delay_bounds if randomize else (0, 0)
        delay = torch.randint(lo, hi + 1, (len(env_ids),), device=holding_target.device)
        self.delay_line.reset(env_ids, holding_target, delay)
        rfi = float(dr.get("rfi_lim", 0.)) if randomize and dr.get("randomize_torque_rfi", False) else 0.
        if not math.isfinite(rfi) or rfi < 0:
            raise ValueError("rfi_lim must be a finite nonnegative effort fraction")
        self.rfi_scale[env_ids] = rfi
        if rfi and dr.get("randomize_rfi_lim", False):
            low, high = map(float, dr["rfi_lim_range"])
            if not 0 <= low <= high:
                raise ValueError("rfi_lim_range must be ordered nonnegative multipliers")
            self.rfi_scale[env_ids] *= torch.empty((len(env_ids), 1), device=holding_target.device).uniform_(low, high)
        self.last_rfi_torque[env_ids] = 0.
        self._entity.set_joint_effort_target(torch.zeros_like(holding_target), joint_ids=self._target_ids, env_ids=env_ids[:, None])

    def apply_actions(self):
        target = self._processed_actions - self._entity.data.encoder_bias[:, self._target_ids]
        self.last_applied_target[:] = self.delay_line.push(target)
        self._entity.set_joint_position_target(self.last_applied_target, joint_ids=self._target_ids)
        self.last_rfi_torque[:] = (torch.rand_like(target) * 2. - 1.) * self.rfi_scale * self.effort_limits
        self._entity.set_joint_effort_target(self.last_rfi_torque, joint_ids=self._target_ids)
