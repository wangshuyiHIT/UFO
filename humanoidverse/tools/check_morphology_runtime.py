"""Small local GPU check of morphology physics, reference consistency and DR.

Does not train a model or approve a dataset. Review motions require the explicit
--allow-review-motions switch. Each saved report records the source file hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from humanoidverse.train import build_ufo_mjlab_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--allow-review-motions", action="store_true")
    parser.add_argument("--exercise-control-dr", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 16 or not 1 <= args.steps <= 1000:
        raise ValueError("This is a bounded physics check: 1–16 environments, 1–1000 policy steps")
    cfg = build_ufo_mjlab_config(
        device=args.device, work_dir=str(args.output.parent), num_envs=args.num_envs,
        num_env_steps=args.steps, seed=31, use_wandb=False, wandb_run_name=None,
        smoke=True, data_path=str(args.motion_file.resolve()), robot_config=args.robot_config.resolve(),
        disable_dr=not args.exercise_control_dr, disable_obs_noise=True)
    morphology = cfg.env.robot_training.get("morphology")
    if not morphology or not morphology.get("enabled"):
        raise ValueError("An explicitly enabled morphology robot config is required")
    if args.allow_review_motions:
        morphology["allow_review_motions"] = True
    if args.exercise_control_dr:
        cfg.env.hydra_overrides.extend([
            "domain_rand.randomize_pd_gain=true", "domain_rand.kp_range=[0.8,1.2]", "domain_rand.kd_range=[0.8,1.2]",
            "domain_rand.randomize_ctrl_delay=true", "++domain_rand.ctrl_delay_ms_range=[0,10]",
            "domain_rand.randomize_torque_rfi=true", "domain_rand.rfi_lim=0.05"])
    env, _ = cfg.env.build(num_envs=args.num_envs)
    try:
        core = env._env
        core._motion_lib.load_all_motions()
        bank = core._motion_lib._motion_morphology_delta_m
        # Choose distinct shapes first, even when the dataset is grouped by shape.
        unique = torch.unique(bank, dim=0)
        first_ids = torch.stack([torch.isclose(bank, shape[None], atol=1e-7, rtol=0).all(-1).nonzero()[0,0] for shape in unique])
        ids = first_ids[torch.arange(args.num_envs, device=core.device) % len(first_ids)]
        times = torch.zeros(args.num_envs, device=core.device)
        state = core._motion_lib.get_motion_state(ids, times, offset=core.env_origins)
        target = {
            "root_states": torch.cat([state[k] for k in ("root_pos", "root_rot", "root_vel", "root_ang_vel")], -1),
            "dof_states": torch.stack((state["dof_pos"], state["dof_vel"]), -1),
            "motion_ids": ids, "motion_start_times": times, "morphology_delta_m": state["morphology_delta_m"],
        }
        obs, _ = env.reset(to_numpy=False, target_states=target)
        assert obs["state"].shape == (args.num_envs, 2*core.num_dof+6+len(morphology["family_names"]))
        for key in ("state", "privileged_state"):
            torch.testing.assert_close(obs[key][:, -len(morphology["family_names"]):], state["morphology_delta_m"]/morphology["observation_scale_m"])
        fk_error = float((core.body_pos-state["rg_pos"]).abs().max())
        assert fk_error < 3e-6, fk_error
        runtime = core.morphology_runtime
        for field, values in runtime.parameters.items():
            if args.exercise_control_dr and field in ("body_mass", "body_inertia", "body_ipos"):
                continue
            torch.testing.assert_close(getattr(core.mjlab_env.sim.model, field)[:], values[runtime.bank_indices], rtol=1e-5, atol=1e-6)
        action = core.mjlab_env.action_manager.get_term("actions")
        kp = torch.cat([a.stiffness/a.default_stiffness for a in core.robot.actuators], -1)
        kd = torch.cat([a.damping/a.default_damping for a in core.robot.actuators], -1)
        # Exercise the actual motor control law independently of contact forces.
        from mjlab.actuator.actuator import ActuatorCmd
        for actuator in core.robot.actuators:
            zero = torch.zeros_like(actuator.stiffness)
            cmd = ActuatorCmd(zero+.001, zero+.002, zero+.1, zero, zero)
            expected = actuator.stiffness*.001 + actuator.damping*.002 + .1
            torch.testing.assert_close(actuator.compute(cmd), expected)
        delay_steps = action.delay_line.delay.tolist()
        record = {
            "schema": "ufo_morphology_gpu_physics_check_v1", "dr_exercised": args.exercise_control_dr,
            "fk_max_error_m": fk_error, "shapes_checked": core.morphology_delta_m.tolist(),
            "observation_shapes": {k:list(v.shape) for k,v in obs.items()},
            "physics_dt": core.sim_dt, "policy_dt": core.dt,
            "kp_multiplier_range": [float(kp.min()), float(kp.max())],
            "kd_multiplier_range": [float(kd.min()), float(kd.max())],
            "delay_steps": delay_steps, "motor_pd_feedforward_law_checked": True,
            "robot_config_sha256": hashlib.sha256(args.robot_config.read_bytes()).hexdigest(),
            "motion_file_sha256": hashlib.sha256(args.motion_file.read_bytes()).hexdigest(),
        }
        rfi_max = 0.
        for step in range(args.steps):
            obs, reward, _, _, _ = env.step(torch.zeros(args.num_envs, core.num_dof, device=core.device), to_numpy=False)
            assert all(bool(torch.isfinite(v).all()) for v in obs.values())
            assert bool(torch.isfinite(reward).all()) and bool(torch.isfinite(core.torques).all())
            rfi_max = max(rfi_max, float(action.last_rfi_torque.abs().max()))
            if step % 40 == 39:
                env.reset(to_numpy=False, target_states=target)
        record.update(steps=args.steps, all_finite=True, rfi_abs_max_nm=rfi_max,
                      training_started=False, dynamics_tracking_validated=False)
        if args.exercise_control_dr:
            assert float((kp-1).abs().max()) > .01 and rfi_max > 0
        else:
            assert torch.equal(kp, torch.ones_like(kp)) and rfi_max == 0 and max(delay_steps) == 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2)+"\n")
        print(json.dumps(record, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
