"""Optional real-data FK/Jacobian/latency validation; does not launch training.

Run from UFO with PYTHONPATH=. using the same environment as training. Provide
an evaluation's physical_metrics.json to --motion-selection to reproduce its
motion keys. Only sampling, immutable MuJoCo FK and timing are performed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time

import joblib
import mujoco
import numpy as np
import torch
import yaml
from easydict import EasyDict
from scipy.spatial.transform import Rotation

from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
from humanoidverse.utils.morphology import morphology_model_parameters


def validate(args):
    robot = yaml.safe_load(args.robot_config.read_text())
    xml = Path(robot['xml_path'])
    if not xml.is_absolute():
        xml = args.robot_config.parent / xml
    source = joblib.load(args.motion_file)
    if args.motion_selection:
        selection = json.loads(args.motion_selection.read_text())
        keys = [row['motion'] for row in selection['trajectories']]
    else:
        keys = list(source)
    morphology = copy.deepcopy(robot['training']['morphology'])
    morphology['motion_purpose'] = args.purpose
    with tempfile.TemporaryDirectory() as tmp:
        motion_path = Path(tmp) / 'selected_motions.pkl'
        joblib.dump({key: source[key] for key in keys}, motion_path)
        config = EasyDict(motion_file=str(motion_path),
            asset=dict(assetRoot=str(xml.parent), assetFileName=xml.name),
            morphology=morphology, extend_config=[], has_mesh=False)
        lib = MotionLibRobot(config, len(keys), args.device)
        lib.load_all_motions()
    rng = np.random.default_rng(4928)
    rows = []
    for motion_id, key in enumerate(keys):
        length = float(lib._motion_lengths[motion_id])
        time50 = np.arange(0, length + 1e-7, .02, dtype=np.float32)
        random_times = rng.uniform(.00001, length - .00001, 256).astype(np.float32)
        times = torch.as_tensor(np.r_[time50, random_times], device=args.device)
        ids = torch.full((len(times),), motion_id, device=args.device, dtype=torch.long)
        sampled = lib.get_motion_state(ids, times)
        state = {name: value.cpu().numpy() for name, value in sampled.items()}
        centered = lib.get_motion_state(ids, times, offset=-sampled['root_pos'])['rg_pos'].cpu().numpy()
        model = mujoco.MjModel.from_xml_path(str(xml))
        for name, value in morphology_model_parameters(model, morphology, state['morphology_delta_m'][0]).items():
            getattr(model, name)[:] = value.reshape(getattr(model, name).shape)
        model.body_sameframe[:] = model.geom_sameframe[:] = model.site_sameframe[:] = 0
        model.body_simple[:] = 0
        data = mujoco.MjData(model)
        mujoco.mj_setConst(model, data)
        bodies = [model.body(name).id for name in lib._morphology_kinematics.body_names]
        joints = [model.joint(name).id for name in lib._morphology_kinematics.joint_names]
        pose_errors, rotation_errors, centered_errors, velocity_errors, angular_errors = [], [], [], [], []
        for frame in range(len(times)):
            data.qpos[:3] = state['root_pos'][frame]
            data.qpos[3:7] = state['root_rot'][frame][[3, 0, 1, 2]]
            data.qpos[model.jnt_qposadr[joints]] = state['dof_pos'][frame]
            data.qvel[:3] = state['root_vel'][frame]
            data.qvel[3:6] = Rotation.from_quat(state['root_rot'][frame]).inv().apply(state['root_ang_vel'][frame])
            data.qvel[model.jnt_dofadr[joints]] = state['dof_vel'][frame]
            mujoco.mj_forward(model, data)
            pose_errors.append(float(np.abs(data.xpos[bodies] - state['rg_pos'][frame]).max()))
            rotation_errors.append(float(np.abs(data.xmat[bodies].reshape(-1, 3, 3)
                                               - Rotation.from_quat(state['rb_rot'][frame]).as_matrix()).max()))
            if frame % 17 == 0:
                for index, body in enumerate(bodies):
                    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                    mujoco.mj_jacBody(model, data, jp, jr, body)
                    velocity_errors.append(float(np.abs(jp @ data.qvel - state['body_vel'][frame, index]).max()))
                    angular_errors.append(float(np.abs(jr @ data.qvel - state['body_ang_vel'][frame, index]).max()))
            data.qpos[:3] = 0
            mujoco.mj_forward(model, data)
            centered_errors.append(float(np.abs(data.xpos[bodies] - centered[frame]).max()))
        row = dict(motion=key, frames50Hz=len(time50), random_non_grid=len(random_times),
            world_pose_max_m=max(pose_errors), root_centered_max_m=max(centered_errors),
            rotation_matrix_max=max(rotation_errors), body_velocity_jacobian_max_m_s=max(velocity_errors),
            angular_velocity_jacobian_max_rad_s=max(angular_errors),
            max_by_phase_mod5_m=[max(pose_errors[phase:len(time50):5]) for phase in range(5)])
        row['passed'] = (row['world_pose_max_m'] < 2e-5 and row['root_centered_max_m'] < 2e-6
            and row['rotation_matrix_max'] < 2e-6 and row['body_velocity_jacobian_max_m_s'] < 2e-5
            and row['angular_velocity_jacobian_max_rad_s'] < 2e-5)
        rows.append(row)
        print(json.dumps(row), flush=True)
    ids = torch.arange(1024, device=args.device) % len(keys)
    times = torch.rand(1024, device=args.device) * lib._motion_lengths[ids]
    synchronize = (lambda: torch.cuda.synchronize(args.device)) if str(args.device).startswith('cuda') else lambda: None
    timing = {}
    for name, enabled in [('consistent_fk', True), ('old_independent_interpolation', False)]:
        saved = lib.morphology_config
        try:
            if not enabled:
                lib.morphology_config = None
            for _ in range(15):
                lib.get_motion_state(ids, times)
            synchronize()
            durations = []
            for _ in range(50):
                start = time.perf_counter()
                lib.get_motion_state(ids, times)
                synchronize()
                durations.append((time.perf_counter() - start) * 1000)
            timing[name] = dict(median_ms=float(np.median(durations)), p95_ms=float(np.percentile(durations, 95)),
                                mean_ms=float(np.mean(durations)))
        finally:
            lib.morphology_config = saved
    report = dict(schema='ufo_morphology_interpolation_validation_v1', passed=all(row['passed'] for row in rows),
        device=torch.cuda.get_device_name(args.device) if str(args.device).startswith('cuda') else 'cpu',
        motion_file_sha256=hashlib.sha256(args.motion_file.read_bytes()).hexdigest(),
        robot_config_sha256=hashlib.sha256(args.robot_config.read_bytes()).hexdigest(),
        motions=len(rows), shape_count=len({tuple(source[key]['metadata']['morphology_delta_m']) for key in keys}),
        samples=sum(row['frames50Hz'] + row['random_non_grid'] for row in rows),
        velocity_semantics='piecewise generalized-coordinate interpolation derivatives with current-pose body-origin world Jacobian; zero at/after final frame or before start',
        thresholds=dict(world_pose_m=2e-5, root_centered_pose_m=2e-6, rotation_matrix=2e-6,
                        body_velocity_m_s=2e-5, angular_velocity_rad_s=2e-5), rows=rows, query1024=timing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(timing), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-config', type=Path, required=True)
    parser.add_argument('--motion-file', type=Path, required=True)
    parser.add_argument('--motion-selection', type=Path)
    parser.add_argument('--purpose', choices=['training', 'evaluation'], default='training')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(validate(parser.parse_args()))
