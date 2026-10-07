"""Watch a UFO FB checkpoint in the Scaler sim2sim MuJoCo window.

The policy still steps the 1-env mjlab rollout used by tracking inference.
The window is mujoco_viewer.MujocoViewer, the same shell as
scripts/sim2sim_scaler.py. It mirrors the policy pose and draws gold spheres
on the reference bodies. It does not replace the Renaissance five-task
workbench, and it does not write into the live training checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import mujoco.viewer
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import MODEL_NAME_TO_CLASS
from humanoidverse.mjlab_inference_utils import load_mjlab_env_cfg, policy_qpos_from_env
from humanoidverse.tracking_inference import (
    _resolve_tracking_robot_config,
    _target_states_from_obs,
    _tracking_z,
)
from humanoidverse.utils.helpers import get_backward_observation
from humanoidverse.utils.robot_spec import load_robot_training_spec

DEFAULT_MODEL_FOLDER = Path("runs/ufo_fb_g1_b1048576_20260927")
DEFAULT_DATA = Path("humanoidverse/data/lafan_29dof_10s-clipped.pkl")


def resolve_motion_ids(requested: list[int] | None, loaded: int, all_motions: bool) -> list[int]:
    """Pick clip ids. An explicit list wins; otherwise all clips, else 555."""
    if requested:
        ids = [int(motion_id) for motion_id in requested]
    elif all_motions:
        ids = list(range(int(loaded)))
    else:
        ids = [555]
    kept = [motion_id for motion_id in ids if 0 <= motion_id < int(loaded)]
    if kept:
        return kept
    print(f"[INFO] Requested motion ids were outside 0..{loaded - 1}; using motion 0", flush=True)
    return [0]


def _checkpoint_step(model_folder: Path) -> int | None:
    status_path = model_folder / "checkpoint" / "train_status.json"
    if not status_path.is_file():
        return None
    return int(json.loads(status_path.read_text())["time"])


def snapshot_model_folder(model_folder: Path) -> tuple[Path, int | None]:
    """Copy the small checkpoint files so a live trainer can overwrite them."""
    model_folder = model_folder.expanduser().resolve()
    source_model = model_folder / "checkpoint" / "model"
    weights = source_model / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing checkpoint weights: {weights}")
    step_before = _checkpoint_step(model_folder)
    snapshot_root = Path(f"/tmp/ufo_fb_view_{step_before}_{os.getpid()}")
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    snapshot_root.mkdir(parents=True)
    shutil.copy2(model_folder / "config.json", snapshot_root / "config.json")
    destination = snapshot_root / "checkpoint" / "model"
    destination.mkdir(parents=True)
    for path in source_model.iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)
    status_path = model_folder / "checkpoint" / "train_status.json"
    if status_path.is_file():
        shutil.copy2(status_path, snapshot_root / "checkpoint" / "train_status.json")
    step_after = _checkpoint_step(model_folder)
    if step_before != step_after:
        raise RuntimeError(f"Checkpoint changed while copying: {step_before} -> {step_after}")
    print(f"[INFO] snapshot step={step_before} dir={snapshot_root}", flush=True)
    return snapshot_root, step_before


def _load_policy(snapshot: Path, device: str):
    config_path = snapshot / "checkpoint" / "model" / "config.json"
    name = json.loads(config_path.read_text())["name"]
    model = MODEL_NAME_TO_CLASS[name].load(str(snapshot / "checkpoint" / "model"), device=device, strict=False)
    model.eval()
    return model


def _viewer_model(xml_path: Path):
    spec = mujoco.MjSpec.from_file(str(xml_path))
    spec.worldbody.add_geom(
        name="view_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0.0, 0.0, 0.0],
        size=[20.0, 20.0, 0.02],
        rgba=[0.45, 0.47, 0.50, 1.0],
        contype=0,
        conaffinity=0,
    )
    spec.worldbody.add_light(
        name="view_key_light",
        pos=[0.0, -3.0, 4.0],
        dir=[0.2, 0.5, -1.0],
        diffuse=[0.8, 0.8, 0.8],
        ambient=[0.35, 0.35, 0.35],
        specular=[0.1, 0.1, 0.1],
    )
    return spec.compile()


def _open_scaler_viewer(view_model, view_data, controls: dict[str, bool]):
    """Open the same MujocoViewer shell used by sim2sim_scaler.py."""
    import glfw
    import mujoco_viewer

    _install_mouse_compat()
    viewer = mujoco_viewer.MujocoViewer(
        view_model,
        view_data,
        mode="window",
        title="UFO FB tracking",
        width=1920,
        height=1080,
    )
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.fixedcamid = -1
    viewer.cam.distance = 3.2
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -18.0
    viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
    viewer._render_every_frame = True
    viewer._paused = None
    for flag in (
        mujoco.mjtVisFlag.mjVIS_INERTIA,
        mujoco.mjtVisFlag.mjVIS_CONTACTPOINT,
        mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
        mujoco.mjtVisFlag.mjVIS_COM,
        mujoco.mjtVisFlag.mjVIS_TRANSPARENT,
    ):
        viewer.vopt.flags[flag] = 0

    raw = viewer._key_callback

    owned = {
        glfw.KEY_RIGHT,
        glfw.KEY_LEFT,
        glfw.KEY_N,
        glfw.KEY_R,
        glfw.KEY_SPACE,
        glfw.KEY_Q,
        glfw.KEY_ESCAPE,
        glfw.KEY_D,
    }

    def _keys(window, key, scancode, action, mods, _raw=raw) -> None:
        if key in owned:
            if action == glfw.RELEASE:
                if key in (glfw.KEY_RIGHT, glfw.KEY_N):
                    controls["pause"] = False
                    controls["next"] = True
                elif key == glfw.KEY_LEFT:
                    controls["pause"] = False
                    controls["prev"] = True
                elif key == glfw.KEY_R:
                    controls["pause"] = False
                    controls["reset"] = True
                elif key == glfw.KEY_SPACE:
                    controls["pause"] = not controls["pause"]
                elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                    controls["quit"] = True
            return
        return _raw(window, key, scancode, action, mods)

    viewer._ufo_key_callback = _keys
    glfw.set_key_callback(viewer.window, viewer._ufo_key_callback)
    _install_marker_compat(viewer)
    original_overlay = viewer._create_overlay

    def _overlay_with_clip() -> None:
        original_overlay()
        text1, text2 = getattr(viewer, "_clip_hud", ("", ""))
        viewer._overlay[mujoco.mjtGridPos.mjGRID_TOPRIGHT] = [text1, text2]

    viewer._create_overlay = _overlay_with_clip
    viewer._clip_hud = ("UFO FB", "loading")
    return viewer


def _install_mouse_compat() -> None:
    """Keep double-click from crashing on MuJoCo 3.8's scalar skinselect."""
    try:
        from mujoco_viewer import callbacks as viewer_callbacks
    except (ImportError, AttributeError):
        return
    raw = viewer_callbacks.Callbacks._mouse_button_callback
    if getattr(raw, "_ufo_compat", False):
        return

    def _mouse_button_callback(self, window, button, action, mods, _raw=raw):
        try:
            return _raw(self, window, button, action, mods)
        except TypeError as exc:
            if "skinselect" not in str(exc) and "incompatible function arguments" not in str(exc):
                raise
            self.pert.skinselect = -1
            return None

    _mouse_button_callback._ufo_compat = True
    viewer_callbacks.Callbacks._mouse_button_callback = _mouse_button_callback


def _install_marker_compat(viewer) -> None:
    """mujoco_viewer still writes texid; MuJoCo 3.8 geoms no longer have it."""

    def _add_marker_to_scene(marker) -> None:
        if viewer.scn.ngeom >= viewer.scn.maxgeom:
            raise RuntimeError(f"Ran out of geoms. maxgeom: {viewer.scn.maxgeom}")
        geom = viewer.scn.geoms[viewer.scn.ngeom]
        geom.dataid = -1
        geom.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        geom.objid = -1
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.emission = 0
        geom.specular = 0.5
        geom.shininess = 0.5
        geom.reflectance = 0
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size[:] = np.ones(3) * 0.1
        geom.mat[:] = np.eye(3)
        geom.rgba[:] = np.ones(4)
        for key, value in marker.items():
            if not hasattr(geom, key):
                continue
            if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                setattr(geom, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(geom, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            else:
                raise ValueError(f"Unsupported marker field {key}: {type(value)}")
        viewer.scn.ngeom += 1

    viewer._add_marker_to_scene = _add_marker_to_scene


def _draw_reference(viewer, points: np.ndarray) -> None:
    rgba = (0.95, 0.75, 0.25, 0.45)
    size = (0.03, 0.03, 0.03)
    count = min(int(points.shape[0]), max(int(viewer.scn.maxgeom) - 1, 0))
    for index in range(count):
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=size,
            pos=np.asarray(points[index], dtype=float),
            mat=np.eye(3),
            rgba=rgba,
        )


def _present(viewer, view_model, view_data, qpos: np.ndarray, points: np.ndarray, hud: tuple[str, str]) -> bool:
    if not viewer.is_alive:
        return False
    view_data.qpos[:] = qpos
    view_data.qvel[:] = 0.0
    mujoco.mj_forward(view_model, view_data)
    viewer.cam.lookat[:] = np.asarray(qpos[:3], dtype=float)
    viewer._clip_hud = hud
    _draw_reference(viewer, points)
    viewer.render()
    return bool(viewer.is_alive)


def _reference_points(obs_dict: dict, step: int) -> np.ndarray:
    positions = obs_dict["ref_body_pos"].detach().cpu().numpy()
    frame = min(step + 1, positions.shape[0] - 1)
    return np.asarray(positions[frame], dtype=np.float64)


def view_tracking(
    *,
    model_folder: Path,
    data_path: Path,
    motion_list: list[int] | None,
    all_motions: bool,
    device: str,
    max_steps: int | None,
) -> None:
    snapshot, snapshot_step = snapshot_model_folder(model_folder)
    robot_config = _resolve_tracking_robot_config(None, None)
    robot_training = load_robot_training_spec(robot_config)
    robot_xml = Path(robot_training.robot.xml_path).expanduser().resolve()
    num_dof = len(robot_training.robot.control_joint_names)
    model = _load_policy(snapshot, device)

    env_cfg, use_root_height_obs = load_mjlab_env_cfg(
        snapshot,
        data_path=data_path,
        robot_config=robot_config,
        device=device,
        headless=True,
        disable_dr=True,
        disable_obs_noise=True,
        max_episode_length_s=10000.0,
    )
    wrapped_env, _info = env_cfg.build(num_envs=1)
    env = wrapped_env._env
    env._motion_lib.load_all_motions()
    env.is_evaluating = True
    loaded = int(env._motion_lib._motion_lengths.shape[0])
    names = list(getattr(env._motion_lib, "_motion_data_keys", []))
    print(f"[INFO] Loaded motions={loaded}", flush=True)
    motion_list = resolve_motion_ids(motion_list, loaded, all_motions)

    view_model = _viewer_model(robot_xml)
    view_data = mujoco.MjData(view_model)
    expected_qpos = 7 + num_dof
    if view_model.nq != expected_qpos:
        raise ValueError(f"Viewer qpos size {view_model.nq} does not match 7+{num_dof}")

    controls = {"next": False, "prev": False, "quit": False, "reset": False, "pause": False}
    print(
        "[INFO] Scaler-style MuJoCo window (mujoco_viewer). Solid robot is the FB policy, "
        "gold spheres are the reference motion. Top right shows clip index, motion id, and name. "
        f"Playlist={len(motion_list)} of {loaded}. "
        "Keys: Right arrow or N next clip, Left arrow previous clip, Space pause, R restart clip, Q or Esc close. "
        "Mouse: left drag orbits, right drag pans, wheel zooms. "
        "This is not the five-task workbench.",
        flush=True,
    )
    viewer = _open_scaler_viewer(view_model, view_data, controls)
    try:
        motion_cursor = 0
        while viewer.is_alive and not controls["quit"]:
            motion_id = int(motion_list[motion_cursor % len(motion_list)])
            motion_name = names[motion_id] if motion_id < len(names) else str(motion_id)
            controls["next"] = False
            controls["prev"] = False
            controls["reset"] = False
            outcome = _play_motion(
                viewer=viewer,
                view_model=view_model,
                view_data=view_data,
                wrapped_env=wrapped_env,
                env=env,
                policy=model,
                device=device,
                motion_id=motion_id,
                motion_name=motion_name,
                clip_index=motion_cursor % len(motion_list),
                clip_total=len(motion_list),
                snapshot_step=snapshot_step,
                use_root_height_obs=use_root_height_obs,
                num_dof=num_dof,
                max_steps=max_steps,
                controls=controls,
            )
            if outcome == "prev":
                motion_cursor = max(0, motion_cursor - 1)
            elif outcome != "reset":
                motion_cursor += 1
    finally:
        if viewer.is_alive:
            viewer.close()
        wrapped_env.close()


@torch.no_grad()
def _play_motion(
    *,
    viewer,
    view_model,
    view_data,
    wrapped_env,
    env,
    policy,
    device: str,
    motion_id: int,
    motion_name: str,
    clip_index: int,
    clip_total: int,
    snapshot_step: int | None,
    use_root_height_obs: bool,
    num_dof: int,
    max_steps: int | None,
    controls: dict[str, bool],
) -> None:
    viewer._clip_hud = (
        "clip\nid\nname\nframe\nckpt",
        f"{clip_index + 1}/{clip_total}\n{motion_id}\n{motion_name}\nloading\n{snapshot_step}",
    )
    if viewer.is_alive:
        viewer.render()
    backward_obs, obs_dict = get_backward_observation(env, motion_id, use_root_height_obs=use_root_height_obs)
    z = _tracking_z(policy, tree_map(lambda value: value[1:].to(device) if hasattr(value, "to") else value, backward_obs))
    target_states = _target_states_from_obs(obs_dict, device=device, num_dof=num_dof)
    observation, _reward = wrapped_env.reset(to_numpy=False, target_states=target_states)
    episode_len = int(z.shape[0])
    if max_steps is not None:
        episode_len = min(episode_len, int(max_steps))
    step_dt = float(env.dt)
    print(
        f"[INFO] clip={clip_index + 1}/{clip_total} motion_id={motion_id} "
        f"name={motion_name} steps={episode_len} dt={step_dt}",
        flush=True,
    )
    wall_start = time.perf_counter()
    shown = 0
    low_since: int | None = None
    for step in range(episode_len):
        if controls["quit"] or controls["next"] or controls["prev"] or controls["reset"] or not viewer.is_alive:
            if controls["reset"]:
                return "reset"
            if controls["prev"]:
                return "prev"
            return "stop"
        started = time.perf_counter()
        action = policy.act(observation, z[step].unsqueeze(0), mean=True)
        observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)
        qpos = policy_qpos_from_env(wrapped_env, expected_qpos_size=view_model.nq)
        shown += 1
        elapsed = max(time.perf_counter() - wall_start, 1e-6)
        speed = (shown * step_dt) / elapsed
        hud = (
            "clip\nid\nname\nframe\nckpt",
            f"{clip_index + 1}/{clip_total}\n{motion_id}\n{motion_name}\n"
            f"{step + 1}/{episode_len}  {speed:.2f}x\n{snapshot_step}",
        )
        alive = _present(viewer, view_model, view_data, qpos, _reference_points(obs_dict, step), hud)
        if not alive:
            controls["quit"] = True
            return "stop"
        while controls["pause"] and viewer.is_alive and not controls["quit"] and not controls["next"] and not controls["prev"] and not controls["reset"]:
            alive = _present(viewer, view_model, view_data, qpos, _reference_points(obs_dict, step), hud)
            if not alive:
                controls["quit"] = True
                return "stop"
            time.sleep(0.02)
        if controls["quit"] or controls["next"] or controls["prev"] or controls["reset"]:
            if controls["reset"]:
                return "reset"
            if controls["prev"]:
                return "prev"
            return "stop"
        remaining = step_dt - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)
        if float(qpos[2]) < 0.3:
            low_since = step if low_since is None else low_since
            if (step - low_since) * step_dt >= 1.0:
                print(f"[INFO] motion_id={motion_id} pelvis stayed below 0.3 m; next clip", flush=True)
                return "fell"
        else:
            low_since = None
        if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
            print(f"[INFO] motion_id={motion_id} ended at step={step}", flush=True)
            return "done"
    return "done"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a UFO FB checkpoint in a MuJoCo window.")
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL_FOLDER)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--all-motions", action="store_true", help="Play every loaded clip from 0 upward.")
    parser.add_argument("--motion-list", type=int, nargs="+", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    view_tracking(
        model_folder=args.model_folder,
        data_path=args.data_path,
        motion_list=None if args.motion_list is None else list(args.motion_list),
        all_motions=bool(args.all_motions),
        device=args.device,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
