"""CPU checks of real FK, physical morphology and substep control contracts."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import joblib
import mujoco
import numpy as np
import torch
from easydict import EasyDict
from scipy.spatial.transform import Rotation

from humanoidverse.agents.envs.morphology_actions import PhysicsStepDelay
from humanoidverse.utils.morphology import (
    MjlabMorphologyRuntime, MORPHOLOGY_MODEL_FIELDS,
    morphology_model_parameters, profile_properties,
)
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot


def fixture(root):
    profile = {"body_name": "length", "geom_name": "profile", "family_index": 0,
        "nominal_length_m": .2, "section_xy_m": [.02, .04], "axis": [0, 0, 1],
        "nominal_center_m": [0, 0, -.1], "linear_density_kg_m": 2.,
        "carrier_mass_kg": 1., "carrier_com_m": [0, 0, 0],
        "rotation_matrix": np.eye(3).tolist(), "carrier_inertia_kg_m2": (np.eye(3)*.001).tolist()}
    cfg = {"enabled": True, "family_names": ["thigh", "calf", "torso", "upperarm", "forearm"],
        "family_bounds_m": [[-.05, .05]]*5, "observation_scale_m": .05,
        "body_offsets": [{"body_name": "length", "family_index": 0, "axis": [0, 0, 1]}],
        "profiles": [profile]}
    def xml(delta):
        p = profile_properties(profile, delta)
        vec = lambda x: " ".join(str(float(a)) for a in x)
        return f'''<mujoco model="morphology_test"><compiler angle="radian" fusestatic="false"/>
        <worldbody><body name="base" pos="0 0 1"><joint name="root" type="free"/>
        <geom type="sphere" size=".05" mass="1"/>
        <body name="hinge" pos="0 0 .1"><joint name="hinge_joint" axis="-1 0 0" range="-1 1"/>
        <geom type="sphere" size=".03" mass=".1"/>
        <body name="length" pos="0 0 {.2 + delta}">
        <inertial mass="{p['mass']}" pos="{vec(p['com'])}" diaginertia="{vec(p['inertia'])}" quat="{vec(p['iquat'])}"/>
        <geom name="profile" type="box" pos="{vec(p['center'])}" size="{vec(p['size'])}"/>
        </body></body></body></worldbody><actuator><motor name="hinge_joint" joint="hinge_joint"/></actuator></mujoco>'''
    path = root / "nominal.xml"
    path.write_text(xml(0))
    return cfg, path, xml


class MorphologyRuntimeTest(unittest.TestCase):
    def test_physical_parameters_equal_separately_compiled_robot(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, path, xml = fixture(Path(tmp))
            nominal = mujoco.MjModel.from_xml_path(str(path))
            for delta in (-.04, 0, .035):
                expected = mujoco.MjModel.from_xml_string(xml(delta))
                fields = morphology_model_parameters(nominal, cfg, [delta, 0, 0, 0, 0])
                actual = copy.copy(nominal)
                for field, value in fields.items():
                    getattr(actual, field)[:] = value.reshape(getattr(actual, field).shape)
                a, b = mujoco.MjData(actual), mujoco.MjData(expected)
                mujoco.mj_setConst(actual, a)
                for field in MORPHOLOGY_MODEL_FIELDS:
                    if field not in ("body_inertia", "body_iquat"):
                        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), atol=1e-8)
                for angle in (-.6, .2, .7):
                    a.qpos[-1] = b.qpos[-1] = angle
                    mujoco.mj_forward(actual, a); mujoco.mj_forward(expected, b)
                    np.testing.assert_allclose(a.xpos, b.xpos, atol=1e-8)
                    np.testing.assert_allclose(a.geom_xpos, b.geom_xpos, atol=1e-8)
                    np.testing.assert_allclose(a.qM, b.qM, atol=1e-8)

    def test_partial_reset_has_correct_shape_no_cross_env_or_dr_accumulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, path, _ = fixture(Path(tmp))
            model = mujoco.MjModel.from_xml_path(str(path))
            fields = morphology_model_parameters(model, cfg, [0]*5)
            sim = SimpleNamespace(mj_model=model, device="cpu", num_envs=3,
                model=SimpleNamespace(**{k:torch.tensor(v).repeat((3,)+(1,)*v.ndim) for k,v in fields.items()}),
                expand_model_fields=lambda _: None, recompute_constants=lambda _: None)
            dr = EasyDict(randomize_link_mass=True, link_mass_range=[1.2, 1.2],
                randomize_base_com=True, base_com_range={a:[.01,.01] for a in 'xyz'})
            deltas = torch.tensor([[0,0,0,0,0],[.03,0,0,0,0]])
            runtime = MjlabMorphologyRuntime(sim, cfg, deltas, robot_body_ids=[1,2,3],
                                            torso_body_id=1, domain_rand=dr, prefix="")
            untouched = sim.model.body_pos[1].clone()
            runtime.apply(torch.tensor([0,2]), deltas)
            mass = sim.model.body_mass.clone(); inertia = sim.model.body_inertia.clone(); com = sim.model.body_ipos.clone()
            runtime.apply(torch.tensor([0,2]), deltas)
            torch.testing.assert_close(sim.model.body_mass, mass)
            torch.testing.assert_close(sim.model.body_inertia, inertia)
            torch.testing.assert_close(sim.model.body_ipos, com)
            torch.testing.assert_close(sim.model.body_pos[1], untouched)
            self.assertAlmostEqual(float(sim.model.body_pos[2,3,2]), .23, places=7)
            with self.assertRaisesRegex(ValueError, "no unique retargeted"):
                runtime.apply(torch.tensor([0]), torch.tensor([[.01,0,0,0,0]]))
            runtime.apply(torch.tensor([2]), deltas[:1], randomize=False)
            torch.testing.assert_close(sim.model.body_mass[2], torch.tensor(fields['body_mass']))

    def test_motion_lib_shape_fk_metadata_and_raw_joint_sign(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, path, xml = fixture(Path(tmp))
            motions = {}
            angles = np.array([-.3, .1, .5], dtype=np.float32)
            for index, delta in enumerate((0., .04)):
                pose = np.zeros((3, 3, 3), dtype=np.float32)
                pose[:,1,0] = -angles
                motions[str(index)] = {"fps":30, "pose_aa":pose,
                    "root_trans_offset":np.tile(np.array([0,0,1], np.float32),(3,1)),
                    "dof_pos":angles[:,None], "metadata":{"morphology_delta_m":[delta,0,0,0,0],
                    "morphology_reference":"retargeted", "valid_for_training":True}}
            motion_path = Path(tmp)/"motions.pkl"; joblib.dump(motions,motion_path)
            lib_cfg = EasyDict(motion_file=str(motion_path), morphology=cfg,
                asset={"assetRoot":str(path.parent),"assetFileName":path.name}, extend_config=[], has_mesh=False)
            lib = MotionLibRobot(lib_cfg, num_envs=2, device='cpu'); lib.load_all_motions()
            state = lib.get_motion_state(torch.tensor([0,1,1]), torch.tensor([0.,1/30,2/30]))
            np.testing.assert_allclose(state['dof_pos'], angles[:,None], atol=1e-7)
            np.testing.assert_allclose(state['morphology_delta_m'][:,0], [0,.04,.04], atol=1e-7)
            scalar = lib.get_motion_state(1, torch.tensor([0.,1/30,2/30]))
            self.assertEqual(scalar['morphology_delta_m'].shape, (3,5))
            model = mujoco.MjModel.from_xml_string(xml(.04)); data = mujoco.MjData(model)
            for i,angle in enumerate(angles):
                data.qpos[-1]=angle; mujoco.mj_forward(model,data)
                np.testing.assert_allclose(scalar['rg_pos'][i], data.xpos[1:], atol=2e-7)
            np.testing.assert_allclose(lib.mesh_parsers._offsets[0,-1], [0,0,.2])
            motions['0']['metadata']['review_only']=True; joblib.dump(motions,motion_path)
            with self.assertRaisesRegex(ValueError,'review-only'):
                MotionLibRobot(lib_cfg, num_envs=2, device='cpu')
            # A held-out motion is usable only for explicitly qualified evaluation.
            motions['0']['metadata'].update(review_only=False, valid_for_training=False, kinematic_qualified=True)
            joblib.dump(motions,motion_path)
            with self.assertRaisesRegex(ValueError,'not qualified for training'):
                MotionLibRobot(lib_cfg, num_envs=2, device='cpu')
            lib_cfg.morphology.motion_purpose='evaluation'
            MotionLibRobot(lib_cfg, num_envs=2, device='cpu')
            motions['0']['metadata']['kinematic_qualified']=False
            joblib.dump(motions,motion_path)
            with self.assertRaisesRegex(ValueError,'not qualified for training'):
                MotionLibRobot(lib_cfg, num_envs=2, device='cpu')

    def test_delay_is_physics_steps_and_partial_reset_clears_history(self):
        delay = PhysicsStepDelay(2, 1, 2, 'cpu')
        delay.reset(torch.tensor([0,1]), torch.tensor([[10.],[20.]]), torch.tensor([0,2]))
        torch.testing.assert_close(delay.push(torch.tensor([[1.],[2.]])), torch.tensor([[1.],[20.]]))
        torch.testing.assert_close(delay.push(torch.tensor([[3.],[4.]])), torch.tensor([[3.],[20.]]))
        torch.testing.assert_close(delay.push(torch.tensor([[5.],[6.]])), torch.tensor([[5.],[2.]]))
        delay.reset(torch.tensor([1]), torch.tensor([[90.]]), torch.tensor([2]))
        torch.testing.assert_close(delay.push(torch.tensor([[7.],[8.]])), torch.tensor([[7.],[90.]]))


if __name__ == '__main__':
    unittest.main()
