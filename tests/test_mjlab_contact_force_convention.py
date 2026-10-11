"""CPU regressions for the public force-on-body adapter convention.

The native physical oracle uses MJLab's installed sensor builder but never
constructs a GPU environment. No implementation of _read_contact_forces is
copied here: every case invokes the production class's real method.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from humanoidverse.agents.envs.humanoidverse_mjlab import HumanoidVerseMjlabCore


def _core(body_names, *, force=None, primary_names=(), sensor_present=True, num_envs=1):
    sensors = {}
    if sensor_present:
        sensors['body_contact'] = SimpleNamespace(
            primary_names=primary_names,
            data=SimpleNamespace(force=force),
        )
    return SimpleNamespace(
        num_envs=num_envs,
        num_bodies=len(body_names),
        body_names=tuple(body_names),
        device='cpu',
        mjlab_env=SimpleNamespace(scene=SimpleNamespace(sensors=sensors)),
    )


def _read(core):
    return HumanoidVerseMjlabCore._read_contact_forces(core)


class MjlabContactForceConventionTest(unittest.TestCase):
    def test_supported_body_force_points_upward(self):
        # Native body1/primary reports force exerted by body on ground.
        raw = torch.tensor([[[0., 0., -19.62]]])
        core = _core(['foot'], force=raw, primary_names=['robot/foot'])
        force = _read(core)
        torch.testing.assert_close(force, torch.tensor([[[0., 0., 19.62]]]))
        self.assertGreater(float(force[0, 0, 2]), 10.)

    def test_world_xyz_direction_and_primary_name_mapping(self):
        # Explicit physical directions, two worlds and shuffled body columns.
        # The robot exerts +x,-y,-z on its contact partners in the first slot;
        # action-reaction therefore gives the body -x,+y,+z.
        raw = torch.tensor([
            [[3., -4., -20.], [99., 98., 97.], [-2., 5., -30.]],
            [[-7., 8., 9.], [1., 2., 3.], [6., -5., -40.]],
        ])
        core = _core(
            ['torso', 'left_foot', 'right_foot', 'unobserved'],
            force=raw,
            primary_names=['robot/right_foot', 'other/unknown', 'robot/left_foot'],
            num_envs=2,
        )
        expected = torch.tensor([
            [[0., 0., 0.], [2., -5., 30.], [-3., 4., 20.], [0., 0., 0.]],
            [[0., 0., 0.], [-6., 5., 40.], [7., -8., -9.], [0., 0., 0.]],
        ])
        torch.testing.assert_close(_read(core), expected)

    def test_absent_sensor_or_force_is_zero(self):
        for present in (False, True):
            with self.subTest(sensor_present=present):
                core = _core(['torso', 'foot'], sensor_present=present, num_envs=2)
                torch.testing.assert_close(_read(core), torch.zeros(2, 2, 3))

    def test_raw_sensor_is_not_mutated_or_aliased(self):
        raw = torch.tensor([[[3., -4., -20.]]])
        before = raw.clone()
        core = _core(['foot'], force=raw, primary_names=['robot/foot'])
        converted = _read(core)
        torch.testing.assert_close(raw, before)
        converted[0, 0, :] = 123.
        torch.testing.assert_close(raw, before)
        self.assertNotEqual(converted.data_ptr(), raw.data_ptr())


class MjlabNativeContactForceOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import mujoco
            from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
        except ImportError as error:
            raise unittest.SkipTest(f'Native MuJoCo/MJLab physical oracle unavailable: {error}')
        cls.mujoco = mujoco
        cls.ContactMatch = ContactMatch
        cls.ContactSensorCfg = ContactSensorCfg
        if not hasattr(mujoco.mjtSensor, 'mjSENS_CONTACT'):
            raise unittest.SkipTest('Native MuJoCo CONTACT sensors unavailable')

    def _settled_box(self, horizontal_force_N):
        mujoco = self.mujoco
        spec = mujoco.MjSpec.from_string('''
        <mujoco model="contact_force_oracle">
          <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
          <worldbody>
            <geom name="ground" type="plane" size="3 3 .1" friction="1 .005 .0001"/>
            <body name="box" pos="0 0 .6">
              <freejoint/>
              <geom name="box_geom" type="box" size=".1 .12 .1" mass="2"
                    friction="1 .005 .0001"/>
            </body>
          </worldbody>
        </mujoco>''')
        sensor = self.ContactSensorCfg(
            name='body_contact',
            primary=self.ContactMatch(mode='body', pattern='box'),
            fields=('force',), reduce='netforce',
        ).build()
        sensor._add_contact_sensor_to_spec(spec, 'box_native_primary', 'box', None, 'force')
        model = spec.compile()
        data = mujoco.MjData(model)
        body = model.body('box').id
        data.xfrc_applied[body, 0] = horizontal_force_N
        for _ in range(2000):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        mujoco.mj_rnePostConstraint(model, data)
        contact_on_body = np.zeros(3)
        self.assertGreater(data.ncon, 0)
        for index in range(data.ncon):
            contact = data.contact[index]
            local = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, local)
            force_on_geom2 = contact.frame.reshape(3, 3).T @ local[:3]
            if model.geom_bodyid[contact.geom2] == body:
                contact_on_body += force_on_geom2
            elif model.geom_bodyid[contact.geom1] == body:
                contact_on_body -= force_on_geom2
        external = np.array([horizontal_force_N, 0., 0.])
        weight = model.body_mass[body] * model.opt.gravity
        np.testing.assert_allclose(
            contact_on_body + external + weight,
            model.body_mass[body] * data.qacc[:3], atol=1e-5,
        )
        # cfrc_ext includes applied external force as well as contact force.
        np.testing.assert_allclose(data.cfrc_ext[body, 3:], contact_on_body + external, atol=1e-5)
        self.assertGreater(float(contact_on_body[2]), 10.)
        np.testing.assert_allclose(contact_on_body, [-horizontal_force_N, 0., 19.62], atol=1e-3)
        sid = model.sensor('box_native_primary').id
        native = data.sensordata[model.sensor_adr[sid]:model.sensor_adr[sid] + 3].copy()
        # Oracle also locks the native side's convention independently.
        np.testing.assert_allclose(native, -contact_on_body, atol=1e-5)
        raw = torch.from_numpy(native.astype(np.float32)).reshape(1, 1, 3)
        before = raw.clone()
        core = _core(['box'], force=raw, primary_names=['robot/box'])
        actual = _read(core).numpy()[0, 0]
        torch.testing.assert_close(raw, before)
        np.testing.assert_allclose(actual, contact_on_body, atol=1e-5)
        self.assertFalse(torch.cuda.is_initialized())

    def test_box_ground_support_matches_independent_force_balance(self):
        self._settled_box(horizontal_force_N=0.)

    def test_box_friction_force_direction_matches_independent_force_balance(self):
        self._settled_box(horizontal_force_N=3.)


if __name__ == '__main__':
    unittest.main()
