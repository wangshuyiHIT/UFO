"""Rendered models must use the rollout's physical shape and qpos order."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from humanoidverse.mjlab_inference_utils import (
    MujocoQposRenderer, policy_qpos_from_env, render_policy_frame,
)
from humanoidverse.utils.morphology import morphology_model_parameters
from test_morphology_runtime import fixture


class MorphologyRenderingTest(unittest.TestCase):
    def test_renderer_switches_actual_shape_without_nominal_accumulation(self):
        with tempfile.TemporaryDirectory() as folder, patch("mujoco.Renderer") as pixels:
            pixels.return_value.render.return_value = np.zeros((16, 16, 3), dtype=np.uint8)
            cfg, path, xml = fixture(Path(folder))
            renderer = MujocoQposRenderer(path, render_size=16, morphology=cfg)
            baseline = renderer._nominal_model.body_pos.copy()
            try:
                for delta in (.04, -.035, 0.):
                    renderer.set_morphology(torch.tensor([delta, 0, 0, 0, 0]))
                    expected = mujoco.MjModel.from_xml_string(xml(delta))
                    data = mujoco.MjData(expected)
                    data.qpos[-1] = .4
                    mujoco.mj_forward(expected, data)
                    renderer.render_qpos(data.qpos)
                    np.testing.assert_allclose(renderer.data.xpos, data.xpos, atol=3e-8)
                    index = renderer.model.geom("profile").id
                    other = expected.geom("profile").id
                    np.testing.assert_allclose(renderer.model.geom_size[index], expected.geom_size[other], atol=3e-8)
                    np.testing.assert_array_equal(renderer._nominal_model.body_pos, baseline)
            finally:
                renderer.close()

    def test_policy_fallback_uses_environment_shape(self):
        with tempfile.TemporaryDirectory() as folder, patch("mujoco.Renderer") as pixels:
            pixels.return_value.render.return_value = np.zeros((16, 16, 3), dtype=np.uint8)
            cfg, path, xml = fixture(Path(folder))
            renderer = MujocoQposRenderer(path, render_size=16, morphology=cfg)
            core = SimpleNamespace(morphology_config=cfg, morphology_delta_m=torch.tensor([[.04, 0, 0, 0, 0]]),
                                   dof_names=["hinge_joint"])
            qpos = renderer.model.qpos0.copy(); qpos[-1] = -.3
            wrapped = SimpleNamespace(_env=core, _get_qpos_qvel=lambda **_: (qpos[None], None))
            try:
                image, used_native = render_policy_frame(wrapped, renderer, use_env_render=False)
                self.assertEqual(image.shape, (16, 16, 3)); self.assertFalse(used_native)
                expected = mujoco.MjModel.from_xml_string(xml(.04)); data = mujoco.MjData(expected)
                data.qpos[:] = qpos; mujoco.mj_forward(expected, data)
                np.testing.assert_allclose(renderer.data.xpos, data.xpos, atol=3e-8)
            finally:
                renderer.close()

    def test_policy_control_order_is_converted_to_mjcf_order(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><freejoint/>
          <geom type="sphere" size=".1"/><body><joint name="a"/><geom type="sphere" size=".1"/>
          <body><joint name="b"/><geom type="sphere" size=".1"/></body></body></body></worldbody></mujoco>''')
        qpos = np.array([0., 0., 1., 1., 0., 0., 0., .2, .6])
        wrapped = SimpleNamespace(_env=SimpleNamespace(dof_names=["b", "a"]),
                                  _get_qpos_qvel=lambda **_: (qpos[None], None))
        result = policy_qpos_from_env(wrapped, expected_qpos_size=9, model=model)
        np.testing.assert_array_equal(result[:7], qpos[:7])
        np.testing.assert_array_equal(result[7:], [.6, .2])

    def test_native_mjlab_offscreen_syncs_selected_world_geometry(self):
        # Exercise the installed MJLab renderer's actual sync/update path using
        # CPU tensors and fake pixels; no graphics driver or CUDA is required.
        from mjlab.viewer.offscreen_renderer import OffscreenRenderer
        from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
        with tempfile.TemporaryDirectory() as folder:
            cfg, path, xml = fixture(Path(folder))
            nominal = mujoco.MjModel.from_xml_path(str(path))
            banks = [morphology_model_parameters(nominal, cfg, [delta, 0, 0, 0, 0]) for delta in (0., .04)]
            renderer = OffscreenRenderer.__new__(OffscreenRenderer)
            renderer._model = copy.copy(nominal)
            disable_model_sameframe_shortcuts(renderer._model)
            renderer._data = mujoco.MjData(renderer._model)
            renderer._sim_model = SimpleNamespace(**{key: torch.tensor(np.stack([bank[key] for bank in banks])) for key in banks[0]})
            renderer._expanded_fields = set(banks[0])
            renderer._cfg = SimpleNamespace(env_idx=1)
            renderer._cam = mujoco.MjvCamera()
            renderer._renderer = SimpleNamespace(update_scene=lambda *args, **kwargs: None)
            renderer._get_extra_env_ids = lambda *args: []
            qpos = torch.tensor(np.tile(nominal.qpos0, (2, 1))); qpos[:, -1] = .5
            state = SimpleNamespace(nworld=2, qpos=qpos, qvel=torch.zeros((2, nominal.nv)))
            renderer.update(state)
            expected = mujoco.MjModel.from_xml_string(xml(.04)); data = mujoco.MjData(expected)
            data.qpos[:] = qpos[1].numpy(); mujoco.mj_forward(expected, data)
            np.testing.assert_allclose(renderer._data.xpos, data.xpos, atol=3e-8)
            np.testing.assert_allclose(renderer._model.geom_size, expected.geom_size, atol=3e-8)


if __name__ == "__main__":
    unittest.main()
