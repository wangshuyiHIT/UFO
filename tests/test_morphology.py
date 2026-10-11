"""CPU contract checks; these deliberately do not claim online physics support."""
from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from humanoidverse.utils.morphology import (
    append_morphology_state, fk_motion_with_morphology, morphology_offsets,
    motion_morphology_delta, profile_properties, validate_morphology_config,
)


def config():
    return {"enabled": True, "family_names": ["thigh", "calf", "torso", "upperarm", "forearm"],
            "family_bounds_m": [[-.05, .05], [-.05, .05], [-.005, .05], [-.01, .05], [-.05, .05]],
            "observation_scale_m": .05,
            "body_offsets": [{"body_name": "child", "family_index": 0, "axis": [0, 0, -1]}],
            "profiles": [profile()]}


def profile():
    return {"nominal_length_m": .2, "section_xy_m": [.02, .04], "axis": [0, 0, 1],
            "nominal_center_m": [0, 0, -.1], "linear_density_kg_m": 2.,
            "carrier_mass_kg": 1., "carrier_com_m": [0, 0, 0],
            "rotation_matrix": np.eye(3).tolist(), "carrier_inertia_kg_m2": (np.eye(3)*.001).tolist()}


class MorphologyTest(unittest.TestCase):
    def test_synthetic_static_is_explicit_and_not_human_retargeting(self):
        metadata = dict(morphology_delta_m=[0]*5, morphology_reference="synthetic_static",
                        source_kind="synthetic_constant", reference_kind="synthetic_static", synthetic=True)
        np.testing.assert_array_equal(motion_morphology_delta({"metadata":metadata}, config()), np.zeros(5))
        for field in ('source_kind','reference_kind','synthetic'):
            bad={k:v for k,v in metadata.items() if k!=field}
            with self.assertRaisesRegex(ValueError,'explicit'):
                motion_morphology_delta({'metadata':bad},config())

    def test_requires_retargeted_finite_bounded_metadata(self):
        cfg = validate_morphology_config(config())
        motion = {"metadata": {"morphology_delta_m": [.03, 0, 0, 0, 0], "morphology_reference": "retargeted"}}
        np.testing.assert_allclose(motion_morphology_delta(motion, cfg), [.03, 0, 0, 0, 0])
        for invalid in ([0, 0, -.01, 0, 0], [float('nan')]*5, [0]*4):
            with self.assertRaises(ValueError):
                motion_morphology_delta({"metadata": {**motion["metadata"], "morphology_delta_m": invalid}}, cfg)
        with self.assertRaisesRegex(ValueError, "not retargeting"):
            motion_morphology_delta({"metadata": {"morphology_delta_m": [0]*5}}, cfg)

    def test_fk_offsets_preserve_nominal_and_other_bodies(self):
        offsets = torch.tensor([[[0., 0, 0], [0, 0, -.2], [1, 0, 0]]])
        actual = morphology_offsets(offsets, ["root", "child", "sibling"], config(), np.array([.03, 0, 0, 0, 0]))
        self.assertAlmostEqual(float(actual[0, 1, 2]), -.23, places=6)
        self.assertTrue(torch.equal(actual[:, [0, 2]], offsets[:, [0, 2]]))
        self.assertAlmostEqual(float(offsets[0, 1, 2]), -.2, places=6)

    def test_online_and_reference_token_contract(self):
        delta = torch.tensor([[.02, -.03, .005, 0, .05], [0, 0, 0, 0, 0]])
        env = SimpleNamespace(morphology_config=config(), morphology_delta_m=delta)
        state = torch.zeros(2, 54)
        self.assertTrue(torch.equal(append_morphology_state(env, state), append_morphology_state(env, state, delta)))
        self.assertEqual(append_morphology_state(env, state).shape, (2, 59))
        self.assertTrue(torch.allclose(append_morphology_state(env, state)[:, -5:], delta/.05))
        self.assertIs(append_morphology_state(SimpleNamespace(), state), state)

    def test_profile_mass_inertia_matches_composite_parallel_axis(self):
        p = profile()
        result = profile_properties(p, .04)
        self.assertAlmostEqual(result["mass"], 1.48)
        np.testing.assert_allclose(result["center"], [0, 0, -.12])
        np.testing.assert_allclose(result["size"], [.01, .02, .12])
        np.testing.assert_allclose(result["com"], [0, 0, -.12*.48/1.48])
        from scipy.spatial.transform import Rotation
        r = Rotation.from_quat(result["iquat"][[1, 2, 3, 0]]).as_matrix()
        actual = r @ np.diag(result["inertia"]) @ r.T
        expected_x = .001 + .48/12*(.04**2+.24**2) + (1*.48/1.48)*.12**2
        self.assertAlmostEqual(actual[0, 0], expected_x)
        self.assertGreater(np.min(np.linalg.eigvalsh(actual)), 0)
        with self.assertRaises(ValueError):
            profile_properties(p, -.2)

    def test_fk_restores_nominal_offsets_even_on_failure(self):
        class Parser:
            body_names = ["root", "child"]
            _offsets = torch.zeros(1, 2, 3)
            def fk_batch(self, *args, **kwargs):
                self.seen = self._offsets.clone()
                raise RuntimeError("intentional FK failure")
        parser = Parser()
        nominal = parser._offsets
        motion = {"metadata": {"morphology_delta_m": [.03, 0, 0, 0, 0], "morphology_reference": "retargeted"},
                  "pose_aa": np.zeros((3, 2, 3)), "root_trans_offset": np.zeros((3, 3)), "fps": 50}
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            fk_motion_with_morphology(parser, motion, config())
        self.assertIs(parser._offsets, nominal)
        self.assertAlmostEqual(float(parser.seen[0, 1, 2]), -.03)


if __name__ == "__main__":
    unittest.main()
