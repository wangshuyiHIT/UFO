"""Exercise real expert/reference construction with distinct source morphologies."""
from __future__ import annotations

from types import SimpleNamespace as NS
import unittest

import torch

from humanoidverse.agents.envs.expert_motion_loader import load_expert_trajectories_from_motion_lib
from humanoidverse.agents.evaluations.humanoidverse_mjlab import get_backward_observation as eval_reference
from humanoidverse.goal_inference import _target_states_from_obs as goal_target
from humanoidverse.tracking_inference import _target_states_from_obs as tracking_target
from humanoidverse.utils.helpers import get_backward_observation as helper_reference


class MotionLibrary:
    _num_unique_motions = 2
    _motion_lengths = torch.tensor([.2, .2])
    _motion_data_keys = ["compact", "extended"]

    def __init__(self, include_delta=True):
        self.include_delta = include_delta
        self.deltas = torch.tensor([[-.05, -.05, -.005, -.01, -.05], [.05]*5])

    def load_motions_for_training(self):
        pass

    def get_motion_state(self, ids, times):
        ids = torch.as_tensor(ids, dtype=torch.long).expand_as(times)
        count = len(times)
        pos = torch.zeros(count, 3, 3)
        pos[:, :, 2] = 1.
        rot = torch.zeros(count, 3, 4)
        rot[:, :, 3] = 1.
        velocity = torch.zeros_like(pos)
        result = dict(rg_pos_t=pos, rg_rot_t=rot, body_vel_t=velocity,
                      body_ang_vel_t=velocity, dof_pos=torch.zeros(count, 24),
                      dof_vel=torch.zeros(count, 24), root_pos=pos[:, 0],
                      root_rot=rot[:, 0], root_vel=velocity[:, 0], root_ang_vel=velocity[:, 0])
        if self.include_delta:
            result["morphology_delta_m"] = self.deltas[ids]
        return result


def environment(enabled=True, include_delta=True):
    obs = NS(use_obs_filter=True, root_height_obs=True, obs_scales={"base_ang_vel": .25},
             obs_auxiliary={"history_actor": {}}, obs_dims={})
    return NS(config=NS(obs=obs), device="cpu", dt=.02,
              default_dof_pos=torch.zeros(1, 24), gravity_vec=torch.tensor([[0., 0., -1.]]),
              use_contact_in_obs_max=False, _motion_lib=MotionLibrary(include_delta),
              morphology_config={"family_names": ["thigh", "calf", "torso", "upperarm", "forearm"],
                                 "family_bounds_m": [[-.05,.05],[-.05,.05],[-.005,.05],[-.01,.05],[-.05,.05]],
                                 "observation_scale_m": .05} if enabled else None,
              # Deliberately different from both source motions: this must not
              # leak into expert/reference observations.
              morphology_delta_m=torch.tensor([[.011, -.013, .002, .003, -.004]]))


class MorphologyReferencePathsTest(unittest.TestCase):
    def test_expert_and_both_reference_paths_agree_and_use_source_shape(self):
        env = environment()
        cfg = NS(model=NS(seq_length=1))
        expert = load_expert_trajectories_from_motion_lib(env, cfg)
        for motion_id in (0, 1):
            evaluation, ref = eval_reference(env, motion_id, include_last_action=True)
            helper, _ = helper_reference(env, motion_id, use_root_height_obs=True)
            rows = expert.storage["motion_id"] == motion_id
            for key in ("state", "privileged_state", "last_action"):
                torch.testing.assert_close(expert.storage["observation"][key][rows], evaluation[key])
                torch.testing.assert_close(helper[key], evaluation[key])
            for key in ("state", "privileged_state"):
                expected = (env._motion_lib.deltas[motion_id]/.05).expand(len(evaluation[key]), -1)
                torch.testing.assert_close(evaluation[key][:, -5:], expected)
            self.assertEqual(evaluation["state"].shape[-1], 59)
            for reset_target in (goal_target, tracking_target):
                target = reset_target(ref, "cpu", num_dof=24)
                torch.testing.assert_close(target["morphology_delta_m"][0], env._motion_lib.deltas[motion_id])
                self.assertEqual(target["motion_ids"].tolist(), [motion_id])
                self.assertEqual(target["motion_start_times"].tolist(), [0.])

    def test_missing_source_metadata_fails_closed(self):
        env = environment(include_delta=False)
        for construct in (lambda: eval_reference(env, 0, True),
                          lambda: helper_reference(env, 0),
                          lambda: load_expert_trajectories_from_motion_lib(env, NS(model=NS(seq_length=1)))):
            with self.assertRaises(KeyError):
                construct()

    def test_disabled_robot_keeps_original_observation_contract(self):
        env = environment(enabled=False, include_delta=False)
        expert = load_expert_trajectories_from_motion_lib(env, NS(model=NS(seq_length=1)))
        observation, reference = eval_reference(env, 0, True)
        self.assertEqual(observation["state"].shape[-1], 54)
        self.assertEqual(expert.storage["observation"]["state"].shape[-1], 54)
        self.assertNotIn("morphology_delta_m", reference)
        self.assertNotIn("morphology_delta_m", tracking_target(reference, "cpu", num_dof=24))


if __name__ == "__main__":
    unittest.main()
