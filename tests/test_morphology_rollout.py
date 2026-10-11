"""Expert rollout contexts must refer to the online robot's actual shape."""
import unittest
from types import SimpleNamespace

import torch

from humanoidverse.agents.buffers.trajectory import TrajectoryDictBuffer
from humanoidverse.agents.fb.agent import FBAgent


def buffer():
    episodes = []
    for index, value in enumerate((0., .03)):
        delta = torch.tensor([value,0,0,0,0]).repeat(12,1)
        state = torch.stack((delta[:,0], index*100 + torch.arange(12).float()), -1)
        episodes.append({"observation":{"state":state}, "morphology_delta_m":delta,
                         "motion_id":torch.full((12,),index), "terminated":torch.zeros(12,dtype=torch.bool)})
    return TrajectoryDictBuffer(episodes, device='cpu')


def agent(percentage=1.):
    obj = object.__new__(FBAgent)
    obj._model = SimpleNamespace(device='cpu', sample_z=lambda n,device:torch.zeros(n,2),
        backward_map=lambda obs:obs['state'].clone(), project_z=lambda z:z)
    obj.cfg = SimpleNamespace(train=SimpleNamespace(use_mix_rollout=False, update_z_every_step=10,
        rollout_expert_trajectories=True, rollout_expert_trajectories_percentage=percentage,
        rollout_expert_trajectories_length=3),model=SimpleNamespace(seq_length=1))
    return obj


class MorphologyRolloutTest(unittest.TestCase):
    def test_sequence_sampling_matches_shape_and_preserves_complete_order(self):
        b = buffer()
        requested = torch.tensor([[.03,0,0,0,0],[0,0,0,0,0],[.03,0,0,0,0],[0,0,0,0,0]])
        for _ in range(8):
            sample = b.sample(12,seq_length=3,morphology_delta_m=requested)
            values = sample['observation']['state'].reshape(4,3,2)
            torch.testing.assert_close(values[:,:,0],requested[:,0,None].expand(-1,3))
            torch.testing.assert_close(values[:,1:,1]-values[:,:-1,1],torch.ones(4,2))
            torch.testing.assert_close(sample['next']['observation']['state'][:,1]-sample['observation']['state'][:,1],torch.ones(12))
        with self.assertRaisesRegex(ValueError,'No expert trajectory'):
            b.sample(3,seq_length=3,morphology_delta_m=torch.tensor([[.01,0,0,0,0]]))
        with self.assertRaisesRegex(ValueError,'long enough'):
            b.sample(15,seq_length=15,morphology_delta_m=requested[:1])

    def test_shape_changes_and_resets_only_resample_affected_envs(self):
        a,b = agent(),buffer()
        delta = torch.tensor([[0.,0,0,0,0],[.03,0,0,0,0],[0.,0,0,0,0]])
        z0 = a.maybe_update_rollout_context(None,torch.zeros(3,1),{'expert_slicer':b},morphology_delta_m=delta)
        z1 = a.maybe_update_rollout_context(z0,torch.ones(3,1),{'expert_slicer':b},morphology_delta_m=delta)
        torch.testing.assert_close(z1[:,1],z0[:,1]+1)
        torch.testing.assert_close(z1[:,0],delta[:,0])
        delta[0,0] = .03
        z2 = a.maybe_update_rollout_context(z1,torch.full((3,1),2),{'expert_slicer':b},morphology_delta_m=delta)
        torch.testing.assert_close(z2[1:,1],z0[1:,1]+2)
        torch.testing.assert_close(z2[:,0],delta[:,0])
        self.assertEqual(len(set(a.env_idx_with_expert_rollout.tolist())),3)
        # Reset environment 0, while the others have not advanced: their latent
        # frame must not jump to a newly sampled trajectory.
        z3 = a.maybe_update_rollout_context(z2,torch.tensor([[0],[2],[2]]),{'expert_slicer':b},morphology_delta_m=delta)
        torch.testing.assert_close(z3[1:],z2[1:])

    def test_expert_subset_has_no_duplicate_environment_ids(self):
        a,b = agent(.5),buffer()
        delta = torch.zeros(6,5)
        a.maybe_update_rollout_context(None,torch.zeros(6,1),{'expert_slicer':b},morphology_delta_m=delta)
        self.assertEqual(len(a.env_idx_with_expert_rollout),3)
        self.assertEqual(len(set(a.env_idx_with_expert_rollout.tolist())),3)

    def test_episode_cannot_change_shape_partway(self):
        ep={'observation':{'state':torch.zeros(5,2)},'motion_id':torch.zeros(5,dtype=torch.long),
            'morphology_delta_m':torch.zeros(5,5),'terminated':torch.zeros(5,dtype=torch.bool)}
        ep['morphology_delta_m'][2,0]=.03
        with self.assertRaisesRegex(ValueError,'constant'):
            TrajectoryDictBuffer([ep])


if __name__=='__main__':unittest.main()
