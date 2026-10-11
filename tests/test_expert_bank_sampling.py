import pytest
import torch

from humanoidverse.agents.buffers.trajectory import TrajectoryDictBuffer, get_idxs


def buffer(monkeypatch, labels=('base','base','independent','standing')):
    monkeypatch.setattr(torch,'compile',lambda f,**_:f)
    episodes=[]
    for i in range(4):
        episodes.append({'observation':{'state':torch.stack((torch.full((12,),float(i)),torch.arange(12).float()),-1)},
            'motion_id':torch.full((12,),i),'terminated':torch.zeros(12,dtype=torch.bool),
            'morphology_delta_m':torch.tensor([0 if i in (0,2,3) else .03,0,0,0,0]).repeat(12,1)})
    return TrajectoryDictBuffer(episodes,trajectory_source_labels=list(labels),
                                source_weights={'base':.75,'independent':.20,'standing':.05})


def test_prioritization_preserves_bank_mass_and_zero_bank_falls_back(monkeypatch):
    b=buffer(monkeypatch)
    for priority in (torch.tensor([100.,1,10000,0]),torch.zeros(4)):
        b.update_priorities(priority,[0,1,2,3])  # Workspace prioritization sends a list.
        torch.testing.assert_close(torch.stack([b.priorities[:2].sum(),b.priorities[2],b.priorities[3]]),torch.tensor([.75,.20,.05]))
    with pytest.raises(ValueError,match='Invalid expert'):b.update_priorities(torch.tensor([1.,-1,1,1]),torch.arange(4))


def test_sampling_uses_configured_mass_and_conditional_available_banks(monkeypatch):
    b=buffer(monkeypatch); b.update_priorities(torch.tensor([1.,8,500.,.001]),torch.arange(4))
    captured=[]
    def sample(*args,**kwargs):
        captured.append(kwargs.get('priorities') if kwargs else args[-1])
        return get_idxs(*args,**kwargs)
    b._get_idxs=sample
    b.sample(12,seq_length=3)
    torch.testing.assert_close(captured[0],b.priorities)
    # Shape zero excludes base episode 1; original base mass remains .75.
    selected=b._source_priorities(torch.tensor([True,False,True,True]))
    torch.testing.assert_close(selected,torch.tensor([.75,.20,.05]))
    # A shape absent from the independent bank renormalizes available mass.
    torch.testing.assert_close(b._source_priorities(torch.tensor([True,False,False,True])),
                               torch.tensor([.9375,.0625]))
    out=b.sample(6,seq_length=3,morphology_delta_m=torch.tensor([[.03,0,0,0,0]]).repeat(2,1))
    assert bool((out['observation']['state'][:,0]==1).all())
    assert bool((out['next']['observation']['state'][:,1]-out['observation']['state'][:,1]==1).all())
    with pytest.raises(ValueError,match='known label'):buffer(monkeypatch,('bad','base','independent','standing'))


def test_existing_online_motion_library_also_retains_bank_mass():
    from humanoidverse.utils.motion_lib.motion_lib_base import MotionLibBase
    lib = object.__new__(MotionLibBase)
    lib._device = 'cpu'; lib._num_unique_motions = 4
    lib._motion_source_ids = torch.tensor([0, 0, 1, 2])
    lib._motion_source_weights = torch.tensor([.75, .2, .05])
    for raw in (torch.tensor([100., 1, 5000, .001]), torch.zeros(4)):
        p = lib._source_mixed_prob(raw)
        torch.testing.assert_close(torch.stack((p[:2].sum(),p[2],p[3])),torch.tensor([.75,.2,.05]))


def test_no_source_configuration_retains_legacy_priority_behavior(monkeypatch):
    monkeypatch.setattr(torch,'compile',lambda f,**_:f)
    episodes = [{'observation': {'state': torch.full((8,1), float(i))},
                 'motion_id': torch.full((8,), i), 'terminated': torch.zeros(8, dtype=torch.bool)} for i in range(2)]
    b = TrajectoryDictBuffer(episodes)
    b.update_priorities(torch.tensor([1., 9.]), torch.tensor([0, 1]))
    torch.testing.assert_close(b.priorities, torch.tensor([.1, .9]))
    assert b.source_weights is None


@pytest.mark.parametrize('weight', [0., -1., float('nan')])
def test_invalid_bank_probability_is_rejected(monkeypatch, weight):
    monkeypatch.setattr(torch,'compile',lambda f,**_:f)
    episode = {'observation': {'state': torch.zeros(8,1)}, 'motion_id': torch.zeros(8),
               'terminated': torch.zeros(8,dtype=torch.bool)}
    with pytest.raises(ValueError, match='positive finite weights'):
        TrajectoryDictBuffer([episode], trajectory_source_labels=['bad'], source_weights={'bad':weight})
