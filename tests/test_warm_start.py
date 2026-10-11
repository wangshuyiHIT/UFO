"""Small CPU models exercise full-state initialization without a simulator."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import safetensors.torch
import torch

from humanoidverse.agents.envs.utils.gym_spaces import space_to_json
from humanoidverse.training.checkpoint_io import file_digest
from humanoidverse.training.warm_start import assert_fresh_target, inspect_source, initialize_model, validate_interface


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in ('actor', 'backward_map', 'forward_map', 'critic', 'discriminator', 'aux_critic'):
            setattr(self, '_'+name, torch.nn.Linear(3, 2))
        self._target_forward_map = copy.deepcopy(self._forward_map)
        self._target_critic = copy.deepcopy(self._critic)
        self._obs_normalizer = torch.nn.BatchNorm1d(3, affine=False)
        self.register_buffer('_aux_reward_mean', torch.tensor([2.]))


def agent():
    model = TinyModel()
    obj = SimpleNamespace(_model=model, obs_space=gym.spaces.Dict(state=gym.spaces.Box(-np.inf, np.inf, (3,), np.float32)), action_dim=2)
    for name, module in (('actor','actor'), ('backward','backward_map'), ('forward','forward_map'),
                         ('critic','critic'), ('discriminator','discriminator'), ('aux_critic','aux_critic')):
        setattr(obj, name+'_optimizer', torch.optim.Adam(getattr(model, '_'+module).parameters(), lr=.004))
    return obj


def config():
    return {'agent': {'name':'FBcprAuxAgent', 'model':{'name':'tiny', 'device':'cpu'},
                     'train':{'discount':.98,'lr_actor':.004}, 'aux_rewards':['q'], 'aux_rewards_scaling':{'q':-1}},
            'env':{'robot_training':{'robot':{'control_joint_names':['a','b'], 'xml_path':'/old/robot.xml'},
                     'config_path':'/old/robot.yaml','action_scale':.25,
                     'morphology':{'family_names':['thigh','calf','torso','upperarm','forearm'],'observation_scale_m':.05}},
                   'name':'mjlab','hydra_overrides':['domain_rand.kp_range=[.9,1.1]']},
            'training_data_provenance':{'banks':[{'name':'new-data','weight':1}]}}


def source(tmp_path):
    old = agent()
    with torch.no_grad():
        for p in old._model.parameters(): p.add_(2)
        old._model._obs_normalizer.running_mean.fill_(3)
        old._model._obs_normalizer.num_batches_tracked.fill_(12)
    # Save a real nonempty Adam state; initialization must not consume it.
    old._model._actor(torch.ones(1,3)).sum().backward(); old.actor_optimizer.step()
    path = tmp_path/('generation-'+'a'*32); path.mkdir(); (path/'model').mkdir()
    cfg = config()
    payloads = {'config.json':cfg['agent'], 'effective_train_config.json':cfg,
                'init_kwargs.json':{'obs_space':space_to_json(old.obs_space),'action_dim':2},
                'model/config.json':cfg['agent']['model'],
                'train_status.json':{'global_time':123456,'optimizer_steps':222}}
    for name, value in payloads.items(): (path/name).write_text(json.dumps(value))
    torch.save({'actor_optimizer':old.actor_optimizer.state_dict()},path/'optimizers.pth')
    safetensors.torch.save_model(old._model,path/'model/model.safetensors')
    files={str(p.relative_to(path)):{'bytes':p.stat().st_size,'sha256':file_digest(p)} for p in path.rglob('*') if p.is_file()}
    (path/'COMMITTED.json').write_text(json.dumps({'version':1,'files':files}))
    spec={'generation':str(path),'manifest_sha256':file_digest(path/'COMMITTED.json'),
          'model_sha256':file_digest(path/'model/model.safetensors')}
    return old,path,spec


def test_complete_state_targets_normalizers_load_but_optimizers_remain_fresh(tmp_path):
    old,path,spec=source(tmp_path); before={str(p):file_digest(p) for p in path.rglob('*') if p.is_file()}
    new=agent(); inspected=inspect_source(spec); target=config()
    target['agent']['train']['lr_actor']=.001  # New optimizer settings are independent.
    target['env']['robot_training']['robot']['xml_path']='/new/robot.xml'
    receipt=initialize_model(new,inspected,target)
    for k,v in old._model.state_dict().items(): torch.testing.assert_close(v,new._model.state_dict()[k],rtol=0,atol=0)
    for name in ('actor','forward','backward','critic','discriminator','aux_critic'):
        assert not getattr(new,name+'_optimizer').state
    x=torch.randn(4,3)
    torch.testing.assert_close(old._model._actor(x),new._model._actor(x),rtol=0,atol=0)
    assert receipt['optimizers']=='fresh' and receipt['source_train_status']['optimizer_steps']==222
    assert before=={str(p):file_digest(p) for p in path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('change',['joint_order','family_order','scale','normalizer','reward'])
def test_equal_dimensions_with_different_semantics_are_rejected(tmp_path,change):
    _,_,spec=source(tmp_path); inspected=inspect_source(spec); target=config()
    r=target['env']['robot_training']
    if change=='joint_order':r['robot']['control_joint_names'].reverse()
    elif change=='family_order':r['morphology']['family_names'].reverse()
    elif change=='scale':r['morphology']['observation_scale_m']=.1
    elif change=='normalizer':target['agent']['model']['normalizer']='changed'
    else:target['agent']['aux_rewards_scaling']['q']=-2
    with pytest.raises(ValueError,match='semantics differ'):validate_interface(inspected,target)


def test_corrupt_or_extra_files_and_existing_target_fail_closed(tmp_path):
    _,path,spec=source(tmp_path)
    (path/'extra').mkdir()
    with pytest.raises(ValueError,match='uncommitted'):inspect_source(spec)
    (path/'extra').rmdir()
    with (path/'model/model.safetensors').open('r+b') as f:f.seek(-1,2);f.write(b'x')
    with pytest.raises(RuntimeError,match='digest mismatch'):inspect_source(spec)
    work=tmp_path/'new';work.mkdir();assert_fresh_target(work)
    (work/'training_progress.json').write_text('{}')
    with pytest.raises(ValueError,match='fresh training state'):assert_fresh_target(work)


def test_launch_receipt_is_sha_pinned_and_matches_committed_run(tmp_path):
    _,path,spec=source(tmp_path)
    effective=json.loads((path/'effective_train_config.json').read_text())
    effective['work_dir']='/original/run'
    (path/'effective_train_config.json').write_text(json.dumps(effective))
    manifest=json.loads((path/'COMMITTED.json').read_text())
    p=path/'effective_train_config.json'
    manifest['files'][p.name]={'bytes':p.stat().st_size,'sha256':file_digest(p)}
    (path/'COMMITTED.json').write_text(json.dumps(manifest))
    spec['manifest_sha256']=file_digest(path/'COMMITTED.json')
    receipt=tmp_path/'source_launch.json'
    payload={'schema':'scaler_ufo_stage_launch_v1','command':['python','--work-dir','/original/run'],
             'asset_contract':{'sha256':'fixture'}}
    receipt.write_text(json.dumps(payload))
    spec.update(launch_receipt=str(receipt),launch_receipt_sha256=file_digest(receipt))
    assert inspect_source(spec)['source_asset_contract']==payload['asset_contract']
    payload['command'][-1]='/different/run';receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='receipt path/SHA256'):inspect_source(spec)
    spec['launch_receipt_sha256']=file_digest(receipt)
    with pytest.raises(ValueError,match='committed source run'):inspect_source(spec)


def test_tensor_and_space_mismatch_or_existing_adam_state_rejected(tmp_path):
    _,_,spec=source(tmp_path); inspected=inspect_source(spec); new=agent()
    new.action_dim=3
    with pytest.raises(ValueError,match='spaces differ'):initialize_model(new,inspected,config())
    new=agent();new._model._actor(torch.ones(1,3)).sum().backward();new.actor_optimizer.step()
    with pytest.raises(ValueError,match='empty optimizers'):initialize_model(new,inspected,config())
    new=agent();new._model.register_buffer('unexpected',torch.tensor(0))
    with pytest.raises(ValueError,match='keys/shapes'):initialize_model(new,inspected,config())


def test_real_aux_agent_workspace_initialization_has_fresh_clocks_and_context(tmp_path):
    from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
    from humanoidverse.distributed import agent_optimizers
    from humanoidverse.training.workspace import TrainConfig, create_agent_or_load_checkpoint
    from humanoidverse.agents.envs.humanoidverse_mjlab import HumanoidVerseMjlabConfig
    # Actual production classes, tiny networks and CPU only. No simulator,
    # model-size production allocation, legacy BaseModel.load or optimizer load.
    raw = FBcprAuxAgentConfig().model_dump()
    raw['model']['device'] = 'cpu'
    raw['model']['archi']['z_dim'] = 4
    for name in ('actor', 'b', 'f', 'critic', 'aux_critic', 'discriminator'):
        raw['model']['archi'][name]['hidden_dim'] = 8
        raw['model']['archi'][name]['hidden_layers'] = 1
    raw['model']['obs_normalizer']['normalizers'] = {'name': 'BatchNormNormalizerConfig'}
    raw['train'].update(batch_size=4, z_buffer_size=8)
    cfg = FBcprAuxAgentConfig.model_validate(raw)
    space = gym.spaces.Box(-np.inf, np.inf, (3,), np.float32)
    old = cfg.build(space, 2)
    with torch.no_grad():
        old._model._obs_normalizer(torch.randn(4, 3) + 5)
        next(old._model._target_forward_map.parameters()).add_(.5)
    old.z_buffer.add(torch.ones(4, 4))
    # One actual Adam update makes the old optimizer nonempty.
    loss = sum(p.sum() for p in old._model._actor.parameters())
    loss.backward(); old.actor_optimizer.step()
    generation = tmp_path / ('generation-' + 'c' * 32)
    old.save(generation)
    target = TrainConfig(agent=cfg, env=HumanoidVerseMjlabConfig(
        lafan_tail_path='new-expert.pkl', device='cpu', robot_training=config()['env']['robot_training']))
    effective = target.model_dump()
    (generation / 'effective_train_config.json').write_text(json.dumps(effective))
    (generation / 'train_status.json').write_text(json.dumps({'global_time': 9999, 'optimizer_steps': 8}))
    files = {str(p.relative_to(generation)): {'bytes': p.stat().st_size, 'sha256': file_digest(p)}
             for p in generation.rglob('*') if p.is_file()}
    (generation / 'COMMITTED.json').write_text(json.dumps({'version': 1, 'files': files}))
    spec = {'generation': str(generation), 'manifest_sha256': file_digest(generation / 'COMMITTED.json'),
            'model_sha256': files['model/model.safetensors']['sha256']}
    target = target.model_copy(update={'warm_start': spec})
    new, updated, status = create_agent_or_load_checkpoint(tmp_path / 'fresh', target,
                                                          {'obs_space': space, 'action_dim': 2})
    assert all(status[k] == 0 for k in ('local_time', 'global_time', 'optimizer_steps'))
    assert updated.warm_start is None and updated.initialization_provenance['source_train_status']['global_time'] == 9999
    assert target.warm_start == spec  # Frozen config copied, not mutated.
    assert new.z_buffer.empty() and new.env_idx_with_expert_rollout is None
    assert all(not optimizer.state for optimizer in agent_optimizers(new))
    for key, value in old._model.state_dict().items():
        torch.testing.assert_close(value, new._model.state_dict()[key], rtol=0, atol=0)
    old._model.eval(); new._model.eval()
    x, z = torch.randn(4, 3), torch.randn(4, 4)
    torch.testing.assert_close(old._model.act(x, z, mean=True), new._model.act(x, z, mean=True), rtol=0, atol=0)
