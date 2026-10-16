"""Closed, opt-in restore policy. A pinned contract is evidence, not run authorization."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import re
from pydantic import BaseModel, ConfigDict, StrictStr, Field

FIELD = 'freeze_obs_normalizer_stats'
BINDING = 'obs_normalizer_resume_policy'
SCHEMA = 'scaler_ufo_obs_normalizer_resume_policy_v1'
FULL9 = {'config.json','distributed_sync.json','effective_train_config.json','init_kwargs.json',
         'model/config.json','model/init_kwargs.json','model/model.safetensors','optimizers.pth','train_status.json'}
OPTIMIZERS = {'actor_optimizer','backward_optimizer','forward_optimizer','critic_optimizer',
              'discriminator_optimizer','aux_critic_optimizer'}
CONTRACT_KEYS = {'schema','scope','source','target_implementation','target_effective_sha256',
                 'source_policy','target_policy','comparison'}

class PolicyBinding(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    path: StrictStr
    sha256: StrictStr = Field(pattern=r'^[0-9a-f]{64}$')

def need(value, message):
    if not value: raise ValueError(message)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def load_pin(pin):
    need(isinstance(pin,dict) and set(pin)=={'path','sha256'},'Pin keys differ')
    p=Path(pin['path']);need(p.is_absolute() and not p.is_symlink() and p.is_file(),'Pin path differs')
    need(type(pin['sha256']) is str and re.fullmatch('[0-9a-f]{64}',pin['sha256']) is not None,'Pin digest type differs')
    need(digest(p)==pin['sha256'],'Pin digest differs: '+str(p))
    return json.loads(p.read_text())

def policy(agent):
    need(isinstance(agent,dict) and agent.get('name')=='FBcprAuxAgent','Policy supports exact FBcprAuxAgent only')
    value=agent['train'].get(FIELD,False)
    need(type(value) is bool,'Observation policy must be a strict boolean')
    return value

def normalized_agent(agent):
    result=copy.deepcopy(agent);policy(result);result['train'].pop(FIELD,None);return result

def normalized_configs(source,target):
    """Only policy/admin binding normalization; old comparator handles everything else."""
    a,b=copy.deepcopy(source),copy.deepcopy(target)
    old,new=policy(a['agent']),policy(b['agent'])
    need(new is True and old in (False,True),'Only explicit false-to-true or true-to-true is supported')
    need(normalized_agent(a['agent'])==normalized_agent(b['agent']),'Other agent/model/Adam configuration differs')
    a['agent']=normalized_agent(a['agent']);b['agent']=normalized_agent(b['agent'])
    a.pop(BINDING,None);b.pop(BINDING,None)
    return a,b

def target_payload(cfg):
    result=copy.deepcopy(cfg);result.pop(BINDING,None);return result

def implementation_paths():
    root=Path(__file__).resolve().parents[1]
    from scaler_ufo_adapter import obs_normalizer_resume, dr_migration, h20_ee8_update16_resume
    return [Path(__file__).resolve(),root/'training/workspace.py',root/'distributed.py',root/'training/obs_normalizer_sync.py',root/'training/obs_rank_contract.py',root/'training/obs_rank_evidence.py',root/'train.py',root/'freeze_train.py',root/'agents/fb/agent.py',root/'agents/fb_cpr_aux/agent.py',Path(obs_normalizer_resume.__file__).resolve(),Path(dr_migration.__file__).resolve(),Path(h20_ee8_update16_resume.__file__).resolve(),Path(obs_normalizer_resume.__file__).with_name('freeze_training_entry.py'),Path(obs_normalizer_resume.__file__).with_name('freeze_source_gate.py'),Path(obs_normalizer_resume.__file__).with_name('freeze_segment.py'),Path(obs_normalizer_resume.__file__).with_name('freeze_retained.py')]

def implementation_pins():return [{'path':str(p),'sha256':digest(p)} for p in implementation_paths()]

def verify_source(source,checkpoint):
    need(isinstance(source,dict) and set(source)=={'generation','marker','lineage','files'},'Source keys differ')
    p=Path(checkpoint).resolve(strict=True)
    need(str(p)==source['generation'] and p.name.startswith('generation-'),'Source generation differs')
    need(source['marker']['path']==str(p/'COMMITTED.json'),'Source marker path differs')
    marker=load_pin(source['marker'])
    need(marker.get('version')==1 and set(marker.get('files',{}))==FULL9,'Source full9 closure differs')
    need(source['files']==marker['files'],'Source file identities differ')
    actual={str(x.relative_to(p)) for x in p.rglob('*') if x.is_symlink() or not x.is_dir()}
    need(actual==FULL9|{'COMMITTED.json'},'Source tree differs')
    for name,record in source['files'].items():
        file=p/name
        need(not file.is_symlink() and file.is_file() and file.stat().st_size==record['bytes'],'Source file type/size differs')
        need(digest(file)==record['sha256'],'Source file digest differs: '+name)
    lineage=load_pin(source['lineage'])
    need(set(lineage)=={'schema','scope','source_parent_commit','source_child_commit','generation','marker_sha256'},'Source lineage keys differ')
    need(lineage['schema']=='scaler_ufo_obs_policy_source_lineage_v1','Source lineage schema differs')
    need(lineage['generation']==str(p) and lineage['marker_sha256']==source['marker']['sha256'],'Source lineage differs')
    for name in ('source_parent_commit','source_child_commit'):
        need(type(lineage[name]) is str and re.fullmatch('[0-9a-f]{40}',lineage[name]) is not None,'Source code identity differs')
    config=json.loads((p/'config.json').read_text());effective=json.loads((p/'effective_train_config.json').read_text())
    need(effective['agent']==config,'Source saved agent and effective configuration differ')
    return config,effective,lineage

def verify_contract(binding,checkpoint,target_cfg):
    binding=PolicyBinding.model_validate(binding).model_dump()
    contract=load_pin(binding)
    from . import obs_rank_contract as ranked
    is_ranked=contract.get('schema') in (ranked.SCHEMA,ranked.ENTRY_SCHEMA)
    if is_ranked: ranked.verify_static(contract)
    else: need(set(contract)==CONTRACT_KEYS and contract['schema']==SCHEMA,'Policy contract schema/keys differ')
    need(contract['scope'] in ('synthetic_cpu_fixture','full_checkpoint_preparation'),'Policy scope differs')
    need(contract['target_implementation']==implementation_pins(),'Target implementation differs')
    config,effective,lineage=verify_source(contract['source'],checkpoint)
    need(lineage['scope']==contract['scope'],'Source lineage scope differs')
    need(type(contract['source_policy']) is bool and type(contract['target_policy']) is bool,'Contract policy type differs')
    need(contract['source_policy'] is policy(config) and contract['target_policy'] is True,'Source/target policy differs')
    need(policy(target_cfg['agent']) is True,'Target effective policy differs')
    need(normalized_agent(config)==normalized_agent(target_cfg['agent']),'Target changes another agent/model/Adam field')
    if is_ranked:
        comparison_target,_,_=ranked.verify_rank_target(contract,target_cfg)
        status=json.loads((Path(checkpoint)/'train_status.json').read_text())
        need(contract['source_baseline']['start_coordinate']=={k:status[k] for k in ('optimizer_steps','local_time','global_time')},'Source baseline coordinate differs')
    else:
        need(canonical(target_payload(target_cfg))==contract['target_effective_sha256'],'Target effective configuration differs')
        comparison_target=target_cfg
    need(target_cfg.get('warm_start') is None and target_cfg.get('transactional_checkpoints') is True,'Policy requires transactional normal resume')
    from scaler_ufo_adapter.obs_normalizer_resume import verify_comparison
    verify_comparison(effective,comparison_target,contract['comparison'])
    return contract

def checked_load_config(path,loaded_config,authorization,*,return_contract=False):
    need(type(authorization) is dict and set(authorization)=={'binding','target_effective'},'Load authorization keys differ')
    c=verify_contract(authorization['binding'],path,authorization['target_effective'])
    original=json.loads((Path(path)/'config.json').read_text())
    need(loaded_config==original,'Loaded config changed before policy authorization')
    result=copy.deepcopy(loaded_config);result['train'][FIELD]=c['target_policy']
    return (result,c) if return_contract else result

def verify_restored(agent,target_cfg):
    need(agent.cfg.model_dump()==target_cfg['agent'],'Restored agent differs from target effective config')
    need(policy(agent.cfg.model_dump()) is True,'Restored policy was not applied')
    need(set(agent.optimizer_dict)==OPTIMIZERS,'Restored optimizer closure differs')

def verify_no_unbound_change(path,target):
    saved=json.loads((Path(path)/'config.json').read_text())
    need(policy(saved) is False and policy(target['agent']) is False,'Policy change or frozen-policy continuation requires an explicit contract')
