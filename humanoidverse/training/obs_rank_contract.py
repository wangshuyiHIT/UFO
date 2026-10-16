"""Exact single-node rank projection; source creation and candidate base are distinct."""
from __future__ import annotations
import copy,json,os,re,socket
from pathlib import Path
import torch
import torch.distributed as dist
from . import obs_normalizer_resume as p
SCHEMA='scaler_ufo_obs_normalizer_rank_policy_v2'
ENTRY_SCHEMA='scaler_ufo_obs_normalizer_training_rank_policy_v3'
SOURCE_PARENT='b54305e405f55fb71a0bd91cee5b8fd560addabe'
SOURCE_CHILD='d563c95c17055d05101d48de867fa61684966d27'
BASE_PARENT='a12c10c3a3f8db4a23f319f1f6e7cc8ed167afd7'
BASE_CHILD='f7d924f60433f6cc26e5102b19ce5b87a98e668f'
LAUNCH_SHA='3a69c366361a78a2808a3b3408d28b01bf4a672293fd06741e93120fa4360d8b'
MARKER_SHA='42bad96dd2f9f576d806589997f0f0a5a2f2dd209945004f215b9301fb4c6cca'
GENERATION='generation-ee8bf4079b4342599d3bcaec9483ac5e'
RANK_KEYS={'schema','canonical_effective','launch_intent','target_base_parent','target_base_child'}
INTENT_KEYS={'schema','scope','session_id','run_id','base_seed','use_wandb_cli','world_size','local_rank_by_rank','cuda_visible_devices'}
EVIDENCE_KEYS={'schema','session_id','run_id','output_root','first_updates','max_record_bytes','max_total_bytes_per_rank'}
ORIGIN_KEYS={'schema','mode','launch','source_parent','source_child'}

def integer(x,label,minimum=0):p.need(type(x) is int and x>=minimum,'Invalid integer '+label)
def id_text(x):return type(x) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,95}',x) is not None

def validate_projection(projection):
    p.need(set(projection)==RANK_KEYS and projection['schema']=='scaler_ufo_obs_rank_projection_v1','Projection schema differs')
    p.need(projection['target_base_parent']==BASE_PARENT and projection['target_base_child']==BASE_CHILD,'Target base source differs; no new Git commit claimed')
    canonical=p.load_pin(projection['canonical_effective']);intent=p.load_pin(projection['launch_intent'])
    p.need(set(intent)==INTENT_KEYS and intent['schema']=='scaler_ufo_obs_rank_launch_intent_v1','Launch intent schema differs')
    p.need(intent['scope'] in ('single_node_cuda','synthetic_cpu_gloo'),'Projection scope differs')
    integer(intent['world_size'],'world',1);world=intent['world_size'];integer(intent['base_seed'],'base seed')
    p.need(type(intent['use_wandb_cli']) is bool and id_text(intent['session_id']) and id_text(intent['run_id']),'Launch intent values differ')
    p.need(intent['local_rank_by_rank']==list(range(world)) and all(type(x) is int for x in intent['local_rank_by_rank']),'Only exact single-node rank mapping supported')
    p.need(canonical['distributed_rank']==0 and type(canonical['distributed_rank']) is int and canonical['tags']['distributed_rank']==0 and type(canonical['tags']['distributed_rank']) is int,'Canonical configuration is not rank0')
    for v in (canonical['distributed_world_size'],canonical['tags']['distributed_world_size'],canonical['seed'],canonical['env']['seed']):integer(v,'canonical rank/seed value')
    p.need(canonical['distributed_world_size']==world and canonical['tags']['distributed_world_size']==world,'Canonical world differs')
    p.need(canonical['seed']==intent['base_seed'] and canonical['env']['seed']==intent['base_seed'],'Base seed differs from pinned canonical config')
    p.need(canonical['use_wandb'] is intent['use_wandb_cli'],'Canonical rank0 differs from separately pinned original wandb flag')
    p.need(canonical['distributed_sync'] is True and canonical['rank0_only_writes'] is True,'Ranked policy requires original shared writer/sync')
    if intent['scope']=='single_node_cuda':
        p.need(world==8,'Production projection requires world8')
        p.need(type(intent['cuda_visible_devices']) is str and len(intent['cuda_visible_devices'].split(','))==8 and len(set(intent['cuda_visible_devices'].split(',')))==8 and all(re.fullmatch('0|[1-9][0-9]*|GPU-[A-Za-z0-9-]+',x) for x in intent['cuda_visible_devices'].split(',')),'CUDA visibility mapping differs')
        p.need(canonical['env']['device']=='cuda:0' and canonical['agent']['model']['device']=='cuda' and canonical['buffer_device']=='cuda','Canonical CUDA device strings differ')
    else:
        p.need(world in (1,2,8) and intent['cuda_visible_devices']=='','Synthetic CPU scope differs')
        p.need(canonical['env']['device']==canonical['agent']['model']['device']==canonical['buffer_device']=='cpu','Synthetic devices must remain CPU')
    return canonical,intent

def project(canonical,intent,rank,local_rank):
    integer(rank,'rank');integer(local_rank,'local rank');p.need(rank<intent['world_size'] and local_rank==intent['local_rank_by_rank'][rank],'Rank mapping differs')
    result=copy.deepcopy(p.target_payload(canonical))
    result['distributed_rank']=rank;result['tags']['distributed_rank']=rank
    result['seed']=intent['base_seed']+rank;result['env']['seed']=intent['base_seed']+rank
    result['env']['device']=f'cuda:{local_rank}' if intent['scope']=='single_node_cuda' else 'cpu'
    result['use_wandb']=bool(intent['use_wandb_cli'] and rank==0)
    return result

def proc_identity():
    raw=Path('/proc/self/stat').read_text();return {'pid':os.getpid(),'start_ticks':int(raw[raw.rindex(')')+2:].split()[19]),'hostname':socket.gethostname()}

def runtime_identity(intent):
    def envint(key):
        value=os.environ.get(key);p.need(type(value) is str and re.fullmatch('0|[1-9][0-9]*',value) is not None,'Missing/invalid launcher '+key);return int(value)
    rank=envint('RANK');local=envint('LOCAL_RANK');world=envint('WORLD_SIZE');localworld=envint('LOCAL_WORLD_SIZE')
    p.need(world==localworld==intent['world_size'] and rank<world and local==intent['local_rank_by_rank'][rank],'Actual launcher rank mapping differs')
    actual=(dist.get_rank(),dist.get_world_size()) if dist.is_initialized() else (0,1)
    p.need(actual==(rank,world),'Actual process group differs from launcher')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES');p.need(visible==intent['cuda_visible_devices'],'Actual CUDA visibility differs')
    device=None
    if intent['scope']=='single_node_cuda':
        p.need(os.environ.get('MUJOCO_EGL_DEVICE_ID')==str(local),'Actual EGL mapping differs')
        device=torch.cuda.current_device();p.need(device==local and torch.cuda.device_count()==world,'Actual CUDA ordinal mapping differs')
    else:p.need(not torch.cuda.is_initialized(),'Synthetic CPU fixture initialized CUDA')
    return dict(proc_identity(),rank=rank,local_rank=local,world_size=world,local_world_size=localworld,cuda_visible_devices=visible,
        cuda_current_device=device,egl_device=os.environ.get('MUJOCO_EGL_DEVICE_ID'),scope=intent['scope'])

def verify_origin(contract):
    if contract.get('schema')==ENTRY_SCHEMA:
        from scaler_ufo_adapter.freeze_training_entry import verify_rank_origin
        return verify_rank_origin(contract)
    origin=contract['source_origin'];p.need(set(origin)==ORIGIN_KEYS and origin['schema']=='scaler_ufo_obs_source_origin_v1','Source origin keys differ')
    p.need(origin['source_parent']==SOURCE_PARENT and origin['source_child']==SOURCE_CHILD,'Source creation is not b543/d563')
    lineage=p.load_pin(contract['source']['lineage']);p.need(lineage['source_parent_commit']==SOURCE_PARENT and lineage['source_child_commit']==SOURCE_CHILD,'Pinned lineage source differs')
    launch=p.load_pin(origin['launch'])
    p.need(launch.get('schema')=='scaler_ufo_stage_launch_v1' and launch.get('scaler_commit')==SOURCE_PARENT and launch.get('ufo_commit')==SOURCE_CHILD and launch.get('ufo_dirty') is False and launch.get('status')=='running' and launch.get('dry_run') is False,'Original launch identity/status differs')
    if contract['scope']=='full_checkpoint_preparation':
        p.need(origin['mode']=='original_ee8' and origin['launch']['sha256']==LAUNCH_SHA and contract['source']['marker']['sha256']==MARKER_SHA and Path(contract['source']['generation']).name==GENERATION,'Exact original ee8 source pins differ')
    else:p.need(contract['scope']=='synthetic_cpu_fixture' and origin['mode']=='synthetic_fixture' and launch.get('synthetic') is True,'Synthetic origin not explicitly labelled')
    return launch

def verify_static(contract):
    extra={'entry_context'} if contract.get('schema')==ENTRY_SCHEMA else set()
    p.need(set(contract)==p.CONTRACT_KEYS|{'rank_projection','source_origin','source_baseline','evidence'}|extra and contract['schema'] in (SCHEMA,ENTRY_SCHEMA),'Ranked policy schema/keys differ')
    canonical,intent=validate_projection(contract['rank_projection']);verify_origin(contract)
    p.need((contract['scope']=='full_checkpoint_preparation')==(intent['scope']=='single_node_cuda'),'Source/runtime scope differs')
    p.need(p.canonical(p.target_payload(canonical))==contract['target_effective_sha256'],'Canonical effective hash differs')
    evidence=contract['evidence'];p.need(set(evidence)==EVIDENCE_KEYS and evidence['schema']=='scaler_ufo_obs_rank_evidence_spec_v1','Evidence spec schema differs')
    p.need(evidence['session_id']==intent['session_id'] and evidence['run_id']==intent['run_id'],'Evidence session/run differs')
    p.need(evidence['first_updates']==16 and type(evidence['first_updates']) is int and evidence['max_record_bytes']==32768 and evidence['max_total_bytes_per_rank']==4194304,'Evidence bounds differ')
    root=Path(evidence['output_root']);p.need(root.is_absolute() and str(root)==str(root.resolve()) and root.name==intent['run_id'] and root.parent.name==intent['session_id'],'Evidence namespace differs')
    from .obs_normalizer_sync import KEYS
    baseline=contract['source_baseline'];p.need(set(baseline)=={'schema','model_sha256','entries','start_coordinate'} and baseline['schema']=='scaler_ufo_obs_source_typed_baseline_v1','Source baseline schema differs')
    p.need(baseline['model_sha256']==contract['source']['files']['model/model.safetensors']['sha256'],'Source baseline model differs')
    validate_typed(baseline['entries']);p.need([x['name'] for x in baseline['entries']]==list(KEYS),'Source baseline keys/order differs')
    start=baseline['start_coordinate'];p.need(set(start)=={'local_time','global_time','optimizer_steps'} and all(type(x) is int and x>=0 for x in start.values()),'Source coordinate differs')
    return canonical,intent

def validate_typed(entries):
    from .obs_normalizer_sync import KEYS
    p.need(type(entries) in (list,tuple) and len(entries)==12,'Typed baseline length differs')
    for name,e in zip(KEYS,entries):
        p.need(type(e) is dict and set(e)=={'name','dtype','shape','sha256'} and e['name']==name,'Typed baseline key differs')
        counter=name.endswith('num_batches_tracked');p.need(e['dtype']==('torch.int64' if counter else 'torch.float32'),'Typed baseline dtype differs')
        p.need(e['shape']==[] if counter else type(e['shape']) is list and len(e['shape'])==1 and type(e['shape'][0]) is int and e['shape'][0]>0,'Typed baseline shape differs')
        p.need(type(e['sha256']) is str and re.fullmatch('[0-9a-f]{64}',e['sha256']) is not None,'Typed baseline digest differs')

def verify_rank_target(contract,target_cfg,*,require_runtime=True):
    canonical,intent=verify_static(contract)
    runtime=runtime_identity(intent) if require_runtime else None
    rank=runtime['rank'] if runtime else target_cfg['distributed_rank'];local=intent['local_rank_by_rank'][rank]
    expected=project(canonical,intent,rank,local)
    p.need(p.target_payload(target_cfg)==expected,'Actual rank effective config differs from exact projection')
    return canonical,intent,runtime

def bind_train_config(cfg,binding,*,base_seed,use_wandb_cli,rank,local_rank,world_size):
    """Explicit keyword caller only; no parser or generic target-config merge."""
    contract=p.load_pin(binding);canonical,intent=verify_static(contract);runtime=runtime_identity(intent)
    p.need((base_seed,use_wandb_cli,rank,local_rank,world_size)==(intent['base_seed'],intent['use_wandb_cli'],runtime['rank'],runtime['local_rank'],intent['world_size']) and type(use_wandb_cli) is bool,'Actual run_train arguments differ from pinned launch intent')
    expected=project(canonical,intent,rank,local_rank);expected['agent']['train'][p.FIELD]=False;expected['initialization_provenance']=None
    p.need(p.target_payload(cfg.model_dump())==expected,'Pre-policy train.py configuration differs; only the explicit field may change')
    raw=cfg.model_dump();raw['agent']['train'][p.FIELD]=True;raw[p.BINDING]=binding
    return type(cfg)(**raw)

def evidence_context(contract,binding,target_cfg):
    canonical,intent,runtime=verify_rank_target(contract,target_cfg)
    return {'contract':contract,'binding':binding,'runtime':runtime,'projected_effective_sha256':p.canonical(p.target_payload(target_cfg)),
        'canonical_effective_sha256':contract['target_effective_sha256'],'launch_intent_sha256':contract['rank_projection']['launch_intent']['sha256']}
