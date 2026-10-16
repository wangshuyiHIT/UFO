"""Typed source-derived protection; no arbitrary exclusion list or buffer repair."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from contextlib import contextmanager
import torch
import torch.distributed as dist
from safetensors import safe_open
from . import obs_normalizer_resume as policy

GROUPS=('history_actor','last_action','privileged_state','state')
KINDS=('running_mean','running_var','num_batches_tracked')
KEYS=tuple('_obs_normalizer._normalizers.'+g+'._normalizer.'+k for g in GROUPS for k in KINDS)
_AUTHORITY=object()

class FrozenObservationError(RuntimeError):pass

@dataclass(frozen=True)
class AdmittedRestore:
    source_generation: str
    source_marker_sha256: str
    source_model_sha256: str
    source_model_path: str
    source_stat: tuple
    policy_contract_sha256: str
    target_implementation_sha256: str
    target_agent_sha256: str
    target_policy: bool
    _authority: object = field(repr=False,compare=False)
    rank_evidence: dict | None = field(default=None,repr=False,compare=False)

@dataclass(frozen=True)
class RuntimeRank:
    rank: int
    world: int
    def validate(self):
        policy.need(type(self.rank) is int and type(self.world) is int and self.world>0 and 0<=self.rank<self.world,'Invalid typed runtime rank/world')
        actual=(dist.get_rank(),dist.get_world_size()) if dist.is_initialized() else (0,1)
        policy.need(actual==(self.rank,self.world),'Runtime rank/world differs from initialized process group')

def file_stat(path):
    p=Path(path);policy.need(not p.is_symlink() and p.is_file(),'Source model is not a regular file')
    s=p.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def admit_restore(contract,binding,target_effective,stat_before):
    """Called only after actual load authorization and original model/Adam load."""
    path=Path(contract['source']['generation'])/'model/model.safetensors'
    policy.need(file_stat(path)==stat_before,'Source model changed across checked restore')
    policy.need(contract['target_policy'] is True,'Frozen synchronization requires explicit policy')
    from . import obs_rank_contract as ranked
    context=ranked.evidence_context(contract,binding,target_effective) if contract['schema'] in (ranked.SCHEMA,ranked.ENTRY_SCHEMA) else None
    return AdmittedRestore(contract['source']['generation'],contract['source']['marker']['sha256'],
        contract['source']['files']['model/model.safetensors']['sha256'],str(path),stat_before,binding['sha256'],
        policy.canonical(contract['target_implementation']),policy.canonical(target_effective['agent']),True,_AUTHORITY,context)

def typed(tensor,name):
    expected=torch.int64 if name.endswith('.num_batches_tracked') else torch.float32
    policy.need(tensor.dtype==expected,'Protected dtype differs: '+name)
    policy.need(tuple(tensor.shape)==() if expected==torch.int64 else tensor.ndim==1 and tensor.numel()>0,'Protected shape differs: '+name)
    value=tensor.detach().cpu().contiguous()
    raw=value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {'name':name,'dtype':str(tensor.dtype),'shape':list(tensor.shape),'sha256':hashlib.sha256(raw).hexdigest()}

def protected(module):
    result={k:v for k,v in module.named_buffers(remove_duplicate=False) if k.startswith('_obs_normalizer.')}
    policy.need(set(result)==set(KEYS),'Protected buffer key closure differs')
    policy.need(len({v.data_ptr() for v in result.values()})==len(KEYS),'Protected buffers alias')
    return result

class SourceBufferGuard:
    def __init__(self,agent,admission,identity,coordinator):
        policy.need(type(admission) is AdmittedRestore and admission._authority is _AUTHORITY and admission.target_policy is True,'Actual checked restore token is required')
        policy.need(type(identity) is RuntimeRank,'Typed runtime identity required')
        self.agent=agent;self.admission=admission;self.identity=identity;self.coordinator=coordinator
        self.records=[];self.last_record=None;self.check_count=0;self.last_phase=None;self.last_coordinate=None
        self._admitted_token=admission
        self._runtime_world=coordinator.world_size
        self._contract_sha256=admission.policy_contract_sha256
        identity.validate()
        policy.need(coordinator is not None and coordinator.world_size==identity.world and coordinator.rank==identity.rank,'Bounded coordinator rank/world differs')
        policy.need(policy.policy(agent.cfg.model_dump()) is True and policy.canonical(agent.cfg.model_dump())==admission.target_agent_sha256,'Actual loaded agent policy/config differs')
        source=Path(admission.source_model_path);policy.need(file_stat(source)==admission.source_stat,'Admitted source stat differs')
        values=protected(agent._model)
        with safe_open(source,framework='pt',device='cpu') as f:
            names={k for k in f.keys() if k.startswith('_obs_normalizer.')}
            policy.need(names==set(KEYS),'Source protected tensor closure differs')
            self.baseline=tuple(typed(f.get_tensor(k),k) for k in KEYS)
        policy.need(file_stat(source)==admission.source_stat,'Source changed during baseline reading')
        self.baseline_sha256=policy.canonical(self.baseline)
        policy.need(tuple(typed(values[k],k) for k in KEYS)==self.baseline,'Loaded protected state differs from admitted source')
        self.common={'source_generation':admission.source_generation,'marker_sha256':admission.source_marker_sha256,
            'model_sha256':admission.source_model_sha256,'implementation_sha256':admission.target_implementation_sha256,
            'agent_sha256':admission.target_agent_sha256,'baseline_sha256':self.baseline_sha256,
            'policy_contract_sha256':admission.policy_contract_sha256,'policy':True}
        self.evidence_writer=None;self.runtime_cfg=None
        if admission.rank_evidence is not None:
            from .obs_rank_evidence import RankEvidenceWriter
            context=admission.rank_evidence
            policy.need(context['runtime']['rank']==identity.rank and context['runtime']['world_size']==identity.world,'Admitted process identity differs')
            self.evidence_writer=RankEvidenceWriter(context,self.baseline)

    def _local(self,phase,coordinate,module):
        self.identity.validate()
        if self.admission.rank_evidence is not None:
            context=self.admission.rank_evidence
            policy.need(self.runtime_cfg is not None and policy.canonical(policy.target_payload(self.runtime_cfg.model_dump()))==context['projected_effective_sha256'],'Current full rank effective config differs')
            from .obs_rank_contract import proc_identity
            process=proc_identity()
            policy.need(all(process[k]==context['runtime'][k] for k in process),'Admitted process changed')
            if self.last_phase is None:policy.need(coordinate==context['contract']['source_baseline']['start_coordinate'],'Initial coordinate differs from source')
        policy.need(module is self.agent._model,'Guard belongs to a different model')
        policy.need(type(coordinate) is dict and set(coordinate)=={'optimizer_steps','local_time','global_time'} and
                    all(type(v) is int and v>=0 for v in coordinate.values()),'Invalid typed update coordinate')
        policy.need(policy.policy(self.agent.cfg.model_dump()) is True and
                    policy.canonical(self.agent.cfg.model_dump())==self.admission.target_agent_sha256,'Actual agent policy/config changed')
        policy.need(self.admission is self._admitted_token and type(self.admission) is AdmittedRestore and self.admission._authority is _AUTHORITY,'Restore token changed')
        expected={None:'initial_broadcast_before','initial_broadcast_before':'initial_broadcast_after',
                  'initial_broadcast_after':'update_before','update_before':'update_after','update_after':'sync_before',
                  'sync_before':'sync_after','sync_after':'update_before'}[self.last_phase]
        policy.need(phase==expected,'Guard phase sequence differs')
        if self.last_coordinate is not None:
            old=self.last_coordinate
            if phase=='update_before':
                policy.need(coordinate['optimizer_steps']==old['optimizer_steps'] and coordinate['local_time']>=old['local_time'] and coordinate['global_time']>=old['global_time'],'Update entry coordinate differs')
            elif phase=='update_after':
                policy.need(coordinate==dict(old,optimizer_steps=old['optimizer_steps']+1),'Post-update coordinate differs')
            else:policy.need(coordinate==old,'Synchronization coordinate differs')
        values=protected(module);observed=tuple(typed(values[k],k) for k in KEYS)
        policy.need(observed==self.baseline,'Protected state changed; no broadcast/repair permitted')
        policy.need(policy.canonical(self.baseline)==self.baseline_sha256,'Typed source baseline changed')
        return observed

    def check(self,phase,coordinate,module=None):
        """All ranks exchange local status before any protected sync can mutate state."""
        module=self.agent._model if module is None else module
        error=None;observed=None
        try:observed=self._local(phase,coordinate,module)
        except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)[:2048]}
        rank=dist.get_rank() if dist.is_initialized() else 0
        payload={'schema':'scaler_ufo_obs_buffer_check_v1','rank':rank,'world':self._runtime_world,'pid':os.getpid(),
            'phase':phase,'coordinate':coordinate,'common':self.common,'policy_contract_sha256':self._contract_sha256,
            'error':error,'observed':observed}
        # Reuse existing bounded Gloo control group. Failure/timeout propagates;
        # there is no retry/barrier/recovery and no continuation into NCCL.
        results=self.coordinator.gather(payload)
        problems=[r for r in results if r.get('error') is not None]
        common_ok=(len(results)==self._runtime_world and {r.get('rank') for r in results}==set(range(self._runtime_world))
            and all(r.get('schema')==payload['schema'] and r.get('world')==self._runtime_world and
                    r.get('phase')==phase and r.get('coordinate')==coordinate and r.get('common')==self.common for r in results))
        record={'local':payload,'all_ranks_agree':common_ok,'rank_errors':problems}
        self.check_count+=1;self.last_record=record
        if len(self.records)<128:self.records.append(record)
        if self.evidence_writer is not None and not self.evidence_writer.closed:
            self.coordinator.stage('durable first16 observation evidence',lambda: self.evidence_writer.append(record))
        if problems or not common_ok:raise FrozenObservationError('Frozen observation check rejected before synchronization: '+json.dumps(record,default=str)[:12000])
        self.last_phase=phase;self.last_coordinate=dict(coordinate)

    def protected_ids(self,module,required_phase):
        # Called only after check(); this is fixed internal closure, never a caller list.
        policy.need(module is self.agent._model and self.last_phase==required_phase,'No matching successful guard check')
        return frozenset(id(v) for v in protected(module).values())

    @contextmanager
    def coordinate_local_errors(self,name):
        error=None
        try:yield
        except Exception as exc:error={'rank':self.identity.rank,'type':type(exc).__name__,'message':str(exc)[:2048]}
        results=self.coordinator.gather({'name':name,'error':error})
        if any(r.get('name')!=name or r.get('error') is not None for r in results):
            raise FrozenObservationError('Coordinated local failure: '+json.dumps(results)[:12000])


def require_guard(guard):
    policy.need(type(guard) is SourceBufferGuard, "Only the fixed source-buffer guard is supported")
    return guard
