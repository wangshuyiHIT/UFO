"""Private bounded first16 evidence. No owner, process control, retry, or admission release."""
from __future__ import annotations
import hashlib,json,os,stat
from pathlib import Path
from . import obs_normalizer_resume as p
from . import obs_rank_contract as rank_contract
HEADER_KEYS={'schema','session_id','run_id','policy_contract_sha256','source_generation','source_model_sha256','source_marker_sha256','source_launch_sha256','implementation_sha256','canonical_effective_sha256','projected_effective_sha256','runtime','source_baseline','start_coordinate'}
EVENT_KEYS={'schema','index','previous_sha256','rank','session_id','run_id','policy_contract_sha256','phase','coordinate','observed','local_error','all_ranks_agree','rank_errors_count'}
COMPLETE_KEYS={'schema','session_id','run_id','rank','policy_contract_sha256','record_count','last_sha256','window_updates','scope'}
RECORDS=66

def private(path,*,directory=False):
    path=Path(path);p.need(not path.is_symlink() and path.exists(),'Missing or aliased evidence path')
    s=path.stat();p.need(s.st_uid==os.getuid() and stat.S_IMODE(s.st_mode)&0o077==0,'Evidence is not private/current user owned')
    p.need(stat.S_ISDIR(s.st_mode) if directory else stat.S_ISREG(s.st_mode) and s.st_nlink==1,'Evidence type/link count differs')

def fsync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)

def make_private_root(root):
    missing=[];current=root
    while not current.exists():missing.append(current);current=current.parent
    for path in reversed(missing):
        path.mkdir(mode=0o700,exist_ok=True);private(path,directory=True);fsync_directory(path.parent)
    private(root,directory=True);fsync_directory(root.parent)

def header(contract,binding,runtime,projected_hash,baseline):
    spec=contract['evidence'];return {'schema':'scaler_ufo_obs_rank_header_v1','session_id':spec['session_id'],'run_id':spec['run_id'],
        'policy_contract_sha256':binding['sha256'],'source_generation':contract['source']['generation'],
        'source_model_sha256':contract['source']['files']['model/model.safetensors']['sha256'],'source_marker_sha256':contract['source']['marker']['sha256'],
        'source_launch_sha256':contract['source_origin']['launch']['sha256'],'implementation_sha256':p.canonical(contract['target_implementation']),
        'canonical_effective_sha256':contract['target_effective_sha256'],'projected_effective_sha256':projected_hash,
        'runtime':runtime,'source_baseline':list(baseline),'start_coordinate':contract['source_baseline']['start_coordinate']}

class RankEvidenceWriter:
    def __init__(self,context,baseline):
        self.contract=context['contract'];self.binding=context['binding'];self.runtime=context['runtime'];self.spec=self.contract['evidence']
        rank_contract.verify_static(self.contract)
        p.need(list(baseline)==self.contract['source_baseline']['entries'],'Independent source baseline differs from contract')
        root=Path(self.spec['output_root']);p.need(not root.is_relative_to(Path(self.contract['source']['generation'])),'Evidence cannot write into source checkpoint')
        p.need(str(root)==str(root.resolve()),'Evidence root traverses a symlink')
        make_private_root(root)
        self.path=root/f"rank-{self.runtime['rank']:02d}";self.path.mkdir(mode=0o700,exist_ok=False);fsync_directory(root)
        self.bytes=0;self.count=0;self.closed=False
        h=header(self.contract,self.binding,self.runtime,context['projected_effective_sha256'],baseline)
        self.last=self._write('header.json',h)
    def _write(self,name,value):
        raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        p.need(len(raw)<=self.spec['max_record_bytes'] and self.bytes+len(raw)<=self.spec['max_total_bytes_per_rank'],'Evidence byte bound exceeded')
        path=self.path/name;fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        try:
            with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            d=os.open(self.path,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(d)
            finally:os.close(d)
        finally:pass
        self.bytes+=len(raw);return p.digest(path)
    def append(self,record):
        if self.closed:return
        local=record['local'];event={'schema':'scaler_ufo_obs_rank_event_v1','index':self.count,'previous_sha256':self.last,
            'rank':self.runtime['rank'],'session_id':self.spec['session_id'],'run_id':self.spec['run_id'],'policy_contract_sha256':self.binding['sha256'],
            'phase':local['phase'],'coordinate':local['coordinate'],'observed':local['observed'],'local_error':local['error'],
            'all_ranks_agree':record['all_ranks_agree'],'rank_errors_count':len(record['rank_errors'])}
        self.last=self._write(f'event-{self.count:03d}.json',event);self.count+=1
        if local['error'] is not None or not record['all_ranks_agree'] or record['rank_errors']:
            self.closed=True;return
        if self.count==RECORDS:
            self._write('first16_complete.json',{'schema':'scaler_ufo_obs_rank_first16_complete_v1','session_id':self.spec['session_id'],
                'run_id':self.spec['run_id'],'rank':self.runtime['rank'],'policy_contract_sha256':self.binding['sha256'],
                'record_count':RECORDS,'last_sha256':self.last,'window_updates':16,'scope':'first16_observed_window_only'})
            self.closed=True


def file_identity(path):
    s=Path(path).stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)

def read_small(path,limit):
    private(path);before=file_identity(path);p.need(before[2]<=limit,'Evidence record too large')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:
        fs=os.fstat(f.fileno());p.need((fs.st_dev,fs.st_ino)==before[:2],'Evidence inode changed before read')
        raw=f.read(limit+1);p.need(len(raw)<=limit,'Evidence read bound exceeded')
    p.need(file_identity(path)==before,'Evidence changed while reading')
    return json.loads(raw),hashlib.sha256(raw).hexdigest(),before


def collect(binding,header_pins):
    """Read verified header pins and chained files; never rehashes source full-model or grants training."""
    contract=p.load_pin(binding);canonical,intent=rank_contract.verify_static(contract);world=intent['world_size'];spec=contract['evidence']
    p.need(type(header_pins) is list and len(header_pins)==world and len({x['path'] for x in header_pins})==world,'Missing or duplicate rank header')
    roots={};runtimes=[];all_events=[];snapshots=[];source=contract['source_baseline'];expected_entries=source['entries'];start=source['start_coordinate']
    for hp in header_pins:
        p.need(type(hp) is dict and set(hp)=={'path','sha256'},'Header pin schema differs')
        path=Path(hp['path']);private(path.parent,directory=True);h,header_sha,hs=read_small(path,spec['max_record_bytes']);snapshots.append((path,hs))
        p.need(header_sha==hp['sha256'],'Header pin SHA differs')
        p.need(set(h)==HEADER_KEYS and h['schema']=='scaler_ufo_obs_rank_header_v1','Header schema differs')
        runtime=h['runtime'];expected_runtime_keys={'pid','start_ticks','hostname','rank','local_rank','world_size','local_world_size','cuda_visible_devices','cuda_current_device','egl_device','scope'}
        p.need(type(runtime) is dict and set(runtime)==expected_runtime_keys,'Recorded process identity keys differ')
        for name in ('rank','local_rank','world_size','local_world_size'):rank_contract.integer(runtime[name],'recorded '+name)
        p.need(type(runtime['hostname']) is str and 0<len(runtime['hostname'])<=255,'Recorded hostname differs')
        rank=runtime['rank'];p.need(rank<world and rank not in roots,'Rank missing/duplicated')
        for key in ['pid','start_ticks']:rank_contract.integer(runtime[key],key,1)
        p.need(runtime['local_rank']==intent['local_rank_by_rank'][rank] and runtime['world_size']==runtime['local_world_size']==world and runtime['scope']==intent['scope'] and runtime['cuda_visible_devices']==intent['cuda_visible_devices'],'Recorded rank mapping differs')
        if intent['scope']=='single_node_cuda':p.need(runtime['cuda_current_device']==runtime['local_rank'] and runtime['egl_device']==str(runtime['local_rank']),'Recorded CUDA mapping differs')
        else:p.need(runtime['cuda_current_device'] is None,'Synthetic record claims CUDA')
        projected=p.canonical(rank_contract.project(canonical,intent,rank,runtime['local_rank']))
        p.need(h==header(contract,binding,runtime,projected,expected_entries),'Header differs from pinned source/config/typed baseline')
        expected_root=Path(spec['output_root'])/f'rank-{rank:02d}';p.need(path==expected_root/'header.json' and str(path)==str(path.resolve()),'Rank evidence path differs')
        expected_names={'header.json','first16_complete.json'}|{f'event-{n:03d}.json' for n in range(RECORDS)}
        p.need({x.name for x in path.parent.iterdir()}==expected_names,'Incomplete/extra evidence files')
        p.need(sum(x.stat().st_size for x in path.parent.iterdir())<=spec['max_total_bytes_per_rank'],'Rank evidence total too large')
        last=hp['sha256'];events=[];previous=dict(start)
        for index in range(RECORDS):
            ep=path.parent/f'event-{index:03d}.json';e,event_sha,es=read_small(ep,spec['max_record_bytes']);snapshots.append((ep,es))
            p.need(set(e)==EVENT_KEYS and e['schema']=='scaler_ufo_obs_rank_event_v1' and type(e['index']) is int and e['index']==index and e['previous_sha256']==last,'Event schema/chain differs')
            p.need(type(e['rank']) is int and e['rank']==rank and e['session_id']==spec['session_id'] and e['run_id']==spec['run_id'] and e['policy_contract_sha256']==binding['sha256'],'Event identity differs')
            p.need(e['observed']==expected_entries and e['local_error'] is None and e['all_ranks_agree'] is True and type(e['rank_errors_count']) is int and e['rank_errors_count']==0,'Observed values/errors differ from admitted source')
            c=e['coordinate'];p.need(type(c) is dict and set(c)==set(start) and all(type(x) is int and x>=0 for x in c.values()),'Event coordinate types differ')
            if index<2:
                phase=['initial_broadcast_before','initial_broadcast_after'][index];p.need(c==start,'Initial source coordinate differs')
            else:
                update=(index-2)//4;part=(index-2)%4;phase=['update_before','update_after','sync_before','sync_after'][part]
                p.need(c['optimizer_steps']==start['optimizer_steps']+update+(part!=0),'First16 optimizer coordinate differs')
                if part==0:p.need(c['local_time']>=previous['local_time'] and c['global_time']>=previous['global_time'],'Update coordinate regressed')
                else:p.need(c['local_time']==previous['local_time'] and c['global_time']==previous['global_time'],'Within-update coordinate differs')
                scale=world if canonical['distributed_global_steps'] else 1
                p.need(c['global_time']==c['local_time']*scale,'Global/local coordinate scale differs')
            p.need(e['phase']==phase,'Event phase differs');previous=c;last=event_sha;events.append(e)
        cp=path.parent/'first16_complete.json';completion,_,cs=read_small(cp,spec['max_record_bytes']);snapshots.append((cp,cs))
        p.need(set(completion)==COMPLETE_KEYS and type(completion['rank']) is int and completion=={'schema':'scaler_ufo_obs_rank_first16_complete_v1','session_id':spec['session_id'],'run_id':spec['run_id'],'rank':rank,'policy_contract_sha256':binding['sha256'],'record_count':RECORDS,'last_sha256':last,'window_updates':16,'scope':'first16_observed_window_only'},'Completion differs')
        roots[rank]=hp;runtimes.append(runtime);all_events.append(events)
    p.need(set(roots)==set(range(world)),'Rank closure differs')
    p.need(len({(x['hostname'],x['pid'],x['start_ticks']) for x in runtimes})==world and len({x['hostname'] for x in runtimes})==1,'Process map is not unique single-node')
    reference=[x['coordinate'] for x in all_events[0]]
    p.need(all([x['coordinate'] for x in events]==reference for events in all_events),'Cross-rank coordinates differ')
    p.need(all(file_identity(path)==value for path,value in snapshots),'Evidence changed across collection')
    return {'schema':'scaler_ufo_obs_rank_first16_collection_v1','complete_observed_window':True,'session_id':spec['session_id'],'run_id':spec['run_id'],
        'policy_contract':binding,'world_size':world,'rank_headers':[roots[i] for i in range(world)],'scope':intent['scope'],
        'window_updates':16,'source_full9_rehashed_by_collector':False,'checkpoint_or_end_covered':False,'training_authorized':False}
