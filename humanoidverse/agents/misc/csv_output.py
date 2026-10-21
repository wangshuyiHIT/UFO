"""Exact two campaign CSV writers, cumulative across segment processes.

Only normal complete serialization is appended. Overflow is a training failure;
no rows are dropped. A failed append leaves pending evidence, never a reset.
"""
from __future__ import annotations
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import stat

LIMIT = 16 * 1024**2
STATE_LIMIT = 4096
NAMES = {'train_log.txt', 'humanoidverse_tracking_eval.csv'}
SCHEMA = 'ufo_campaign_csv_extent_v1'

class CSVAllowanceExceeded(RuntimeError): pass

def directory_sync(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)

def identity(path):
    s=Path(path).lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:raise ValueError('CSV regular independent file required')
    values={k:getattr(s,'st_'+k) for k in ('dev','ino','size','mtime_ns','ctime_ns')}
    if any(type(v)is not int or not 0<=v<2**64 for v in values.values()):raise ValueError('CSV stat domain')
    return values

def small(path):
    before=identity(path)
    if before['size']>STATE_LIMIT:raise ValueError('CSV state extent')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:b=f.read(STATE_LIMIT+1)
    if identity(path)!=before or len(b)!=before['size']:raise ValueError('CSV state changed')
    return json.loads(b)

def persist(path, obj, *, initial=False):
    b=(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if len(b)>STATE_LIMIT:raise CSVAllowanceExceeded('CSV control receipt extent')
    tmp=path.with_name(path.name+'.tmp')
    if os.path.lexists(tmp):raise ValueError('Unknown prior CSV temporary; no retry')
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    if initial:
        os.link(tmp,path,follow_symlinks=False);os.unlink(tmp)
    else:os.replace(tmp,path)
    directory_sync(path.parent)

class LimitedText(io.TextIOBase):
    def __init__(self, remaining):self.remaining=remaining;self.chunks=[];self.size=0
    def writable(self):return True
    def write(self,text):
        if not isinstance(text,str):raise TypeError('CSV text serialization required')
        raw=text.encode('utf-8')
        if len(raw)>self.remaining-self.size:raise CSVAllowanceExceeded('CSV cumulative 16MiB allowance exceeded before append')
        self.chunks.append(raw);self.size+=len(raw);return len(text)

def append(path, fields, frame, *, write_header):
    path=Path(path)
    if path.name not in NAMES or path.parent.resolve(strict=True)!=path.parent:
        raise ValueError('Only exact canonical campaign CSV paths')
    if type(fields)is not list or any(type(f)is not str for f in fields) or len(fields)!=len(set(fields)):
        raise ValueError('Exact finite CSV field schema required')
    schema_raw=json.dumps(fields,ensure_ascii=False,separators=(',',':')).encode()
    if len(schema_raw)>LIMIT:raise CSVAllowanceExceeded('CSV schema exceeds cumulative allowance')
    fields_sha=hashlib.sha256(schema_raw).hexdigest()
    state_path=path.with_name('.'+path.name+'.output_state.json')
    lock_path=path.with_name('.'+path.name+'.output.lock')
    fd=os.open(lock_path,os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb',buffering=0) as lock:
        if os.fstat(lock.fileno()).st_nlink!=1:raise ValueError('CSV lock alias')
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity(lock_path)
        if os.path.lexists(path.with_name('.'+path.name+'.output_failed')):raise ValueError('Failed CSV transaction; original graph must terminate')
        if os.path.lexists(state_path.with_name(state_path.name+'.tmp')):raise ValueError('Unknown prior CSV temporary; no retry')
        exists=os.path.lexists(path)
        if exists:
            before=identity(path);state=small(state_path)
            if (set(state)!={'schema','fields_sha256','status','identity','sequence','limit'} or
                state['schema']!=SCHEMA or state['fields_sha256']!=fields_sha or state['status']!='committed' or
                state['identity']!=before or state['limit']!=LIMIT or type(state['sequence'])is not int or
                not 0<=state['sequence']<=LIMIT or write_header):raise ValueError('CSV continuity/pending/schema mismatch')
            size=before['size'];sequence=state['sequence']
        else:
            if os.path.lexists(state_path):raise ValueError('CSV missing with prior state; no reset')
            before=None;size=0;sequence=0
        if size>LIMIT:raise CSVAllowanceExceeded('Existing CSV exceeds allowance')
        sink=LimitedText(LIMIT-size)
        frame.to_csv(sink,header=write_header,index=False)
        if not sink.size:raise ValueError('Empty CSV append is not a new transaction')
        pending={'schema':SCHEMA,'fields_sha256':fields_sha,'status':'pending','identity':before,'sequence':sequence,'limit':LIMIT}
        persist(state_path,pending,initial=not exists)
        flags=os.O_WRONLY|os.O_NOFOLLOW|(os.O_APPEND if exists else os.O_CREAT|os.O_EXCL)
        out=os.open(path,flags,0o600)
        with os.fdopen(out,'wb',buffering=0) as f:
            current=os.fstat(f.fileno())
            if exists and (current.st_dev,current.st_ino,current.st_size)!=(before['dev'],before['ino'],size):raise ValueError('CSV changed before append')
            if not stat.S_ISREG(current.st_mode) or current.st_nlink!=1:raise ValueError('CSV write identity')
            for raw in sink.chunks:
                view=memoryview(raw)
                while view:
                    count=f.write(view)
                    if not count:raise OSError('CSV short write')
                    view=view[count:]
            f.flush();os.fsync(f.fileno())
        after=identity(path)
        if after['size']!=size+sink.size or after['size']>LIMIT:raise ValueError('CSV extent mismatch')
        pending.update(status='committed',identity=after,sequence=sequence+1)
        try: persist(state_path,pending)
        except BaseException as original:
            # A failed final directory sync may leave a visible committed JSON.
            # Latch a separate fixed fail fence; it never authorizes recovery.
            # If even this write fails, the original graph must still terminate
            # and may not relaunch from this state (no crash-recovery contract).
            fence=path.with_name('.'+path.name+'.output_failed')
            try:
                descriptor=os.open(fence,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                try:os.fsync(descriptor)
                finally:os.close(descriptor)
                directory_sync(path.parent)
            except BaseException:pass
            original.csv_failure_fence_observed=os.path.lexists(fence)
            raise
        return {'bytes_appended':sink.size,'cumulative_bytes':after['size'],'sequence':sequence+1}
