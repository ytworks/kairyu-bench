# Homogeneous AV serving-record correction

After the user explicitly approved the four chat-container replacement, all four
were verified healthy with the AV checkpoint and unchanged runtime. The global
`chat-checkpoint.json` still described the old h0397 checkpoint. This host
metadata was corrected using the actual owner's `chat_checkpoint_record` schema.
No model payload, container, command/environment, capture configuration or
running WSP input changed. The official dev30 owner continued uninterrupted.

The update reread all four exact CIDs, healthy states, read-only AV mounts and
`kairyu.chat-checkpoint-tree` labels; checked all nine current file stats against
the fresh full-hash HTTP-trace identity; reconstructed the canonical sorted
file tree; preserved the old record; and replaced it atomically only while the
old bytes still matched. This reused the just-completed full hashes rather than
claiming another payload hash. No environment or credentials were dumped.

- Old tree: `ba5e6ab526c8c08182a030f459f4acb532d5f50ebc1d35e4745b09fe59a8768d`.
- Correct served tree: `5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf`.
- Before SHA: `afb253fbd4134fefee2ae0a23b593134617023801154fcf8f2a5eb34df9825c9`.
- After SHA: `8cd66daf213fe900c9dcd418e374f10111361a7d31ecb36f677e4ef1cb4bcf81`.

[Before](artifacts/av-serving-record-before-20261107.json),
[after](artifacts/av-serving-record-after-20261107.json) and
[receipt](artifacts/av-serving-record-update-20261107.json) retain the full safe
metadata and exact four CIDs. This is administrative consistency, not a new
capability measurement or a change to which checkpoint won dev10.

## Executed one-off source

The fixed old-tree precondition and fresh output directory intentionally prevent
blind repetition. The code ran remotely through the existing CPU interpreter.

```python
import hashlib,json,os,pathlib,subprocess,tempfile,time
r=pathlib.Path('/mnt/nvme/kairyu/wsp/gemma4-coding-20261104'); n=r/'analysis/fresh-purpose-20261107-av-dev30-dp4-v1'
out=n/'served-record-update';out.mkdir(mode=0o700)
p=pathlib.Path('/mnt/nvme/kairyu/model-volumes/gemma-4-31b-8gpu/chat-checkpoint.json')
assert p.is_file() and not p.is_symlink()
old=p.read_bytes(); before=json.loads(old)
assert set(before)=={'files','path','schema_version','source','tree_sha256'} and before['schema_version']==1
assert before['tree_sha256']=='ba5e6ab526c8c08182a030f459f4acb532d5f50ebc1d35e4745b09fe59a8768d'
plan=json.loads((n/'launch-plan-approved.json').read_text()); serving=plan['expected_serving']; expected=serving['tree_sha256']
assert expected=='5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf'
config=json.loads((n/'http-trace/capture.json').read_text());rows=[]
for f in config['checkpoint']['files']:
 st=(pathlib.Path(serving['checkpoint'])/f['path']).stat()
 assert st.st_size==f['size'] and f['stat']=={'device':st.st_dev,'inode':st.st_ino,'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns}
 rows.append({k:f[k] for k in ('path','size','sha256')})
rows.sort(key=lambda x:x['path']);assert len(rows)==9
assert hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()==expected
checked=[]
for name,cid in serving['service_cids'].items():
 d=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
 assert d['Id']==cid and d['State']['Running'] and d['State']['Health']['Status']=='healthy'
 assert d['Config']['Labels']['com.docker.compose.service']==name
 labels=[k for k,v in d['Config']['Labels'].items() if v==expected];assert labels
 assert any(m['Source']==serving['checkpoint'] and not m['RW'] for m in d['Mounts'])
 checked.append({'service':name,'cid':cid,'tree_label_keys':labels,'healthy':True})
assert len(checked)==4
record={'schema_version':1,'source':'CHAT_CHECKPOINT_DIR','path':serving['checkpoint'],'tree_sha256':expected,'files':rows}
new=(json.dumps(record,indent=2,sort_keys=True)+'\n').encode()
(out/'before.json').write_bytes(old);(out/'after.json').write_bytes(new)
assert p.read_bytes()==old
fd,tmp=tempfile.mkstemp(prefix='.chat-checkpoint-fresh-av-',dir=p.parent)
try:
 with os.fdopen(fd,'wb') as f:f.write(new);f.flush();os.fsync(f.fileno())
 os.chmod(tmp,p.stat().st_mode&0o777);assert p.read_bytes()==old;os.replace(tmp,p)
finally:
 if os.path.exists(tmp):os.unlink(tmp)
assert p.read_bytes()==new
receipt={'epoch':time.time(),'path':str(p),'schema_source':'existing control.py chat_checkpoint_record','before_sha256':hashlib.sha256(old).hexdigest(),'after_sha256':hashlib.sha256(new).hexdigest(),'old_tree':before['tree_sha256'],'new_tree':expected,'services':checked,'payload_hashes_reused_from_fresh_init':True,'all_nine_stats_unchanged':True,'weight_writes':0,'container_changes':0,'benchmark_config_changes':0,'atomic_replace':True}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
```
