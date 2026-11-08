"""Fixed T002 post-chain span verification and stat freeze; no full payload reread."""
from pathlib import Path
from types import SimpleNamespace
import datetime,hashlib,importlib.util,json,math,os,sys
ROOT=Path('/mnt/nvme/kairyu/wsp/gemma4-coding-20261104')
T=ROOT/'restart-direct-20261108/trial-002-final-branch-ablation'; WT=ROOT/'worktrees/kairyu-bench-weight-tools'
sys.path.insert(0,str(WT/'src'))
from kairyu_bench.wsp.weights import editor
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def save_new(p,value):
 with p.open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')
reg=read(T/'registration.json'); source=Path(reg['source_checkpoint']); candidate=Path(reg['candidate_path']); manifest_path=Path(reg['manifest_path'])
chain_path=T/'chain-verification.json'; execution_path=T/'chain-execution.json'
execution=read(execution_path); chain=read(chain_path); manifest=read(manifest_path); proof=read(T/'literal-zero-proof.json')
assert execution['exit_code']==0 and chain['verified'] is True and chain['checkpoint_writes']==0
assert chain['original']==str(source) and chain['candidate']==str(candidate) and chain['stages']==1
assert chain['manifests']==[{'path':str(manifest_path),'sha256':sha(manifest_path)}]
assert manifest['source']==str(source) and manifest['candidate']==str(candidate)
assert manifest['spec_sha256']==sha(T/'edit-spec.json')==reg['spec_sha256']=='86a0facdfff4c556ec1c01ec832cbda0a4418c45fb2a15107d225dc3022e1abd'
assert manifest['spec']==read(T/'edit-spec.json')
editor.validate_manifest(manifest)
jobs=read(T/'creation-receipt.json')['jobs']; assert len(jobs)==1 and jobs[0]['exit_code']==0 and jobs[0]['candidate']==str(candidate) and jobs[0]['manifest']==str(manifest_path)
helper_path=WT/'docs/experiments/scripts/redeploy-replica.py'; assert sha(helper_path)=='2e9da820fe30ba9b2436fb600cf14a533b1c496b2fed48ac3649fffb9d2bc9ae'
module=importlib.util.spec_from_file_location('t002_freeze_helper',helper_path); helper=importlib.util.module_from_spec(module); module.loader.exec_module(helper)
source_inventory=helper.inventory(source); candidate_inventory=helper.inventory(candidate)
assert set(source_inventory)==set(candidate_inventory) and len(source_inventory)==12
for row in proof['source_files']:
 s=source_inventory[row['path']]; assert [s[k] for k in ('device','inode','size','mtime_ns','ctime_ns')]==row['stat_stamp']
finished_ns=int(datetime.datetime.fromisoformat(execution['finished_at']).timestamp()*1e9)
for rel,state in candidate_inventory.items():
 assert state['nlink']==1 and state['uid']==os.getuid() and state['size']==source_inventory[rel]['size']
 assert (state['device'],state['inode'])!=(source_inventory[rel]['device'],source_inventory[rel]['inode'])
for states in (source_inventory,candidate_inventory):
 assert all(max(s['ctime_ns'],s['mtime_ns'])<=finished_ns for s in states.values())
for name,digest in chain['implementation_sha256'].items(): assert sha(WT/'src/kairyu_bench/wsp/weights'/name)==digest
expected_names=[r['tensor'] for r in proof['rows']]; assert [e['tensor'] for e in manifest['edits']]==expected_names
zero_rows=[]; total_changed_bytes=0
for edit,p in zip(manifest['edits'],proof['rows'],strict=True):
 before=bytes.fromhex(edit['before_hex']); after=bytes.fromhex(edit['after_hex'])
 assert edit['dtype']=='BF16' and edit['shape']==[5376] and edit['size']==10752
 assert edit['operation']=='set' and edit['requested_value']==0.0
 assert edit['shard']==p['shard'] and edit['offset']==p['file_offset']
 assert hashlib.sha256(before).hexdigest()==p['before_sha256']
 assert after==bytes(10752) and hashlib.sha256(after).hexdigest()==p['after_sha256']
 with (candidate/edit['shard']).open('rb') as stream: stream.seek(edit['offset']); stored=stream.read(edit['size'])
 assert stored==after
 changed=sum(a!=b for a,b in zip(before,stored,strict=True)); total_changed_bytes+=changed
 assert changed==p['changed_bytes'] and edit['changed_count']==5376
 zero_rows.append({'tensor':edit['tensor'],'shard':edit['shard'],'offset':edit['offset'],'size':edit['size'],'after_sha256':p['after_sha256'],'all_positive_bf16_zero':True,'changed_elements':5376,'changed_bytes':changed})
assert total_changed_bytes==21476 and manifest['changed_elements']==10752 and manifest['changed_tensors']==2
assert math.isclose(manifest['delta_l2'],268.1548464079132,rel_tol=1e-15)
manifest_files={r['path']:r for r in manifest['files']}; chain_files={r['path']:r for r in chain['files']}
assert set(manifest_files)==set(chain_files)==set(source_inventory) and len(chain['files'])==12
for rel,row in chain_files.items():
 assert row['size']==source_inventory[rel]['size']==manifest_files[rel]['size']
 assert row['stage_sha256']==[manifest_files[rel]['source_sha256'],manifest_files[rel]['candidate_sha256']]
assert helper.inventory(source)==source_inventory and helper.inventory(candidate)==candidate_inventory
provenance={'kind':'fresh_full_chain_hashes_with_postverification_exact_stat_freeze','chain_verification':str(chain_path),'chain_verification_sha256':sha(chain_path),'chain_execution':str(execution_path),'chain_execution_sha256':sha(execution_path),'chain_exit_code':0,'chain_finished_at':execution['finished_at'],'manifest':str(manifest_path),'manifest_sha256':sha(manifest_path),'reconstructor_version':chain['reconstructor_version'],'implementation_sha256':chain['implementation_sha256'],'all_file_metadata_not_newer_than_chain_completion':True,'scope':chain['scope'],'extra_checkpoint_created':False,'full_hash_repeated_for_freeze':False,'limitations':'Uses normal filesystem stat continuity from fresh full chain completion; no GPU weight readback.'}
freezes=[]
for role,path,states,index in [('original',source,source_inventory,0),('candidate',candidate,candidate_inventory,1)]:
 files=[{'path':rel,'size':row['size'],'sha256':row['stage_sha256'][index]} for rel,row in sorted(chain_files.items())]
 tree=helper.digest(files)
 if role=='original': assert tree=='cab440f871a32b5ea1c12364db7dd35ac8768a83b413ca24bbc18d6478eccbd9'
 out=T/(role+'-freeze.json')
 value={'schema_version':1,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint_path':str(path),'candidate_inventory_stat':states,'all_checkpoint_files':files,'control_tree_sha256':tree,'proof':{**provenance,'model_role':role,'hash_field':'source_sha256' if index==0 else 'candidate_sha256'},'helper_source':str(helper_path),'helper_sha256':sha(helper_path)}
 save_new(out,value)
 ident=helper.identity(SimpleNamespace(candidate=path,freeze=out))
 assert ident['tree_sha256']==tree and ident['stat']==states
 freezes.append({'kind':role,'path':str(out),'sha256':sha(out),'tree_sha256':tree,'files':len(files),'inventory_stable':True,'helper_identity_passed':True})
assert helper.inventory(source)==source_inventory and helper.inventory(candidate)==candidate_inventory
result={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed','verifier_source_sha256':sha(Path(__file__)),'freezes':freezes,'chain':provenance,'literal_zero_rows':zero_rows,'changed_tensors':2,'changed_elements':10752,'changed_bytes':total_changed_bytes,'delta_l2':manifest['delta_l2'],'outside_declared_spans_verified_by_full_chain':True,'all_12_file_stage_hashes_match_manifest':True,'candidate_payload_reread_bytes':21504,'full_payload_reread_for_freeze':False,'service_operations':0,'checkpoint_writes':0,'weight_chmod':False}
save_new(T/'freeze-validation.json',result)
print(json.dumps({'status':result['status'],'freezes':freezes,'changed_bytes':total_changed_bytes,'full_payload_reread_for_freeze':False}))
