import datetime, hashlib, json, pathlib, re, subprocess, sys
root=pathlib.Path('/mnt/nvme/kairyu/wsp/gemma4-coding-20261104')
trial=root/'restart-direct-20261108/trial-002-final-branch-ablation'
assert len(sys.argv) in (1,2,3), 'usage: audit-live-load.py [output-filename] or [relative-deployment-receipt output-filename]'
receipt_relative=sys.argv[1] if len(sys.argv)==3 else 'deployment-apply/receipt.json'
output_name=sys.argv[-1] if len(sys.argv)>1 else 'candidate-live-load-receipt.json'
assert output_name in ('candidate-live-load-receipt.json','candidate-live-load-after-restart.json')
receipt_path=(trial/receipt_relative).resolve()
assert receipt_path.is_relative_to(trial) and receipt_path.name=='receipt.json'
assert not (trial/output_name).exists(), 'output must be fresh'
deploy=json.loads(receipt_path.read_text())
if deploy['status']!='completed':
 print(json.dumps({'status':'waiting_for_deployment','deployment_status':deploy['status']}));raise SystemExit(2)
name='gemma-4-31b-8gpu-gemma-0-1'
def field(key):return json.loads(subprocess.check_output(['docker','inspect','--format','{{json '+key+'}}',name],text=True))
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
before=deploy['before'];identity=deploy['candidate_identity'];freeze_path=trial/'candidate-freeze.json';freeze=json.loads(freeze_path.read_text());manifest_path=trial/'edit-manifest.json';manifest=json.loads(manifest_path.read_text())
configcmd=field('.Config.Cmd');entry=field('.Config.Entrypoint');env=dict(v.split('=',1) for v in field('.Config.Env') if '=' in v)
mounts=sorted([{'type':m['Type'],'source':m['Source'],'target':m['Destination'],'read_only':not m['RW']} for m in field('.Mounts')],key=lambda m:m['target'])
after={'id':field('.Id'),'image_id':field('.Image'),'command_sha256':digest({'entrypoint':entry,'cmd':configcmd}),'environment_sha256':digest(env),'running':field('.State.Running'),'health':field('.State.Health.Status'),'started_at':field('.State.StartedAt'),'tree':field('.Config.Labels').get('kairyu.chat-checkpoint-tree')}
expectedmounts=[dict(m) for m in before['mounts']]
for m in expectedmounts:
 if m['target']=='/checkpoint':m['source']=identity['path'];m['read_only']=True
checks={'deployment_completed':True,'new_container':after['id']!=before['id'],'same_image':after['image_id']==before['image_id'],'same_command_hash':after['command_sha256']==before['command_sha256'],'same_environment_hash':after['environment_sha256']==before['environment_sha256'],'only_checkpoint_mount_changed':mounts==expectedmounts,'tree_matches_frozen_candidate':after['tree']==identity['tree_sha256']==freeze['control_tree_sha256'],'candidate_checkpoint_path_matches':identity['path']==freeze['checkpoint_path'],'running_healthy':after['running'] and after['health']=='healthy'}
initial_load_path=trial/'candidate-live-load-receipt.json'
if output_name=='candidate-live-load-after-restart.json':
 initial_load=json.loads(initial_load_path.read_text())
 checks['initial_load_verified']=initial_load['status']=='verified_saved_mount_and_runtime_identity'
 checks['same_container_id_after_restart']=after['id']==initial_load['container']['id']
 checks['new_started_at_after_restart']=datetime.datetime.fromisoformat(after['started_at'].replace('Z','+00:00'))>datetime.datetime.fromisoformat(initial_load['container']['started_at'].replace('Z','+00:00'))
stat_mismatches=[]
for rel,wanted in freeze['candidate_inventory_stat'].items():
 s=(pathlib.Path(identity['path'])/rel).stat();actual={'device':s.st_dev,'inode':s.st_ino,'size':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,'uid':s.st_uid,'gid':s.st_gid,'mode':s.st_mode,'nlink':s.st_nlink}
 if actual!=wanted:stat_mismatches.append(rel)
checks['candidate_file_stats_match_completed_chain_freeze']=not stat_mismatches
selected=[{k:e[k] for k in ['tensor','shard','offset','size','candidate_tensor_sha256']} for e in manifest['edits']]
code='''import hashlib,json,pathlib,sys
selected=json.loads(sys.argv[1]);out=[]
for row in selected:
 p=pathlib.Path('/checkpoint')/row['shard']
 with p.open('rb') as f:f.seek(row['offset']);data=f.read(row['size'])
 out.append({'tensor':row['tensor'],'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'matches_saved_manifest':len(data)==row['size'] and hashlib.sha256(data).hexdigest()==row['candidate_tensor_sha256'], 'all_positive_bf16_zero':data==bytes(row['size'])})
print(json.dumps(out))'''
proc=subprocess.run(['docker','exec','--user','1003',name,'python3','-B','-c',code,json.dumps(selected)],capture_output=True,text=True)
assert proc.returncode==0,'read-only mounted-vector verification failed'
mounted=json.loads(proc.stdout);checks['mounted_target_saved_bytes_match']=all(x['matches_saved_manifest'] and x['all_positive_bf16_zero'] for x in mounted)
logs=subprocess.run(['docker','logs','--since',after['started_at'],after['id']],capture_output=True)
logbytes=logs.stdout+logs.stderr; text=logbytes.decode(errors='replace')
startup={'docker_logs_exit_code':logs.returncode,'bytes':len(logbytes),'sha256':hashlib.sha256(logbytes).hexdigest(),'loading_checkpoint_shards_seen':'Loading safetensors checkpoint shards' in text,'checkpoint_shards_100_percent_seen':bool(re.search(r'Loading safetensors checkpoint shards[^\n\r]*100%',text)),'loading_weights_timing_seen':'Loading weights took' in text,'model_loading_timing_seen':'Model loading took' in text,'application_startup_complete_seen':'Application startup complete' in text,'traceback_seen':'Traceback (most recent call last)' in text,'engine_initialization_failed_seen':'Engine core initialization failed' in text,'raw_log_saved':False}
assert after['id']==field('.Id') and after['started_at']==field('.State.StartedAt'),'container changed during audit'
refs=[receipt_path,freeze_path,manifest_path,trial/'chain-verification.json',trial/'mtp-kv-prefix-chunked-addon.json',trial/'runtime-zero-ablation-audit.json',trial/'review-condition-resolution.json']
if output_name=='candidate-live-load-after-restart.json':refs.append(initial_load_path)
result={'schema_version':1,'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'verified_saved_mount_and_runtime_identity' if all(checks.values()) else 'verification_failed','checks':checks,'container':{'name':name,**after},'checkpoint_mount':next(m for m in mounts if m['target']=='/checkpoint'),'candidate_tree_sha256':identity['tree_sha256'],'candidate_stat_mismatches':stat_mismatches,'mounted_saved_target_vectors':mounted,'startup_log_summary':startup,'limitations':['Mounted saved vector bytes and source/config loader evidence do not constitute live GPU parameter-byte readback.','Whole-file candidate hashes are reused from the completed streaming chain proof while all frozen file stat metadata are unchanged; no redundant complete checkpoint or full hash pass was created.','No weight API, model import, GPU operation, evaluation, or service mutation was used by this audit.'],'live_gpu_bytes_verified':False,'evidence':[{'path':str(q),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in refs]}
out=trial/output_name
with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'path':str(out),'status':result['status'],'checks':checks,'startup_log_summary':startup,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'bytes':out.stat().st_size}))
