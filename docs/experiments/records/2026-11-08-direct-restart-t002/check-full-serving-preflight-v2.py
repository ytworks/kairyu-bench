"""Fixed T002 all-four serving preflight; read-only and bounded saved-vector reads."""
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
import datetime,hashlib,importlib.util,json,re,subprocess,sys
import yaml
R=Path('/mnt/nvme/kairyu/wsp/gemma4-coding-20261104'); T=R/'restart-direct-20261108/trial-002-final-branch-ablation'; W=R/'worktrees/kairyu-bench-weight-tools'
sys.path.insert(0,str(W/'src')); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
OUT=T/'full-serving-preflight-v2.json'; assert not OUT.exists()
result={'schema_version':1,'status':'checking','recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'replicas':{},'service_operations':0,'checkpoint_writes':0,'new_gpu_observation':False,'full_payload_rehashed':False}
def main():
 hp=W/'docs/experiments/scripts/redeploy-replica.py';spec=importlib.util.spec_from_file_location('t002_full_preflight_helper',hp);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
 plan=json.loads((T/'full-replica-switch-plan.json').read_text()); assert sha(hp)==plan['helper']['sha256'] and sha(Path(plan['control']['path']))==plan['control']['sha256']
 cf=Path(plan['candidate']['freeze']); assert sha(cf)==plan['candidate']['freeze_sha256']; freeze=json.loads(cf.read_text()); candidate=Path(freeze['checkpoint_path']); tree=freeze['control_tree_sha256']
 identity=h.identity(SimpleNamespace(candidate=candidate,freeze=cf)); assert identity['tree_sha256']==tree
 baseline_path=T/'full-candidate-baseline.json'; baseline=json.loads(baseline_path.read_text()); assert set(baseline['replicas'])=={'gemma-0','gemma-1','gemma-2','gemma-3'}
 manifest_path=T/'edit-manifest.json';manifest=json.loads(manifest_path.read_text());selected=[{k:e[k] for k in ('tensor','shard','offset','size','candidate_tensor_sha256')} for e in manifest['edits']]; assert len(selected)==2 and all(e['size']==10752 for e in selected)
 snippet="import pathlib,json,hashlib,sys; rows=json.loads(sys.argv[1]); out=[]\nfor r in rows:\n p=pathlib.Path('/checkpoint')/r['shard']\n with p.open('rb') as f: f.seek(r['offset']); raw=f.read(r['size'])\n out.append({'tensor':r['tensor'],'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'positive_zero_bytes':raw==bytes(r['size']),'manifest_hash_matches':hashlib.sha256(raw).hexdigest()==r['candidate_tensor_sha256']})\nprint(json.dumps(out))"
 evidence=[cf,baseline_path,manifest_path,T/'freeze-validation.json',T/'candidate-live-load-after-restart.json',T/'full-replica-switch-plan.json']
 containers={}
 for index in range(4):
  service=f'gemma-{index}';name=f'gemma-4-31b-8gpu-{service}-1';current=h.inspect(name);snapshot=h.compare_after(baseline['replicas'][service],current,candidate,tree)
  receipt_argv=plan['services'][service]['apply_argv'] if index else None;receipt_path=(Path(receipt_argv[receipt_argv.index('--output')+1])/'receipt.json') if index else T/'deployment-apply/receipt.json';receipt=json.loads(receipt_path.read_text());assert receipt['status']=='completed' and receipt['after']['id']==current['Id']
  assert receipt['candidate_identity']['tree_sha256']==tree and receipt['candidate_identity']['path']==str(candidate)
  if index==0:assert current['Id']==plan['prepared_runtime_snapshots'][service]['id'] and current['State']['StartedAt']==plan['gemma0_started_at'] and snapshot==plan['prepared_runtime_snapshots'][service]
  assert snapshot['checkpoint_readonly'] and snapshot['checkpoint_path']==str(candidate) and snapshot['tree']==tree and snapshot['running'] and snapshot['health']=='healthy'
  h.idle(current)
  proc=subprocess.run(['docker','exec','--user','1003',name,'python3','-B','-c',snippet,json.dumps(selected)],capture_output=True,timeout=30);assert proc.returncode==0
  mounted=json.loads(proc.stdout);assert len(mounted)==2 and all(v['positive_zero_bytes'] and v['manifest_hash_matches'] and v['bytes']==10752 for v in mounted)
  logs=subprocess.run(['docker','logs','--since',current['State']['StartedAt'],current['Id']],capture_output=True,timeout=30);assert logs.returncode==0
  logbytes=logs.stdout+logs.stderr; text=logbytes.decode(errors='replace'); startup={'log_bytes':len(logbytes),'log_sha256':hashlib.sha256(logbytes).hexdigest(),'loading_weights_timing_seen':'Loading weights took' in text,'model_loading_timing_seen':'Model loading took' in text,'application_startup_complete_seen':'Application startup complete' in text,'engine_initialization_failed_seen':'Engine core initialization failed' in text,'traceback_seen':'Traceback (most recent call last)' in text,'raw_log_saved':False}
  assert startup['application_startup_complete_seen'] and not startup['engine_initialization_failed_seen']
  result['replicas'][service]={'snapshot':snapshot,'started_at':current['State']['StartedAt'],'mounted_saved_vectors':mounted,'startup':startup,'helper_receipt':str(receipt_path),'helper_receipt_sha256':sha(receipt_path),'idle_at_check':True}; containers[service]=current;evidence.append(receipt_path)
 gateway_name='gemma-4-31b-8gpu-kairyu-1';gateway=h.inspect(gateway_name);old_gateway_path=R/'restart-direct-20261108/current-gateway-identity.json';old_gateway=json.loads(old_gateway_path.read_text());assert gateway['Image']==old_gateway['image'] and gateway['Created']==old_gateway['container_created'] and gateway['State']['Running']
 mounts=sorted([{'source':m['Source'],'destination':m['Destination'],'rw':m['RW']} for m in gateway['Mounts']],key=lambda m:m['destination']);assert mounts==sorted(old_gateway['mounts'],key=lambda m:m['destination'])
 config_mount=next(m for m in mounts if m['destination']=='/etc/kairyu/kairyu.yaml'); assert not config_mount['rw'];config_path=Path(config_mount['source']);config_stat=config_path.stat(); config_bytes=config_path.read_bytes(); config=yaml.safe_load(config_bytes)
 launch_path=R/'restart-direct-20261108/baseline-launch-preflight.json'; oldlaunch=json.loads(launch_path.read_text()); old_ns=int(datetime.datetime.fromisoformat(oldlaunch['at']).timestamp()*1e9);assert max(config_stat.st_mtime_ns,config_stat.st_ctime_ns)<=old_ns
 routes=config['pools']['gemma-4-31b']['replicas'];assert len(routes)==4;whitelist=[]
 for row in routes:
  url=urlsplit(row['options']['base_url']);health=urlsplit(row['health_url']);assert url.scheme=='http' and url.port==8000 and url.path=='/v1' and not url.username and not url.password and not url.query and not url.fragment;assert health.hostname==url.hostname and health.port==8000 and health.path=='/health';whitelist.append(url.hostname)
 assert sorted(whitelist)==['gemma-0','gemma-1','gemma-2','gemma-3']
 dns_code="import json,socket; print(json.dumps({s:sorted(set(x[4][0] for x in socket.getaddrinfo(s,8000,socket.AF_INET,socket.SOCK_STREAM))) for s in ['gemma-0','gemma-1','gemma-2','gemma-3']}))"
 dns=json.loads(subprocess.check_output(['docker','exec','--user','1003',gateway_name,'python3','-B','-c',dns_code],timeout=30,text=True))
 network='gemma-4-31b-8gpu_default';assert network in gateway['NetworkSettings']['Networks']
 for service,current in containers.items(): assert dns[service]==[current['NetworkSettings']['Networks'][network]['IPAddress']]
 result['gateway']={'container_id':gateway['Id'],'image':gateway['Image'],'container_created':gateway['Created'],'started_at':gateway['State']['StartedAt'],'original_image_and_container_creation_match':True,'original_mounts_match':True,'config_path':str(config_path),'config_sha256':hashlib.sha256(config_bytes).hexdigest(),'config_unchanged_since_original_preflight_by_stat':True,'config_stat':{'device':config_stat.st_dev,'inode':config_stat.st_ino,'size':config_stat.st_size,'mtime_ns':config_stat.st_mtime_ns,'ctime_ns':config_stat.st_ctime_ns},'route_hosts':sorted(whitelist),'route_dns_matches_candidate_ips':dns,'pool':'gemma-4-31b','raw_env_or_config_saved':False,'limits':'No historical full YAML hash in original gateway receipt; uses same image/container creation/RO mount and ordinary filesystem stat continuity plus current exact route whitelist/DNS.'}
 evidence.extend([old_gateway_path,launch_path]); assert h.inventory(candidate)==freeze['candidate_inventory_stat']
 for service,record in result['replicas'].items():
  c=h.inspect('gemma-4-31b-8gpu-'+service+'-1');assert h.snapshot(c)==record['snapshot'] and c['State']['StartedAt']==record['started_at']
 assert h.inspect(gateway_name)['Id']==gateway['Id'] and config_path.stat().st_ctime_ns==config_stat.st_ctime_ns
 result.update(status='passed',candidate_tree=tree,candidate_checkpoint=str(candidate),helper_sha256=sha(hp),checker_sha256=sha(Path(__file__)),candidate_payload_read_bytes=4*21504,evidence=[{'path':str(p),'sha256':sha(p)} for p in evidence],limitations=['Mounted saved-byte identity does not verify live GPU-resident weights.','Existing exact full-file chain hashes are reused through unchanged freeze metadata; no whole-checkpoint hash pass or copy was added.'])
try: main()
except BaseException as error:
 result.update(status='failed',failure_type=type(error).__name__)
 raise
finally:
 with OUT.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps({'status':result['status'],'path':str(OUT),'sha256':sha(OUT),'replicas':list(result['replicas'])}),flush=True)
