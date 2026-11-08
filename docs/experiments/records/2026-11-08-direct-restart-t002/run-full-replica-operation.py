"""Run one fixed full-phase replica operation; root invokes it sequentially."""
import pathlib,json,os,sys,subprocess,hashlib,time,fcntl
t=pathlib.Path(__file__).resolve().parent;r=t.parent.parent;w=r/'worktrees/kairyu-bench-weight-tools'
assert len(sys.argv)==3,'usage: run-full-replica-operation.py gemma-N preview|apply|restore_preview|restore_apply'
service,mode=sys.argv[1:];assert mode in ('preview','apply','restore_preview','restore_apply')
plan=json.loads((t/'full-replica-switch-plan.json').read_text());args=plan['services'][service][mode+'_argv']
out=t/f'full-{service}-{mode}-execution.json';assert not out.exists()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for key in ('helper','control'):assert sha(plan[key]['path'])==plan[key]['sha256']
for key in ('candidate','original'):assert sha(plan[key]['freeze'])==plan[key]['freeze_sha256']
lock=(r/'candidate-evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if not mode.startswith('restore'):
 assert sha(plan['screen']['path'])==plan['screen']['sha256']
 order=plan['candidate_order'];index=order.index(service)
 if index:
  previous=plan['services'][order[index-1]]['apply_argv'];previous_output=pathlib.Path(previous[previous.index('--output')+1]);assert json.loads((previous_output/'receipt.json').read_text())['status']=='completed'
 c=json.loads(subprocess.check_output(['docker','inspect','gemma-4-31b-8gpu-gemma-0-1']))[0]
 assert c['Id']==plan['prepared_runtime_snapshots']['gemma-0']['id'] and c['State']['StartedAt']==plan['gemma0_started_at'] and c['State'].get('Health',{}).get('Status')=='healthy'
if mode in ('apply','restore_apply'):
 preview='restore_preview' if mode=='restore_apply' else 'preview';a=plan['services'][service][preview+'_argv'];p=pathlib.Path(a[a.index('--output')+1])/'receipt.json';assert json.loads(p.read_text())['status']=='preview'
d=json.loads(subprocess.check_output(['docker','inspect','gemma-4-31b-8gpu-chat-ui-1']))[0];available=dict(x.split('=',1) for x in d['Config']['Env'] if '=' in x)
env=dict(os.environ,PYTHONPATH=str(w/'src'),PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(r/'tool-validation/ci-tmp'))
for k in ('WEBUI_ADMIN_EMAIL','WEBUI_ADMIN_PASSWORD'):env[k]=available[k]
record={'service':service,'mode':mode,'argv':args,'status':'running','started_epoch':time.time(),'environment_saved':False,'plan_sha256':sha(t/'full-replica-switch-plan.json')}
with out.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
try:
 p=subprocess.run(args,cwd=w,env=env,capture_output=True)
 record.update(status='completed' if p.returncode==0 else 'process_failed',returncode=p.returncode,stdout_bytes=len(p.stdout),stdout_sha256=hashlib.sha256(p.stdout).hexdigest(),stderr_bytes=len(p.stderr),stderr_sha256=hashlib.sha256(p.stderr).hexdigest())
except BaseException as error:
 record.update(status='launcher_failed',error_type=type(error).__name__);raise
finally:
 record['seconds']=time.time()-record['started_epoch'];out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'service':service,'mode':mode,'returncode':p.returncode,'seconds':record['seconds']}),flush=True)
raise SystemExit(p.returncode)
