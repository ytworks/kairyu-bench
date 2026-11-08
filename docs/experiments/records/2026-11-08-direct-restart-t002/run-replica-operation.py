"""Run one preregistered replica operation, keeping credential values in memory."""
import pathlib,json,os,sys,subprocess,hashlib,time,fcntl
t=pathlib.Path(__file__).resolve().parent;r=t.parent.parent;w=r/"worktrees/kairyu-bench-weight-tools"
mode=sys.argv[1];assert mode in ("preview","apply","restore_preview","restore_apply")
plan=json.loads((t/"single-replica-switch-plan.json").read_text());args=plan[mode+"_argv"]
lock=(r/"candidate-evaluation.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
d=json.loads(subprocess.check_output(["docker","inspect","gemma-4-31b-8gpu-chat-ui-1"]))[0]
available=dict(x.split("=",1) for x in d["Config"]["Env"] if "=" in x)
env=dict(os.environ,PYTHONPATH=str(w/"src"),PYTHONDONTWRITEBYTECODE="1")
for k in ("WEBUI_ADMIN_EMAIL","WEBUI_ADMIN_PASSWORD"):env[k]=available[k]
start=time.time();p=subprocess.run(args,cwd=w,env=env,capture_output=True)
(t/(mode+"-execution.json")).write_text(json.dumps({"argv":args,"returncode":p.returncode,"seconds":time.time()-start,"stdout_bytes":len(p.stdout),"stdout_sha256":hashlib.sha256(p.stdout).hexdigest(),"stderr_bytes":len(p.stderr),"stderr_sha256":hashlib.sha256(p.stderr).hexdigest(),"environment_saved":False},indent=2)+"\n")
print(json.dumps({"mode":mode,"returncode":p.returncode,"seconds":time.time()-start}),flush=True)
raise SystemExit(p.returncode)
