"""One CPU pass over existing AV gradients; no solver, candidate, model or service call."""
from pathlib import Path
import json, re, sys, time
import numpy as np
R=Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104");B=R/"worktrees/kairyu-bench-weight-tools"
N=R/"analysis/fresh-purpose-20261107-av-behavior12-native-gpu4-v2"
D=R/"analysis/fresh-purpose-20261107-native-relative-energy-v1"
sys.path[:0]=[str(B/"scripts/experiments/gemma4-purpose"),str(B/"src")]
import joint
started=time.time()
receiptpath=B/"docs/experiments/records/2026-11-07-native-dependency-v2/receipt.json"
old=json.loads(receiptpath.read_text())["all_input_sha256"]
closure=N/"measurement-closure.json"
assert joint.common.sha(closure)=="e889b331d80e724c4313cda5f9cec672d3f1752364e268e4c9b415d12eb897c3"
constraints=[dict(purpose=p,fold=f,minimum_delta=0.)for p in "ABCDEF"for f in range(2)]
arrays=[];pins={str(receiptpath):joint.common.sha(receiptpath),str(closure):joint.common.sha(closure)}
rowlabels=[]
for objective in ["correct-logprob","margin"]:
    base,g,_,layout,coords,labels,hashes,mode,provenance=joint.read_collection(N/"work"/objective,{"constraints":constraints})
    assert len(layout)==421 and base.size==1331516 and g.shape==(12,1331516) and coords is None
    assert mode=="chat-public-answer" and provenance["objective_kind"]==objective
    assert all(old[p]==h for p,h in hashes.items())
    if arrays:assert np.array_equal(base,previous) and layout==priorlayout
    previous=base.copy();priorlayout=layout;arrays.append(g);pins.update(hashes)
    rowlabels.extend(dict(objective=objective,id="behavior-"+r["purpose"]+str(r["fold"]+1))for r in labels)
g=np.concatenate(arrays);del arrays
scale=np.maximum(np.abs(base),1e-4)
family={};offset=0
for entry in layout:
    name=re.sub(r"^model.language_model.layers.\d+\.","",entry["tensor"])
    name=name.removeprefix("model.language_model.")
    count=entry["count"];sl=slice(offset,offset+count);part=g[:,sl];rel=part*scale[sl]
    if name not in family:family[name]=dict(coordinates=0,tensors=0,source_l2_squared=0.,floor_coordinates=0,raw=np.zeros((24,24)),relative=np.zeros((24,24)))
    f=family[name];f["coordinates"]+=count;f["tensors"]+=1;f["source_l2_squared"]+=float(base[sl]@base[sl]);f["floor_coordinates"]+=int(np.count_nonzero(np.abs(base[sl])<1e-4))
    f["raw"]+=part@part.T;f["relative"]+=rel@rel.T;offset+=count
assert offset==base.size and len(family)==8
def geometry(gram):
    norm=np.sqrt(np.diag(gram));assert np.all(norm>0)
    return dict(energy=np.diag(gram).tolist(),gradient_l2=norm.tolist(),cosine=(gram/np.outer(norm,norm)).tolist())
metrics={k:geometry(sum(f[k]for f in family.values()))for k in ["raw","relative"]}
for f in family.values():
    for key in ["raw","relative"]:
        energy=np.diag(f[key]);f[key]=dict(energy=energy.tolist(),fraction=(energy/np.asarray(metrics[key]["energy"])).tolist())
    f["source_l2"]=float(np.sqrt(f.pop("source_l2_squared")))
success=list(range(1,11))
focus={key:{rowlabels[i]["id"]:{"LL_against_successful_cases":[dict(id=rowlabels[j]["id"],cosine=value["cosine"][i][j])for j in success],"LL_margin_cosine":value["cosine"][i][i+12]}for i in [0,11]}for key,value in metrics.items()}
for mod in list(sys.modules.values()):
    p=getattr(mod,"__file__",None)
    if p and str(B) in p and p.endswith(".py"):pins[p]=joint.common.sha(Path(p))
pins[str(Path(__file__).resolve())]=joint.common.sha(Path(__file__))
assert all(joint.common.sha(Path(p))==h for p,h in pins.items())
report=dict(status="completed",seconds=time.time()-started,definition="g_relative = g * max(abs(weight), 1e-4); coordinate x=delta_weight/scale; no proposed delta",source_checkpoint="AV",native_tensors=421,native_coordinates=1331516,row_labels=rowlabels,full_24_by_24=metrics,families=family,focus=focus,input_sha256=pins,model_calls=0,gpu_calls=0,solver_calls=0,candidate_created=False,base_selected=False,limitations=["Local AV teacher-forced derivatives, not finite BF16 or free-generation gains.","Relative cost depends on explicit floor and has no demonstrated superiority.","All eight families and all 24 objective rows retained; no support selection."])
p=D/"report.json";assert not p.exists();p.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(json.dumps(dict(status="completed",seconds=report["seconds"],report_sha256=joint.common.sha(p),family_count=8,row_count=24,source_pins=len(pins))))
