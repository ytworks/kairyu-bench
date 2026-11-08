"""Fixed T002 literal-positive-zero proof; read-only source weights, no model loading."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,math,os,stat,struct,sys
ROOT=Path('/mnt/nvme/kairyu/wsp/gemma4-coding-20261104')
T=ROOT/'restart-direct-20261108/trial-002-final-branch-ablation'
WT=ROOT/'worktrees/kairyu-bench-weight-tools'
sys.path.insert(0,str(WT/'src'))
from kairyu_bench.wsp.weights.editor import plan_edits
sha=lambda b:hashlib.sha256(b).hexdigest()
def stamp(p):
 s=p.stat(); return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def inventory(root):
 rows={}
 for p in sorted(root.rglob('*')):
  rel=p.relative_to(root)
  if '.cache' in rel.parts or rel.name=='.kairyu-model-attestation.json': continue
  assert not p.is_symlink()
  if p.is_dir(): continue
  assert stat.S_ISREG(p.stat().st_mode)
  rows[str(rel)]=stamp(p)
 return rows
registration=json.loads((T/'registration.json').read_text())
source=Path(registration['source_checkpoint'])
candidate=Path(registration['candidate_path'])
manifest=Path(registration['manifest_path'])
assert not candidate.exists() and not manifest.exists()
spec_path=T/'edit-spec.json'; spec_bytes=spec_path.read_bytes(); spec=json.loads(spec_bytes)
assert sha(spec_bytes)==registration['spec_sha256']=='86a0facdfff4c556ec1c01ec832cbda0a4418c45fb2a15107d225dc3022e1abd'
names=['model.language_model.layers.59.post_attention_layernorm.weight','model.language_model.layers.59.post_feedforward_layernorm.weight']
assert [x['tensor'] for x in spec['edits']]==names
for edit in spec['edits']:
 assert set(edit)=={'tensor','set'} and type(edit['set']) is float
 assert struct.pack('<d',edit['set'])==bytes(8)
source_before=inventory(source)
index=json.loads((source/'model.safetensors.index.json').read_text())
plans=plan_edits(source,spec)
assert len(plans)==2
rows=[]; total_square=Fraction(0); all_changed_bytes=0
for name,plan in zip(names,plans,strict=True):
 rel=index['weight_map'][name]; p=source/rel
 with p.open('rb') as f:
  headsize=struct.unpack('<Q',f.read(8))[0]
  assert 0<headsize<=16*2**20
  header=json.loads(f.read(headsize)); entry=header[name]
  assert entry['dtype']=='BF16' and entry['shape']==[5376]
  lo,hi=entry['data_offsets']; assert hi-lo==10752
  offset=8+headsize+lo; f.seek(offset); before=f.read(hi-lo)
 assert len(before)==10752
 codes=struct.unpack('<5376H',before)
 assert all(0<code<0x7f80 and (code>>7)!=0 for code in codes)
 values=[math.ldexp(128+(code&127),(code>>7)-134) for code in codes]
 assert all(math.isfinite(value) and value>0 for value in values)
 after=bytes(len(before))
 changed_bytes=sum(byte!=0 for byte in before)
 sq=sum((Fraction.from_float(value)**2 for value in values),Fraction(0))
 l2=math.sqrt(float(sq)); total_square+=sq; all_changed_bytes+=changed_bytes
 assert plan['tensor']==name and plan['offset']==offset and plan['shard']==rel
 assert plan['before_hex']==before.hex() and plan['after_hex']==after.hex()
 assert plan['candidate_tensor_sha256']==sha(after) and plan['source_tensor_sha256']==sha(before)
 assert plan['changed_count']==5376 and plan['changed_indices']==list(range(5376))
 assert math.isclose(plan['delta_l2'],l2,rel_tol=1e-15,abs_tol=0)
 rows.append(dict(tensor=name,shard=rel,file_offset=offset,dtype='BF16',shape=[5376],before_sha256=sha(before),after_sha256=sha(after),before_positive_normal_finite=True,after_all_positive_zero=True,after_bf16_code='0x0000',span_bytes=len(before),changed_elements=5376,changed_bytes=changed_bytes,delta_l2=l2,delta_l2_squared_exact={'numerator':sq.numerator,'denominator':sq.denominator},before_min=min(values),before_max=max(values),editor_proposed_bytes_match=True))
assert inventory(source)==source_before and spec_path.read_bytes()==spec_bytes
preview=json.loads((T/'editor-preview.stdout').read_text())
copy_bytes=sum(v[2] for v in source_before.values())
assert copy_bytes==preview['copy_bytes_upper_bound']==62578686256
assert preview['applied'] is False and preview['changed_elements']==10752 and preview['changed_tensors']==2
assert math.isclose(preview['delta_l2'],math.sqrt(float(total_square)),rel_tol=1e-15,abs_tol=0)
assert not candidate.exists() and not manifest.exists()
proof=dict(schema_version=1,trial_id=registration['trial_id'],scope='Independent literal positive-zero saved-byte construction for exactly two BF16 vectors; compares native planned bytes, not a created checkpoint',source=str(source),candidate_not_created=True,manifest_not_created=True,spec_sha256=sha(spec_bytes),proof_source_sha256=sha(Path(__file__).read_bytes()),source_module_sha256={name:sha((WT/'src/kairyu_bench/wsp/weights'/name).read_bytes()) for name in ('editor.py','checkpoint.py')},editor_preview_sha256=sha((T/'editor-preview.stdout').read_bytes()),changed_tensors=2,changed_elements=10752,changed_bytes=all_changed_bytes,edited_span_bytes=sum(x['span_bytes'] for x in rows),delta_l2=math.sqrt(float(total_square)),delta_l2_squared_exact={'numerator':total_square.numerator,'denominator':total_square.denominator},source_files_count=len(source_before),source_files=[dict(path=k,size=v[2],stat_stamp=v) for k,v in source_before.items()],copy_bytes_upper_bound=copy_bytes,source_stat_stable=True,full_file_payload_hashes_recomputed=False,rows=rows)
with (T/'literal-zero-proof.json').open('x') as out: json.dump(proof,out,indent=2,allow_nan=False); out.write('\n')
print(json.dumps({k:proof[k] for k in ('changed_tensors','changed_elements','changed_bytes','edited_span_bytes','delta_l2','source_files_count','copy_bytes_upper_bound','candidate_not_created','source_stat_stable')},allow_nan=False))
