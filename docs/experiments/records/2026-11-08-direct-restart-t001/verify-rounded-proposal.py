"""Independent exact-rational check of preregistered BF16 proposal; no writes to weights."""
from fractions import Fraction
import pathlib,json,struct,hashlib,collections,datetime
from kairyu_bench.wsp.weights.editor import tensor_location,read_at,decode,encode
root=pathlib.Path(__file__).resolve().parent
spec=json.loads((root/"edit-spec.json").read_text())
base=pathlib.Path("/mnt/nvme/kairyu/model-volumes/gemma-4-31b-8gpu/models/gemma-4-31b-it")
def rational(code):
    exponent=(code>>7)&255;sign=-1 if code&32768 else 1
    assert 0<exponent<255
    return sign*Fraction(128+(code&127))*Fraction(2)**(exponent-134)
rows=[]
for edit in spec["edits"]:
    assert edit["scale"]==127/128
    shard,offset,size,dtype,shape=tensor_location(base,edit["tensor"])
    assert dtype=="BF16" and shape==[5376]
    before=read_at(base/shard,offset,size)
    after=encode([v*edit["scale"] for v in decode(before,dtype)],dtype)
    words=struct.unpack("<5376H",before);expected=[];ties=0;hist=collections.Counter()
    ratios=[]
    for b in words:
        assert not b&32768 and 0<(b>>7)<255
        exact=rational(b)*Fraction(127,128)
        distances={c:abs(rational(c)-exact) for c in range(b-4,b+1)}
        minimum=min(distances.values());nearest=[c for c,d in distances.items() if d==minimum]
        ties+=len(nearest)==2
        chosen=min(nearest,key=lambda c:c&1)
        expected.append(chosen);hist[b-chosen]+=1;ratios.append(float(rational(chosen)/rational(b)))
    expected_raw=struct.pack("<5376H",*expected)
    assert after==expected_raw
    rows.append({"tensor":edit["tensor"],"dtype":dtype,"shape":shape,"shard":shard,"offset":offset,"bytes":size,"before_sha256":hashlib.sha256(before).hexdigest(),"after_sha256":hashlib.sha256(after).hexdigest(),"bitwise_independent_match":True,"changed_elements":sum(a!=b for a,b in zip(words,expected)),"changed_bytes":sum(a!=b for a,b in zip(before,after)),"positive_normal_elements":len(words),"zeros":0,"subnormals":0,"ties":ties,"stored_code_decrement_histogram":dict(hist),"actual_factor_min":min(ratios),"actual_factor_max":max(ratios),"actual_factor_mean":sum(ratios)/len(ratios)})
receipt={"at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"spec_sha256":hashlib.sha256((root/"edit-spec.json").read_bytes()).hexdigest(),"script_sha256":hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),"method":"Exact integer-significand rational multiplication by127/128, nearest representable BF16 with even tie; independent from editor FP32 bit rounding", "rows":rows,"weights_written":False}
(root/"rounded-proposal-verification.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
