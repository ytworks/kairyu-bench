# Native stored-weight scale observation, 2026-11-07

This is a CPU reread of the previously complete AV native statistics, not a new
checkpoint scan, derivative, activation measurement or model trial. The original
uncompressed observation SHA is `e86674f2df762023885ea84fd4dd01cbb2fb43a74db40a354f1d25119abb2d58`.
All421 text-native tensors contain1,331,516 finite values, combined L2
70,134.96462348501. Family L2 values differ substantially:

| Family | Tensors | Elements | L2 |
|---|---:|---:|---:|
| input_layernorm | 60 | 322560 | 40876.254276 |
| pre_feedforward_layernorm | 60 | 322560 | 56934.280568 |
| post_attention_layernorm | 60 | 322560 | 558.637725 |
| post_feedforward_layernorm | 60 | 322560 | 781.580567 |
| self_attn.q_norm | 60 | 17920 | 134.593258 |
| self_attn.k_norm | 60 | 17920 | 14.753693 |
| layer_scalar | 60 | 60 | 6.239438 |
| final text norm | 1 | 5376 | 2364.630838 |

Each family norm is the square root of the sum of its complete tensor squared
norms. Per-tensor quantiles in the source are explicitly sampled: for example,
layer11 pre-feedforward has full L2 21386.452742 and full range[-10.4375,544],
but its recorded median268 is a4096-element sample. These must not be called
full quantiles or a causal coding assignment. The largest weight alone does
not select the next update support.

A small ratio using the whole native norm can hide larger changes in a
post-attention norm or layer scalar. Any later proposal must therefore report
actual BF16 changed values and per-tensor/per-family denominators, not only a
whole-native ratio or changed-element count. No fixed number of tensors, edit
amplitude, objective or candidate is selected by this observation.

Source: [complete native statistics](evidence-2026-11-07-fresh-audit/av-native-stats.json.gz).
The exact read-only aggregation is:

```python
from collections import defaultdict
import gzip, json, math
from pathlib import Path
p = Path("docs/experiments/evidence-2026-11-07-fresh-audit/av-native-stats.json.gz")
d = json.loads(gzip.decompress(p.read_bytes()))
groups = defaultdict(lambda: {"tensors": 0, "elements": 0, "squared_norm": 0.0})
for tensor in d["tensors"]:
    name = tensor["name"]
    family = name.split(".", 4)[-1] if ".layers." in name else name
    group = groups[family]
    group["tensors"] += 1
    group["elements"] += tensor["numel"]
    group["squared_norm"] += tensor["statistics"]["l2"] ** 2
for group in groups.values():
    group["l2"] = math.sqrt(group.pop("squared_norm"))
```
