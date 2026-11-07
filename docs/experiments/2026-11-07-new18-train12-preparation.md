# New18: train12 native generation and CPU frame preparation

Preparation only: model/API/GPU calls0, new containers0, service changes0,
lock acquisition0. Final authored bundle and its correct/foil fixture audit are
pending. Initial generation uses train12 only; transfer6 generation waits until
candidate spec/amplitude freeze. Local length inspection of transfer is not a
behavior measurement or an optimization input.

The [non-executable plan template](records/2026-11-07-new18-train12-preparation/plan-template.json)
has approval=false and missing case SHA/IDs. [Read-only evidence](records/2026-11-07-new18-train12-preparation/preparation-evidence.json)
records four exact AV CIDs, image af9e746b…, read-only checkpoint mounts, default
thinking=true, nine AV file-stat continuities and 23 protected containers.
AV freeze SHA0fbf034b… refers to prior full-content hashes; no new payload hash
pass occurred. Free NVMe71.14GB, all8GPU utilization0% at preparation; actual
execution must recheck idle. No current sandbox source pin is finalized while its
empty-array transport correction is under audit.

## Single train12 owner recipe

Keep the fixed120 owner unchanged. A small dedicated one-batch owner will:

1. Pin final train envelope/source/Python/owner/generator plus AV identity and
   serving evidence. Require exactly behavior-A1,A2,B1,B2,C1,C2,D1,D2,E1,E2,F1,F2
   (every suffix has the behavior- prefix), six purposes × two singleton folds,
   all split=train, unchanged authored row fields. Transfer A3…F3 is excluded.
   Validate through science_generate.public_cases; preview sends no HTTP.
2. After concrete plan approval, acquire nonblocking candidate-evaluation.lock.
   Recheck pins and all9 AV stats, exact4 Compose members/CIDs/images, healthy
   state, read-only AV mounts/tree5d056eb0…, unchanged command hashes, TP2/MTP4/
   native flags and default thinking=true. Check pinned gateway CID/image,
   direct model IDs and running/waiting=0 on all4 before requests.
3. Issue one generator subprocess, no retry. On SIGINT/SIGTERM set stop-requested,
   mask repeated termination while waiting for this child and issued requests
   to finish, and keep the lock. Do not kill the generator or start another batch.
4. In finally, retain all12 response/missing/error IDs, request IDs, token counts,
   finish reasons, receipt, return code and elapsed time. Verify source/input/
   checkpoint continuity and protected23/four-serving closure; confirm requests
   idle before releasing lock. If closure is unresolved, report it and retain
   ownership rather than start a successor. SIGKILL cannot be handled: successor
   must independently prove idle. Errors and missing rows are unknown, not scores.

Preview command, with final frozen TRAIN, unique PREVIEW_OUT, existing CI PY and
pinned science_generate.py GEN (the generator creates its preview directory):

~~~bash
"$PY" "$GEN" --cases "$TRAIN" --split train \
 --endpoint http://172.26.0.9:8000/v1 --model gemma-4-31b \
 --checkpoint-sha256 5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf \
 --output "$PREVIEW_OUT" --concurrency 16 --max-tokens 32768 \
 --timeout 1200 --seed 0 --thinking-control server-default
~~~

Only after root approval append --run with a different new output. This preserves
native120-v2 conditions: concurrency ceiling16 (actual12), native server sampling
without temperature/top_p overrides, seed0, timeout1200, max_tokens32768, retry0.
Only original prompts are sent. server-default omits unsupported request kwargs;
serving command evidence separately proves thinking=true. Seed0 is not a promise
of future bitwise reproducibility. Public answers only are stored.

## CPU-only length check, all18

After bundle freeze, use local immutable image
sha256:af9e746b913088cbf6c4464a31d6904a6b7e1d3ba62f0895362ddf13595f7abe
(9,738,554,834 bytes). Flags: --pull=never --runtime=runc --network=none
--user 1003:1003 --read-only --cap-drop=ALL --security-opt=no-new-privileges
--pids-limit 128 --memory 4g --cpus 2. No GPU/device/socket mounts; override entrypoint
with python, never vLLM. Pin/mount package source, AV checkpoint and case inputs
read-only; only unique NVMe output/cache is writable. CUDA_VISIBLE_DEVICES empty,
NVIDIA_VISIBLE_DEVICES=void, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1,
PYTHONDONTWRITEBYTECODE=1, TOKENIZERS_PARALLELISM=false, OMP_NUM_THREADS=1;
TMPDIR/HF_HOME/XDG_CACHE_HOME under output.

Before/after: verify source/input hashes and AV all9 stat continuity against full
attestation, actual image ID, small tokenizer/config/template hashes. Save package
versions and bounded/noGPU/no-network container inspection. No new full weight
hash, model load, forward, gradient or generation. Do not call read_cases on
transfer: its gradient exclusion stays intact. Read both JSON envelopes solely
for this length operation; verify exact12+6 IDs, splits and full source equality.

Core code to freeze with final inputs, executed only after separate CPU approval:

~~~python
from pathlib import Path
import json
from transformers import AutoTokenizer
from kairyu_bench.wsp.weights.finite_replay import prepare_frames, validate_frame_rows
train = json.loads(Path("/inputs/train-cases.json").read_text())
transfer = json.loads(Path("/inputs/transfer-cases.json").read_text())
assert transfer["split"] == "transfer" and transfer["gradient_allowed"] is False
groups = train.get("cases", train)
tr = [r for folds in groups.values() for fold in folds for r in fold]
te = transfer["cases"]
assert [r["id"] for r in tr] == [f"behavior-{a}{i}" for a in "ABCDEF" for i in (1,2)]
assert [r["id"] for r in te] == [f"behavior-{a}3" for a in "ABCDEF"]
assert all(r["split"] == "train" for r in tr)
assert all(r["split"] == "transfer" for r in te)
assert len({r["id"] for r in tr+te}) == 18
assert not list(Path("/dev").glob("nvidia*"))
tok = AutoTokenizer.from_pretrained("/checkpoint", local_files_only=True, trust_remote_code=False)
prepared, evidence = prepare_frames(tok, {**groups,"transfer_length_only":[[r] for r in te]},
                                    max_frame_tokens=None)
rows = [r for folds in prepared.values() for fold in folds for r in fold]
evidence["lengths"] = [{"id":r[0],"prefix":len(r[1]),"correct":len(r[2]),"foil":len(r[3]),
                      "correct_complete":len(r[1])+len(r[2]),
                      "foil_complete":len(r[1])+len(r[3])} for r in rows]
evidence.update(expected_count=18,model_forwards=0,generation_calls=0,gradient_calls=0)
evidence["over_limit_ids"] = [r[0] for r in rows if len(r[1])+max(len(r[2]),len(r[3]))>1024]
Path("/out/frame-evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
validate_frame_rows(rows,1024)  # Records all lengths before rejecting oversize.
~~~

Never delete or truncate cases. Retain source/template/input/output SHA, complete
lengths for both answers, all18IDs, maximum, failures, elapsed time and continuity.
The shared protocol is gemma4-public-answer-v2-forced-empty-thought: complete
prefix plus its existing stripped answer, excluding closing-turn tokens.
Its existing strip removes boundary whitespace; record that convention. No token
sequence is cut. This differs from native generated-thinking context and does
not certify native answer lengths or capability.

## Evidence and pending work

Generator SHA df88767ebcd5a3138747a36fe7e0976dbb0c2ceb5b0f1eece74740818f0b7f89;
frame source SHA71c424bffac775ba112e709632df0a37351a79c3121827be7c9844f321c7c4d3,
finite_replay.py lines90–92 and312–375. AV identity receipt
analysis/fresh-purpose-20261107-av-live10-v1/checkpoint-identities.json,
SHA a679fa4a7b85a2465108da794e0d9cf089acb370b9f5e616ffa76f0dd040e4eb,
uses its current member. Full source pins and prior serving closures are in plan.

Still pending: completed authored audit/bundle SHA, all18 length measurement,
dedicated one-batch owner implementation/review/preview, final approval. The
existing120 owner is a flock/child-wait reference only. No capability result is
claimed. Read-only source audit once requested nonexistent weights/reader.py and
then used actual checkpoint.py; a documentation tool-call string had a JavaScript
parse error before any execution and was rewritten. Neither caused requests,
benchmark retries or service changes.


Actual CPU results and the finished dedicated owner are recorded in [the readiness follow-up](2026-11-07-behavior18-frame-and-train12-readiness.md). The af9 Python entrypoint is /usr/bin/python3; the initial python assumption failed before execution and is retained.
