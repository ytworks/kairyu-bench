# Actual Opus 5.5 outcome review; strict format failure retained

This follow-up concerns the already measured F02 candidate. It does not create,
adopt, rerank or rerun a candidate. The full AV dev30 run continued independently.
The [anonymous prompt](anonymous-f02-outcome-review-2026-11-07.md) includes the
complete code tie, other-area gains and losses, failed proxy constraints,
case-margin concentration, all-case contract audit, stored-weight denominators,
activation coverage and missingness, and the distinct pending dev30 cohort.
No credential, private path, URL, raw response stream or reasoning text was sent.
Prompt SHA256: `bb1656564e953947520345bd6b308435c82198b0f83acda1b3755f0f9aa1c0d2`.

One external dispatch returned in 51.628986 seconds, cost USD 0.141616. CLI return
code was zero, is_error false, tool uses zero. Assistant models, usage key and
canonicalModel all identify **claude-opus-5-5**. Nevertheless the strict review
schema failed: it required one reviews entry labeled A, but the response contains
two entries both labeled A. Accordingly the original receipt retains
`verified_requested_model=false`, `review_usable=false`, `labels_match=false` and
`review_format=unusable`. This combined flag includes format acceptance and is
not a finding that another model responded. No flag or response was rewritten,
and no retry was sent to obtain a preferred answer or format.

The [original receipt](artifacts/f02-opus-outcome-receipt-20261107.json) SHA is
`e7df0f19293817b54cdf0bddf1f0910e5404ce6d387a3bfa1663189df2651393`.
A separate [identity-only audit](artifacts/f02-opus-outcome-identity-audit-20261107.json)
checks the three model fields, return code, error/tool status and the exact
format mismatch. It does not convert the receipt to a passed review. The final
structured advice was retained by the existing tool despite the failed label
count, so root can read it as nonconforming advice with its limitation explicit.
The earlier pre-materialization F01/F02 successful reviews are separate records.

## Advice and disposition

The reviewer regards F02 as a code null result and the margin objective as a
plausible weak point, while correctly withholding a conclusion that sparse
support itself was insufficient. It recommends first measuring executable
native-thinking behavior on a frozen authored catalog, including a parent
ceiling check, with all cases and predeclared validity strata retained. Depending
on headroom and paired behavioral changes, later work could revisit the objective,
per-case influence and generated-context alignment. It does not recommend adopting
F02 from authored outcomes or enlarging support solely because official code tied.

Root accepts the need for a behavioral headroom check and rejects treating the
existing margin gain or changed activation RMS as success. The strict-code-gain
rule still leaves AV as parent. No new solve, support selection or candidate is
bound by this review. The review's assumption that the existing sandbox can
execute all 120 mixed cases is not established: it currently supports five fixed
code catalogs, while the mixed cases include short answers and workflows. A
minimum next measurement must first specify valid executable contracts and an
honest unknown category. The 11 flawed cases must not be silently repaired or
removed. Its proposed parent/candidate comparison is conditional on useful parent
headroom; this document does not authorize another F02 deployment or official
rerun. Development data remain available for the user's authorized tuning;
review advice to reserve the new dev30 baseline does not override that instruction.

## Invocation and failures

All actions ran over the authorized SSH connection on NVMe. The first
`command -v claude` in a noninteractive shell returned 1 with no output, before
the review script or external dispatch. The
[preflight failure](artifacts/f02-opus-outcome-preflight-failure-20261107.json)
was retained. Running the standard interactive shell with startup output suppressed
resolved the existing executable to `/home/y-takagi/.local/bin/claude`. No credential
content was inspected or serialized. Preview validated the anonymous bytes and SHA.

The single dispatch used the existing script from the worktree, inherited standard
shell authentication in memory, and suppressed shell startup output. The script's
summary alone was redirected to a safe NVMe log:

```sh
python scripts/claude-method-review.py \
  --claude-binary /home/y-takagi/.local/bin/claude \
  --prompt docs/experiments/anonymous-f02-outcome-review-2026-11-07.md \
  --output /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/analysis/fresh-purpose-20261107-f02/opus-outcome-review-v1 \
  --model claude-opus-5-5 --labels A --timeout-seconds 300 --max-budget-usd 2 \
  --send --approved-prompt-sha256 bb1656564e953947520345bd6b308435c82198b0f83acda1b3755f0f9aa1c0d2
```

The exact identity-audit and evidence-publication Python used after the receipt:

```python
from pathlib import Path
import hashlib,json
r=Path('/mnt/nvme/kairyu/wsp/gemma4-coding-20261104')
b=r/'worktrees/kairyu-bench-weight-tools'
p=r/'analysis/fresh-purpose-20261107-f02/opus-outcome-review-v1'
x=json.loads((p/'receipt.json').read_bytes())
model='claude-opus-5-5'
assert x['assistant_models']==[model]
assert set(x['model_usage'])=={model}
assert x['model_usage'][model]['canonicalModel']==model
assert x['returncode']==0 and x['is_error'] is False and x['tool_uses']==0
assert x['verified_requested_model'] is False and x['review_usable'] is False and x['labels_match'] is False
assert [t['label'] for t in x['response']['reviews']]==['A','A']
assert x['thinking_blocks_persisted'] is False and x['environment_persisted'] is False
record={'schema_version':1,'original_receipt_sha256':hashlib.sha256((p/'receipt.json').read_bytes()).hexdigest(),'assistant_usage_canonical_identity_matches_requested':True,'requested_model':model,'original_combined_verified_flag':False,'original_review_usable':False,'format_deviation':'two reviews entries both label A; required one entry with label A','original_receipt_unchanged':True,'external_dispatches':1,'new_dispatches_in_this_audit':0,'interpretation':'Identity evidence confirms actual requested model; strict review format remains failed. Root may consider the preserved final structured response as nonconforming advice, not a passed review receipt or adoption decision.'}
with (p/'identity-only-audit.json').open('x') as f: json.dump(record,f,indent=2); f.write('\n')
a=b/'docs/experiments/artifacts'
for src,name in [(p/'receipt.json','f02-opus-outcome-receipt-20261107.json'),(p/'identity-only-audit.json','f02-opus-outcome-identity-audit-20261107.json'),(p.parent/'opus-outcome-preflight-failure-v1.json','f02-opus-outcome-preflight-failure-20261107.json')]:
    with (a/name).open('xb') as f: f.write(src.read_bytes())
print(json.dumps(record))
```

No benchmark score, model weight, service or task selection changed during this
follow-up. Model-review format failure, CLI-discovery failure and all costs remain
part of the trial record.
