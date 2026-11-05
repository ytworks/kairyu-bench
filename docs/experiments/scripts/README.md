# Read-only experiment evidence

These recipes supplement the standard WSP/DeepSWE reports; they do not launch
models, execute agent commands, regrade results, or change score semantics.
Run them from a checkout with its Python dependencies available:

```bash
PYTHONPATH=src python docs/experiments/scripts/wsp-run-evidence.py \
  /data/base-dev /data/edited-dev --partial \
  --baseline-repeat /data/base-dev-repeat \
  --candidate-repeat /data/edited-dev-repeat \
  --historical-dev-reference /data/base-dev-larger-tier \
  --output /data/analysis/new-wsp-evidence.json

PYTHONPATH=src python docs/experiments/scripts/deepswe-recorded-usage.py \
  /data/deepswe-run --output /data/analysis/new-deepswe-usage.json
```

Both outputs must be fresh paths outside all input run directories. Input files
are never modified. JSON outputs contain only whitelisted scoring, identity,
timing and usage metadata, not prompts, answers, reasoning, tool arguments or
environment values. Private paths and experiment-specific controls are not
embedded in either recipe.

- [WSP evidence](wsp-run-evidence.py) delegates the official paired score/CI/probe
  calculation to `wsp.report.compare`. It adds fractional item gains/losses,
  separately reports crossing 0.5, descriptive repeat ranges and recorded
  calls/tokens/steps. Repeat cohorts must have the same manifest, tier, split,
  model, weight label and item IDs. Only dev is accepted. Historical dev
  references can differ in tier/cohort and are explicitly **range-only**; they
  never alter the official estimate or interval. Their metadata is retained,
  and different runtime/concurrent-load conditions remain a limitation.
  Labels are declarations, not new checkpoint-byte attestations. An observed
  repeat range is not a confidence bound or proof of a causal effect. Fields named
  persistent describe only the supplied observations, not statistical
  significance or a guaranteed future effect. The
  recipe imposes no zero-regression or automatic adoption gate.
- [DeepSWE usage](deepswe-recorded-usage.py) delegates canonical scoring to
  `summarize_deepswe`. It counts recorded usage from canonical trials plus
  retry archives once per UUID, rejecting conflicting slot, checksum, usage
  or time records for one UUID. Copies under `jobs/` are excluded. Archived
  attempts add usage, never scores. The official scored denominator and
  unscored count are retained, including incomplete runs. The recipe accepts
  the selected cohort rather than assuming 113 tasks. `n_agent_steps` is
  labelled agent steps, not asserted to equal HTTP or tool-call counts.

Missing measurements have a null total and an explicit measured/expected
count. `sum_recorded` is only the known portion. Concurrent request/agent
durations are sums, not wall time. Multi-call WSP samples contribute their
individual call records once; aggregate sample totals are not added again.
Agent summaries without recorded calls/tokens remain unknown.
