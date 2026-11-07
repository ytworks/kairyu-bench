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

# Optional recorded model API counter; no message bodies are read.
PYTHONPATH=src python docs/experiments/scripts/deepswe-recorded-usage.py \
  /data/deepswe-run --agent-api-metadata \
  --output /data/analysis/new-deepswe-api-metadata.json
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

With `--agent-api-metadata`, DeepSWE adds `usage.agent_api_calls` separately
from `usage.agent_steps`. The source is the agent's recorded
`trajectory.info.model_stats.api_calls` counter, **not independent HTTP
transport/server telemetry**. No count is inferred from steps. The reader
requires `info` to be the first top-level object and stops at its closing
brace, with a fixed 2 MiB prefix limit; it does not scan, decode or parse
`messages`. Only nonnegative integer counters are accepted. Missing files,
unsupported ordering, truncated/oversized metadata and invalid counters remain
unknown with measured/missing coverage and per-observation status.

Metadata beside canonical/retry results and under `jobs/repeat-*/` is linked by
task, repeat, nonempty task checksum and trial UUID, using the sibling `result.json`.
Job copies add no trial or score. Conflicting identities, existing usage/timing
or two recorded API counters for the same UUID are rejected; matching copies
count once. Unmatched job UUIDs are ignored, so unfinished canonical slots stay
missing. Per-file stat stability and the SHA256 of only the read prefix are
recorded; a changed file or path escaping the run is rejected. The source
trajectory is never modified. Existing output without this flag is unchanged.


## Explicit WSP progress snapshot

The historical filename [fresh-av-dev30-progress-20261107.py](fresh-av-dev30-progress-20261107.py)
now requires both target directories; there is no default AV run. Use it after
the existing owner's start receipt and run.json exist:

    python docs/experiments/scripts/fresh-av-dev30-progress-20261107.py --help
    python docs/experiments/scripts/fresh-av-dev30-progress-20261107.py --analysis-root /data/analysis/current-run --run-root /data/results/current-run

The analysis directory must contain a plan matching owner-start.json's exact
plan SHA. Its run ID, working/result directory, capture config and the run.json
observation/label/tier/split must match before the monitor reads trajectories,
queries GET /metrics, or appends one progress.jsonl line to that analysis directory.
Help and missing required arguments do none of these operations. The monitor
does not generate model responses, regrade tasks, or rank incomplete runs.
Unlike the two separate evidence exporters above, this monitor intentionally
appends its own progress record; it never edits canonical result files.
See the [incident and regression evidence](../2026-11-07-wsp-progress-cli-fix.md).
