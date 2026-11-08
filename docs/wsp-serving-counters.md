# Serving counter snapshots

This opt-in CPU tool records a bounded whitelist of vLLM process counters before
and after a run. It does not change the official runner, scoring, sampling, or
model. Use a fresh snapshot **after authored smoke and before the official run**,
then another after completion on the same serving process.

```sh
python scripts/experiments/serving-counters.py snapshot \
  --endpoint http://127.0.0.1:8000/metrics --model gemma-4-31b \
  --container-id ACTUAL_CONTAINER_ID --output before.json
# The default is preview: no HTTP request and no output file.
# Add --capture to take the snapshot; repeat to a fresh after.json.
python scripts/experiments/serving-counters.py diff before.json after.json \
  --output counters-diff.json
```

The endpoint cannot contain credentials, query or fragment. Redirects are
refused. Capture uses one HTTP request, socket-operation timeout at most 10 seconds
and response at most 2 MiB. This is not a total wall-time deadline. No raw metrics, log, prompt, credentials, unknown labels or error
body is saved. Output is fresh-only and mode 0600. The optional
`--attestation-sha256` links a caller's identity receipt; this tool does not
independently inspect Docker or verify that attestation. Record the actual
container ID, not a reusable service name.

Allowed families are `vllm:num_requests_running`,
`vllm:num_requests_waiting`, `vllm:generation_tokens_total`,
`vllm:prompt_tokens_total`, `vllm:request_success_total`
and optional `process_start_time_seconds`. The vLLM series require the expected
`model_name` or `model` label. Only those labels, numeric `engine`,
and success `finished_reason` (stop/length/abort/error/repetition) are accepted.
Unsupported runtime label schemas fail closed rather than silently merging
series. Unknown metric families are ignored, not retained. Optional exposition
timestamps are validated but not used. Numeric samples/deltas are decimal strings
with sufficient subtraction precision. Inputs are limited to 128 characters,
absolute decimal exponent ≤1024 and finite float64 magnitude; values outside
these bounds, NaN, infinity, negatives and duplicates fail.

Diff requires the same endpoint, expected model, caller container identity and
series. Missing series produce `unknown_missing_series` with **all deltas null**,
not zero or a partial total. Counter decreases or a changed process start are
refused. Gauges and process identity keep before/after values with no delta.
Without the process-start metric, increasing counters cannot exclude restart.
Even a matching start time cannot prove there was no unobserved counter reset.

A valid delta is still only an endpoint-process aggregate: it does not prove
exclusive client use, item attribution, retry count, thought-token count, or
accuracy. Saved benchmark sample calls are a different measurement. In the I
experiment a cumulative length event could not initially be assigned because
there was no pre-run baseline; later saved sample metadata established its own
length outcome. This tool prevents subtracting an absent baseline, but does not
turn process telemetry into per-item evidence.

CPU fixtures cover reordered labels, exact large deltas, reset/missing/mixed-model
rejection, malformed selected data, preview with zero network calls and bounded
mock HTTP. No GPU or actual serving endpoint is used by tests.
