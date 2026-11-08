# Offline scoring of all16 authored helper stages

`helper_stage_offline.py` consumes the **unchanged complete72** generation input and response envelope. It scores all eight short and eight long train stages with existing `helper_chain_cases.score`, `long_helper_chain_cases.score`, and `science_sandbox.Docker.run`. It makes no model request. The other56 IDs remain explicitly not targeted. This is an additional known-calibration diagnostic, separate from the existing48 catalog plan and official WSP.

Both case and response SHA256 are required. All16 source-owned rows must match exactly, including prompts, correct/foil, stage and metadata. All72 input IDs must be unique; response IDs outside that set or duplicates are rejected. Missing responses and HTTP failures remain unknown. Shared `science_sandbox.read_public_answers` retains its public-only schema, thinking-control and HTTP-status validation; no reasoning fields are accepted. No response extraction or `cases_sha256` rewriting is needed.

Use the existing scientific CPU Python with NumPy, SciPy and SymPy, and an already installed immutable science executor image. Preview writes a new report without constructing Docker:

```sh
PYTHONDONTWRITEBYTECODE=1 /absolute/scientific-venv/bin/python \
  scripts/experiments/gemma4-purpose/helper_stage_offline.py \
  --cases /nvme/full72-cases.json --cases-sha256 <exact-case-sha256> \
  --responses /nvme/full72-responses.json --responses-sha256 <exact-response-sha256> \
  --output /nvme/new-preview \
  --docker-socket /var/run/docker.sock --daemon-id <exact-daemon-id> \
  --image sha256:<exact-local-image-id> \
  --seed 20261107 --trials 8 --seconds 15 --memory-mib 512 --cpus 1
```

Only after the execution plan is fixed and execution authorized, use a **different fresh output** and add `--run`. Existing socket/daemon/image guards and owned-container cleanup are reused. The worker is runc, network-none, uid1003, read-only, no-new-privileges, cap-drop ALL, bounded CPU/memory/time, no host mounts and no GPUs. Inputs and source identity are checked before every stage. No new dependency or oracle is introduced.

For stage2 the adapter adds only the fixed helper already present in that prompt. A failed generated stage1 never suppresses its independently supplied-helper stage2. Long `score` adds the public support exactly once. `evaluate` / `evaluate_requests` are not used: their generated-helper conditions differ and can make extra model requests. `trusted_run` is **never used by the adapter**; only regression tests use its exact source-owned correct/foil allowlist. Generated programs run only inside the existing Docker worker. Expected numeric values stay on the host; the payload contains code, parameter names, sampled inputs and the existing underflow flag.

The report preserves every raw stage outcome and reports16 totals plus the3 known dtype-contract cases and13 other cases. The known cases are `chain-whitened-fit-1`, `chain-quadrature-mixture-1`, and `chain-boundary-response-1`. Existing float64 fixtures do not cover their float32/integer counterexamples. Numeric pass does not guarantee helper reuse, non-redefinition, every prompt instruction, or correctness over all allowed inputs. Do not discard these cases or present a16/72,64/120, or WSP capability score.

Verification extends `tests/test_gemma4_helper_chains.py` for input/response identity rejection, no-Docker preview, missing/error preservation, no expected values in worker payloads, and all16 existing authored correct/foil cases under the scientific CPU environment. The existing sandbox regression suite covers the shared response-reader extraction. Real Docker execution and model-generated code execution are separate validation and were not performed during implementation.
