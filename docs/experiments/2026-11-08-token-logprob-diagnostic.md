# Fixed ARC10 token-logprob diagnostic

This is a preregistered pair of observer-OFF diagnostic runs on the designated original after rejected candidate001 cleanup. It is not a new scorer, a candidate search, or a full WSP evaluation. Run it only on the SSH host after the trial's storage and serving checks.

The pinned official ARC adapter already receives per-token prompt logprobs but saves only their continuation sum. `scripts/experiments/arc-token-logprob-diagnostic.py` delegates to that same adapter and client. Each item gets its own proxy; its four synchronous choices keep their own tokenization results and indices even when 10 items overlap under concurrency 32. No arrival-order correlation is used. The official scorer/load functions and default source files are unchanged.

The proxy returns the exact client result objects and propagates original exceptions. It makes no extra request. For this fixed scope, 40 completion calls and 81 tokenize calls (80 plus the once-cached BOS tokenization) are expected. It records only actual token IDs/hash, whole/head text hashes, the continuation token boundary, actual continuation-token logprobs, and the adapter's returned sum with an exact equality check. Summation follows the official left-to-right float addition; Python 3.12 `sum()` may round differently. No raw text, gold label, top-k alternatives, generated answer, header, auth value or environment is saved.

Allowed token capacity is 1024 per call, at most 40 call records and **2MiB total JSON**. The bounded records stay in memory until the official run finishes, then flush into one fresh file. Unavailable, oversized or failed captures retain a small item/choice-attributed category/type record where possible and set `capture_complete=false`; missing values never become zeros. They do not change the official request, response or score. Error messages and raw error bodies are omitted. Recording can still change host timing and batching; this is not a numerical-equivalence claim.

```sh
python scripts/experiments/arc-token-logprob-diagnostic.py --help
python scripts/experiments/arc-token-logprob-diagnostic.py \
  /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/restart-direct-20261108/observation-002-token-logprobs/token-logprobs.json \
  -- run http://gemma-0:8000/v1 --tier 10 --split dev --only arc-challenge \
  --logprob-endpoint http://gemma-0:8000/v1 --chat-concurrency 256 \
  --logprob-concurrency 32 --agent-concurrency 4 --run-id REGISTERED_FRESH_RUN \
  --results-dir REGISTERED_RESULTS_DIR [the registered --exclude-item arguments]
```

The last example is schematic: use the complete argv in the SSH trial's `wrapper-launch-proposal.json`, after checking it against the final `registration.json`, including all exclusions. The wrapper rejects a changed manifest, different selection/order, different inspected official source hashes, non-dev/tier10 conditions, or activation capture. It neither chooses new questions nor adjusts labels or coefficients. Validate the exact wrapper SHA256 and launch argv in the registration before execution.

Reuse immutable runner `sha256:f91bdcf997af6d7b194bec229812fd018277158f6795de7d91e58448936e7ad8`, its existing lock venv, and `/app/src`; mount only the wrapper file read-only at `/diagnostic.py`. Keep existing result/cache dependencies on NVMe, add the registered output directory writable, and use the trial-owned NVMe TMPDIR/XDG cache. No source overlay, build, pull, Docker socket, privileged flag or GPU access is needed by the runner. The image's client/ARC adapter/adapter registry/CLI/runner hashes were checked against the wrapper's five expected hashes.

Validation uses the real official adapter/scorer with a fake transport: identical HTTP payloads and 121 call count, identical scores/results, same return-object and exception identity, 32-worker overlap with 10 distinct prompt boundaries, and honest partial records under quota failure. It performs no model inference. Run with NVMe temporary storage:

```sh
TMPDIR=/mnt/nvme/kairyu/wsp/gemma4-coding-20261104/tool-validation/ci-tmp \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/infra/weight-tools-ci/bin/python \
  -m unittest tests.test_arc_token_logprob_diagnostic
```

The first test attempt used Python 3.12 compensated `sum()` and failed its own assertion; the test was corrected to the unchanged official addition order. Preserve that failure and subsequent checks in `observation-002-token-logprobs/validation-history.jsonl`. Both runs use the same unchanged original serving process. Each has 40 completions and a 2MiB limit (4MiB total); the second is not executed if the first capture is incomplete. This two-run count was registered before either ran, because the older OFF evidence retains only sums and cannot supply a token-level comparison. After the registered pair, retain bounded evidence, canonical results and failures; remove only processed trial-owned temporary files and reconcile `storage-state.json`. No checkpoint is created.

The registered pair completed with 40 records, 40 completion calls and 81 tokenize calls per run; both captures are complete and both official scores are 0/10. All 40 paired token sequences/hashes, text hashes, lengths and continuation boundaries match. There are no missing or nonfinite continuation logprobs. All 80 reconstructed sums and all 20 item argmax/score checks exactly match the official results. Nevertheless, 64 of 220 continuation-token positions differ across runs (11 choices, 8 items): maximum absolute token-logprob difference is 5.0473985672 and maximum absolute change in the official correct-option margin is 0.2389380485. The difference is already present in returned token logprobs; this pair found no input, boundary or summation discrepancy. It does not isolate the runtime cause, demonstrate deterministic inference, or show a weight improvement.

The unique evidence remains on the SSH host at `restart-direct-20261108/observation-002-token-logprobs/`: the two captures total 74,581 bytes, and `token-pair-summary.json` has SHA256 `d024204be9ea5a20618659b9802231bd4ec1e756ae32434a277d4f8308b595fb`. No result copy is added to this repository. The executed wrapper SHA256 is `aab6c223eaeb940fbbd4acb156f3dbb94c03e74bc89885a8ce9cd6cddf78c8b8`.

One failure-path limitation remains in this fixed wrapper: an I/O exception while flushing the diagnostic JSON in `finally` can supersede the official CLI return or exception. Both registered flushes succeeded, so this does not invalidate these results. A future use must distinguish such a recording failure from the already-written official evaluation result; this implementation does not guarantee exit-status preservation when its output cannot be written. No extra diagnostic run was performed to investigate it.
