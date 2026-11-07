# Same-AV dev10 repeat audit

The 151 historical runs and the fresh completed AV run contain exactly two complete tier10/dev runs of AV checkpoint tree `5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf`. The other tied models are not AV repeats. Every historical run.json was reread and matched to its original audit hash. The [complete numeric audit](evidence-2026-11-07-f02/av-repeat-audit.json) records every scanned identity, all 26 code outcomes, input hashes and the [reproduction recipe](scripts/fresh-av-repeat-audit.py). No benchmark or model call was added.

| Run | UTC start | Completed/scored | Code source mean | Overall | Wall seconds | Chat/logprob calls |
|---|---|---:|---:|---:|---:|---:|
| gemma4-ad-av-av-c32-dev10-full14-20261107-v1 | 2026-11-07T01:26:29.260520Z | 778/778 | 87.777778 | 77.154948 | 1559.1 | 563/3062 |
| gemma4-fresh-av-observed-dev10-20261107-v1 | 2026-11-07T04:18:20.433906Z | 778/778 | 87.777778 | 77.208631 | 1800.7 | 558/3062 |

Both runs score HumanEval+ 10/10, MBPP+ 8/10 and LiveCodeBench 5/6. All 26 item scores and detailed pass/fail results agree: no gain, loss or partial-score shift. The same failures are LiveCodeBench arc191_c and MBPP 410/576; both MBPP base tests pass and plus tests fail. This two-run observation does not estimate a confidence interval or prove determinism.

## Conditions and repeat limit

**Exact-current-condition prior repeats: none.** The fresh run is the only completed AV dev10 run under its actual live-capture and runner-image conditions. The older AV run is a same-weight observation with condition differences, not an exact replication.

| Condition | Earlier AV | Fresh AV |
|---|---|---|
| Selection | Same manifest, 778 items, all 14 areas | Identical all 778 IDs, source/kind, sample counts and token budgets |
| Code cohort | HumanEval+ 10 / MBPP+ 10 / LCB 6 | Identical 26 items |
| Serving | Ordinary TP2, one replica on GPU0/1 | TP2, one replica on GPU0/1, MTP4 |
| Serving image | af9e746b... | Same af9e746b... |
| Chat/logprob/agent parallelism | 32 / 8 / 1 | 32 / 8 / 1 |
| Endpoints | Direct gemma0 IP for chat and logprob | Direct gemma0 DNS for chat and logprob |
| Capture | Completed HTTP request traces, 256 MiB limit; no worker files | Actual residual/attention/MLP at all 60 layers and both TP ranks, 2 GiB output limit, 8 prefill positions and 16 decode offsets |
| Runner image | e70b1709578f... | f91bdcf997af... |
| Other simultaneous work | AD full WSP on GPU2/3 | F01 finite replay on GPU6 during part of run |

The actual installed package hashes differ only in weights/editor.py and weights/finite_replay.py, with weights/native_faststore.py added. Benchmark selection, client, all source adapters and scorers, report and manifest hashes agree. These unused weight-tool changes do not establish a benchmark scoring change, but the OCI image identity and serving capture conditions are distinct. Runtime differences are also confounded by different overlap and generated workloads; the longer fresh run is not an isolated capture-overhead estimate. Fresh capture overhead has a separate authored-prompt measurement in its observation record.

The full 26-row evidence retains per-run completion-token counts without exporting prompts, answers or reasoning. Historical code-only tier30 AV-labelled/overlay runs and all other checkpoint trees remain excluded from the dev10 repeat count, not silently treated as incomplete dev10 repetitions.

The historical 87.7778-point tie was **13 runs across 10 distinct checkpoint labels**, not 13 AV repetitions and not 13 distinct models. Exactly one of those historical runs is AV; the fresh run provides the second same-weight AV observation under the condition differences above.
