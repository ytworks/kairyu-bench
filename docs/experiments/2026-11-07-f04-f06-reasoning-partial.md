# F04/F06 returned-output comparison — frozen partial snapshot

This is one read-only snapshot at **2026-11-07T14:08:07.534738+00:00** while F06 WSP30 was running. It includes **491/2,338 completed response records**, including **26/90 code items**, with zero recorded response errors. These are response completions, not scored-item counts. Early/short responses are overrepresented. Nothing here estimates the full population, determines adoption, or stops evaluation.

Exact item ID and nested sample/path matched 492 nodes; 15 parent aggregate nodes were retained but excluded from additive statistics. The remaining **477 leaf nodes across 408 items** have both returned reasoning strings observed. The other 83 completed response records remain in the coverage table; missing response nodes are not zero-length reasoning. Matching a path does not establish identical input prefixes or equivalent tool trajectories.

| Matched leaf cohort | All | Code | Non-code |
|---|---:|---:|---:|
| Nodes / items | 477 / 408 | 36 / 26 | 441 / 382 |
| Reasoning characters, F04 → F06 total | 3,858,913 → 1,820 | 289,557 → 0 | 3,569,356 → 1,820 |
| Reasoning characters, median | 5,475 → 0 | 5,717.5 → 0 | 5,475 → 0 |
| Reasoning UTF-8 bytes, total | 3,896,210 → 1,820 | 289,560 → 0 | 3,606,650 → 1,820 |
| Public-content characters, median | 1,619 → 281 | 2,355.5 → 709.5 | 1,536 → 267 |
| Total completion tokens, median | 2,646 → 117 | 2,745 → 252.5 | 2,640 → 111 |
| Empty returned reasoning, F04 → F06 | 2 → 476 | 0 → 36 | 2 → 440 |

All 477 F06 matched leaves ended with stop. F04 had 444 stop, 32 tool_calls and one length. The 32 tool-call transitions cannot be treated as otherwise identical generation tasks.

A prespecified narrow code-first appearance heuristic recognizes a leading code fence, Python definition/class/import, or shebang. It is not semantic correctness or a complete code detector. On the **same 408 non-code leaves with public content observed on both sides**, matches increased from **2/408 to 370/408**. Thirty-three additional non-code F04 contents are unobserved and remain null; among all 441 observed F06 contents, 393 match. On code leaves the comparable count is 36/36 → 35/36. The frozen observed output shows shorter returned reasoning and frequent code-shaped non-code answers without establishing final benchmark effects.

Reasoning-token counts are not saved. Total completion tokens combine generated output and cannot isolate reasoning. Returned reasoning text is not a measurement of internal computation or ability. No absent value is imputed as zero.

## Provenance and continuation

The CPU operation took 8.989 seconds under nice 15, idle I/O, and one BLAS/OpenMP thread. It made zero model/API requests and zero rescoring calls and changed no WSP raw result. Candidate responses were read as a bounded initial file prefix before other inputs: 464,445 bytes, SHA 268a97fc692af0eea81d00a65f0d7add697ed48c7ee615f82bcd17fca15ce84b, with no partial final line. Complete F04 inputs retain their previous full hashes.

- [Frozen summary](records/2026-11-07-f04-f06-reasoning-comparison-partial-v1/summary.json), SHA 815adbc4e44f6d6b2b55ae3b213b9daacb4e908546754c9799a903a1113bf635.
- [All observed IDs and path coverage](records/2026-11-07-f04-f06-reasoning-comparison-partial-v1/all-observed-item-coverage.json), [all matched metadata pairs](records/2026-11-07-f04-f06-reasoning-comparison-partial-v1/all-matched-node-pairs.json), and [same observed denominators](records/2026-11-07-f04-f06-reasoning-comparison-partial-v1/same-observation-denominator.json).
- [Executed source](records/2026-11-07-f04-f06-reasoning-comparison-partial-v1/source.py.txt), [denominator calculation](records/2026-11-07-f04-f06-reasoning-comparison-partial-v1/denominator-source.py.txt), [candidate raw prefix](records/2026-11-07-f04-f06-reasoning-comparison-partial-v1/candidate/responses.jsonl.gz), [artifact index](records/2026-11-07-f04-f06-reasoning-comparison-partial-v1/artifact-index.json), and successful focused Ruff receipt.
- Reusable [comparison recipe](scripts/fresh-f04-f06-reasoning-compare-20261107.py) requires a new output directory. A later --phase completed invocation requires completed status and all 2,338 IDs. This snapshot must remain unchanged; final comparison is pending.

Execution used OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 and MKL_NUM_THREADS=1 with:

    nice -n 15 ionice -c 3 /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/infra/weight-tools-ci/bin/python docs/experiments/scripts/fresh-f04-f06-reasoning-compare-20261107.py --phase partial --output /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/analysis/fresh-purpose-20261107-f04-f06-reasoning-comparison-partial-v1
