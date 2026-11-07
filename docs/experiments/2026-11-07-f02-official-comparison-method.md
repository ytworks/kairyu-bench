# 2026-11-07: F02 official comparison method, prepared while running

This read-only method review does not score or rank the running F02. It changes neither the frozen amendment nor the active owner files. No additional tool is required: reuse [wsp compare](../../src/kairyu_bench/wsp/report.py), the existing [full-result audit function](scripts/fresh-wsp-large-tier-audit.py), and the saved runtime/observer evidence.

The [amendment](2026-11-07-f02-transfer-amendment.md) SHA remains `1b6df2b79abb7d4db023651df632509a49d0b35e816c40c6d342d3884691a1ae`; its freeze envelope SHA is `344993b8b1b2e3622c581d363d2a4ccdfc28b914e7d22e8c51151f43a8fcdfc3`. Frozen time is 2026-11-07T05:05:22.705316+00:00, materialized_before_freeze=false, official_candidate_runs_before_freeze=0. F02's original proxy gate remains FAILED.

## Inputs and completion gate

Under `/mnt/nvme/kairyu/wsp/gemma4-coding-20261104/results/`:

- Base: `gemma4-fresh-av-observed-dev10-20261107-v1`; tree `5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf`.
- Candidate: `gemma4-fresh-f02-observed-dev10-20261107-v1`; saved tree `46680f4066f57c1596a1f4c66b67d23929c96b325943d981c6d74260cdbb3f5c`.

Require completed run.json and report.json, exactly 778 unique canonical dev10 IDs in items.jsonl, every score non-null, every item error null, no declared unscored/exclusions, and the identical source/kind/tier/max_tokens/samples for each ID. The source of truth is manifest SHA `a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b`, including 32 diagnostic probe items, 26 code items and all 14 areas. Check saved identity, actual runner, TP2/DP1, MTP4, 32/8/1 concurrency, generation settings and capture sampling against the owner receipts separately; a matching run label is not a runtime attestation.

The existing audit function `audit_run(path, load_manifest())` in fresh-wsp-large-tier-audit.py checks completion, all unique selected IDs, max_tokens/samples/source, all item scores/errors, source means and code mean without editing the results. Reuse it for each directory, additionally comparing kind/tier/area metadata to the manifest/catalog and the expected labels above. Hash the final run/report/items inputs before writing a comparison receipt.

The current runner normally marks any unscored error as partial. Nonetheless, compare() itself only refuses non-completed status and unequal ID sets in strict mode: it does **not** independently enforce canonical 778 selection, duplicate-free rows, item budgets, empty unscored/error fields, or runtime identity. Do not use `--partial` here. Recovered API retries may not appear as item errors: retain/check attempt-level request trace errors and run logs as well, and disclose any retries rather than claiming error-free from run.json alone.

## Existing comparison command, after those checks

The official compare implementation reads only run.json and items.jsonl; it does not read report.json, responses.jsonl, capture workers or merged activation files. AV items.jsonl is 11,030,832 bytes. It performs 2,000 deterministic stratified item bootstraps plus diagnostic probe KL/NLL calculation. It therefore requires no multi-GB activation merge. Capture completeness/drop/unjoined-request/writer checks remain a separate required report, not inferred from successful score comparison.

The repository wrapper sends `wsp compare` through Docker and may build an image unless pinned. For this CPU-only postprocessing, invoke the exact existing WSP CLI function on the remote host to avoid an unnecessary runner/container:

```sh
cd /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/worktrees/kairyu-bench-weight-tools
TASK_ROOT=/mnt/nvme/kairyu/wsp/gemma4-coding-20261104
TASK_PY="$TASK_ROOT/infra/weight-tools-ci/bin/python"
TASK_BASE="$TASK_ROOT/results/gemma4-fresh-av-observed-dev10-20261107-v1"
TASK_CANDIDATE="$TASK_ROOT/results/gemma4-fresh-f02-observed-dev10-20261107-v1"
TASK_OUTPUT="$TASK_ROOT/analysis/fresh-purpose-20261107-f02/official-comparison-v1"
test ! -e "$TASK_OUTPUT"
mkdir "$TASK_OUTPUT"
PYTHONPATH=src "$TASK_PY" -c 'import sys; from kairyu_bench.wsp.cli import main; raise SystemExit(main(sys.argv[1:]))' compare "$TASK_BASE" "$TASK_CANDIDATE" --output "$TASK_OUTPUT/comparison.json" > "$TASK_OUTPUT/comparison.md"
```

This command is supplied for the owner to execute after completion; it was not run on the in-progress F02 during preparation. Use a new output name on every analysis attempt and retain failures.

## Exact score and decision mapping

Use full-precision JSON, not the Markdown's one-decimal display or rounded 87.777778 threshold. AV's stored code mean is 0.8777777777777778 and overall is 0.772086308565694. Scores are 0–1; multiply deltas by 100 for points.

For each source, mean all its item scores. Code is the equal mean of HumanEval+ (10), MBPP+ (10), LiveCodeBench (6), **not** a pooled 26-item accuracy. Each other area is an equal source mean; overall is the equal mean of all 14 areas. Compare JSON keys are `areas.code.{base,candidate,delta}`, `overall.delta`, `areas[area].delta` and `sources[source].{base,candidate,delta,items}`. Confirm zero `unpaired_items`, empty `missing_areas`, `partial=false`, all 14 areas and 26 capability sources; the 3 probe sources are in `probes` with NLL units.

Let dc be code delta, do be overall delta, and worst=min(all 14 area deltas):

| Condition after completion/identity/error gates | Frozen result |
|---|---|
| dc <= 0 | No advancement (tie/loss). |
| do < -0.01 or worst < -0.10 | Reject, even with code gain. |
| dc > 0, do >= -0.01, worst >= -0.05 | Provisional selection permitted. |
| dc > 0, -0.10 <= worst < -0.05, do >= 0 | Provisional selection permitted only after explicit paired-item tradeoff analysis. |
| dc > 0, -0.10 <= worst < -0.05, do < 0 | Reject. |

The inequalities are mathematical point thresholds. If binary floating arithmetic puts an apparent boundary within 1e-12 of -0.01, -0.05, -0.10 or 0, show the unrounded source/item calculation and resolve representation rounding; do not turn it into an undisclosed scientific tolerance or change a threshold after seeing data. No one-run result establishes a causal or robust ability increase.

Produce all 778 matched rows by ID with source/area, base score, candidate score and delta, retaining diagnostic NLL units separately. Include every positive, negative and equal result, plus exact code IDs and partial scores. Existing compare `flips` counts crossings of 0.5 only; it is **not** an exhaustive count of improved/worsened partial-credit items. Preserve task-specific diagnostics, steps/tool calls, API calls, tokens and wall/load time alongside the score tables.

## Confirmation rule

Only if the initial comparison permits provisional F02 selection, register exactly one matched AV/F02 dev10 confirmation pair **before further weight editing**. Retain both new repeats regardless of outcome. For each model, source and fixed question set, average the source means across all comparable repetitions, including the initial fresh AV/F02 pair; then average the 3 code source means. With two complete repetitions this equals averaging their two code source means, because source weights and item sets are identical. Keep per-repeat/per-source/per-item results visible.

The old AV run with a different capture/runner is preserved as a condition-different repeat, not silently pooled into the matched cohort. Thirteen historical tied runs are neither thirteen AV repeats nor a variability estimate for this pair. No repeat-until->90 procedure is authorized. The amendment prescribes assessing all repetitions but does not specify a new numerical post-confirmation adoption gate; report the pooled and per-run evidence instead of inventing one retrospectively. Wider 30/60, final test, DeepSWE and publication requirements remain separate.

Source versions reviewed: report.py `f708cab18226874ac26d2b9fce642db811fcdbe985f2420eeda40920e2575784`; wsp/cli.py `406b07cb9bed06bd3b814292329cd67d0acf89fcaf5fddca8858cc865bbed090`; reusable audit script `c08ae268ae2e5ec55f690d7e4f935566995f881a183f86ce3bb38d3230cf1462`. Preparation read metadata/source only, with no F02 partial scores, new WSP or service operation.
