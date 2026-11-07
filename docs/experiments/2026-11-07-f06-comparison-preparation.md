# F04 versus F06: full dev30 comparison prepared, not executed

Reuse the existing F04 complete audit and unchanged official CPU `wsp compare`, without the Docker wrapper or a model call. F06 is still running at preparation; no candidate score, partial rank or adoption decision is computed.

Inputs are the existing F04 run `gemma4-fresh-f04-dev30-dp4-20261107-v1` and its `official-comparison-v1/candidate-audit`, paired with F06 `gemma4-fresh-f06-dev30-dp4-20261107-v1`. All 2,338 IDs, 90 code items, 32 sources, 14 areas and 96 diagnostics remain selected. Fresh output will be `analysis/fresh-purpose-20261107-f06-dev30-dp4-v1/official-comparison-v1`. Existing outputs are never overwritten; a failure remains visible.

Execution waits for the root completion notice. Both run status and candidate report status must be completed, owner return code must be 0, and candidate full audit must verify complete before strict comparison. The existing audit first writes every canonical item, missing/error/null/nonfinite anomaly and all90 code rows; an incomplete or invalid run cannot acquire a full-score ranking. No `--allow-partial` or `--partial` is used. The F04 audit/report/item hashes are reused rather than remeasured. The same f91 source, selected IDs, item metadata, topology/concurrency and official source/area means are checked.

A real setup issue required one small audit change: F06's first `launch-plan-approved.json` is an intentionally retained malformed writer-failure artifact. The actual owner used `launch-plan-approved-v2.json`. New optional `--launch-plan` selects that actual file and validates its SHA against `owner-start.json`; the old default remains unchanged. Scoring, item selection, missing-data preservation and compare mathematics are unchanged. Source changed from `51898bdb…` to **e471c64712fdd70f37439f80bb86137889287c823b99b6ff945ee90eec1e75b1**. Existing focused tests now number **5**, all pass (0.054 s); Ruff and help pass. No full CI rerun.

The prepared one-off sources reuse the previous F04 collation and extraction. They will save all2,338 paired numeric rows, all90 code rows, all39 SciCode step outcomes, allsource/area gains and losses, and all10 Terminal-Bench/SWE tasks with their distinct step/tool/API definitions and unknown usage preserved. Additional all90 visible-string metrics count saved reasoning/public characters and UTF-8 bytes per call, with missing/null retained; total completion tokens are never relabeled reasoning tokens. No raw reasoning or benchmark answers are copied into compact records.

After completion, use the existing CPU Python with `PYTHONPATH=<repository>/src` to execute, in sequence:

1. `<N6>/comparison-preparation-v1/collect-completed.py.txt` — candidate full audit with explicit v2 launch plan/tree, then official strict compare and every paired item.
2. `<N6>/comparison-preparation-v1/extract-completed-extras.py.txt` — all39 Sci steps, agent usage and all90 saved text metrics.

These scripts were compiled only at preparation, not run. Exact commands and full input/source pins are in [preparation.json](records/2026-11-07-f06-comparison-preparation-v1/preparation.json). [Source change and validation](records/2026-11-07-f06-comparison-preparation-v1/source-change.json), complete focused logs, old audit source and both new one-off sources are retained with SHA256SUMS. Model calls, GPU/service operations, candidate audits and baseline reruns at preparation: **0**.
