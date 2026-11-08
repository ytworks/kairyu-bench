# Existing F04 reasoning-output evidence — 2026-11-07

Existing full-WSP30 F04 records contain actual returned reasoning text for all90 code items. The unchanged F06 run can compare these without regenerating the parent.

`responses.jsonl` stores `outcome.samples[*].reasoning`; SciCode can contain per-step entries in `model_responses`. Across the90 items,119 reasoning nodes are present and all119 are nonempty. Per-node content length, finish reason, prompt tokens, total completion tokens and response seconds are also saved. The source/item/sample/nested path must be retained rather than collapsing all calls.

The matching client trace has119 code request records covering90 IDs and input hashes/call IDs. Every one of4856 trace records has null `completion_tokens_details`; a reasoning token count is therefore unobserved. Character and UTF8-byte lengths are directly measurable. Total completion tokens cannot isolate reasoning. Returned reasoning text also does not prove how the model reasoned internally.

The small [all90 field inventory](records/2026-11-07-f04-reasoning-evidence-v1/all90-code-reasoning-fields.json) retains every code item, score and observed node; the [summary](records/2026-11-07-f04-reasoning-evidence-v1/summary.json) pins the complete original items/responses/trace. No generation, source change or evaluation occurred. Future comparison must preserve missing fields and different call trajectories; do not invent119 matched calls if F06 produces a different structure.
