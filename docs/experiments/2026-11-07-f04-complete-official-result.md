# F04: complete canonical dev30 comparison — 2026-11-07

**All2,338 selected items were scored, with0 unscored/API errors; all90 code items and all14 named areas are complete.** F04's official four-source code mean is **72.9365079365**, exactly equal to existing AV. Overall is **76.9910864328**, versus AV **76.7200592915** (+0.2710271413 points). The >90 code objective is not achieved. **Root selects F04 as the next editing parent:** primary code exactly ties the highest complete comparable AV, and overall is higher. This tie-break selection is **not a coding improvement**; reasoning/vision losses remain visible below. **No AV inference was repeated.**

## Code: equal source means, six item changes

| Source | Selected | AV | F04 | Delta points |
|---|---:|---:|---:|---:|
| humaneval-plus | 30 | 96.666666667 | 96.666666667 | +0.000000000 |
| livecodebench | 20 | 75.000000000 | 75.000000000 | +0.000000000 |
| mbpp-plus | 30 | 76.666666667 | 76.666666667 | +0.000000000 |
| scicode | 10 | 43.412698413 | 43.412698413 | +0.000000000 |

HE29/30, LCB15/20 and MBPP23/30 are unchanged in aggregate. SciCode averages10 per-problem partial-credit scores; all10 scores **and all39 step pass/fail statuses** are unchanged. All90 paired rows, including84 equal cases and every existing failure, are retained.

| Changed item | AV | F04 | Saved official diagnostic |
|---|---:|---:|---|
| humaneval-plus:HumanEval/151 | 1 | 0 | base remains pass; plus pass → fail |
| humaneval-plus:HumanEval/152 | 0 | 1 | base/plus fail → pass |
| livecodebench:3777 | 0 | 1 | fail → pass |
| livecodebench:abc390_g | 1 | 0 | pass → fail |
| mbpp-plus:Mbpp/641 | 0 | 1 | base/plus fail → pass |
| mbpp-plus:Mbpp/801 | 1 | 0 | base/plus pass → fail |

## All14 areas

All2,242 capability items pair exactly:51 improve,50 worsen,2,141 are equal. The96 diagnostic items are separate. Each area uses an equal-weight mean over its official sources; overall is the equal mean over14 areas. The code difference bootstrap interval is −4.5833 to +5.0000 points and overall interval −2.4031 to +3.0310: a single paired run does not establish a repeatable gain.

| Area | AV | F04 | Delta points |
|---|---:|---:|---:|
| knowledge | 85.555555556 | 85.555555556 | +0.000000000 |
| expert | 83.333333333 | 87.500000000 | +4.166666667 |
| facts | 71.666666667 | 70.833333333 | -0.833333333 |
| math | 91.258012821 | 90.256410256 | -1.001602564 |
| code | 72.936507937 | 72.936507937 | +0.000000000 |
| instructions | 89.916666667 | 90.500000000 | +0.583333333 |
| tools | 79.166666667 | 80.000000000 | +0.833333333 |
| agents | 54.166666667 | 54.166666667 | +0.000000000 |
| long-context | 84.837524373 | 90.605912127 | +5.768387754 |
| reasoning | 84.130434783 | 78.115942029 | -6.014492754 |
| commonsense | 61.666666667 | 61.777777778 | +0.111111111 |
| multilingual | 78.872053872 | 79.284511785 | +0.412457912 |
| vision | 79.166666667 | 77.083333333 | -2.083333333 |
| safety | 57.407407407 | 59.259259259 | +1.851851852 |

Notable measured tradeoffs: long-context +5.7684, expert +4.1667, reasoning −6.0145 and vision −2.0833 points. Code and agents have unchanged source means despite item changes. These are descriptive results, not a localized weight-to-ability attribution.

## Every source, without exclusions

Capability source means below are0–1 (higher is better). Probe source values are per-item NLL (lower is better); their positive raw deltas are losses. All32 sources retain their full canonical selected counts.

| Source | Items/scored | AV mean | F04 mean | Raw + / − / = |
|---|---:|---:|---:|---|
| aime-2026 | 12/12 | 0.937500000 | 0.916666667 | 1 / 2 / 9 |
| arc-challenge | 300/300 | 0.566666667 | 0.576666667 | 6 / 3 / 291 |
| bbeh | 46/46 | 0.782608696 | 0.695652174 | 0 / 4 / 42 |
| bfcl | 120/120 | 0.791666667 | 0.800000000 | 4 / 3 / 113 |
| gpqa-diamond | 48/48 | 0.833333333 | 0.875000000 | 3 / 1 / 44 |
| gsm8k-platinum | 50/50 | 1.000000000 | 1.000000000 | 0 / 0 / 50 |
| hellaswag | 300/300 | 0.610000000 | 0.600000000 | 6 / 9 / 285 |
| hmmt-feb-2026 | 13/13 | 0.846153846 | 0.826923077 | 0 / 1 / 12 |
| humaneval-plus | 30/30 | 0.966666667 | 0.966666667 | 1 / 1 / 28 |
| ifbench | 50/50 | 0.840000000 | 0.860000000 | 2 / 1 / 47 |
| ifeval | 120/120 | 0.958333333 | 0.950000000 | 1 / 2 / 117 |
| jamc-qa | 72/72 | 0.638888889 | 0.652777778 | 3 / 2 / 67 |
| livecodebench | 20/20 | 0.750000000 | 0.750000000 | 1 / 1 / 18 |
| longbench-v2 | 10/10 | 1.000000000 | 1.000000000 | 0 / 0 / 10 |
| math-500 | 60/60 | 0.866666667 | 0.866666667 | 1 / 1 / 58 |
| mbpp-plus | 30/30 | 0.766666667 | 0.766666667 | 1 / 1 / 28 |
| mgsm | 55/55 | 0.927272727 | 0.909090909 | 0 / 1 / 54 |
| mmlu-pro | 180/180 | 0.855555556 | 0.855555556 | 3 / 3 / 174 |
| mmlu-prox-ja | 60/60 | 0.800000000 | 0.816666667 | 1 / 0 / 59 |
| mmmu-pro | 48/48 | 0.791666667 | 0.770833333 | 1 / 2 / 45 |
| mrcr-v2 | 16/16 | 0.670125731 | 0.759844030 | 3 / 2 / 11 |
| musr | 30/30 | 0.900000000 | 0.866666667 | 0 / 1 / 29 |
| probe-code | 40/40 | 4.022500118 | 3.959948848 | 15 / 25 / 0 |
| probe-wikipedia-ja | 40/40 | 6.628099446 | 6.604307689 | 21 / 19 / 0 |
| probe-wikitext | 16/16 | 7.638729488 | 7.615903794 | 7 / 9 / 0 |
| ruler | 24/24 | 0.875000000 | 0.958333333 | 2 / 0 / 22 |
| scicode | 10/10 | 0.434126984 | 0.434126984 | 0 / 0 / 10 |
| swe-bench-pro | 4/4 | 0.250000000 | 0.250000000 | 1 / 1 / 2 |
| terminal-bench | 6/6 | 0.833333333 | 0.833333333 | 0 / 0 / 6 |
| triviaqa | 120/120 | 0.716666667 | 0.708333333 | 3 / 4 / 113 |
| winogrande | 300/300 | 0.673333333 | 0.676666667 | 5 / 4 / 291 |
| xstest | 108/108 | 0.574074074 | 0.592592593 | 2 / 0 / 106 |

Probe-code mean NLL4.0225001185 →3.9599488475 (25 lower,15 higher across40 items), while code90 remains tied. The comparison's probe-position-weighted results are separately retained with their own denominators; they are not substituted for capability points.

## Work, time and closure

| Measure | AV | F04 |
|---|---:|---:|
| wall_seconds | 2018.9 | 1940.2 |
| load_seconds | 111.2 | 109.6 |
| chat_api_calls | 1754 | 1750 |
| logprob_api_calls | 9193 | 9193 |
| report_source_prompt_tokens | 5335614 | 5304567 |
| report_source_completion_tokens | 4094068 | 3999831 |
| terminal_reported_episodes | 57 | 56 |
| terminal_tool_calls | 166 | 140 |
| swe_assistant_messages | 123 | 127 |
| swe_announced_tool_calls | 123 | 127 |
| swe_api_calls | 125 | 129 |

Wall and the report's generation_seconds both start **before data loading**. F04 uses1940.2s (32.34min),78.7s below AV2018.9s. This is observed elapsed time, not an isolated throughput estimate. Source-report token totals exclude agent/logprob usage (reported placeholders are zero); separate persisted agent token counts and missing-usage cases are in usage-summary.json. Do not interpret them as a complete billing total.

Terminal-Bench:5/6 pass (Windows fails), same six item outcomes as AV;56 reported episodes/140 tools. SWE:1/4 pass in both arms, with qutebrowser0→1 and Openlibrary1→0; other two remain0. F04 SWE uses127 saved assistant messages/tools and129 reported API calls. Terminal reported episodes and SWE saved assistant messages are different definitions.

| Agent item | AV score | F04 score | AV/F04 steps | AV/F04 tools |
|---|---:|---:|---:|---:|
| swe-bench-pro:instance_NodeBB__NodeBB-da0211b1a001d45d73b4c84c6417a4f1b0312575-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e | 0.0 | 0.0 | 24/20 | 24/20 |
| swe-bench-pro:instance_ansible__ansible-11c1777d56664b1acb56b387a1ad6aeadef1391d-v0f01c69f1e2528b935359cfe578530722bca2c59 | 0.0 | 0.0 | 41/32 | 41/32 |
| swe-bench-pro:instance_internetarchive__openlibrary-431442c92887a3aece3f8aa771dd029738a80eb1-v76304ecdb3a5954fcf13feb710e8c40fcf24b73c | 1.0 | 0.0 | 16/31 | 16/31 |
| swe-bench-pro:instance_qutebrowser__qutebrowser-ed19d7f58b2664bb310c7cb6b52c5b9a06ea60b2-v059c6fdc75567943479b23ebca7c07b5e9a7f34c | 0.0 | 1.0 | 42/44 | 42/44 |
| terminal-bench:bn-fit-modify | 1.0 | 1.0 | 10/12 | 18/16 |
| terminal-bench:cobol-modernization | 1.0 | 1.0 | 6/7 | 35/31 |
| terminal-bench:install-windows-3.11 | 0.0 | 0.0 | 13/11 | 38/31 |
| terminal-bench:polyglot-c-py | 1.0 | 1.0 | 7/6 | 18/18 |
| terminal-bench:portfolio-optimization | 1.0 | 1.0 | 7/10 | 13/15 |
| terminal-bench:reshard-c4-data | 1.0 | 1.0 | 14/10 | 44/29 |

F04 finished normally: owner returncode0, owner elapsed1946.714824s, services_changed_by_owner=false; all four inference queues were0 at completion. No stopping signal or timeout change was used. Polyglot completed/pass in250.205762s, so the F03 interrupted timeout outcome is not repeated here. HTTP-client traces record4846 requests plus10 non-request trial records, errors0/retry_attempts0. These counters do **not** establish zero nested agent retries.

The separately authorized nine-case authored validation overlapped loading for10.988155s (epoch1794051738.7987063–1794051749.7868617), using CPU1/shares128 and no GPU/model call. This overlap is recorded rather than assumed to have zero timing impact; see the linked authored validation record.

## Exact comparison and validation boundary

Same f91 immutable runner, af9 serving image, TP2×4/MTP4, concurrency256/32/4, thinking/token settings, manifest a3a0b60e…, selected IDs and item budgets were verified. Actual runner CIDd991fa638c9d7446e39500879dbca381b2d3835dd24ba7a2a9a0181474eab469 and four exact serving CIDs were observed43.006s after owner start; the actual observation timestamp is retained, not represented as the launch instant. F04 tree7b9e9488e3cd4e0b6c5f61ae341e5eeccdf5bd31c46d1c8354950dbc29d0f924 is bound to runtime evidence.

Existing fresh-n30-complete-audit.py SHA51898bdb849a632a1837b6c4b6166ff8456854c11fdce6077c15d6106ffe73b5 ran once with --expected-tree and **without --allow-partial** (3.823s). Existing CPU package compare ran once without --partial (4.437s), bypassing wrapper/image builds. All2338 IDs/metadata, source/area means, raw numeric score/error state and exact official report reconstruction pass. Existing AV run/report/items hashes and its prior audit were reused. No new test suite or benchmark was run for packaging.

[Exact commands and all numerical records](records/2026-11-07-f04-official-comparison/README.md) retain full2338 paired rows, all90 code rows, all39 SciCode steps, failures/equal results,32 source means,14 areas, runtime proofs, usage/timing, commands and SHA/gzip identities. No live progress log, credential, reasoning text, benchmark gold or generated public answer is included in this compact package. No DeepSWE result or Hugging Face publication is claimed.

## Parent selection after complete results

The explicit root decision is preserved in [parent-selection.json](records/2026-11-07-f04-official-comparison/parent-selection.json). Selection occurs after complete strict results, with code exactly tied and overall higher; the numerical collation JSON retains its earlier neutral adoption field. No additional model or baseline measurement was introduced. F04 is the parent for subsequent editing, while all AV/F03/F04 results and losses remain preserved.

## Public implementation differences informing later design

This is read-only interpretation of the six changed public programs, not new scoring or a causal weight attribution. Exact row/content hashes are in [code-public-evidence.json](records/2026-11-07-f04-official-comparison/code-public-evidence.json). Official failing hidden inputs and reference implementations were not read.

| Item | AV → F04 public implementation | What follows / limit for later design |
|---|---|---|
| HE151, loss | `isinstance(x,int)` → `type(x)==int`, with explicit text excluding bool. Both retain nonnegative/odd predicates. Base still passes; plus fails. | Python type-domain behavior changes: an independently reasoned `[True]` contributes1 under AV and0 under F04. The exact official failed input is unknown. Existing behavior12 and fixed follow-up9 do not test heterogeneous scalar/subclass contracts. |
| HE152, gain | Two same-name definitions (first `abs(generator)` survives saved sanitizer) → one correct elementwise absolute-difference comprehension. | Existing C1/C2 cover exact entrypoint and uniqueness, but this result does not prove a particular weight family owns extraction correctness. |
| LCB3777, gain | Zero-filtered parity/product DP → separate `dp_zero` states and zero-preserving parity transitions; updates are staged before merging. | F1/F2 zero-position contracts and D state progression address dependencies. They do not cover all reachable subset/product-cap interactions; verifier pass is not universal correctness proof. |
| LCB abc390_g, loss | NTT-based polynomial multiplication → nested Python coefficient loops; the combinatorial formula is also rewritten. | Concrete asymptotic cost changes from transform multiplication to product-of-degrees loops. Saved binary failure does not identify timeout versus wrong output; do not assert the rewritten formula is wrong without proof. Small9-case fixtures do not test algorithm scaling. |
| MBPP641, gain | Renames required `is_nonagonal` to `nonagonal_number` and adds n<1 guard → correct required name and direct formula. | C1/C2 cover the callable-name contract. Both name and guard change; the saved earlier sanitizer already establishes the AV name mismatch. |
| MBPP801, loss | All-distinct branch returns0 → returns maximum frequency, hence1 for all-distinct; F04 self-tests encode that new convention. | Follow-up I1/I2/I3 explicitly specify frequency/tie/sentinel contracts. This supports contract-aware testing, not treating “more confidence in maximum frequency” as sufficient. |

The nine follow-up cases were frozen and validated before these F04 outcomes and remain unchanged. Type-domain and large-input complexity are explicit coverage gaps, not retrospectively added gates or claimed explanations of the whole score. No new F05 target/candidate or model call was computed in this interpretation.
