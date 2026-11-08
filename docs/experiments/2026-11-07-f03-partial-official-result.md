# F03: partial canonical dev30 result — 2026-11-07

F03 retained the full **2,338-item selection** and produced 2,338 response records. The official runner saved **2,337 scores and one unknown**, terminal-bench:polyglot-c-py, after an explicitly authorized targeted interruption. This run is **partial**, cannot enter a complete-run ranking, and is not adopted. Existing AV remains the highest complete parent. **No AV benchmark was rerun.**

The code cohort itself is complete: **90/90 scored**, four-source mean **72.6587301587**, versus existing AV **72.9365079365** (delta **−0.2777777778 points**). This is not a complete 14-area evaluation.

## Complete code cohort

| Source | AV | F03 | Delta (points) |
|---|---:|---:|---:|
| humaneval-plus | 96.66666667 | 100.00000000 | +3.33333333 |
| livecodebench | 75.00000000 | 75.00000000 | +0.00000000 |
| mbpp-plus | 76.66666667 | 73.33333333 | -3.33333333 |
| scicode | 43.41269841 | 42.30158730 | -1.11111111 |

HumanEval+ is 30/30 (AV 29/30), LiveCodeBench 15/20 in both arms, MBPP+ 22/30 (AV 23/30); SciCode averages ten per-problem partial-credit scores. These are all five changed code items; the other **85 are equal**.

| Item | AV | F03 | Evidence |
|---|---:|---:|---|
| humaneval-plus:HumanEval/152 | 0.000000000 | 1.000000000 | base/plus fail → pass |
| livecodebench:3777 | 0.000000000 | 1.000000000 | verifier fail → pass |
| livecodebench:abc395_f | 1.000000000 | 0.000000000 | verifier pass → fail |
| mbpp-plus:Mbpp/801 | 1.000000000 | 0.000000000 | base/plus pass → fail |
| scicode:72 | 0.888888889 | 0.777777778 | 8/9 → 7/9; step 72.9 newly fails AssertionError; 72.5 remains failed |

Full IDs, equal scores, all failures, SciCode step diagnostics and generation-token metadata are retained in the all-90 code artifact. No benchmark reference answers were moved into calibration.

## All areas and sources

Among **2,241 paired capability items**, 44 improve, 54 worsen and 2,143 are equal. One additional capability item is unknown. The 96 diagnostic NLL items are separate. Each area is a source-equal mean. The agents row uses nine paired items with one missing and is **not a complete agent-area result**.

| Area | AV paired | F03 paired | Delta (points) | Unpaired |
|---|---:|---:|---:|---:|
| knowledge | 85.55555556 | 85.55555556 | +0.00000000 | 0 |
| expert | 83.33333333 | 89.58333333 | +6.25000000 | 0 |
| facts | 71.66666667 | 73.33333333 | +1.66666667 | 0 |
| math | 91.25801282 | 92.17147436 | +0.91346154 | 0 |
| code | 72.93650794 | 72.65873016 | -0.27777778 | 0 |
| instructions | 89.91666667 | 86.25000000 | -3.66666667 | 0 |
| tools | 79.16666667 | 79.16666667 | +0.00000000 | 0 |
| agents | 52.50000000 | 30.00000000 | -22.50000000 | 1 |
| long-context | 84.83752437 | 72.79802942 | -12.03949495 | 0 |
| reasoning | 84.13043478 | 81.37681159 | -2.75362319 | 0 |
| commonsense | 61.66666667 | 61.66666667 | +0.00000000 | 0 |
| multilingual | 78.87205387 | 78.31649832 | -0.55555556 | 0 |
| vision | 79.16666667 | 75.00000000 | -4.16666667 | 0 |
| safety | 57.40740741 | 57.40740741 | +0.00000000 | 0 |

Complete-area losses include long-context −12.03949495 points, instructions −3.66666667, vision −4.16666667 and reasoning −2.75362319. Expert improves +6.25, facts +1.66666667 and math +0.91346154. These are descriptive one-pair results, not an isolated causal estimate.

The saved report's overall 73.9488933437 and the existing compare --partial paired-subset overall are **not complete overall scores**. Both remain in JSON but are not used for ranking. Missing polyglot is never assigned zero.

| Source | Selected | F03 scored | AV reported mean | F03 reported mean | Raw + / − / = / unknown |
|---|---:|---:|---:|---:|---|
| aime-2026 | 12 | 12 | 0.937500000 | 0.979166667 | 2 / 0 / 10 / 0 |
| arc-challenge | 300 | 300 | 0.566666667 | 0.580000000 | 5 / 1 / 294 / 0 |
| bbeh | 46 | 46 | 0.782608696 | 0.760869565 | 2 / 3 / 41 / 0 |
| bfcl | 120 | 120 | 0.791666667 | 0.791666667 | 5 / 5 / 110 / 0 |
| gpqa-diamond | 48 | 48 | 0.833333333 | 0.895833333 | 3 / 0 / 45 / 0 |
| gsm8k-platinum | 50 | 50 | 1.000000000 | 1.000000000 | 0 / 0 / 50 / 0 |
| hellaswag | 300 | 300 | 0.610000000 | 0.603333333 | 5 / 7 / 288 / 0 |
| hmmt-feb-2026 | 13 | 13 | 0.846153846 | 0.807692308 | 0 / 1 / 12 / 0 |
| humaneval-plus | 30 | 30 | 0.966666667 | 1.000000000 | 1 / 0 / 29 / 0 |
| ifbench | 50 | 50 | 0.840000000 | 0.800000000 | 0 / 2 / 48 / 0 |
| ifeval | 120 | 120 | 0.958333333 | 0.925000000 | 0 / 4 / 116 / 0 |
| jamc-qa | 72 | 72 | 0.638888889 | 0.638888889 | 3 / 3 / 66 / 0 |
| livecodebench | 20 | 20 | 0.750000000 | 0.750000000 | 1 / 1 / 18 / 0 |
| longbench-v2 | 10 | 10 | 1.000000000 | 0.700000000 | 0 / 3 / 7 / 0 |
| math-500 | 60 | 60 | 0.866666667 | 0.900000000 | 2 / 0 / 58 / 0 |
| mbpp-plus | 30 | 30 | 0.766666667 | 0.733333333 | 0 / 1 / 29 / 0 |
| mgsm | 55 | 55 | 0.927272727 | 0.927272727 | 0 / 0 / 55 / 0 |
| mmlu-pro | 180 | 180 | 0.855555556 | 0.855555556 | 4 / 4 / 172 / 0 |
| mmlu-prox-ja | 60 | 60 | 0.800000000 | 0.783333333 | 2 / 3 / 55 / 0 |
| mmmu-pro | 48 | 48 | 0.791666667 | 0.750000000 | 0 / 2 / 46 / 0 |
| mrcr-v2 | 16 | 16 | 0.670125731 | 0.608940883 | 1 / 2 / 13 / 0 |
| musr | 30 | 30 | 0.900000000 | 0.866666667 | 0 / 1 / 29 / 0 |
| probe-code | 40 | 40 | 4.022500118 | 3.954920417 | 14 / 26 / 0 / 0 |
| probe-wikipedia-ja | 40 | 40 | 6.628099446 | 6.590620263 | 16 / 24 / 0 / 0 |
| probe-wikitext | 16 | 16 | 7.638729488 | 7.575848690 | 1 / 15 / 0 / 0 |
| ruler | 24 | 24 | 0.875000000 | 0.875000000 | 1 / 1 / 22 / 0 |
| scicode | 10 | 10 | 0.434126984 | 0.423015873 | 0 / 1 / 9 / 0 |
| swe-bench-pro | 4 | 4 | 0.250000000 | 0.000000000 | 0 / 1 / 3 / 0 |
| terminal-bench | 6 | 5 | 0.833333333 | 0.600000000 | 0 / 1 / 4 / 1 |
| triviaqa | 120 | 120 | 0.716666667 | 0.733333333 | 3 / 1 / 116 / 0 |
| winogrande | 300 | 300 | 0.673333333 | 0.666666667 | 4 / 6 / 290 / 0 |
| xstest | 108 | 108 | 0.574074074 | 0.574074074 | 0 / 0 / 108 / 0 |

Capability values are 0–1 and higher is better. Probe source means are per-item NLL and **lower is better**; their positive raw deltas are losses. Probe-code mean NLL improves 4.0225001185 → 3.9549204165 while the complete code score decreases. The compare.probes output separately reports shared probe positions (code: 2,171 positions, NLL 3.1890228707 → 3.1440310119); these denominators are not pooled.

## Interruption, time and agent work

Polyglot had two saved assistant actions and nine tools while generation continued. A two-snapshot GET-only check observed +1,305 generation tokens in 10.020 seconds (~130.24 tokens/s), so the engine was progressing. At 11:05:19.210852 UTC the gateway logged “client disconnected; cancelled in-flight work”, about 1,200 seconds after the last saved step; configured LLM-call timeout was 1,200 seconds. Work then moved from gemma1 to gemma3 without a new saved action. This strongly suggests a nested timeout/retry; exact retry count is unknown because request-ID linkage and an attempt counter were unavailable.

The task's original agent timeout is 900 seconds with multiplier 48 (nominal 43,200 seconds). No timeout, token cap, task count or serving condition was changed. Following the user's acceleration instruction, root explicitly authorized ending this attempt as incomplete. At epoch 1794050001.632966 one SIGTERM was sent to exact Harbor host PID 1083709/container PID 602 after checking starttime 69562253, argv hash, parent chain and runner CID. The command ran inside the existing root-owned runner with **no host root/sudo**, no SIGKILL and no repeat signal. Main WSP/scorers and inference services were not signalled.

The adapter recorded CancelledError/null reward for polyglot, preserved five verifier results and allowed WSP to flush its already-computed RAM scores. Earlier progress wording “scoring not started” was inaccurate: scores were **not yet saved/available**, although source scoring had run in RAM. Historical progress records are not rewritten.

F03 wall time is **2975.2 s**, including **108.0 s** loading; AV was **2018.9 s**, including **111.2 s**. Interrupted F03 consumed 956.3 more seconds. The report's generation_seconds also starts before loading. Chat calls 1754 → 1744; logprob calls 9193 → 9193. WSP trace error/retry counters do not cover every nested agent retry.

| Agent task | AV score | F03 score | AV / F03 reported steps | AV / F03 tools |
|---|---:|---:|---:|---:|
| swe-bench-pro:instance_NodeBB__NodeBB-da0211b1a001d45d73b4c84c6417a4f1b0312575-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e | 0.0 | 0.0 | 24 / 23 | 24 / 23 |
| swe-bench-pro:instance_ansible__ansible-11c1777d56664b1acb56b387a1ad6aeadef1391d-v0f01c69f1e2528b935359cfe578530722bca2c59 | 0.0 | 0.0 | 41 / 39 | 41 / 39 |
| swe-bench-pro:instance_internetarchive__openlibrary-431442c92887a3aece3f8aa771dd029738a80eb1-v76304ecdb3a5954fcf13feb710e8c40fcf24b73c | 1.0 | 0.0 | 16 / 23 | 16 / 23 |
| swe-bench-pro:instance_qutebrowser__qutebrowser-ed19d7f58b2664bb310c7cb6b52c5b9a06ea60b2-v059c6fdc75567943479b23ebca7c07b5e9a7f34c | 0.0 | 0.0 | 42 / 55 | 42 / 55 |
| terminal-bench:bn-fit-modify | 1.0 | 1.0 | 10 / 7 | 18 / 14 |
| terminal-bench:cobol-modernization | 1.0 | 1.0 | 6 / 8 | 35 / 35 |
| terminal-bench:install-windows-3.11 | 0.0 | 0.0 | 13 / 6 | 38 / 17 |
| terminal-bench:polyglot-c-py | 1.0 | unknown | 7 / 3 | 18 / 9 |
| terminal-bench:portfolio-optimization | 1.0 | 1.0 | 7 / 5 | 13 / 12 |
| terminal-bench:reshard-c4-data | 1.0 | 0.0 | 14 / 9 | 44 / 21 |

Terminal steps above are reported n_episodes; live monitoring counted persisted assistant trajectory steps. Polyglot has three reported episodes but only two saved assistant actions. SWE steps are saved assistant messages. F03 SWE has 140 saved assistant messages/tools and 141 reported API calls; missing usage is not zero-filled. F03 Terminal-Bench has 38 reported episodes, 37 saved assistant actions and 108 tools including the interrupted attempt.

Owner ended with process returncode 1 for the partial run, elapsed 2,981.634 seconds. Harbor, shell, main WSP and owner PIDs disappeared. All pre-stop running host containers except the owned WSP runner remained running; four inference CIDs and gateway CID were unchanged. No global-lock probe was repeated because the separately authorized F04 CPU build had already acquired it.

## Reproduction and validation

Existing CPU package compare was invoked with explicit --partial, bypassing the wrapper/image build. Existing fresh-n30-complete-audit.py received only an expected-tree argument and opt-in --allow-partial; default completion requirements remain. It retains all 2,338 records before strict checks, preserves the unknown and records full_score_ranking_permitted=false. The unchanged three missing/duplicate/nonfinite tests passed (0.069 seconds); compile/help and diff checks passed. No full-suite rerun was added.

Runtime is matched f91 runner, af9 serving, TP2×4/MTP4, concurrency 256/32/4, all canonical selected IDs/budgets and HTTP-client trace-only mode. Runner identity was captured **mid-run**, 1,783.8 seconds after owner start, as explicitly stated in the receipt. Old AV audit outputs and their run/report/items hashes were reused without new inference.

[All records and exact one-time commands](records/2026-11-07-f03-official-comparison/README.md) include lossless compact numerical outputs and input hashes. The completed-only collector was prepared but **never executed**; the explicit partial collector ran once successfully. One packaging command failed JavaScript parsing before SSH and made no remote change; the corrected packaging retained this failure. Source-discovery misses/permission denials did not alter evaluation. No live progress log, credential, benchmark gold or generated answer is included in this compact PR package.

No candidate adoption, DeepSWE result or Hugging Face publication is claimed.
