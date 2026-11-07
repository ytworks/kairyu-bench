# F06 partial HTTP and TerminalBench observations

This read-only check did not issue model requests, rescore, stop tasks, or change WSP inputs/results. HTTP errors, response-level errors, parser warnings and final task rewards are separate evidence.

| WSP item | Attempt 0 | Observed recovery |
|---|---|---|
| mrcr-v2:1940 | HTTP502 after 1800.601s | Same call/input SHA/seed attempt1 success108.837s; final response error=null |
| longbench-v2:66ebd0ea5a08c7b9b35dff57 | HTTP502 after1800.041s | No completed retry/final response in this snapshot; recovery unknown |
| mmlu-pro:1843 | HTTP502 after1800.020s | Same call/input SHA/seed attempt1 success23.484s; final response error=null |

The gateway recorded three upstream-backend errors at14:33:16,14:33:20 and14:35:11UTC with httpcore/httpx ReadTimeout, corresponding to these completion timestamps. This supports upstream read timeout as the immediate HTTP502 cause; it does not identify the backend reason for delayed responses. These item IDs are separate from the bn-fit-modify TerminalBench trial. [Full attempt metadata](records/2026-11-07-f06-http-errors-readonly-v1/snapshot-1794062392357686677.json) and [safe gateway exception metadata](records/2026-11-07-f06-http-errors-readonly-v1/gateway-readtimeout-evidence.json) retain the evidence. Raw credentials/env were not dumped.

TerminalBench subsequently reached5/6 completed: polyglot-c-py and portfolio-optimization passed; reshard-c4-data, install-windows-3.11 and cobol-modernization failed. bn-fit-modify remained unscored. Counts here are partial.

Agent-step counts include only trajectory rows whose source is agent. Total trajectory rows are retained separately and include the initial user row. Corrected completed counts were polyglot12agent/13total/15tools, reshard8/9/16, windows15/16/23, portfolio12/13/22, cobol250/251/0. An earlier operator short report used total-row counts as steps; this document clarifies the difference.

COBOL recorded250 “No valid JSON object found” parser warnings and no tool calls, with reward0, no terminal exception, agent execution1440.086s and total trial1468.073s. Parser failures are not HTTP errors. The last observed bn-fit trajectory had7agent rows and6tools; saved logs alone do not prove instantaneous active generation or remaining duration.

[Read-only snapshots and sources](records/2026-11-07-f06-terminal-progress-readonly-v1/) preserve the progression, per-task durations and input hashes. The first snapshot source had a missing parenthesis and failed before reading/writing model state; its original source and failure are retained. The corrected source is v2. Monitoring and the final completed-cohort reasoning comparison were handed off to audit_results; no final success or failure is inferred from missing records.
