# F04 train19 input and first public-answer score

F04 is the next parent after complete dev30 code tied AV (72.9365079365) and overall improved to76.9910864328. This authored calibration score is not added to WSP scores.

The input preserves all original A–F12 and G–I6 train rows exactly, in two singleton folds per purpose. J adds one singleton with ID behavior-J1: the original F2 public prompt and correct program, with the observed wrong foil values[0::k] contrasting required values[k-1::k]. Original F2 and its zero-removal foil remain unchanged. These are **19 calibration entries including one repeated prompt with a new foil**, not19 independent tasks. Old transfer6 and follow-up transfer3 files remain byte-identical and unused.

Input SHA256:74e1cd2000fda0abdc03f4454b18f4231b9891360af266391744e13976f199f1. Freeze:975e9087c66b72e15d666fbad8ab08f6b390e48fb2bc4428cc814e89bde85374. Ordered purpose/fold/ID and row hashes are retained. Existing behavior_followup_cases.training_payload()/training_catalog() compose the cohort. Fixture dispatch reuses original A–F and new G–I functions; no new numerical oracle.

Only J1's authored correct/foil were validated: two existing Docker worker calls,16 fixtures each,1.303934s. Correct passed16/16; offset foil passed4/16 and failed a fixed witness. Existing18 were not rerun. Expected values stay outside the worker. Code executes with network none, no host mount/GPU, uid1003, cap-drop-all, CPU1/512MiB/15s.

Root generated all19 once from F04: owner26.399529s,19 responses,0error/empty/length/retry. After normal lock release, the full CPU score ran once in15.267588s: **16 PASS /3 FAIL /0unknown;264/304 fixtures**,19 Docker calls. A1 failed0/16, F2 and J1 each4/16; every other case passed16/16. B1/B2 numeric success does not prove the required SymPy method.

Report SHA256:1599a3c564aa9fe019087618f9373a4d65d268ab2c11d7fb56859dab70cd9c82. Summary:2907a3768ecb3c4370a11ae43bc475309e251122afaa15929490a8f5acec8408. Cleanup confirmed zero science-sandbox containers and total27 host containers before GPU handoff. All pre-existing running services remained present; extra model/service operations0.

Preparation failures are retained: missing repository src import, then an incorrect equality check between raw rows and a reader that intentionally removes split metadata. Both stopped before Docker. The exported input stayed unchanged during resume. One metadata-only cleanup attempt used system Python lacking NumPy and was retried with scientific Python; it issued no Docker command. A pre-SSH documentation quoting error made no remote changes. No authored/generated program was rerun to hide failures.

[Inputs, source, lineage, every public response and every worker/trial result](records/2026-11-07-behavior19-inputs-and-score-v1/) include original/stored SHA256 and all one-off sources. The new preservation test covers old rows/foils, unique19 IDs and transfer exclusion. Shared scorer/worker and old case source are unchanged. Full post-change CI is separately tracked as r10.
