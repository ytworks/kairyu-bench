# AV behavior train12 native generation

All12/12 public answers completed in **18.620s** including
owner preflight/closure. Twelve single POST attempts, retries0, API errors0,
missing0, empty0, length-finished0; every reply finished with stop. Transfer
requests0. Numerical correctness is pending separate offline execution; this is
authored calibration, not a WSP score or unseen evaluation.

The exact approved plan SHA was
e088cf58f25426395a9754094e66e7b4046c452a78acfa71a1baa7aaefe720f1.
The reviewed owner PID801022 started child802345, retained the evaluation flock,
verified four ordinary AV endpoints and protected23 at closure, then released
normally. Source/input pins remained unchanged. No model service changed.
[Plan/preview/frame readiness](2026-11-07-behavior18-frame-and-train12-readiness.md)
and [complete raw records](records/2026-11-07-av-behavior12-native/SHA256SUMS)
preserve all12 public answers, request IDs, usage, receipt and closure.

| Case | Finish | Input tokens | Output tokens | Request seconds |
|---|---|---:|---:|---:|
| behavior-A1 | stop | 114 | 1424 | 12.641 |
| behavior-A2 | stop | 108 | 1119 | 9.864 |
| behavior-B1 | stop | 106 | 1769 | 15.351 |
| behavior-B2 | stop | 117 | 1996 | 16.195 |
| behavior-C1 | stop | 109 | 1256 | 10.818 |
| behavior-C2 | stop | 120 | 1122 | 9.092 |
| behavior-D1 | stop | 109 | 699 | 5.916 |
| behavior-D2 | stop | 117 | 631 | 5.475 |
| behavior-E1 | stop | 121 | 989 | 8.048 |
| behavior-E2 | stop | 107 | 1311 | 10.082 |
| behavior-F1 | stop | 88 | 1661 | 13.937 |
| behavior-F2 | stop | 99 | 1209 | 10.256 |

API-reported totals: 1315 input tokens and
15186 output tokens. Request durations overlap under bounded
concurrency; their sum (127.677s) is not
wall time. Completion counters are not asserted to count public code alone.
Hidden reasoning was neither retained nor inspected.

Conditions: same AV tree5d056eb0…, four native TP2/MTP4 ordinary replicas, gateway
model gemma-4-31b, exact train envelope SHAb42990b0…, source generator df88767e…,
server-default thinking (serving default true separately attested), seed0,
concurrency16/actual12, max_tokens32768, timeout1200, no sampling overrides.
The caller-supplied checkpoint field is backed by the pinned saved identity,
read-only mounts and pre/post serving checks; it alone is not an attestation.

Raw response SHA 5ea0e169922048a2e5bf649ceb6b1add7bb81db5ee19bcf48c620cb00f3de3db.
No hidden thought data or generated code host execution was used in this read-only
closure audit. Offline grading and B1/B2 public-code method inspection are separate
authorized steps; unknown correctness is not represented as twelve passes.
