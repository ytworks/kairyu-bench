# F06 ordinary deployment and WSP handoff

F06 was deployed to all four ordinary replicas in **123.903780 seconds**, with all four helper processes completed, no rollback, and all 23 other containers unchanged. Endpoint verification completed in 4.821478 seconds including launcher overhead (4.616185 seconds inside the endpoint process). Both global-lock scopes released normally. F04 remains the highest completed parent pending the separate official evaluation.

The follow-up used exactly one gateway chat request and one direct logprob request per replica, all with successful API responses. **This is transport/generation success, not 5/5 instruction correctness.** For “Reply with the single word ready.” the saved public answer was:

```python
def ready():
  return "ready"

print(ready())
```

It finished with `stop`, 23 prompt tokens and 15 completion tokens. This is one observed instance of answering a non-code instruction with code, retained as a potential training side effect. It is not a benchmark score. No retry or additional probe was made.

## Identity and invariants

The dense LoRA candidate is `checkpoints/fresh-lora-mbpp-f06-v1/checkpoint`, tree `0fc0ef29346c4d8a06bea8570dee96d8e0ba7038fee0d5aded8db57852474ec5`. Deployment consumed the existing successful full verifier and nine-file identity with fresh stat continuity; it performed no payload rehash. The stored identity is post-verification, not a saved verification-start snapshot.

All four replicas retained image `sha256:af9e746b913088cbf6c4464a31d6904a6b7e1d3ba62f0895362ddf13595f7abe`, native settings, TP2 and MTP4. All 27 container members were accounted for (four new inference containers and 23 unchanged protected containers). Gateway and direct model discovery agreed on `gemma-4-31b`; chat and all four logprob probes reached F06. All four queues were idle at handoff.

| Service | New container ID | Direct endpoint |
|---|---|---|
| gemma-0 | `1ffc862fe747b8f4e6134c72fb7bb322cd6acd7e3fdd012cc1d9614d09932273` | `http://172.26.0.8:8000/v1` |
| gemma-1 | `ee04769375d9e78370d8d7372ad3c5de8223e50ce3a4a233ae8adabdc1bff897` | `http://172.26.0.6:8000/v1` |
| gemma-2 | `48551f7b49ceb64b921ffc986de01b04cf1a53386173a1d15ad6af87c50e22ab` | `http://172.26.0.5:8000/v1` |
| gemma-3 | `657399f41b7c641b860a1c406123be1e75960118d6044d2c0aed367859c05461` | `http://172.26.0.4:8000/v1` |

The unchanged gateway was `http://172.26.0.9:8000/v1`. The existing `chat-checkpoint.json` was atomically updated only in `files`, `path`, and `tree_sha256`, preserving its schema. Exact before and after bytes were retained. Before SHA: `4e4b9a1d6670624cab7e4f53accad87970bf10635599ce2d06a2f6d2b0652552`; after SHA: `2f23d9975c3633cb1e1d0d2cc17e67ba63163a04df60c2ea6997822758797f09`.

## Execution and evidence

The approved plan SHA was `a1ca6fe0e8756b64e8957e7d14361eef50aef2169f6e0ec2c81600d724fca39f`. The existing parallel owner and single-service helper were unchanged. The endpoint one-off reused the previous F04 endpoint source with explicit path, expected hash, process ID and label substitutions preserved in its command record. No generic deployment feature or additional model computation was added.

- [All-four loaded identity and closure](records/2026-11-07-f06-deployment-dp4-v1/four-apply-closure.json): SHA `102d743fbc5ff4c190be6db13c89f076d367f832e05a490eeacd1241351e8098`.
- [Endpoint closure](records/2026-11-07-f06-deployment-dp4-v1/endpoint-closure.json): SHA `1137d4e6f2ba4de6fa2ec141a06e29c8218d67e827a60d3bbd1685ffb191b2ee`.
- [Normal endpoint lock release](records/2026-11-07-f06-deployment-dp4-v1/global-lock-released.json): SHA `dfe2fe321a8debc13a4e41614723215ea400c8dd7477199602e89e091cc6ab41`.
- [All four child outcomes](records/2026-11-07-f06-deployment-dp4-v1/parallel-apply-v1/all-helper-results.json) retain individual times, return codes and exact receipts.
- [Smoke public answer](records/2026-11-07-f06-deployment-dp4-v1/gateway-chat-smoke.json), [endpoint source](records/2026-11-07-f06-deployment-dp4-v1/endpoint-source.py) and [source transformation/launch](records/2026-11-07-f06-deployment-dp4-v1/endpoint-launch-source.py.txt) preserve the operation.
- [Final record index](records/2026-11-07-f06-deployment-dp4-v1/completed-records-index.json) maps every analysis artifact to its PR record and SHA. Earlier snapshots are retained; changed terminal output is appended under `final-snapshot-v1` instead of overwritten.

The official selection remains all 2,338 WSP30 dev items, all 14 areas and 96 probes, with the same immutable runner and concurrency as F04. Deployment made zero official evaluation requests and no new F04 baseline requests.
