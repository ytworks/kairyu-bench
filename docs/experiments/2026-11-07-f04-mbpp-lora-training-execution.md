# F04 MBPP LoRA training execution — 2026-11-07

The one approved run completed all64 optimizer steps and256 forwards on204 distinct TRAIN IDs. HOLDOUT gradient IDs are empty. The immutable F04 parent remained unchanged. The final adapter is an experimental weight update, not an adopted model or a measured WSP improvement.

The owner completed in586.284809 seconds including stop/load/training/approval wait/export/restoration. Trainer elapsed was461.425499 seconds. Root explicitly approved continuation after the four-step pilot; the same model and optimizer continued. The exact source, plan, approval, step log, per-example microbatches, all410 module observations, tokenizer frames, pilot/final export receipts and complete source provenance are retained in the [raw archive](records/2026-11-07-f04-mbpp-lora-training-v1/archive-index.json). Full adapter safetensors remain on NVMe with receipt SHA and size; they are not embedded in Git.

The read-only midrun audit confirmed the actual af9 image, UID1003, GPU4 only, no network,128GiB memory, read-only checkpoint root, exact204 tokenizer frames, current source pins and nine checkpoint stat records, and all26 other containers unchanged. Owner-controlled cleanup finished its child before restarting exact gemma2 CID07e1b17e7c84d8d178703417bb341c3313653eca7122aba1ad7b3d6d893d5abb. Final independent inspection confirmed healthy, running/waiting queues0, all27 expected containers,26 protected identities, owner PID absent and no remaining lock under that PID. Errors, cleanup failures and uncompleted training steps:0.

CPU saved-activation bytes are cumulative save/unpack events, including repetitions; they are not resident RAM or unique storage. Batch loss measurements are training measurements, and source/gradient/activation checks do not establish general coding capability. Independent numeric training-log analysis is recorded separately, without additional model calls.

| Artifact | SHA256 |
| --- | --- |
| owner completion | 6b0e332f85075ead1d4e26c3ce3a3e82e7c39ebb7f82146421b91dbaf5a6afa7 |
| trainer summary | 3a62262a726fd3eee67a1d491a755ae3caebc21762024a3ab0246dc0cc2065f1 |

See the earlier [CPU preparation](2026-11-07-f04-mbpp-lora-owner-preparation.md). No additional baseline generation was performed.
