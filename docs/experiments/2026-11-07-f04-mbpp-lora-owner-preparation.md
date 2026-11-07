# F04 MBPP LoRA owner preparation — 2026-11-07

The existing GPU4 owner now has an explicit `train-lora` mode. This record closes CPU preparation; the later root-approved training execution is recorded separately.

The change reuses the native/finite global and GPU4 locks, exact target stop/restart, protection of the other 26 containers, owned-child cleanup, and the second inactivity check after client termination. One pinned training job replaces the native collection jobs in this mode. Training is not labeled as a native19 observation. The existing native/finite 900-second collection bound remains unchanged.

## Fixed scope

- Parent: adopted full-WSP30 F04, tree `7b9e9488e3cd4e0b6c5f61ae341e5eeccdf5bd31c46d1c8354950dbc29d0f924`. Its read-only nine files use existing full hashes plus exact current stat continuity; this is not a new full hash.
- Exact gemma-2 CID: `07e1b17e7c84d8d178703417bb341c3313653eca7122aba1ad7b3d6d893d5abb`. The safe snapshot contains 27 containers, no temporary sandbox, and healthy idle target queues. Execution checks these again under nonblocking locks.
- All TRAIN 204 original references passed. The disjoint HOLDOUT 51 never enters gradients. Exact JSONL hashes, ordered IDs, split summary, all255 outcomes and the trainer's same reference-quality summary are linked.
- Tokenizer proof retains all204 complete frames, official ON chat framing, prompt-masked answer and official turn terminator loss, no invented empty thought or truncation, all at most1024 tokens.
- All410 dense text projections, FP32 rank8 LoRA factors, BF16 parent, SDPA, eligible saved FP32 activations on CPU; GPU4 only, allocator cap88GiB, container memory128GiB, no network.
- Four pilot steps precede a maximum300-second continuation wait. Root alone can authorize the same model/optimizer to total64 with a `continue.json` tied to the SHA of `pilot-ready.json`. Otherwise the explicit outcome is `pilot_only`.
- Trainer budget1800 seconds includes loading/tokenization but excludes actual approval wait. Owner deadline2400 seconds comprises1800+300 wait+300 startup allowance. Stop/restoration retain separate existing limits.

Training loss, gradient health and export validity are not WSP scores or evidence of adoption.

## CPU checks and preserved evidence

The owner suite increased from23 to26 cases. Three additions cover training metadata/quality validation, rejection of a false64-step completion claim, and a failed training child going through owned cleanup before exact original restart. All26 passed in0.435 seconds; focused Ruff passed. These tests mocked Docker/GPU operations.

Both the intermediate and final schema-aligned CPU validation receipts are retained. A shell quoting failure happened before remote execution; the one-time preparation's first Ruff check rejected an ambiguous variable name before plan creation. Both are retained as preparation failures. No model call or service mutation occurred during preparation.

The frozen trainer metadata preview and complete owner preview passed. The original plan is unapproved; a separate root-approved plan with explicit SHA is required for execution. The previous executed native owner is preserved byte-for-byte.

| Artifact | SHA256 |
| --- | --- |
| owner source | `3e7e9c58b68a679d87623e1507e6508adcc731c3da36d51dd390f9bffcdb2b07` |
| trainer source | `8bfa6f3f5568264417d52e1cf297ebf0f988458c0c075f329a189a0233a7a846` |
| unapproved owner plan | `c3de99ec89a2730d1350d645b557d6d7fe8bb7f94a8a209085b3cf9634eccd67` |
| trainer plan | `5f8de3c395cf249be6d22406f9c0d339c1cf21150a6ea0497967d6805f9b83df` |
| owner preview receipt | `c4f5117a0053f6e5678e0ec158ad7ec2f6d666580590d1f7cf4241b2785c83a6` |
| TRAIN frame proof | `41fc78e1f8d26c52b1f5ce85120aba3049e6f2797c792232d59a9785b6c85cf7` |

See the [record index](records/2026-11-07-f04-mbpp-lora-owner-preparation/archive-index.json), the [one-time CPU preparation](scripts/fresh-f04-mbpp-lora-plan-20261107.py), and the [shared owner](scripts/fresh-av-native-gpu4-20261107.py). Preparation used the frozen trainer SHA above; execution remains `--plan <root-approved-plan> --run --approved-plan-sha256 <exact SHA>`.
