# AV authored 120-case native generation, 2026-11-07

Status: the authorized v1 trial finished with **120/120 HTTP errors and 0/120 public answers**. Gateway access logs identify all 120 as HTTP 400. All attempts and all failures are retained; correctness is unknown for every case. The generator receipt status `completed` means attempt processing ended, not that answers succeeded. No score or weight conclusion is drawn. This generation-only diagnostic changed no case, weight or service and executed no answer. The known authored calibration inputs are not presented as unseen evaluation.

The complete original 120-case ID set is retained as four fixed sequential
batches. All original fields and purpose/fold positions remain unchanged;
56 rows without a split receive an explicit `train` field in the derived
envelope. Only `id` and public `prompt` are selected by the existing generator;
correct/foil answers are not sent. Input mapping and validation are maintained
by the separate authored120 catalog audit.

| Order | Batch | Cases | Input SHA256 |
|---|---|---:|---|
| 1 | basic | 16 | `ed17e06c04c8d6416168e67501b4a07e9490fa4a76d5355a746370447ba99ae4` |
| 2 | advanced-v2 | 16 | `9d51dfdeaa9e61a4d989b23459c4eff8b3d5b7d7d9291de3258bd3c12921797b` |
| 3 | library-train-v1 | 16 | `d3059d4d1c073ba372d103ace9760398341a1add1133e70e306cfefab94db6fc` |
| 4 | unsupported72 | 72 | `30606fd3b14b4c7809a488fda2bc7bdf318fead9543228b50e7617b0754ca7be` |

The 48 catalog-backed cases have existing CPU executors; three contract-question
cases remain a separately marked stratum. The remaining 72 lack a selected
executor and must remain unsupported/unknown, never converted to zero. This
owner performs no correctness grading for any of the 120 cases. Future
comparisons must retain the same batch order and conditions.

## Frozen plan and source

The [complete preview artifact](artifacts/av-authored120-native-preview-20261107.json)
contains every ID, exact command, bound source/input hash, four generator preview
receipts, owner preview and checkpoint stat evidence. Its remote working root is
`analysis/fresh-purpose-20261107-av-authored120-native-v1` (N120). The original
unapproved `launch-plan.json` is retained even if a separately approved plan is
later issued. The preview plan SHA256 is
`97a3da29997bc14734cb461eca6c174385f8f8180d48da8018a65c054fb20e21`.

[The small owner](scripts/fresh-av-authored120-20261107.py) is SHA256
`7cc455a29250f04819bc287397d45b594be742c2f7ff69cfaefb5a81024080d1`.
It calls the existing `scripts/experiments/gemma4-purpose/science_generate.py`,
SHA256 `8d05c960d4b8dbb3112f7fd2214df50b7b31c29c78f07ad1294f021a5fafb8af`.
The generation Python is the existing CPU environment
`/mnt/nvme/kairyu/wsp/gemma4-coding-20261104/infra/weight-tools-ci/bin/python`
(Python 3.12.13); generation uses only the standard library. No packages are
installed and no Docker executor is launched.

Parent checkpoint tree is
`5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf`.
The four ordinary AV CIDs are the unchanged dev30 closure CIDs, with native
BF16, TP2 per replica, four replicas and MTP4. The gateway is the bound ordinary
CID `88943a0985ba97dfb17be6b5fb54040eba2570b70ce05126bd410162d3658ffb`,
addressed from the host as `http://172.26.0.9:8000/v1`. This IP comes from the
actual deployment endpoint closure, not a presumed public host port.

## Generation protocol

Each case receives one attempt and one returned choice, seed 0, thinking enabled,
max_tokens 32768, concurrency 16, request timeout 1200 seconds and zero retries.
The existing generator omits temperature and top_p so the current native server
generation configuration applies. This configuration is preserved in the plan.
Seed and configuration are evidence, not a guarantee of bitwise determinism
across future requests. Within each batch completion order may differ; results
are joined by the unchanged unique case IDs. The first three batches have 16
requests each; the last has 72 through the same bounded pool.

The per-batch command template is:

```sh
<PY> <REPO>/scripts/experiments/gemma4-purpose/science_generate.py   --cases <EXACT_PINNED_INPUT> --split train   --endpoint http://172.26.0.9:8000/v1 --model gemma-4-31b   --checkpoint-sha256 5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf   --output <FRESH_BATCH_OUTPUT> --concurrency 16 --max-tokens 32768   --timeout 1200 --seed 0
```

Without `--run` this existing command makes a CPU preview only. The owner adds
`--run` only when its own `--run` is requested and the separately supplied plan
has explicit approval. All four CPU previews completed with 16/16/16/72 cases,
zero model or network calls, retries 0 and no response file. The owner preview
also passed; it checks source pins, checkpoint stats, the full ID bijection,
fixed commands and fresh output paths without calling Docker or the network.

## Lock, failure records and monitoring

At execution the owner would create one fresh attempt directory, acquire the
existing global evaluation flock nonblocking, and preserve its PID/plan/source
receipt. Before every batch it checks pinned inputs and all nine checkpoint
stats against the earlier fresh full hashes. It checks exact CIDs, healthy
state, read-only AV mounts, all direct model IDs and zero running/waiting request
counters; it also verifies the exact gateway CID, IP and advertised model.
After the fourth batch it repeats the read-only preflight. No service operation
is part of the owner.

The original generator retains public answers, request IDs, finish reason,
usage, request seconds and safe error types; it discards hidden reasoning and
never records an error body. Each subprocess retains its own command/start PID,
stdout/stderr log, receipt and response artifact. Owner batch summaries retain
all metadata, batch elapsed time, error/length counts and missing IDs. A final
all-batch snapshot includes unstarted and incomplete batches. Response errors
are retained with no retry; process/preflight failure prevents the next batch.
For a handled owner termination the owner waits for the already issued generator
process before releasing the lock, then records the interruption without
starting a new batch. An uncatchable process kill cannot guarantee a final
receipt and must remain an explicit uncertainty.

Progress reads the atomically saved per-batch `train-responses.json` and
`receipt.json` at approximately one-minute intervals. It reports received/120,
per-batch completion, each error type, length finishes, known token usage and
elapsed time. A received public answer is not a passed test; all correctness
scores remain unknown until the separately reviewed CPU execution. The full
120-case envelope stays visible even if the run stops. No automatic repeat or
new candidate/official benchmark is authorized by this plan.

CPU validation: ruff passed for the owner; the existing generator's focused
three tests passed. Those tests cover preview without network, omission of
reference/hidden-reasoning content, native sampling fields and a failure with
no retry or private error body. The subsequent authorized request trial is reported below; preview checks did not exercise the gateway request contract.

## Authorized v1 failure and diagnosis

Root authorized plan SHA256
`5a53204ca6f7b60a6a97fbfd7a31d7dea04e7e68ff20f7ffca37455596b894de`
and detached owner PID 569900. The earlier unapproved plan and previews remain
unchanged. Global flock acquisition and all four replica/gateway preflights
passed. The owner ended after 2.1462135314941406 seconds, with batch elapsed
spans below. Every ID has exactly one `HTTPError` record and no public answer,
usage counter or request duration. No retry was made. Missing usage is not zero.

| Batch | Terminal attempts | Public answers | HTTPError | Batch seconds |
|---|---:|---:|---:|---:|
| basic | 16/16 | 0 | 16 | 0.138988495 |
| advanced-v2 | 16/16 | 0 | 16 | 0.128062725 |
| library-train-v1 | 16/16 | 0 | 16 | 0.130786180 |
| unsupported72 | 72/72 | 0 | 72 | 0.219551325 |

The [failure artifact](artifacts/av-authored120-native-failure-20261107.json)
retains all 120 ID/error records, original receipts, source and result hashes,
exact approved plan, safe access-log metadata, source rejection evidence and
thinking flags. Owner process absence and global lock release were verified;
all four exact ordinary AV CIDs were healthy and idle at final preflight.

The original generator only stores exception class, so its response files lack
HTTP numeric status and error body. A read-only, narrowly bounded gateway access
log extraction found exactly 120 `POST /v1/chat/completions` rows at
07:34:45–47 UTC, all status 400. Only timestamp, method, path, code and duration
were saved; no error/request body, credential or environment was displayed.
The initial access parser expected text logs and found no status matches;
inspecting JSON field names established `code` as the actual status field.
That initial diagnostic is preserved and superseded by the exact metadata
extraction, without modifying the original failed response records.

The actual gateway image accepts `chat_template_kwargs` as a protocol field,
but its text upstream-conversation path (`chat_service.py`, lines 805–809)
rejects a nonempty value when a Kairyu text template is absent. The existing
science generator sent `{"enable_thinking": true}` through that field; the
successful WSP client omits it. This source-level incompatibility is consistent
with the 120 observed HTTP 400 responses. The discarded HTTP error body cannot
be reconstructed from the generator records.

All four actual replica commands contain `--default-chat-template-kwargs
{"enable_thinking":true}` and `--reasoning-parser gemma4`. The checkpoint Jinja
alone defaults thinking to false, so the actual server flag is the relevant
true-default evidence. The gateway pool capability allowing `enable_thinking`
does not remove its earlier text-route restriction. A possible future explicit
server-default request mode must be separately reviewed and recorded as a new
trial; v1 is not overwritten or silently retried. The executed generator source
is preserved at `N120/science_generate-executed-v1.py` with the original SHA.
No additional generation request was sent during diagnosis.

The exact v1 owner and generator bytes are also retained in [the v1 source snapshot](records/2026-11-07-av-authored120-v1/source-snapshot.json) with byte counts and SHA256, so a later source change cannot erase the implementation that produced this failure.

## Separately proposed v2 and transport smoke

The v1 owner bytes are preserved before a four-line optional-plan extension:
only `thinking_control=server-default` adds the new generator CLI flag. The
[current owner](scripts/fresh-av-authored120-20261107.py) is now SHA256
`051d39833131c513b44813025ad081f81f6e83c1fdc55bcacd72f282ae242384`.
The generator revision is
`df88767ebcd5a3138747a36fe7e0976dbb0c2ceb5b0f1eece74740818f0b7f89`;
its default request behavior remains unchanged. In server-default mode the
request omits `chat_template_kwargs`, and its own receipt records
`enable_thinking=null`; the separate current-serving command evidence, rather
than an inferred generator setting, establishes the true native default.

[The v2 preview artifact](artifacts/av-authored120-native-v2-preview-20261107.json)
binds all 120 unchanged IDs, the same four input hashes and fixed batch order,
new fresh output paths and the same AV/gateway identities. N120v2 is
`analysis/fresh-purpose-20261107-av-authored120-native-v2`. Its unapproved plan
SHA256 is `1f5f40804358e42e0e725c999d6bcbda736d3d91044aa655a3a62cf8522af84e`.
Four generator CPU previews and owner preview passed with zero model calls.
The independently reviewed generator's five focused CPU tests passed; owner
ruff passed. Neither those tests nor previews establish actual API success.

Before authorizing v2, a separate one-attempt transport smoke is proposed:
`Reply with the single word ready.`, seed 0, max_tokens 256, concurrency 1,
timeout 1200 seconds, no retry, same server-default mode and same current
four-AV gateway identity under the global evaluation lock. This smoke is not
one of the 120 calibration cases. Its unapproved plan SHA256 is
`1d1c28484b43294f1585e23c2a59074c061f1233553731bade3c3975b4cf179a`; CPU
preview passed without a request. Expected success is one non-error public
`ready` answer with stop finish reason. Root review and explicit execution
instruction are required for both this smoke and the later full v2 trial.

Source checks of the actual gateway show model/messages/max_tokens/seed are
accepted fields. `seed=0` enters SamplingParams and the upstream payload uses
`value is not None`, preserving zero; the vLLM capability includes seed and no
configured seed offset/denial was found. Omitted native sampling fields stay
marked omitted. These are source-level transfer checks, not a determinism
claim or a successful runtime measurement. The audit files preserve source
hashes and line excerpts without executing model requests.

The root subsequently authorized exactly the separate smoke. Its approved plan
SHA256 is `3a742fc2e66690f139736147fb0c0e9d633ba45230af98c63449032557b5c846`.
[The smoke result](artifacts/av-authored120-native-v2-smoke-20261107.json) is
successful: the sole public answer is `ready`, finish reason `stop`, zero errors
and retries, 23 input tokens and 47 output tokens, request 0.3880546411 seconds,
owner 1.0442044735 seconds. The generator records server-default control and
unknown/null thinking status; the separately attested command default remains
the identity evidence. Hidden reasoning was not saved.

[The exact smoke operation](records/2026-11-07-av-authored120-v1/ready-smoke-v2-operation.py.txt)
is SHA256 `36d9544114882db4db4887213684365b59e64d7676c7a76273c497520dba76f0`.
It held the global evaluation flock, reused the frozen owner preflight, and
confirmed before/after all four exact AV CIDs healthy/idle plus all nine
checkpoint stats unchanged. Owner PID 600076 disappeared and the global lock
was verified released. No service changed. This one transport success is not
an authored-case or WSP result; full120 v2 still awaits separate authorization.
