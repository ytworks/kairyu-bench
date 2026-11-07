# Behavior18 frame evidence and train12 launch readiness

All18 authored correct/foil frames fit the complete1024-token bound; maximum
**290 tokens**, no truncation/exclusion. Tokenizer-only work took
9.049s including container lifecycle; frame preparation inside
the container reported 2.052s. Model forwards, API calls and
GPU calls were0. All27 existing containers, all9 AV file stats and pinned inputs/
sources were unchanged. This is readiness evidence, not a capability score.

[All records](records/2026-11-07-behavior18-frame-and-train12/SHA256SUMS) preserve
the failed v1 (Python executable absent, Exit127 before program start;0.654s
including cleanup) and successful v2. The correct af9 entrypoint is
/usr/bin/python3, as used by the prior finite owner. Both temporary CPU containers
were recorded and removed by exact ID after being inactive. No model service
was stopped. The [earlier preparation](2026-11-07-new18-train12-preparation.md)
is historical; its unmeasured/pending status is superseded only by these results.

| Case | Complete correct tokens | Complete foil tokens |
|---|---:|---:|
| behavior-A1 | 208 | 202 |
| behavior-A2 | 181 | 181 |
| behavior-B1 | 199 | 193 |
| behavior-B2 | 242 | 229 |
| behavior-C1 | 155 | 153 |
| behavior-C2 | 197 | 220 |
| behavior-D1 | 179 | 179 |
| behavior-D2 | 197 | 200 |
| behavior-E1 | 258 | 290 |
| behavior-E2 | 154 | 154 |
| behavior-F1 | 136 | 141 |
| behavior-F2 | 148 | 152 |
| behavior-A3 | 169 | 169 |
| behavior-B3 | 276 | 243 |
| behavior-C3 | 179 | 179 |
| behavior-D3 | 210 | 210 |
| behavior-E3 | 165 | 165 |
| behavior-F3 | 186 | 190 |

These are shared forced-empty-thought frames (existing boundary strip, no closing
turn tokens), distinct from native generated-thinking responses. Transfer6 was
only tokenized locally. All18 authored fixture validation is [separate](2026-11-07-behavior18-implementation.md);
transfer behavior remains unmeasured until candidate spec/amplitude freeze.

## Fixed implementation and preview

The [single train12 owner](scripts/fresh-av-behavior12-native-20261107.py) reuses
science_generate server-default with concurrency16, seed0, max_tokens32768,
timeout1200, retry0. It requires an approved exact plan SHA, original twelve
IDs/folds, all18 matching frame evidence, all4 ordinary AV exact image/CID/command/
read-only mounts and protected23 unchanged. It holds the global flock through
generator completion and idle closure, including repeated handled stop requests.
Persistent closure errors deliberately retain the lock and write bounded-field
pending receipts for root supervision. It never changes serving state.
SIGKILL cannot preserve a Python lock; successors must independently recheck idle.

The [CPU tokenizer script](scripts/fresh-behavior18-frames-20261107.py) uses local
AutoTokenizer and shared prepare_frames only. Its full18 evidence is retained;
the train12 view preserves exact group/fold/row correspondence with a subset
proof. Existing120 owner and original input envelopes are unchanged.
No dependency installation was needed: immutable af9 image provides
{"numpy": "2.2.6", "tokenizers": "0.23.3", "torch": "2.14.0+cu130", "transformers": "5.20.2"}.

Three focused guards passed: reject transfer/duplicate/non-singleton input;
reject missing/wrong exact-plan approval; and actual repeated SIGTERM during a
mock child preserves flock through child completion and serving closure before
release. Both owner and science_generate CPU previews passed with0 network/model
calls. Ruff and git diff checks passed. This validates guards and tokenizer,
not an actual new model run.

Working root N is analysis/fresh-purpose-20261107-av-behavior12-native-v1.
Its launch-plan.json is **approved=false**:
SHA 71cbd3cc035a8e2e5d895e0dcbe618e2de437d79b574821b791c93d845043d71.
Owner SHA 5de93a34ab2a45b0309eddbb8eb73b9919968bc6c9d9fd18579ad0b926ed2264.
Full frame SHA 7e6b18abc78a79e09899838ee047da715696baf6d8dc0e5597ee5c3a2ab5239a;
train frame SHA 21ae203f395c51f7d973436b42cfeab19f9d72b1c2db18e8c0faa4613fa26f2d.
Owner preview SHA 8ccec8ee6488325e17d22f75156e9306dfe62bc80ec670a62fd4101b7127d8ac;
generator preview receipt SHA 5f75fe664bcdf76ee84dc1174cda8ecef82dfa6a1b30848d5fdf254bc3740a91.

Root must create/review a new approved plan copy and explicitly authorize the
following, substituting its actual absolute path and SHA:

~~~bash
R=/mnt/nvme/kairyu/wsp/gemma4-coding-20261104
B="$R/worktrees/kairyu-bench-weight-tools"
"$R/infra/weight-tools-ci/bin/python" \
 "$B/docs/experiments/scripts/fresh-av-behavior12-native-20261107.py" \
 --plan "$APPROVED_PLAN" --run --approved-plan-sha256 "$APPROVED_SHA"
~~~

The unapproved original and previews remain untouched. Model submission and
offline answer execution are still pending separate root instructions.
