# CPU validation r8（2026-11-07）

**564 collected = 507 passed + 57 optional/explicit-enablement skips、failure/error0。**
unittest報告48.484秒、実wall48.989秒。Ruff、mypy（74 source files）、data consistency、31 shell構文確認もすべてexit0。全体49.403秒。固定した新しいsourceに対して一度だけ実施し、再実行していない。

| Check | 結果 | wall秒 |
|---|---|---:|
| unittest discover | 507 pass / 57 skip / 0 fail | 48.989 |
| Ruff | PASS | 0.035 |
| mypy src | PASS、74 files | 0.248 |
| model-comparison data --check | PASS | 0.042 |
| sh -n | 31/31 PASS | 個別exitをreceiptへ保存 |

固定対象はfinite対応GPU4 owner `39707865f4cc4851290d218d0f4cb6a3fa8768a940a124a9f0ea953ee0d2c5d7` と native_dependency `8aefc1862e964ea9d2f821a1bcb413ad8f421cecc3e56b140583ed56da37a7a9`。開始時に両者のexact SHAをassertし、Gitが列挙するtracked/untracked（ignore対象除外）のPython/shell/TOML/CI workflow全319fileの前後SHA mapが完全一致した。途中の追加・削除・変更は0。
親が直前に追加したF03 CPU準備script `fresh-f03-prepare-20261107.py` は開始時点で既に含まれ、SHA `36ac2be3c119624b382e0cf37b40127c6a4d5a6a7a3b4b59b3022afb08eb7aad` のまま終了した。この実行の後に変更されたsourceまで検証済みと称さない。

既存r7入口をr8の新規出力先へ再利用した。`TMPDIR=R/tmp-ci-r3`、`infra/weight-tools-ci/bin`をPATH先頭とし、そのPythonをKAIRYU_BENCH_PYTHONへ設定。PYTHONDONTWRITEBYTECODE=1、HOMEは変更していない。Rは `/mnt/nvme/kairyu/wsp/gemma4-coding-20261104`。実モデル/API/GPU/serving操作は0。このCPU検証は別CI jobのDocker build/locked runtime実行やモデル能力の検証を代替しない。

[実行原文](records/2026-11-07-cpu-validation-r8/run.py.txt)、[receiptと前後source map](records/2026-11-07-cpu-validation-r8/receipt.json)、[全数/skip理由summary](records/2026-11-07-cpu-validation-r8/summary.json)、[lossless圧縮検証](records/2026-11-07-cpu-validation-r8/compression.json)、[SHA一覧](records/2026-11-07-cpu-validation-r8/SHA256SUMS)をPRへ保存。unittest/Ruff/mypy/data/shellの全5ログをgzipし、空logも含めdecompress後の全bytesを元logと比較した。元の未圧縮logは `analysis/fresh-purpose-20261107-cpu-validation-r8` に保持。

## Skipの全理由

| 理由 | 件数 |
|---|---:|
| requires the locked DeepSWE runtime | 18 |
| requires the locked dsh runtime venv | 2 |
| optional Torch CPU fixture | 3 |
| optional SciPy authored reference environment | 1 |
| optional scientific CPU environment | 1 |
| optional SciPy/SymPy CPU reference environment | 1 |
| optional SciPy/SymPy authored execution environment | 1 |
| optional CPU Torch byte-view contract | 1 |
| optional Torch CPU joint restoration contract | 2 |
| optional Torch and C++ CPU RAM contract | 1 |
| optional Torch CPU metric contract | 1 |
| optional saved-tensor CPU fixture | 1 |
| optional CPU Torch fixture; no model download | 4 |
| optional Torch CPU contract | 1 |
| the pinned Harbor runtime is not installed | 5 |
| optional pinned vLLM/Torch CPU runtime | 1 |
| optional Torch RAM-replay boundary | 1 |
| optional finite-response solver requires scipy | 7 |
| explicit coordinated CUDA smoke only | 2 |
| optional serving PyTorch unavailable | 3 |

CPU CIの前に行った[F03容量のread-only調査](records/2026-11-07-f03-capacity-readonly.json)では、変更対象が2shardのため現editorでpayload計62,546,338,264 bytesのprivate copyが必要と確認した。shard_poolは未変更shard限定、reflink/clone機能無し、native vector非対応のload-time overlayは標準物理checkpointの代わりにしない。実copy/delete/saveは行っていない。
