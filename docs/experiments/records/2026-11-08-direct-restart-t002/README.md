# T002固定scriptの再現記録

これは2026-11-08 direct restartの登録済みT002に使った非学習script13本の固定版である。汎用CLIではなく、SSH作業root・trial・候補・出力pathを固定した再現記録。元sourceとここに収録したscriptは同一device/inodeのhardlinkであり、追加のsource payloadコピーはない。識別情報・SHA256・AST検査は `script-snapshot.json` を参照。

T002の正規778問評価はsnapshot時点でRUNNING。候補未採用、採用判断済みbaselineの未解決169。終了、全問比較、採否、復元、cleanupはPENDING。この収録は終了や改善の証拠ではない。

## 実行方法と順序

作業はSSH先 `/mnt/nvme/kairyu/wsp/gemma4-coding-20261104` のみで行う。操作の詳細と固定hashは [実験記録](../../2026-11-08-direct-restart.md) のT002表を参照。既存trialや実行中のlauncherを再実行しない。再現する場合は、新trial・fresh候補/manifest/output・最大一時容量と上限を先に登録し、必要なpath変更は別inodeの新versionへ行う。

環境は `PYTHONDONTWRITEBYTECODE=1`、`TMPDIR=<root>/tool-validation/ci-tmp`、必要時 `PYTHONPATH=<root>/worktrees/kairyu-bench-weight-tools/src`。Pythonは依存込みの `<root>/infra/weight-tools-ci/bin/python -B` を使う。以下の引数は登録を済ませた新versionに対する使い方であり、ここにある完了/実行中T002を再起動する指示ではない。

| script | 引数・実行の前提 |
| --- | --- |
| `verify-literal-zero.py` | 引数なし。生成前の保存+0 proof。 |
| `create-and-verify.py` | 引数なし。容量・科学guard後に生成しstreaming chain検証。 |
| `verify-and-freeze.py` | 引数なし。chain exit0後、対象bytesと12file証拠からfreeze。 |
| `run-replica-operation.py` | `preview` / `apply` / `restore_preview` / `restore_apply`。単一replicaの登録argv。 |
| `audit-live-load.py` | 初回は引数なし。再起動後は `candidate-live-load-after-restart.json`。 |
| `run-diagnostic.py` | `1` / `2` / `3` を順に各1回。40 continuation/回、既存公式scorer維持。 |
| `restart-before-r3.py` | 引数なし。r1/r2終了後に同候補process再起動。 |
| `compare-fixed-three-token-runs.py` | 引数なし。3回の記録完成後のみ。 |
| `check-full-serving-preflight-v2.py` | 引数なし。4候補replica切替後の同一runtime/RO/readiness確認。 |
| `run-canonical-full.py` | 引数なし。全guard後のみ。snapshot時点で実行中。 |
| `compare-canonical-full.py` | 引数なし。正規全問終了後のみ。通信失敗・未採点を不正解と混同しない。 |
| `run-full-replica-operation.py` | `gemma-N` と `preview` / `apply` / `restore_preview` / `restore_apply`。登録plan内argvに従う。 |
| `partial-response-observation-002.py` | 引数なし。001が固定したprefixだけをcontentで訂正観測する一回限りのscript。出力が既存のため再実行不可。 |

各scriptが参照する登録JSON、helper `docs/experiments/scripts/redeploy-replica.py`、token診断wrapper、chain再構成器の固定版と証拠は、実験記録およびSSH trial内の登録・receiptに従う。sourceを読むだけなら実行やimportは不要で、AST解析を使う。scriptによってはimport時にも本処理が走るので `--help` のために起動しない。

## 固定版と証拠の扱い

hardlinkのどちら側もin-placeで編集しない。修正は別inodeの新しいversion名で作成し、revisionに旧新path/hash、理由、適用する新trial/operationを記録する。実行中の `run-canonical-full.py` とこのsnapshotは変更しない。hardlinkはファイルシステム上の書込禁止を保証しないため、この運用とhash検証で固定する。

raw評価、レビュー、manifest、activation、モデル重みはここへ複製していない。唯一の保持先は `<root>/restart-direct-20261108/trial-002-final-branch-ablation/` および登録で参照する `<root>/results/`。部分観測001の誤参照、preflight初回失敗、分類訂正の原記録も削除・上書きしていない。002は001を置換せず、元path/hashで訂正履歴を示す。

容量は元モデル＋採用中1＋候補1まで、追加weight150GiB、再生成cache100GiB、NVMe空き200GiB以上を維持する。sourceの同一inodeを二重計上しない。snapshotを理由に新候補・再評価・不要な第3の完全コピーを生成しない。
