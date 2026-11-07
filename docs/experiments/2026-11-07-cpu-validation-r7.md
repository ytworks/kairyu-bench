# CPU validation r7（2026-11-07）

**552 collected = 495 passed + 57 optional-dependency skips、failure/error0。**
unittest報告50.237秒、実wall50.753秒。
Ruff、mypy（74 source files）、data consistency、31 shellの構文確認もすべてexit0。
全担当のsource/test最終freeze通知後に1回だけ実施した。再実行はしていない。

| Check | 結果 | wall秒 |
|---|---|---:|
| unittest discover | 495 pass / 57 skip / 0 fail | 50.753 |
| Ruff | PASS | 0.036 |
| mypy src | PASS、74 files | 0.251 |
| model-comparison data --check | PASS | 0.057 |
| sh -n | 31/31 PASS | 個別exitをreceiptへ保存 |

全体51.212秒。実行前後の対象Python/shell/TOML/CI workflowの全hash mapは一致。
今回のbehavior18 case/adapter、共通空配列修正、新generation/GPU ownerと
既存テスト拡張を含む。これはCPUコード検証で、GPU/model能力の検証ではない。

`TMPDIR=R/tmp-ci-r3`、既存`infra/weight-tools-ci/bin`をPATH先頭とし、
そのPythonを`KAIRYU_BENCH_PYTHON`へ指定した。HOMEは変更していない。
R=`/mnt/nvme/kairyu/wsp/gemma4-coding-20261104`。
モデルAPI、GPU、サービス操作は0。通常生成/採点の結果とは別に扱う。

[実行原文](records/2026-11-07-cpu-validation-r7/run.py.txt)、
[receipt](records/2026-11-07-cpu-validation-r7/receipt.json)、
[summary](records/2026-11-07-cpu-validation-r7/summary.json)、
[圧縮のbyte検証](records/2026-11-07-cpu-validation-r7/compression.json)、
[SHA一覧](records/2026-11-07-cpu-validation-r7/SHA256SUMS)を保存。
unittest、Ruff、mypy、data、shellの**全5ログ**をlossless gzipでPRへ含め、
各々decompressした全bytesが元logと一致することを確認した。
空logも省略しない。skipや失敗の行の選別はしていない。

元の未圧縮logは`analysis/fresh-purpose-20261107-cpu-validation-r7`へ保持。
PRの`*.log.gz`は通常add可能。未圧縮`*.log`を追加する場合はgitignoreのため
`git add -f`が必要だが、同じ全bytesをgzipで既に保存しており強制追加は不要。

後続のGPU計測ownerにはlate-create cleanup raceの局所修正が追加された。r7で検証したownerはc70af9c1…、その後のowner9a8d9255…/testa26f75bd…は別の13 focused testsとRuffで検証されており、このr7の全suite再実行とは称しない。r7の原receipt/log/source mapは不変。
