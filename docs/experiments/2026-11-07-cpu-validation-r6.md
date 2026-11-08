# CPU validation r6（2026-11-07）

**532 collected = 475 passed + 57 optional-dependency skips、失敗0。** unittest報告47.444秒、実wall47.963秒。Ruff、mypy（74 source files）、data consistency、31 shell filesのsyntax確認も全てexit0だった。6つの対象source/test SHAは実行前後一致し、この監査時点のrepo内容とも一致した。再実行は行っていない。

| Check | 結果 | wall秒 |
|---|---|---:|
| unittest discover | 475 pass / 57 skip / 0 fail | 47.963 |
| Ruff | PASS | 0.051 |
| mypy src | PASS、74 files | 0.348 |
| model-comparison data --check | PASS | 0.064 |
| shell syntax | 31/31 PASS | 個別exitをreceiptに保存 |

元結果は `analysis/fresh-purpose-20261107-cpu-validation-r6`。[元receipt](records/2026-11-07-cpu-validation-r6/receipt.json)、[検算summary](records/2026-11-07-cpu-validation-r6/summary.json)、[全unittest logのlossless gzip](records/2026-11-07-cpu-validation-r6/unittest.log.gz)、[全SHA](records/2026-11-07-cpu-validation-r6/SHA256SUMS)を同PRに保存した。圧縮前96,796 bytesのログを全bytes復元できることも確認し、skipや失敗の行を選別していない。元の未圧縮logもNVMeに保持する。

初回の報告用assertは同一行にあるokだけを数え、別行に出力された12件を見落として停止した。[集計失敗記録](records/2026-11-07-cpu-validation-r6/reporting-audit-failure-v1.json)。463同一行＋12単独行=475と最終unittest集計を照合して訂正した。これはCIテストの失敗ではなく、CIの再実行や元logの変更はしていない。

これはCPUコード検証であり、GPUやモデル能力の検証ではない。追加helperの科学CPU E2Eと実隔離数値診断、通常生成や公式WSPの得点はそれぞれ別の記録・条件で保持する。過去の検証失敗・修正履歴も変更していない。
