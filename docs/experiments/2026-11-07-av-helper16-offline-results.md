# AV の追加全16 helper stage 数値診断（2026-11-07）

**16/16合格、失敗0、unknown0。各8 trials、合計128/128合格。** 15.449秒、固定条件で1回だけ実行した。既知dtype契約疑義3件は3/3、その他13件は13/13。これは公開promptに含まれる固定helper/supportを与えた既存有限数値fixtureの結果であり、dtype反例の解消や公開契約全域の保証ではない。既存48件の採点と公式WSPとは別の診断として保持する。

入力は完了済み通常生成v2の全72 responseそのまま（SHA `9803f7a2fd8f331c74805b7dd8cf5111e9efbfa4c15415bb10988edf50c54876`）。case SHA `30606fd3b14b4c7809a488fda2bc7bdf318fead9543228b50e7617b0754ca7be` の書換え・回答抽出・追加生成はない。全16 rowを既存sourceと完全照合し、全56 not-targeted IDも [raw report](records/2026-11-07-helper16-offline-results/report.json) に残した。56を失敗と数えたり、16/72・64/120の全能力得点を作ったりしない。

[固定plan](records/2026-11-07-helper16-offline-results/plan.json) SHA `dfb8263dbf2a1ea501831464b4d16e1deef98134ec58ce9ef90138da9c4d0d61`。adapter・helper/long・sandbox・generator・basicの6sourceをNVMeのread-only snapshotへ保存し、preview後にrootの明示指示で1回実行した。source全文も [freeze](records/2026-11-07-helper16-offline-results/source-freeze/helper_stage_offline.py.txt) としてPRへ保存した。

条件はseed20261107、8 trials、15秒/case、512 MiB、CPU1。既存科学Pythonとc4ab5022… image、daemon3e518d2b…、NVMe DockerRootDirを固定。既存Docker.runのnetwork none・uid1003・cap-drop ALL・read-only・no host mount・no GPUを使用した。期待値はhostで計算・比較し、生成codeは隔離workerだけで実行した。モデル/API呼出0、serving変更0、再試行0。実際の起動argvは [child-start](records/2026-11-07-helper16-offline-results/child-start.json)。

| ID | 合格trials | dtype契約疑義 |
|---|---:|---|
| chain-whitened-fit-1 | 8/8 | あり・未解消 |
| chain-whitened-fit-2 | 8/8 | 既存3反例には非該当 |
| chain-boundary-response-1 | 8/8 | あり・未解消 |
| chain-boundary-response-2 | 8/8 | 既存3反例には非該当 |
| chain-quadrature-mixture-1 | 8/8 | あり・未解消 |
| chain-quadrature-mixture-2 | 8/8 | 既存3反例には非該当 |
| chain-log-mixture-score-1 | 8/8 | 既存3反例には非該当 |
| chain-log-mixture-score-2 | 8/8 | 既存3反例には非該当 |
| long-weighted-constraint-1 | 8/8 | 既存3反例には非該当 |
| long-weighted-constraint-2 | 8/8 | 既存3反例には非該当 |
| long-conditional-covariance-1 | 8/8 | 既存3反例には非該当 |
| long-conditional-covariance-2 | 8/8 | 既存3反例には非該当 |
| long-affine-state-history-1 | 8/8 | 既存3反例には非該当 |
| long-affine-state-history-2 | 8/8 | 既存3反例には非該当 |
| long-affine-mixture-moments-1 | 8/8 | 既存3反例には非該当 |
| long-affine-mixture-moments-2 | 8/8 | 既存3反例には非該当 |

全source/input SHAとstatは前後一致。owned science container残存0。保護27containerのID/name/image/running/status/healthは前後一致し、daemon/image/Pythonの事後読取りも一致した。[closure](records/2026-11-07-helper16-offline-results/closure.json)、[事後runtime](records/2026-11-07-helper16-offline-results/post-runtime-readonly.json)。失敗・タイムアウト・欠測は今回0であり、全件のraw試行情報と空の実行logも保存している。

検算用 [集計](records/2026-11-07-helper16-offline-results/results-summary.json)、[全artifact SHA](records/2026-11-07-helper16-offline-results/SHA256SUMS)。report SHA `eb1c46234cad2a60258c4d36ed6e2ff48325cb6820d674b5f1d6908a47d46b05`。実装・初回テスト失敗・正答/foil検証・独立reviewは [実装記録](2026-11-07-helper16-offline-implementation.md) に分けて保持した。
