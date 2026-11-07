# 全16 helper stage の offline adapter 実装（2026-11-07）

[helper_stage_offline.py](../../scripts/experiments/gemma4-purpose/helper_stage_offline.py) を追加した。元の全72 case/response SHAと公開回答schemaを検査し、既存sourceの全16 rowと完全照合する。残る56は明示的な未対象。各stageを既存short/long scoreとDocker.runへ渡し、生成失敗・欠測はunknownのまま保存する。元promptの固定helper/supportだけを使用し、生成helperへの差替え・追加生成・新oracleはない。48件の既存採点・分母とは別の追加診断であり、今回は実行していない。[使用方法と契約の限界](../wsp-helper-stage-offline.md)、tool inventoryも更新した。依存は既存科学PythonのNumPy/SciPy/SymPyで、新installなし。

公開response検証は science_sandbox.read_inputs から `read_public_answers` へ抽出して共有した。既存5 catalogの検証条件は維持。commit4e088df比で WORKER文字列、Docker/prepare/compare/extract/bounded_command/main ASTが同一、既存helper/long/basic/advanced/libraryの5oracle sourceは全bytes一致した。[不変確認](records/2026-11-07-helper16-offline-validation/closure.json)。

科学CPUの focused10 testsは全てPASS、9.021秒。新規3 regression methodsはidentity/SHA/未知ID/重複/思考field拒否、Dockerを作らないpreview、HTTP失敗全16をunknownで保持、全16著者correctと全16著者foilのE2Eを検証した（各4 trials、正答16合格・foil16失敗）。E2Eは既存trusted_runの厳密なsource-owned allowlistだけを使用し、adapterの実行経路にはtrusted_runを含めていない。既存sandbox regressionでは固定されたsource-owned診断fixtureも実行した。モデル生成codeや外部任意codeのhost実行、実Docker実行は0。[結果log/receipt](records/2026-11-07-helper16-offline-validation/focused-v2.log)。CI用CPU環境でも2 tests PASS、SciPy E2E1 skip。対象Ruffとdiff checkもPASS。

初回検証は新テストのpayload key assertが既存 `allow_underflow=True` を省いたため、16 subtests全てがworker呼出し前に停止した。oracleやadapterを変えず、テストの期待keysだけ修正した。[失敗記録](records/2026-11-07-helper16-offline-validation/failure-v1.json)と修正前テスト全文を保持している。これはモデル失敗・採点失敗ではない。

完了済み通常生成v2の全72 response（SHA `9803f7a2fd8f331c74805b7dd8cf5111e9efbfa4c15415bb10988edf50c54876`）をそのまま入力したCPU previewも成功した。全16 target、56未対象、結果配列は空で、モデル/API/Docker/生成code実行なし。[正確なpreview commandとreceipt](records/2026-11-07-helper16-offline-validation/actual-v2-preview-receipt.json)。実採点には別の固定planとroot明示指示を待つ。

3件の既知dtype契約疑義と残り13件のstratumを全16raw結果に併記する実装である。既存float64 samplerはその反例域を網羅せず、helper再利用・再定義禁止など全契約の保証もしていない。未知・失敗・未対象を削って得点化しない。[source識別と検証概要](records/2026-11-07-helper16-offline-validation/implementation-summary.json)、[全証拠SHA](records/2026-11-07-helper16-offline-validation/SHA256SUMS)。audit_resultsの独立read-only reviewは追加blockerなし。記録保存時のSSH宛先誤記は自動承認レビューが実行前に拒否し、正しい承認済みhostへ訂正した同操作だけを再実行した。遠隔の誤宛先操作や権限拡張はない。
