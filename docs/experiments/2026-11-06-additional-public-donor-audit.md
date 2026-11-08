# 追加の公開donor候補：metadata上の不採用理由（2026-11-06）

今回の改善候補として採らなかった3件を記録する。既に取得済みの公開model cardとAPI metadataだけを確認し、新しいdownload・重み適用・生成評価は行っていない。以下は各作者の宣言と今回の適用判断を分けたもので、一般にその学習方法が無効だと結論するものではない。

## fbobe3：一つのbenchmark課題への過適合

[公開card](https://huggingface.co/fbobe3/gemma-4-31b-condensed-selfdistill-lora/blob/main/README.md)は、condensed self-distillationを検証するrank32 QLoRAの研究用成果物と説明し、一つのTerminal-Bench課題に過適合したことを明記している。宣言された3/3成功は訓練した同じ課題の再現であり、未学習課題への一般化ではない。

API一覧にはadapterファイルがあるが、この成果物を一般coding改善の根拠として導入しない。今回、訓練traceや課題の解法は取得していない。学習targetを短く実行可能にするという方法論と、当該adapterの一般能力を混同しない。

- APIで観測したrevision：`345d314bfd192c8184ec11496704e456746f4293`
- [API取得URL](https://huggingface.co/api/models/fbobe3/gemma-4-31b-condensed-selfdistill-lora)
- cached API SHA256：`bb816be4452f5e9a440d2a95572e98c633e73a940ffced8c686dbe96efa9882d`
- cached card SHA256：`aaeaeb29e78949e1ac375de0df15e23918afda2a4e80943e73569e84f53fe7a5`

## eexian：訓練baseの不一致と最終精度の未公表

[公開card](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/blob/main/README.md)が指定するbaseは `google/gemma-4-31B-it-qat-w4a16-ct`、revision `52f3f65bc7a02d555763bc923bd1d9094898219d`。このW4 baseをBF16へ展開して学習し、配信は元のpacked W4にadapterを載せる設計と記載されている。今回の公式BF16元重みへ単純mergeする条件と一致しない。

cardは180訓練task / 2,951 decision rowsと別20-task validationを記載する。一方、公開されている確認はloading・logit変化・tool call等で、最終repair solve rateは未公表と明記されている。adapterの存在はAPI一覧で確認できるが、現在のbaseへそのまま使える改善候補とは扱わない。互換性検証や精度測定は未実施である。

- APIで観測したrevision：`f66802e3cefb6c7db4b8ff883e3c28c48d6d4093`
- [API取得URL](https://huggingface.co/api/models/eexian/gemma4-31b-w4-weighted-reasoning-lora)
- cached API SHA256：`148430c24118b4bf5690f381434d504352ca9aed55f56c4e59e14f4a648bc40d`
- cached card SHA256：`80a8608dd72cac14d23f0e4b89f89bd4c20e6757bc3de5cbc7c1d18a964bea14`

## Fable：第三者FP8配布と比較条件の不一致

確認できたのは [vwdubbの第三者FP8配布card](https://huggingface.co/vwdubb/gemma4-31B-Fable-5-Distilled-FP8/blob/main/README.md)。API metadataは元repoを `autotrust/gemma4-31B-Fable-5-Distilled` とするquantized modelを示し、ファイル一覧はfull-model shard等で、独立adapterは載っていない。元repoのdownload可能なadapterは今回の保存metadataから確認できていない。「存在しない」とは断定しない。

転載されたcardはHumanEval 152/164（92.7%）を主張するが、表には編集モデルthinking OFF、引用する元モデルthinking ONという条件差がある。温度や配信経路も記載されており、同一条件のpaired改善ではない。card自身も学習時thinking ONと評価時OFFの違いを記す。今回のnative thinking ON・4 source公式code >90目標の達成証拠として採用しない。主張する上位30層へのLoRAやvision維持も、今回独立検証した事実ではない。

- APIで観測した第三者repo revision：`0925edf17759796eef53e211f28bdc2c18dc0fd4`
- [第三者API取得URL](https://huggingface.co/api/models/vwdubb/gemma4-31B-Fable-5-Distilled-FP8)
- [原repo API照会先](https://huggingface.co/api/models/autotrust/gemma4-31B-Fable-5-Distilled?blobs=true)：ownerはHTTPErrorを観測したが保存receiptはない。HTTP statusや原因・恒久的な不在は主張せず、本監査で再照会していない。
- cached第三者API SHA256：`594efccfab25f7209e202f3c4112720be4317c395ae90d39dc49bce96ce0b1b1`
- cached第三者card SHA256：`84dc8fd3a6972396ce638a13be0a6a49c93e0c96a44b294f8060b85be4cf38a0`

## 保存範囲

cacheは実験artifactの `additional-public-donor-metadata/` にあり、各 `<owner>--<repo>--receipt.json` がAPI / READMEの取得URL・SHA256・byte数を保持する。READMEはmainへの取得、revisionは別のAPI応答から得たため、両者を原子的なcommit固定downloadと扱わない。

この監査から新しい候補specやcheckpointは作成していない。既存Tsunamayo7 donorの実WSP履歴と一様係数の未測定範囲は、[JOINTとdonor監査](2026-11-06-joint-rounding-and-donor-audit.md)を参照する。
