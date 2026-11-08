# 公開回答の境界を修正した校正監査（2026-11-06）

**旧校正は、思考ONの応答開始に答案を直接置く条件を測っていた。公開finalの尤度とは区別が必要だった。** 元モデルの短い正答「0」が約−30 nats/tokenだったことを調べ、公式ソース・template・3問の実モデル数値比較で確認した。公式WSPの実行・採点結果はこの独自proxyとは別であり、そのまま保持する。能力90達成の証拠ではない。

## 境界と出力経路

pinned imageのTransformers 5.20.2では、text submodelがembedding倍率と最終RMSNormを適用し、lm_head後に30×tanh(logits/30)を適用する。追加のlogit倍率の欠落はなかった。公式full-modelはheadのBF16値のままsoftcapする一方、旧custom経路はその前にFP32へ変換していた。この精度差とframe差は別に測った。

公式templateのthinking ONはsystemにthink flagを入れる。assistant本文にreasoningが無い場合、元のprefixは次の位置で終わっていた。

~~~text
<|turn>model
~~~

system内にthink flagがあることを検査しても、公開回答の境界を検査したことにはならない。公式templateはthoughtを<channel|>で閉じ、その後へ公開本文を続ける。独立したfinal-channelトークンは使わない。thinking OFFの生成prefixでは、次の空thought区間を明示的に挿入する。

~~~text
<|channel>thought
<channel|>
~~~

新版はONのsystem flagを保持してこの空区間を強制挿入し、
frame_protocol=gemma4-public-answer-v2-forced-empty-thoughtと記録する。
**生成された思考本文を条件にした通常の公開finalとは異なるteacher-forced条件**であり、隠れたreasoningを生成・保存・参照していない。旧104caseの改行も確認し、最初の正答「0」の問題はliteralのbackslash-nによる誤入力ではなかった。

## 3問の公式full-model照合

元モデルで3つの自作caseを、新旧frameとpadding0/+8の計12条件で測定した。24forward、モデル読込み込み25.062秒。各条件内で公式full-modelとtext submodel＋head＋同じBF16 softcapのlogits差、平均logP差は**すべて0**だった。

下表はpadding0の公式logP。codeはtoken平均で、正答率ではない。

| 自作case | 答案tokens | 旧frame | 新frame | 新−旧 | 最初の正答token順位 |
|---|---:|---:|---:|---:|---|
| 空rangeの長さ「0」 | 1 | −28.132813 | −0.074154 | +28.058658 | 27,752→1 |
| 加重平均・分散 | 100 | −0.927734 | −1.045158 | −0.117424 | 194→5 |
| 同値を保持するイベント列merge | 115 | −1.229422 | −0.914502 | +0.314920 | 875→6 |

旧frameで「0」の前に来るchannel-open tokenのlogPは約−0.000000715、ほぼ確実だった。公開答案を思考開始位置へ直置きした不一致を強く示す。長いcodeは境界修正で平均が上がる例と下がる例があり、「修正したから全目的が改善した」とは扱わない。

新frameにおける旧custom FP32 softcapと公式BF16 softcapの平均logP差は最大0.003456、padding0/+8の差は最大0.009362（科学2例は最大0.007515）。小さい0.01程度の局所目標と比べても無視できるとは限らない。収集・有限測定のpadding、microbatch、dtype経路を明示して揃える。全logitsや生成本文は保存していない。

## 既存結果と次の測定の扱い

旧frameの微分・finite・6positive案は削除せず、意図した公開final proxyとしての選抜から退役した。6案は未適用である。境界だけを後から読み替えて新しい結果に見せず、修正frameで両目的の勾配を再収集する。136caseの答案token列は同一、prefixは4tokens増え、最大509tokens。transferは引き続きgradientへ渡さない。

公式WSPでB・F・Gが示した得失は有効な実行結果として残る。旧proxyの修正は、Bの不採用や既知64未改善を取り消す理由にはならない。また、新proxyの改善も生成コードの実行成功を保証しない。

[再利用診断CLIとcollectorの境界](../../scripts/experiments/gemma4-purpose/README.md#public-answer-boundary-diagnostic)、
[道具台帳](reusable-tool-inventory.md)、
[6案の退役](2026-11-06-positive-correct-donor-amendment.md#retirement-of-the-intended-proxy)、
[WSP実測](2026-11-06-code90-matched-screens.md)を参照。

## 証跡

実験artifact rootからの相対名。既存原本は変更していない。

- analysis/code90-resume-20261106/public-objective-source-audit.json
- analysis/code90-resume-20261106/public-answer-diagnostic-validation.json
- analysis/code90-resume-20261106/public-final-frame-v3/legacy-source/
- analysis/code90-resume-20261106/public-final-frame-v3/{frame-evidence,tokenizer-readiness,oracle-independent-analysis}.json
- analysis/code90-resume-20261106/science-gpu-runs/public-final-oracle-v1/report/report.json
- analysis/code90-resume-20261106/positive-correct-v2/proxy-retirement.json

独立oracle分析SHA256: 3c634ea45b6258c4ab01aeae93dd3af839270f248fda297283f0376cd77dc83a。
