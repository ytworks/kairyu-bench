# K: 保存済み native B と coherent H の byte 合成計画（2026-11-06）

K は、同じ元モデルから作った B の native 変更と H の係数1/16 LoRA 変更を、保存済み BF16 のまま一つに集める固定候補である。追加学習・係数探索・GEMM・再丸めは行わない。**本節は作成前の入力監査と評価計画であり、K の保存完了、公式成績、採用、code90達成はまだ示さない。**

[J の公式dev90](2026-11-06-weighted-reasoning-c1-review.md) はfresh original比 +3.055556 ppだったが、95% CIは0を跨ぎ、既存B/H/Iを全て上回る固有成功は0だった。次はJを足すのではなく、既存BとHの成功を単一モデルに残せるかを検証する。異なる時刻の単体利得を足し算せず、B/H/K/originalを同時に測る。

## 固定した保存重みと変更量

元treeは cab440f871a32b5ea1c12364db7dd35ac8768a83b413ca24bbc18d6478eccbd9。
入力監査は元／B／Hの各12ファイルの記録と現在stat、元file hashの記録一致、
全571選択tensorのheader・shape・dtype・byte spanと非重複を確認した。
この事前監査自体は新しい全payload hashではない。実合成では後述のbytes検証を行う。

| 成分 | 変更tensor数 | 保存後の変更要素数 | 保存Δ L2 |
|---|---:|---:|---:|
| B: science-native-contracts-v1 | 161 | 49,053 | 0.003735683695257374 |
| H: coherent donor 1/16 | 410 | 4,328,720,639 | 1.888537495617679 |
| Kの非重複合成期待値 | 571 | 4,328,769,692 | 1.888541190360053 |

Bはlayer scalar 2、pre-feedforward norm 29、post-attention norm 58、
post-feedforward norm 56、K norm 5、input norm 11 tensor。
Hは既に元BF16へ1回mergeしたtext linear 410 tensorであり、Bとの重なりはない。
全tensorをコピーするが、上表の変更要素は元bytesと異なる値だけを数えた既存component実測である。

非重複なら変更数は加算、Δ L2は二乗和の平方根になる。
これは独立した数値再計算ではなく、hash固定したcomponent実測とbyte同一性に基づく継承である。
少ないnative要素の機能効果が小さいとは限らず、normとlinearが同じ計算経路で相互作用し得る。
Bの旧代理指標を新frameの学習証拠へ読み替えず、Bの既存公式実測と
[広い領域の損失](2026-11-06-code90-matched-screens.md)も保持する。

| 固定対象 | SHA256 |
|---|---|
| B tree | 564ae5fc1efdedd77180a91764308ccea222311e19c5e0a8c7a2262069a8138b |
| B manifest | aaf7ca16b9bf6df1f5052fac9b3e72db303d7cd069c42d8776ed9b20bb410f16 |
| B spec | 05fd80d19cfc56467e172e0f7bf14a74dae5f562d80b09228eca6b72204d7128 |
| H tree | 0f9775ba6a2406989407d86d365eed69e1684a4400faddf56bb3822fad6b98d3 |
| H manifest | b1eb4249f07baebb4796ec12e8d1097ea50b330f46a9e3128c0279dc801fc6fd |
| H spec | a72aecba942de401d814c06162065ea822c44268f9cacf5b76b5501a7f42f409 |
| H CPU recipe freeze | 8941b3271db0d2593871f114a53759d3ad4741741b10f0fd8d089bca6af37a8b |
| b-plus-h-composition-input-audit.json | 9b7c816bd012704c21d012807b0d6e14fe6ed15c1893f8daf120b0bcc6cb2ddc |

artifact名は analysis/code90-resume-20261106/ からの相対名である。
Bには独立freezeが無いため、合成時はfresh full-file hashを用いる。
HはCPU再生成済みfreezeとexact stat継続を照合し、最初のprivate copyでも全file hashを計算する。

## 合成の検証契約

再利用する [composer](../../scripts/experiments/compose-verified-checkpoints.py) は
明示planとSHA、同じ元tree、非重複full-tensor spans、同じlayout/configを要求する。
Hから独立ファイルを作り、Bの保存済みtensor bytesを指定spanへコピーする。
入力は変更せず、outputのsymlink・alias・上書きを拒否する。

output全fileと各選択tensorのhashを計算し、次の三つの仮想差戻しを検証する。

1. B/Hの全spanを元bytesに戻すとoriginalの全file hashになる。
2. Bのspanだけを戻すとHの全file hashになる。
3. Hのspanだけを戻すとBの全file hashになる。

既存のKと元bytesの同じstreamを使ってhashersを更新するため、この三条件のための追加payload読取やGEMMは不要である。
全差戻しはKの未編集領域を固定し、部分差戻しは各componentの宣言外変更も検出する。
たとえばBの宣言外変更をコピーしなくても、三番目と実Bのfull-file hashが不一致になる。
SHAの衝突耐性と入力の前後stat/hash継続を前提としたbytesの保証であり、関数出力の保証ではない。

manifest/freezeは排他的に公開し、失敗時はこの呼出しが所有する未公開出力だけを掃除する。
既存redeployが読めるfreezeへ実施したhash方式を記録する。
[実装と公開境界](../wsp-checkpoint-composition.md)を参照。
実Kでの証跡は作成後に別途記録し、CPU fixtureの成功と混同しない。

## Opus 5.5 の匿名方法レビュー

actual assistantとusageはともに claude-opus-5-5 のみ、
verified_requested_model=true、review_usable=true、tools 0、60.889秒、$0.116412。
Kをキーとするpublic final JSONは通常のlist schemaと異なり labels_match=false だが、
単一Kのpublic-final-textとして受領した。raw stream・thinking blocks・環境は保存していない。
匿名の公開一般機構だけの意見であり、非公開の候補測定や能力を査読したものではない。

固定済み保存成分のbyte合成は、余分な丸めを避けて累積仮説を直接調べる方法として妥当との意見だった。
採用するのは入力／配信同一性、実span非重複、三つの差戻し、同時4条件比較、
課題別得失と単回測定の不確実性の区別である。

入力normがprojectionの入力を変え、K normがattentionを変え、
post-normが残差の寄与を変えるため、parameterが非重複でも機能は線形加算ではない。
出力長・cap・形式の変化、同じ成功への冗長な寄与も確認する。
レビューが挙げたfinal normの一般論を、今回Bがfinal normを変更したという事実にはしない。

追加logit probe、独立した全重みnorm走査、係数探索、反復や新しい損失閾値は
このレビューによって必須化しない。ユーザーのcoding優先・他領域の多少の損失許容を維持する。
成功集合のunionを超える新成功は必要条件ではない。両単体の成功を一つに残して
best soloを上回れば正当な累積改善であり、同じ成功は二重計上しない。

receipt opus-native-coherent-splice-k-review/receipt.json のSHAは
efad51b13b2fe8d302358899f76652480a5867c0785e539e30892b898436b4d6、
匿名prompt SHAは b3bf4d50ded444b2cc3d00ccb37632f5ff177f6f5b37fb413d0b6511d5067978。

## 予定する4条件の公式比較と数え方

保存検証と既存authored smokeの構造確認後、fresh B、H、K、originalを
4つのdirect endpointで同時に実行する。各endpointはTP2、chat concurrency32。
native thinking ON、公式sampling/template/tokenizer/caps/scorer、固定bench source
5e3f2df0c7f455013af3fa2d7abb3719858f808f を揃える。
source選択はHE30／MBPP30／LCB20／SciCode10の同じ90項目、
既存manifest SHA a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。
run IDs、実際のloaded tree、container/image/config/layer列、開始時刻は実行証跡で固定する。

主指標は公式codeの4-source等重み平均である。K対B、K対H、K対originalを個別に
公式paired比較し、成績の良い単体との関係も明示する。過去単体や元モデルは揺らぎの文脈に限る。
B/Hの双方成功・Bのみ・Hのみ・双方未成功のセル別にKの維持／獲得／喪失を示し、
SciCodeのpartial scoreはそのまま別記する。binary thresholdのflipとfractional変化を混同しない。
cap/timeout/unknown、sampleのfinish reason、使用量、wall timeも併記する。
固定seedでもc32の出力同一性は保証されず、単回のdev再利用は未見確認ではない。
このdev90の90超も、既に成績を見たknown64の目標達成と同一視しない。

開始前／終了後に [serving counters](../wsp-serving-counters.md) を採り、
process identity・reset・missing・running/waitingを保ったまま保存sampleと照合する。
現在の公式sourceは、run作成前にmax_tokens=1のchat discovery probeを1回送る。
全39 SciCode stepまで通常完了する場合、保存leaf callsは119、probeを含むprocess終了は120となり得る。
probeのlength終了をbenchmarkのcap失敗へ足さず、実際の各run recordsとcounter差分で確認する。
単なるcounter差分からHTTP retry不在・他client不在・課題帰属・思考token数を断定しない。
startup probeの根拠は[Jのsource監査](2026-11-06-weighted-reasoning-c1-review.md#採点分離と追加process-eventのsource監査)に残した。

## CPU統合検証と固定plan

新しい合成工具を含む全CPU検証は495件中443成功、任意依存52skip、失敗0。
Ruff全treeとmypy 73 source filesはPASS、実行中のsource変更0だった。
6件の追加テストはbyte合成→配信freezeの契約、入力破損・未申告component差分、
出力競合と失敗時cleanupを保護する。追加の模型／GPU callsは0。
CI receipt cpu-integration-checkpoint-composition-v1/ci-results.json SHAは
536bfa458f158e285ceb33e1341c347a4549fca78b30dda1673e30d2f3d16438。

実作成用planは native-b-plus-coherent-h-v1/plan.json、SHA
085cd6df19b8d410e8278cbb86ece932099bf3f182f82dd0de59ded932b5e6c0。
同フォルダのpreview.jsonは571 tensors／4,328,769,692 changes、
62,578,686,256 bytesの独立コピーと300 GiB予約条件を確認した。
実作成・全hash検証を開始しており、その完了と配信・得点は後続の証跡で報告する。
