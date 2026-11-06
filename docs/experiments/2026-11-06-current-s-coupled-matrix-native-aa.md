# AA：採用済みSへの疎行列・native連成方法（2026-11-06）

AAは採用済み[S](2026-11-06-current-p-native-science-step-s.md)を親に、注意出力とMLPの対応単位、nativeパラメータを一緒に扱う**1候補の事前方法**である。方法・費用・目的の式を新しいS観測前に固定した。観測値、proposal、有限応答、公式accuracyはこの記録では未確定。元モデルやPへ戻さず、旧P/Rの勾配もSの勾配として再利用しない。

Sはfresh dev90で+1.666667 ppだった一方、SciCodeは不変で、歴史P比full14にはtools／long／reasoning／multilingualの損失がある。科学teacher-forcingの局所改善だけでは不足するため、科学と一般実装をともに主目的にする。短い校正で90超や汎化を約束せず、最終目標と集合の違いは[全公式台帳](2026-11-06-all-official-scores.md)に残す。

## 固定した編集支持と観測

親state SHA1049e67a536317876716b516f6abe90b8b0770d5a011e080340a5a8b12ae561a、S tree60520ea1e63b25a69b10da1f39f63339221b1a0597c358d60a7cc6b0001d3be1、freeze a978a2cb7b36c0a4ebab3e160ca7ba729e20e0d0063b18379d98ca8413db3e92。Pの採用済みattention編集とSのnative増分を含む完全な保存Sが入力となる。

独自作成・CPU実行検証済みの136 train case／21目的／42foldを用い、correct-logprobとcorrect-minus-foil marginをnative／matrixで各1回観測する。transfer caseは勾配入力にしない。serial 1回答、元pair由来padding、全回答token平均、seed0／threads4／microbatch1を揃える。frameはnative ON templateに明示空thoughtを置いたpublic-final条件で、通常の生成思考contextとは異なる。

- 観測：native421 tensors／1,331,516座標、matrix410 tensors／209,920座標（各512）。
- 編集可能：native全421、o_proj60とMLP gate/up/down180。Q/K/Vは観測のみで編集0。
- MLP sampling：各層で同じ16 stratified hidden indexをgate/upの行とdownの列へ対応させ、32 input/output indexとの積で各512座標。o_projは既存head-aware samplingを保持し、MLPとの残差行の厳密な対応を偽称しない。
- これは3投影の対応した**疎な支持集合**であり、全MLP weightや完全なneuron/circuit編集ではない。BF16丸め後に各行列・tripleで何座標が残ったかを報告する。

収集前に追加固定した input-contract-decision.json により、native／matrix間の各case・各objectiveの絶対baseline差の許容値は既存参照と同じ1e−6をplanへ明示し、最大絶対差を必ず報告する（入力整合性の条件であり、accuracy・採用gateではなく、method-decision.jsonは変更しない）。

既存のinput activation×backward outputによる真のサンプル行列微分、明示zero-additive probe、FP32積／FP64和を再利用する。新しい数値kernel、full Fisher、全重み勾配は追加しない。旧Oの粗いlayer/role activationを、現在Sのchannel別証拠へ読み替えない。

## 事前固定した費用・目的・予算

費用は既存relative metricで、各座標について cost_i = block_scale / max(abs(current_S_i), 1e-6) とし、native／matrixのblock scaleはともに1。目的は実BF16差分の ||cost * delta||₂。これは相対パラメータ空間の費用であり、activation/function-spaceの等価性やparameterization biasの消滅を意味しない。absolute L2、実変更数、family別参加率も別に報告する。

| 目的群 | correct-LLの局所下限 | marginの局所下限 |
|---|---|---:|
| 主目的：code、implementation、workflow、algorithm、science_numerics、science_contracts、science_advanced_state | min(.01, .25 × headroom) | 0 |
| 関連：tools、tool_call、tool_no_call、tool_arguments、reasoning、lookup、referent、fact_retention、japanese | min(.005, .125 × headroom) | −.01 |
| その他 | −.01 | −.02 |

headroom = − current-S correct-answer mean log probability を目的/foldごとに用い、headroom≤.004の主／関連LL targetは0とする。式は観測前固定、数値の代入のみを新S観測後に行う。短いreferent／fact-retentionは長文能力の検証ではなく、21目的にはXSTest safety目的もない。

radiusはrelative .025／absolute1e−6、BF16補正16回を既存方式で上限固定する。候補1、CPU solve最大300秒、候補finite pass1、amplitude gridなし。丸め後の実deltaで局所予測を再計算し、実finite応答との差を報告する。予測下限は探索条件で、accuracy保証・全領域無損失・有意差の採用gateではない。

行列への実参加が消えた場合は「広い行列仮説は実現しなかった」と記録し、意味のないepsilon変更や2回目の調整で体裁を整えない。finite候補が選ばれた場合に、既存の固定された独自transfer集合でnative ON実行を1回、公式dev90のfresh親ペアを1回行う計画。勾配／proxyでの改善を実行成功と同一視しない。matrix変更はnative-faststoreの共有bulk範囲外なので、実faststore準備は保留した。

## Opus 5.5の匿名方法レビュー

公開一般方法だけの1回、匿名label A。actual assistantとmodel usageはともにclaude-opus-5-5のみ、verified／usable／labels_match true、tools0、fallbackなし、101.9826秒、$0.20388。private score／norm／座標／モデルidentity／case本文／回答／reasoningは送っていない。最初のsubagent送信は実行前に自動レビューで止まり、rootがユーザーの明示したClaude・Opus5.5指示と同じpromptを確認して唯一の実dispatchを行った。

採用した要点は、現在親・目的単位・shapeを揃えること、shared hidden indexを支持集合として扱うこと、実matrix参加と丸め消失を報告すること、実delta上の予測と通常native ON実行を区別すること。独立BF16丸めで全座標が非ゼロになる保証はなく、review文中の「atomic group」を全座標強制変更の要件へ読み替えない。

E[x²]費用、per-example G、低rank empirical-Fisher、cross-fitting、新しいfinite-difference sweep、追加empty-thought生成、thought-length instrumentationは任意助言で、この固定方法には追加していない。E[x²]だけでも下流感度・相関・gate積・norm再正規化を欠くため、厳密な共通function-space metricとは呼べない。追加approval、cosine閾値、全領域zero-lossのgateも導入しない。

## 証跡

analysis/code90-resume-20261106/からの相対パス。既存回答・問題・hidden reasoning本文は本記録に使用していない。

| Artifact | SHA256 |
|---|---|
| current-s-coupled-matrix-native-aa-v1/method-decision.json | d89548e11223f15bc2fa3d96978c47b8ec8f61724744411448fa711b5147a052 |
| coupled-matrix-native-method-prompt.txt | 7a420c8b48ffff6502287537346c6183983efd8381f8aa46d6f32697d2299f63 |
| opus-coupled-matrix-native-method-review/receipt.json | 6b774a50c60a7742da52015bde7086fb6090e1026426953bc4029dd519885f13 |
| 同method-summary.json | 0e398bab47ee65bda498368e5715bdd2af7a66b04e5058e0d0ebca7ab6e43600 |

実装・観測ownerの最終source pins、実測値、候補結果は完成後に別証跡として追記する。これは方法の固定で、能力改善の結果ではない。


2026-11-06 16:24:49 UTC、固定済みSの4段階観測ownerを起動し、native correct-logprob収集の開始を確認した。候補AAの重み編集・公式評価はまだ行っていない。plan SHA a98a5aa8125a3daecc63dcda15ad85834819f3e4fd0f80577c945c0ee2d8d954、runtime SHA 81ec55172c2be0ed67e15fd84cbd5d6e461390fed8067b2828ea462056a98a0c。起動前の4 CPUメタデータ確認、実Torch CPU13件、独立レビュー、全CPU検証510件（458成功・52 optional skip）を通過した。全CI証跡SHA e91a22fe63841405185003f2b449d4ead9705ed273eea0d1d18a27e859d40c5b。結果は未確定のまま記録する。


## Sの新規観測完了と候補計算開始

4収集すべて42/42foldが終了し、全体478.435秒（実計算263.128秒）で同じO2 CIDのhealthy復帰まで完了した。Sの9ファイル、state1049、6 source pins、S/P/R/embedding各サービスは不変。モデル重みの編集はない。

| 収集 | 実forward数 | 計算秒 | 最大reserved GiB |
|---|---:|---:|---:|
| native correct-LL |136|57.829|84.19|
| native margin |272|87.433|84.95|
| matrix correct-LL |136|45.047|79.82|
| matrix margin |272|72.819|80.65|

同じS・frame・136caseで、native/matrix間のcorrect-LL差は0、margin最大差は9.5367431640625e-7で事前固定1e-6以内。matrixのmarginはnative値をFP32へ保存した値と全caseで一致し、この保存境界を数値差として残す。以前のS有限応答との42fold NLLも完全一致した。これは観測の整合性であり、能力向上の証拠ではない。

完了summary SHA24ca2e9ec381900cc93995e4b3022eef21ca9b09e3fb664b7fb2cddab4b61621、completion7d792ce68e992cae80d0522d05f08daf81ef8a59ac248b0c6275a16f5134b5e8、postchecksa8a85283f9d9bd047010b71c8970b0cdfebc61fc0553c09f0965c32ff916e647、独立数値review4aed13610cf2d09e43a26b10ebb398c746763d189fe2c244f16589a3e1cc6496。

AAの84制約は固定式へ新Sのheadroomだけを代入し、独立再計算で一致した。plan SHA40fae9d8152e5730e765bb5a6ed91062069eb9404d5065b113c79d76e3fc05f8、独立plan review4857ab00dccc63fee4653c1e31f89e8a1437752d85f8b92cfbda1284b35fdbd8。上限300秒・1候補・再試行なしのCPU計算を開始した。AAの有限応答、保存checkpoint、公式accuracyは未実施で、Sの採用状態は変わらない。
