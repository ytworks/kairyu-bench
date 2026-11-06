# L: K1 の保存text因子を係数1で移す固定試行（2026-11-06）

Lは、公開K1 adapterのtext因子だけをbyte同一で取り出し、exported coefficient 1で元Gemma 4 31B BF16へ移す一つの固定試行である。**全499件CPU CI、実text因子抽出、固定specのCPU作成・独立CPU検証を完了した。実BF16変更は410 tensor／23,862,088,988値、Δ L2は42.46527094830241。保存重みは検証済みだが、authored smokeは21/24でhistorical originalの23/24を下回った。公式dev90ペアも完了し、Lは67.182540、fresh original比−3.517857 pp（95% CI −11.875000〜+4.107143）だった。code90未達・不採用とし、別基点との組合せ仮説はMで分離して検討する。** K（B＋H byte合成）とは別の候補である。

## 公開由来と基点の限界

対象は [win10/gemma-4-31B-K1](https://huggingface.co/win10/gemma-4-31B-K1/tree/712fa2808d60465fa0bde72a69fc6affcb018ffc)、
固定revision 712fa2808d60465fa0bde72a69fc6affcb018ffc。
cardはSFTと、iannicity/KIMI-K2.5-1000000x、iannicity/Hunter-Alpha-SFTの利用を述べる。
**これはdataset名であり、実際に100万件を学習したという意味ではない。**
利用件数、filter、split、重複、学習長、native thinking設定、coding成績はcardに記載されていない。
学習答案やdataset本体は取得・参照していない。

configの基点は coder3101/gemma-4-31B-it-heretic、revisionはnull。
改変基点が元BF16と同じweightsであるとは確認できず、今回の処理はその学習の再現ではなく
**未知revisionの別基点からのdelta移植**になる。
headerでnames/shapesが合うことは基点のbyte同一性や意味的互換性を示さない。
著者がcoding/native ONの改善を実証したとは述べない。

## 保存因子の観測と固定係数

全payloadのhashとheaderから、1,198 factors（599 A/B pairs）が全てF32であり、
text410組とvision189組に分かれ、余剰の未知factorは0と確認した。
通常LoRAのr32 / alpha32 / scale=alpha/r=1、dropout0。
rsLoRA、DoRA、QALoRA、rank/alpha pattern、modules_to_saveなどの別形式は無い。

| 観測項目 | 値 |
|---|---:|
| 公開payload全体 | 1,068,039,288 B |
| 保持text factors / matrices | 820 / 410 |
| 明示除外vision factors / matrices | 378 / 189 |
| 保持textのF32 data bytes（headerを除く） | 979,435,520 |
| 係数1のtext連続Δ L2 | 42.25605854013242 |
| H原donorの連続unit方向とのcosine | 0.0030925902179596535 |
| I連続方向とのcosine | 0.00003641384013645106 |
| J連続方向とのcosine | 0.007613069059393288 |
| MLPの連続Δ norm²比 | 80.71169478% |
| 上位10層の連続Δ norm²比 | 32.90107916% |

方向観測は既存FP64 factor-Gram observerによるtext410のみで、
vision378因子の数値は調べていない。H比較のnormは原donorの係数1であり、
保存済みH（係数1/16）のΔ L2と同一ではない。
小さいcosineは能力の独立性・相補性の証明ではなく、
大きいnormも単独の採用／棄却理由にしない。

係数1は学習済みのexported operating pointをそのまま最初の仮説にする選択である。
元BF16で最適な係数とは主張せず、保守的係数とのgrid比較や目的別の係数再fitを追加しない。
実BF16では要素の丸め消失や変位の分布が変わるため、
保存後の実変更数・Δ L2・対象projection内の変更率は末尾に別途記録した。
保存／連続norm比は方向cosine・保持率・丸め誤差normではない。

## lossless text選択の契約

専用 [extract_text_factors.py](../../scripts/experiments/gemma4_lora/extract_text_factors.py) は、公開payload全factorを明示selected/excluded listsへ完全分割する。
text A/B pairsは元checkpointの全60層410 projectionへ一意に対応させ、
除外factorも認識済みvision projectionのA/B pairsとshapeを照合する。
片側欠落、未知target、重複、未宣言drop、unsupported LoRAを拒否する。

保持するF32値はfloatへ変換せず、raw bytesを8MiB以下のchunkで新しいsafetensorsへコピーする。
各factorの元／派生SHAを照合し、adapter_config.jsonもbyte同一で保持する。
configのHeretic training-base記述を元モデル名へ書き換えない。
派生payloadは新しいfull/header SHAを持ち、元repository payloadと区別する。
新しいderived SHA、元payload/header/config pins、選択・除外全factor、source stat、
実装SHAをreceiptへ記録する。

元checkpointはheader/indexによるlayout確認のみで、
full_weight_identity_verified=falseを記録する。
抽出時のheader互換を学習基点の完全なweights証明へ読み替えない。
数値の有限性やmergeは後続の既存dense recipeが担当する。
抽出そのものはNaNの判定や浮動小数点演算、BF16丸め、model callを行わない。

no-write previewは実K1で820保持／378除外、F32 r32、text410、
979,435,520 data bytesと元payload full SHAを確認した。出力作成0、model call0。
stagingの4 CPU契約tests、Ruff、mypyと独立read-only reviewを経て、K4条件の起動後にPRへ移した。
全CPU suiteは **499件（447 pass / 52 optional skip、失敗・error 0）**、Ruff全tree、mypy73 filesがPASS。
検証前後source変更は0、CI中の実モデル／benchmark requestも0だった。
[抽出契約](../wsp-text-adapter-extraction.md)はheader互換とfull weight identityの違いを明示している。

Lの後続作成は既存factor-format v2で、原BF16＋trained scale×B@Aを一度だけBF16へ丸める。
元／候補のfull-file hashesと非対象spanを既存CPU verifyで確認する。
text-only selectionは元のmultimodal adapterを再現したという主張ではない。
元vision weightsを維持しても、変更text stackが関わる画像性能まで保持したとは言えない。

## Opus 5.5 の方法レビュー

公開一般機構だけを匿名化し、既存CLIで1回依頼した。
actual assistant/model usageはclaude-opus-5-5のみ、label L一致、
verified_requested_model=true、review_usable=true、tools0。
57.4597秒、$0.115812。raw stream・thinking blocks・環境は保存していない。
非公開成績、物理測定値、座標、実験IDを送信していない。

単一のexported-unit text-only試行はcross-base transferとして妥当との意見。
普通LoRA scale、完全なfactor分類、byte保持、元baseから一度の丸め、
配信同一性、fresh originalとの公式実行比較を重視した。
Heretic基点で学んだ補償、text/vision共有経路、丸め消失、native ONの長さや形式は留保する。

助言を自動的な新要件にはしない。

- exportにtext/vision因子があるだけでは、両方に意味のあるjoint trainingがあったとは証明しない。
- 保持pair数と実変更tensor数は、全量丸めno-opなら異なり得るため別記する。
- 振幅grid、全donor-base取得、反復seed、perplexityや別precision mergeは初回の必須条件にしない。
- thought/visible token分離、独立HTTP retryは未計測ならunknownのままにする。
- capや抽出形式の改善による公式実行成功もユーザー目的上は利得になり得る。新しい推論能力の証明とは区別する。
- 将来の組合せはbest soloを上回る累積を測り、成功unionを超える新規成功を必須としない。
- 新しいユーザー承認待ち、全領域非退行gate、根拠のないnorm/cosine cutoffを設けない。

## 固定評価の範囲

native thinking ONと元モデルの公式sampling、tokenizer/template、caps、scorerを保つ。
既存authored advanced24の構造smokeは破損検知に限定し、回答を見て係数を調整しない。
historical originalの23/24はfresh causal比較ではない。

公式dev90はfresh originalと同条件で測り、4-source code平均、source別得失、
paired uncertainty、partial SciCode、cap/unknown、wall time、使用量を個別に記録する。
保存call、process counter、独立HTTP telemetry、思考tokenを混ぜない。
現在のCLIには保存sample外のmax_tokens=1 discovery probeがあり、
process上の追加length eventをbenchmark失敗と数えない。
dev90は繰り返し利用した開発集合で、既知64のcode90超という目標の代用ではない。
他の関連能力の損失を報告し、coding優先・多少の他分野損失を許容する方針を維持する。

## 証拠（analysis/code90-resume-20261106からの相対名）

| artifact | SHA256 |
|---|---|
| public-k1-donor-audit/physical-audit-summary.json | ee4f799c1ea894c756e2893b67695d9d35f8b18110f565aaf39783bc246e0407 |
| 公開adapter payload | 45d7c9933c1286595072f671978753109abd2d4c24b119661db256ae39c1539f |
| adapter raw header | e9166e8f37e1b7b4ab06569f56af1b8ad58534625f8d5f933af884c1831ee66b |
| 公開adapter_config.json | e5a0f995c9e0fc5689ad9ad526a30e307d9162c614e89666f0571f77dff0b981 |
| 公開README.md | 37ca40a3381eec16dfe5b89a1393abcd4dd25f57d83984d773c82a673b5f773b |
| k1-text-factor-extraction-v1/plan.json | b49fe8ddbd2aa595e9b6de205a573cc863750fa3631b1330d4329db87b593b1d |
| k1-text-factor-extraction-v1/preview.json | 3b34c431a585aaee1196cb1b0c7edbbef7506e66494b67f124ef3045d493ebfc |
| opus-k1-exported-text-c1-review/receipt.json | 909992e857fa4382b9b1faeeee83d582096891c9dc523ad473677b8d15e849f8 |
| 同independent-summary.json | 7b9a15b8bc2183bfe0070ddb61121a91c984bc3ddd81f8dac21d0829df2e4736 |

派生payloadの実抽出後SHAと固定specは次節に記録した。
保存checkpoint作成・verifyの完了値は末尾に追記した。配信・smokeの評価と公式scoreはそれぞれ別の証跡・判定で記録する。
この文書統合では既存Python/testsや稼働中のbenchmarkを変更していない。


## 完了した実抽出とCPU recipe preview

元repository payloadから820 text factorsだけを実抽出し、
全820 source/derived raw-factor SHAの一致と378 vision factorsの明示除外を確認した。
派生adapterは **979,558,760 B**（header込み）、data部979,435,520 B。
F32値の変換・GEMM・丸めは0。configは元公開configとbyte同一で、
Heretic基点の宣言もそのまま残している。
これは公開repositoryの元payloadとは別の派生artifactである。

| 確定artifact | SHA256 |
|---|---|
| 派生adapter full | f33772cab59b931e5ade830686157789fbe6b9163f3ce81539b70ee00b55cb78 |
| 派生adapter raw header | 5e72e0b621922fe78f710abb54395961e3b7a3383a5bf6f60703e41763c4abb2 |
| 派生config（元と同一） | e5a0f995c9e0fc5689ad9ad526a30e307d9162c614e89666f0571f77dff0b981 |
| k1-text-factor-extraction-v1/derived-text-only/extraction.json | 148cc69adb8fa6d4c1f9ea93b689aca22d6740cbc2b7561067fc2e91ed42a484 |
| cpu-integration-text-factor-extraction-v1/ci-results.json | dfa1f432794ecdfa1105b6ff8cbc62519e2394ed47098527acc7c928bed46d34 |
| k1-exported-text-c1-v1/spec.json | 83aa8544b622b1a77145f47eb3affc66a99ff7095af5db19af6a2a72919230a9 |
| k1-exported-text-c1-v1/launch-plan.json | 5569d02b9bc716932a05c15315c54ad71d49ce45bfdc41a4c5ad3c61944e424d |
| k1-exported-text-c1-v1/cpu-preview.json | ee7de21cf138d2002d9684a4ff55c61fdd89d7469c1ec88c692f75e1d3a3b46c |

固定specはcoefficient1、F32 r32、alpha32/scale1、scope native_text410。
元公開payloadと派生payloadを別のprovenance欄へ保存し、
元full/header/configと派生full/header/extraction receiptを混同しない。
previewでの保持410 pairsはmapping数であり、それだけでは実BF16変更tensor数・変更要素数を表さない。後続の実作成値は末尾の表で区別する。
元treeはcab440f871a32b5ea1c12364db7dd35ac8768a83b413ca24bbc18d6478eccbd9。

既存dense factor-format v2のCPU previewは **1.6878秒** で成功した。
410対象、62,578,686,256 Bのcheckpointコピーと300 GiB reserveを確認したが、
このpreviewはsource_full_hash_verified=falseであり、元weightsの全hash再検証ではない。
preview時checkpoint write / model callは0。

ownerはこのspecでCPU作成＋既存CPU verifyを1回ずつ行うことを承認した。
固定af9 image、UID1003、runc、network none、CPU0–7、BLAS4を継承し、
数値kernelやrecipeを新しく変更しない。
preview段階を検証済みcheckpointと扱わず、後続の全CPU verify完了後にfreezeを公開した。
新しいモデル生成や公式成績を抽出成功・重み検証から主張しない。

## 実CPU作成・保存検証の完了

既存dense recipeによる作成は **431.380300秒**、独立CPU verifyは
**299.582166秒** で、各1回・両exit0で完了した。
新規の数値実装や追加振幅試行はなく、固定F32因子と係数1を同じkernelで使用した。
元checkpointと候補はそれぞれ全12ファイルのhashを確認し、
全410対象をCPUで再生成して保存BF16 bytesと照合した。
非対象spanと前後inventory/statも全PASSだった。

これは「CPU recipe regenerated against persisted BF16 bytes」の証明であり、
RAM/GPU上の候補との比較、D2H readback、モデルforwardの証明ではない。
検証はcheckpointを書き換えず、GPU/model callsは0。
freeze公開後にownerへ配信工程を引き継いだ。

| 保存後の実測 | 値 |
|---|---:|
| 保持対象 / 実変更tensor | 410 / 410 |
| 元BF16と値が異なる要素 | 23,862,088,988 |
| 対象410 projectionの全要素 | 29,286,727,680 |
| 上記対象内の変更率 | 81.477484% |
| 保存Δ L2 | 42.46527094830241 |

変更率の分母は対象projection内の要素であり、全モデルparameter数ではない。
同じoriginalからの保存Δ L2はHの22.4858倍、Iの13.9939倍、Jの16.4843倍だった。
これらは既存manifestのmetadata比較で、追加payload走査は0。
能力比・方向保持率・丸め誤差率ではなく、historical H/I/Jの現在の配信可用性も示さない。
公開連続方向のnormと保存後のnormは別の値として保持する。

| 完了証拠（k1-exported-text-c1-v1/以下） | SHA256 |
|---|---|
| create-completion.json | 2a71ce63dfe3f6b707622b23b727533e8a631b76ac8c8ea21540c3c7c5e86880 |
| verify-completion.json | 8b24cb88ccd5f612cdd33dc5f4d3b3e30266c47d4c7900025ef40e3333473720 |
| 作成manifest | 534fbcd4bd29ec8458bd231904362fbc5c1d44e1780b9cdff97ac082d9b57f21 |
| 候補全file tree | f8f936eee95316d560361b5301a1bde8a892138f0907084eb7569242eefc7f90 |
| freeze.json | 9dad2d445ee4ccab1ea2d4bd92da21c9d911d73a018c5f8798bff66b7054526d |
| verified-handoff.json | a47972e2d782cb384cf8b6940067f7b2b7c86dd01a60d9292395bab0693c396a |
| saved-footprint-comparison.json | ec7b3b4420fdaacfd61844b808070243fd548bb8706bd6b31e4846f59efa55c1 |

### 起動前guardの形式違いと修正

初回の配信前guardはOpusのtext-fallback専用review_context_labelを要求したが、
実L receiptは正常なjson_reviewsで、labels=L、labels_match=true、
review_context_label=nullだった。レビュー失敗やモデル不一致ではなく、
呼出し側が正常JSONとtext fallbackのschemaを混同した運用不具合である。

AssertionErrorで配信・モデル処理前に停止し、失敗receiptを保持した。
guardを実形式のlabels／labels_match／review_formatへ修正し、
verified_requested_model、review_usable、tools0の要求は維持した。
実receipt受理と誤label・不一致・text形式・未検証・使用不可・tools使用の拒否を
CPU fixtureで確認し、独立読取レビューでも実receiptとの対応を確認した。
この修正にモデル再作成・再verify・追加Opus問い合わせは不要だった。

修正版pipelineはPID965025で開始した。これはruntime orchestrationだけの修正で、
499件CIに束縛されたWのPython/testsや数値recipeを変更していない。
前節は保存検証の範囲を記録する。配信後のsmokeと公式評価の進行は以下に分けて記録する。

| 運用証拠（同フォルダ） | SHA256 |
|---|---|
| deployment-smoke-preflight-failed-01.json | 08b1533c531c0498412d7cef9fc6e07e3f0fcb108f3f76373414225675781c08 |
| deployment-pipeline-launch-v2.json | 90ca4114d818fb58e6d1030207db7ec3ea7891d3aec8dc614aea0947c9f315c1 |
| dev90-pair-json-review-guard-validation.json | 55e2791cbfe7de2642ccf37b004476a6d8b234f427243f27d5127feee2090243 |

後続dev90-pair runtimeの同形式guard修正版SHAは
d5a647d7ad884bbd3f06482ce47bf6975e18cd84f0d762e6c38e452e53e817d1。
その後のdense source形式guard修正と公式ペアの開始証跡を以下に記録する。

## 配信後のauthored smokeと公式ペア開始

Lの配信と固定advanced-v2 24件の生成・sandbox実行が完了した。
同じcase/seed/validatorと生成設定を用いたが、originalはhistorical referenceであり、
この比較をfresh causalな重み改善とはしない。

| 数値観測 | Historical original-v2 | L |
|---|---:|---:|
| Train | 15/16 | 14/16 |
| Transfer | 8/8 | 7/8 |
| 合計成功 | 23/24 | 21/24 |
| Unknown | 0 | 0 |
| 生成finish stop | 24 | 24 |
| Input tokens | 2,864 | 2,864 |
| Output tokens | 43,599 | 89,912 |

新規成功0、喪失2。categorical-without-replacementは8数値trial中7件でruntime失敗、
obstacle-lattice-pathsはformat_or_syntax_errorだった。
既に失敗していたreflecting-wall-flightは、originalの1/8 numeric失敗に対してLは8/8失敗。
この数値metadataから生成本文の具体的原因は推定しない。
正常にstopした生成でも、コード形式・実行・数値要件の成功は保証されない。

出力tokenは約 **2.06225倍** だった。これは保存された総出力の比で、
思考token／公開回答を分離した測定ではなく、品質向上を意味しない。
今回の事前構造停止規則は「追加format/syntax/missing/execution-unknownが2件以上、
またはcapでusable finalが無い場合」であり、追加構造1、unknown0、cap0のため未発火だった。
categoricalのruntime test失敗は採点可能な意味的失敗であり、execution-unknownとは区別する。

rootは不利なsmoke結果を保持したまま、**既定条件の公式dev90ペアを一度だけ実行する**
判断を記録した。これは改善認定・採用・full14への許可ではない。
smokeに応じた係数変更や回答を使った修正は行っていない。

### 2件目の起動前形式guard修正

公式ペアのbindで、dense schema2の元file一覧がsource_files[].sha256であるのに、
native用のfiles[].source_sha256を参照するruntime側の誤りがKeyErrorで検出された。
モデル・API・scorer呼出し前の失敗であり、重みを再作成・再検証しなかった。

実dense schemaへ合わせ、さらに12行をpath順にcanonicalizeしたfull source treeが
検証済みoriginal treeと一致する条件を追加した。
独立読取レビューでも実manifestの12 source_filesとdigest一致を確認した。
WのPython/tests・数値recipe・公式scorerは変更せず、499件CIの対象sourceは保持した。
最終runtime SHAはf23bd30d3b4c89a124cfc4dfc6ee63e04943a447fc52d62bde2cc570e25ec6fa。

### 公式dev90の開始記録

2026-11-06 **11:03:22 UTC** に2 childを起動し、
run metadataは両方11:03:26 UTC開始・runningを記録した。
Lはgemma-0、fresh originalはgemma-2の別direct TP2 endpoint、各c32／MTP4。
native ONと公式default sampling、tokenizer/template/caps/scorerを保持し、
固定source5e3f2df、同じHE30／MBPP30／LCB20／SciCode10を使用した。

| Arm | Run ID |
|---|---|
| L | gemma4-code90-k1-exported-text-c1-c32-dev30-code-pairl-20261106 |
| Original | gemma4-code90-original-base-c32-dev30-code-pairl-20261106 |

両runのtier30/dev、90項目、manifest
a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b、
model IDとそれぞれの意図したtree labelを確認した。
開始前counter・実CID/配信構成・元weights statをbindingへ固定し、
終了後に保存callとdiscovery probeを区別して照合する。
開始時点では成績を推定せず、以下の終了証跡で結果を確定した。

| 証拠（k1-exported-text-c1-v1/以下） | SHA256 |
|---|---|
| smoke-numeric-summary.json | ce4a8a9ee27310a763972debb6a70de6ec08fb5d771173366d8bee91556a3863 |
| structural-smoke-decision.json | 48f431d800db11bdde599f76672e24306129db444e53ee9536c565cbdde0a52a |
| dev90-pair-dense-source-guard-fix.json | 93c7568ff4ed6db02079f1cce70ab7aa9ddceb8c37b2a9cb587c373f90339f8d |
| dev90-pair-plan.json | 48df4b1f444167971980e21240c0e5ffa2ace7abc671d9a3697b6325e62eaf54 |
| dev90-pair-binding.json | a578708b993e4feee06691f30c524b90e61599db0ee9b1422d027449587fe91f |
| dev90-pair-launch.json | 303cd33c69bf02102e711faf9571106c8401ff4044926af6f3a337041e7ff814 |

## 公式dev90の最終結果：不採用

Lとfresh originalは各90/90採点・未採点0、合計180採点でcompletedになった。
同じ90 ID、manifest、source、runtime/配信identityと終了時継続を確認した。
公式codeは4-source等重み平均であり、90問の単純正解率ではない。
SciCode10問は各問のstep成功率平均を保持する。

| 指標 | Fresh original | L | 差 (pp) |
|---|---:|---:|---:|
| HE+ (30問) | 29/30 | 29/30 | 0 |
| MBPP+ (30問) | 23/30 | 22/30 | −3.333333 |
| LCB (20問) | 15/20 | 15/20 | 0 |
| SciCode mean (10問) | 34.468254% | 23.730159% | −10.738095 |
| 公式code | 70.700397% | 67.182540% | **−3.517857** |

既存公式paired比較の95% CIは **[−11.875000, +4.107143] pp**、
2000 resamples・seed20261104。全90件がpairedで、partialはcode以外13領域が
未測定であることを示す。overallはnullのままで、欠測を0にしていない。
単回の不確実性を残すが、この測定でLの純改善は示されなかった。

### 課題別の実数得失

HEは全問同値。二値80問ではoriginal67成功→L66成功、獲得2・喪失3だった。
SciCodeは部分点を含む獲得3・喪失2、残5問同値。
既存公式の0.5境界flipは全90問で獲得4・喪失5であり、
実数差の獲得5・喪失5や完全成功数とは異なる定義である。

| 変化したID | Original | L |
|---|---:|---:|
| LCB/3777 | 0 | 1 |
| LCB/abc390_g | 1 | 0 |
| Mbpp/607 | 1 | 0 |
| Mbpp/641 | 0 | 1 |
| Mbpp/792 | 1 | 0 |
| SciCode/14 | 1 | 0 |
| SciCode/35 | 1/3 | 2/3 |
| SciCode/46 | 1/4 | 1/2 |
| SciCode/58 | 4/5 | 0 |
| SciCode/80 | 2/7 | 3/7 |

この表は保存されたID・scoreだけから作り、問題本文、生成final、
hidden reasoning、goldを分析していない。喪失原因を重みの特定要素へ帰属しない。

### 生成・使用量と起動probeの分離

| 記録値 | Fresh original | L |
|---|---:|---:|
| 保存chat calls / logprob calls | 119 / 0 | 119 / 0 |
| 保存input tokens | 96,661 | 91,862 |
| 保存output tokens | 343,116 | 505,277 |
| report generation seconds | 318.5 | 808.8 |
| report wall seconds | 382.0 | 854.3 |
| 開始→終了の実時刻差 (s) | 382.057530 | 854.312411 |
| 終了UTC | 11:09:48.665598 | 11:17:40.920228 |

開始は双方2026-11-06 11:03:26 UTC。L出力tokenは約1.47261倍、report wallは約2.23639倍だった。
思考／公開回答tokenを分離した値ではなく、単回のruntime・採点時間を含むwall比である。

processは各120終了、保存119 callsとの差は各prompt21／completion1 tokenで、
固定のmax_tokens=1起動discovery probeと一致する。
originalはprocess stop119＋length1で、probeを除くと保存生成は全119stop。
Lはprocess stop118＋length2なので、probeを除いても**実生成1件のlength終了が残る**。
Lの全capをprobe扱いして消してはいけない。

LのSciCode reportはtruncated_rate=0.1、empty_rate=0.0256410256、
他sourceは両rateとも0だった。originalは全sourceで両rate0。
process error/abort/repetitionは双方0、process start identity不変・counter比較可能。
これらは独立HTTP retryや別clientの不存在、思考token数の証明ではなく、
保存callsとprocess counterを同じ指標へ読み替えない。

### 次の仮説との境界

**Lは不採用、code90未達。** rootの正式判断はfull14・DeepSWE・HFにも進めず、Mを別の診断仮説として検討する。
学習基点が未知のHeretic derivativeであるため、O+Dの失敗だけで
同じDがその公開base上でも無効と結論しない。一方、base不一致が今回の原因だとも断定しない。

次の[M事前契約](2026-11-06-public-base-alignment-fourway-m.md)は、
original O、O+D（L）、公開改変base P単独、P+Dを同じ元protocolで比較する。
これは係数gridやLの採点変更ではなく、base効果とadapter/base相互作用を分離する別仮説である。
真のtraining revisionや著者の推論protocol再現は未確認のまま保持する。

| 最終証跡（k1-exported-text-c1-v1/以下） | SHA256 |
|---|---|
| root-final-decision.json | 3f5d7c022cbccaad22b89629eae88fbe23b07339cad42568715472e821f7be92 |
| dev90-pair-final-summary.json | ef270535e19af2e9174987c33ee601e3ab42ed3f625a57c05d0753585aee92e2 |
| dev90-l-versus-base-evidence.json | c8f8bacf0f6910f332e2a1b8faf9a81f249cea0f6bb9e82c9436130ca1bce9da |
| dev90-pair-numeric-evidence.json | d4ed348f1841016fda2235a6cfc5ac2522f408bdc3c2079ca71e1a2e302d658c |
| dev90-runner-parity.json | a6b323aad8c2b416dfee07ea92c4d022789715b4745c90dd8d0729202d07a54a |
| serving-counters/l-diff.json | eb8f02ff0ba19141a630fb6d0bbe6b708c3f2c8e4bee93a458ac6a7b03816228 |
| serving-counters/base-diff.json | 56a775530598d256815ca800058f9aa01c30b3a2e9eb8a5e560253b416a869aa |

独立レビューで全90 IDとofficial capability値、source平均、公式flip、
二値80件とSciCode部分点の整合を確認した。新規採点・bootstrap再実行・モデル呼出しは0。
