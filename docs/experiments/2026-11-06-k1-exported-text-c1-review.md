# L: K1 の保存text因子を係数1で移す固定試行（2026-11-06）

Lは、公開K1 adapterのtext因子だけをbyte同一で取り出し、exported coefficient 1で元Gemma 4 31B BF16へ移す一つの固定試行である。**全499件CPU CI、実text因子抽出、固定specのCPU previewを完了した。owner承認でモデルのCPU作成・検証工程へ進んでいるが、保存後変更量・freeze・配信・公式成績はまだ確定していない。** K（B＋H byte合成）とは別の候補である。

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
保存後の実変更数・Δ L2・対象元重みとの比を作成後に別途記録する。
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
保存checkpoint作成・verify・配信・smoke・公式scoreの完了値は、各操作の完了証跡で追記する。
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
保持410 pairsはmapping数であり、実BF16変更tensor数・変更要素数をまだ表さない。
元treeはcab440f871a32b5ea1c12364db7dd35ac8768a83b413ca24bbc18d6478eccbd9。

既存dense factor-format v2のCPU previewは **1.6878秒** で成功した。
410対象、62,578,686,256 Bのcheckpointコピーと300 GiB reserveを確認したが、
このpreviewはsource_full_hash_verified=falseであり、元weightsの全hash再検証ではない。
preview時checkpoint write / model callは0。

ownerはこのspecでCPU作成＋既存CPU verifyを1回ずつ行うことを承認した。
固定af9 image、UID1003、runc、network none、CPU0–7、BLAS4を継承し、
数値kernelやrecipeを新しく変更しない。
作成・freezeの完了証跡がまだ無い段階を、配信可能・検証済みと扱わない。
新しいモデル生成や公式成績をこの抽出成功から主張しない。
