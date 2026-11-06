# N: 改善済みPへ固定K1方向の1/16を積み上げる（2026-11-06）

**公式N−fresh Pまで完了。Nは71.603175、Pは74.242063で、Nへ更新せずPを保持する。**
起点は[4条件比較で更新したP](2026-11-06-public-base-alignment-fourway-m.md)である。
original Oへ戻って候補を作り直さず、Pの保存weightsへ固定差分を加える。

## 目的と固定変更

同時期の公式dev90でPは73.825397点、Oは71.186508点だった。
同じPへK1のexported-unit text方向を全量加えたMは67.908730点、
P比−5.916667 ppで更新しなかった。
Nは同じ方向全体を固定係数1/16（0.0625）にする1試行である。
振幅grid、項目別調整、1/16の最適性や単調な安全性を主張しない。

| 固定項目 | 内容 |
|---|---|
| 作成元 | P、checkpoints/code90-heretic-public-base-official-protocol-v1 |
| source tree | 3dfb3bd1db0ab2ba4fff843616bb0d3109e0b97a6c9349658620908f2a24bebc |
| 親state SHA256 | 609190932c606e5b9219be072344f68fd2b8e2fdb695109eec63581ed8a75f21 |
| 公開donor | [win10/gemma-4-31B-K1の固定revision](https://huggingface.co/win10/gemma-4-31B-K1/tree/712fa2808d60465fa0bde72a69fc6affcb018ffc) |
| factor | 同じlossless text820因子、F32、rank32、alpha32、ordinary alpha/r=1 |
| 対象・係数 | text projection全410へ一律0.0625、vision378因子は除外 |
| 派生factor SHA256 | f33772cab59b931e5ade830686157789fbe6b9163f3ce81539b70ee00b55cb78 |
| factor header SHA256 | 5e72e0b621922fe78f710abb54395961e3b7a3383a5bf6f60703e41763c4abb2 |
| config SHA256 | e5a0f995c9e0fc5689ad9ad526a30e307d9162c614e89666f0571f77dff0b981 |

410は対応target数であり、丸め後の実変更tensor数ではない。
既存CPU FP32 factor演算を使い、保存Pから一度だけBF16へ丸める。
norm・変更数はP比として後で記録し、Oからの累積normへ読み替えない。
小振幅でも丸めで消失・偏りが生じ得るため、norm比から方向cosineを推定しない。
Pが宣言された学習baseと同じ系列でも、donor著者の真の学習revisionは未確定である。

## 作成と検証の状態

metadata preview後、rootの固定操作承認により12:41:23.103640 UTCに
CPU作成を1回開始し、12:48:29.904348 UTCに426.8015秒で完了した。
続いて12:48:29.920122 UTCに既存[独立CPU verifyとfreeze](../wsp-dense-checkpoint-freeze.md)を
1回開始し、12:53:27.248394 UTCに297.3289秒で完了した（いずれもexit0）。
全410 targetのCPU再生成、対象外byte、source/candidate全file hash、
前後inventory/stat一致を確認した。新しい数値実装や第三の410行列再計算は導入していない。

| 保存BF16の実測 | 結果 |
|---|---:|
| 実変更tensor | 410 |
| 実数値変更要素／bit変更要素 | 両方5,267,811,348 |
| Pに対するΔL2 | 2.434033420321218 |
| candidate tree | 355629b43ad592b0a830208874c04cc0a11e2d51b4b177aeee0e502dc599c5a5 |
| manifest SHA256 | f1ad1307324166bb63e82264557fd8c6d305cf058b4b9697d22d0c2b0e5eda7e |
| freeze SHA256 | d47151fc72df48e0ba84a3ea9240d1c8fd8a9b2d194909c3f094e07ea346ed31 |

このproofは保存bytesをCPU recipeで再生成して照合したもの。
GPU/RAMとの比較や能力確認ではなく、作成・verifyともGPU/model callsは0である。
P比のnormをOからの累積normや意図した連続方向のcosineへ置き換えない。

最初の配信・smoke待機pipeline（PID1300668）は適用前のassertionで停止した。
Docker Mountsの3要素は同じで配列順序だけが異なることをpermutation照合で確認し、
mount配列をcanonical sortして比較する運用修正だけを行った。
失敗receiptはdeployment-smoke-preflight-failed-01.jsonとして保持し、
再起動r2（PID1330467）でgemma0へのN配信が成功した。
配信後とsmoke後の両方で、非対象P1／O2／M3の保存identityが開始前と一致した。
CPU作成・検証やGEMMは再実行していない。

固定advanced-v2 smokeは22/24（train14/16、transfer8/8）で完了した。
全24件evaluated／stop、unknown0・generation errors0、
追加構造失敗0／capかつusable final無し0。記録input2,864／output39,120 tokens。
比較親はPの既存21/24（output41,241）、originalの23/24はsecondary歴史参照である。
case・source・生成設定・sandbox seed/trialsの一致は数値summaryで確認したが、
Pはfresh同時再生成ではなく、この+1を公式WSPや一般能力の改善としない。
構造停止条件は非該当で、その後の公式N−P結果を末尾へ記録した。

UID1003、固定image、CPU0〜7／BLAS4、128GiB memory、networkなし・GPUなし。
Pの9ファイル入力はread-only、空き容量reserveは300GiB。
各phase前後に親state SHA、source path/tree/statを再確認する。
初回host launcherはNumPyを読み込めず数値処理前に失敗し、checkpoint書込0の証跡を保持した。
managed Pythonへ切り替えた後の実createはこの1回である。
既存506件CI（454 pass／52 optional skip）と数値source pinsを継承し、
今回の記録作成によるsource変更・再テストはない。

## Opus方法レビューの範囲

匿名一般方法を1回送信し、actual modelはclaude-opus-5-5のみ、
label N、verified_requested_model／review_usable／labels_matchはtrue、tools0。
43.278秒、記録費用$0.098524。私的score・座標・問題・回答・reasoningは送らず、
raw stream／thinking本文も保存していない。候補そのものをレビューしたものではない。

採用した助言は、現在の親・factor・演算の固定、mapped targetと実変更数の区別、
freshな親との比較で得失・不確実性・truncation／empty／unknown／tokensを報告すること、
累積退行を追うこと。小振幅での丸め、未知の学習baseとの相互作用、
同じdev集合を繰り返す適応的選抜、単回生成の揺れを限界として残す。

追加の全行列cosine計測、greedy生成、別precision、stochastic rounding、
振幅gridは今回の必須工程にしない。既存native ON構造smokeと保存byte検証を用いる。
reviewの助言を新たな承認gateやCI有意差の要求へ変換しない。

## 比較と作業モデル更新

このN coding stepのprimaryは、fresh Pに対する同時期の公式dev30 code90差分。
HE30／MBPP30／LCB20／SciCode10のsource等重みと全90項目の採点を保持する。
native thinking ON、TP2／MTP4／c32、公式template・tokenizer・sampler・caps・scorerは不変。
未採点を0補完せず、SciCode部分点とbinary flipを区別する。
Nの事前計画は差が正なら作業更新、同点・負ならP保持である。

一般の[積み上げ方針](2026-11-06-code90-plan.md)では、コードに限らず関連能力の
net gainも改善の対象となる。今回のprimaryがcodingであることを、今後の進歩すべてを
code scoreだけで決める規則へ拡張しない。多少の関連領域損失は得失を示して許容し、
全領域非退行やCI有意差を追加条件にしない。
作業更新は最終90点超確認、未知問題への一般化、全領域保持、DeepSWE、HF適格とは別である。

Pの保持状況を測るfull14 dev10は778/778採点・未採点0で完了した。
overall78.47510、code87.77778（SciCodeを含まない26項目）。
歴史original比overall−1.68906 pp、XSTest31/36→21/36等の損失も保持し、[全台帳](2026-11-06-all-official-scores.md)へ記録する。
runはgemma4-code90-incumbent-p-c32-dev10-full14-20261106。
wrapper開始12:41:10.358070 UTC、公式run開始12:42:02.587775 UTCで、
NのCPU作成開始後に公式runが始まり、その後CPU/NVMe処理が重なった。
wall時間を隔離された比較と扱わない。
このrunはPの保持状況でありNの保持の代用ではなく、Oの既存full14は歴史参照である。
runnerの同一性は担当者の証拠範囲で別途確認し、本記録では過去imageとの完全同一を仮定しない。

全期間のscoreは[公式スコア台帳](2026-11-06-all-official-scores.md)、
factor抽出とunit試行は[Lの記録](2026-11-06-k1-exported-text-c1-review.md)を参照。

## 固定証拠

起点はanalysis/code90-resume-20261106/。初期planのprepared状態は保持し、
実作成承認・開始を別receiptで記録する。

| 証拠 | SHA256 |
|---|---|
| 作成時working-incumbent（現在はworking-incumbent-history/p-before-n-609190…json） | 609190932c606e5b9219be072344f68fd2b8e2fdb695109eec63581ed8a75f21 |
| heretic-public-base-k1-c00625-v1/spec.json | 2d97d005268521d1123e2b604bbdd08e513593318f4abd7a4e9274850ce49472 |
| 同launch-plan.json | 8091ed26c49d1572c73c7a1d7ebb66213cfef589ffddeb47eafb474bc72c41a3 |
| 同proposal-n-plan.json | 2c5b5aeb5b876affb5999e654afadf8e975cfe6fa1c2d7e3b37ffe7085f13dac |
| 同root-create-authorization.json | adb63bf897232e85dd879e38ddc07b414a7123662bad82d019795888dd9f6227 |
| 同create-launch.json（開始時） | f44a605568f2e1f63d1a34d942f28e29f664e1cc587b6cf387c808b5b10f33cd |
| opus-incumbent-coherent-step-n-review/receipt.json | a8dd5b0398b4e79a37a6a5265e794de13a669d8ef40def881a08daa52b6f28d5 |
| 同method-summary.json | 3f635bf5b82960ca85c35c29034211460ebdf886328a0314d8a2c6ea49bfb093 |
| incumbent-p-retention-full14-v1/plan.json | 257d375ee671a169fabd08127faee0429828d3d3d7c2ec726604e6455f1018bd |
| 同launch.json（開始時） | 328707b9eb41fe4a48ba76a96f7ca5d2780313249f22909023dcf5765a612524 |
| heretic-public-base-k1-c00625-v1/create-completion.json | 58f57a1d1255b240683121e06574a9bfa1c6980db5e3f1692ff1262824424f95 |
| heretic-public-base-k1-c00625-v1/verify-launch.json | 07434973e44ba12e6f64a4d8a5e204cc0753a71b1ab85a2a6808dbf0ee03772c |
| heretic-public-base-k1-c00625-v1/verify-completion.json | d67f1758fe154613902bed245334c3ba8b4ac636c4d2b95ce1203040605924c0 |
| heretic-public-base-k1-c00625-v1/freeze.json | d47151fc72df48e0ba84a3ea9240d1c8fd8a9b2d194909c3f094e07ea346ed31 |
| heretic-public-base-k1-c00625-v1/native-on-smoke-plan.json | 359b2e97327b87143c0af21471b979b5d032b4a56997bcb4a51770e2f94b5ebf |
| heretic-public-base-k1-c00625-v1/deployment-pipeline-launch.json | 901c41d7c4b809fbb09e16b59994071339e3dc5f8572aa7d0b6c639c525c77f0 |
| heretic-public-base-k1-c00625-v1/deployment-smoke-preflight-failed-01.json | e9462f0940116e53f51491deef2b44807d390febca11ba6407724393617d843d |
| heretic-public-base-k1-c00625-v1/deployment-mount-order-diagnosis.json | 76c9412a76b0e0075b8fb839f9057c0aeddd1e6931edb0c4d99de59d6698a6ce |
| heretic-public-base-k1-c00625-v1/deployment-pipeline-launch-r2.json | a07a595add70f66f52535de2489a0feb22553a19402b2cb702b66fb348d74b2f |
| heretic-public-base-k1-c00625-v1/deployment-smoke-launch.json | 7fbe7dbf46805f0dc5d8894d8a230294dead2ced0ed1c0ecd679205293491eaa |
| heretic-public-base-k1-c00625-v1/smoke-numeric-summary.json | ff908d512a810b5e7bec34bb3d96c931064250b698c1822dd8898219bd5194fd |

## 最終追補：公式N−Pと作業モデル保持

同一dev30コード90項目（HE30／MBPP30／LCB20／SciCode10）で双方completed、
90/90採点・未採点0。既存公式source等重みcodeを比較した。

| 条件 | HE | MBPP | LCB | SciCode | code |
|---|---:|---:|---:|---:|---:|
| fresh P | 96.666667 | 80.000000 | 80.000000 | 40.301587 | 74.242063 |
| N | 96.666667 | 73.333333 | 75.000000 | 41.412698 | 71.603175 |

N−Pは-2.638889 pp、95% CIは
[-7.083333, +1.111111] pp
（既存公式paired bootstrap、2,000回、seed20261104）。
HEは29/30のまま、116を失い95を得た。
MBPP641／801を失い24/30→22/30、LCB abc390_gを失い16/20→15/20。
SciCode72は7/9→8/9の部分改善で、他のSciCode項目は同点だった。
公式0.5閾値flipはgain1／loss4であり、このSciCode部分改善を含む
生score差の正負とは別の定義である。

| 記録量 | fresh P | N |
|---|---:|---:|
| report wall秒 | 351.2 | 345.4 |
| 記録chat API calls | 119 | 119 |
| 記録input tokens | 92,513 | 93,210 |
| 記録output tokens | 318,215 | 324,540 |
| source truncation／empty | 全source0／0 | 全source0／0 |

serving intervalは各119 stop＋1 lengthで、最後の1件は固定wrapperの
1-token readiness request。公式119 callsとの差は入力21／出力1 tokenで一致し、
benchmark itemの打切りとは扱わない。
終了時のsource weight stat・serving identity・incumbent pin継続性を確認した。
counterだけから未知の他clientやitem別HTTP回数を証明しない。

事前のparent-relative規則により、Nは作業モデルへ昇格せずPを保持する。
理由は今回の主比較のnet lossであり、CIの有意差や全分野無損失を新gateにしたものではない。
自作smokeの+1もこの公式結果を上書きしない。新規にNのfull14／DeepSWE／HFへ進む成果は得ていない。
rootのworking-decision.jsonでP保持が確定し、working-incumbent.jsonを更新した。
Pのpath/tree/保存weightsは不変で、73.8254→74.2421は再測定差であり追加の重み改善ではない。
次の[R構造subset仮説](2026-11-06-cumulative-mlp-math-step-r.md)はこの更新stateからPを固定し、
作成直前にもSHA/path/tree継続性を確認する。

対象run：

- gemma4-code90-heretic-public-base-c32-dev30-code-pairn-20261106
- gemma4-code90-heretic-public-base-k1-c00625-c32-dev30-code-20261106

| 最終証拠（同analysis起点） | SHA256 |
|---|---|
| heretic-public-base-k1-c00625-v1/n-vs-current-p-evidence.json | a202364f5e2747089234feb631beaadb5a333dec08dddd2b4d77bd3b4becefa3 |
| 同dev90-final-summary.json | 97140ae8416e4b11bbc4a88e123ebc6c105943d40ca9d52921c26ea6f15530c6 |
| 同working-decision.json | e4c71e80f5eba7a85d64c9562cf2ef36c92f215ce3e6b1c81de671d58a06dc86 |
| 更新後working-incumbent.json | 3bc3b04014732f799987888d171d4f70116724af3e31147aa243095950a5294c |
