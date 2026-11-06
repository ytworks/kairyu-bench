# R: 現作業モデルのattentionを保持するMLP数学方向（2026-11-06）

**全14領域の得失を確認し、Rの暫定採用を撤回した。改善済みPを維持して次の編集を積み上げる。90点目標は未達。**
公式dev90はfresh P73.269841→R73.626984（+0.357143 pp）だったが、後続full14では全体78.475099→76.382319、推論・長文・code等が低下した。以下には暫定採用と撤回の両方を時系列の証拠として残す。
[公式N−P](2026-11-06-incumbent-coherent-step-n.md)はfresh P74.242063に対しN71.603175、
差−2.638889 pp（95% CI [−7.083333,+1.111111]）で、Nを採らずPを保持する判断となった。
R作成時の親はPで、その時点のstate SHAは3bc3b04014732f799987888d171d4f70116724af3e31147aa243095950a5294c。
sourceはcheckpoints/code90-heretic-public-base-official-protocol-v1、treeは
3dfb3bd1db0ab2ba4fff843616bb0d3109e0b97a6c9349658620908f2a24bebc。
作成直前にもSHA/path/treeを確認し、original Oを作成元へ戻さない。
Pの73.8254→74.2421は重み不変の再測定差であり、新しい重み改善とは数えない。

ツールのCPU suiteは507件収集、455 pass／52 optional skip、失敗0。前版506件から、r128/F32の範囲指定作成→独立verify→freezeと対象外byte保持を検証する1件を追加した。既定の410行列経路も維持し、Ruff／mypy73もPASS。証跡はanalysis/code90-resume-20261106/cpu-integration-structured-mlp-scope-v1/ci-results.json（SHA256 5a47f8fe58ec82f525b62f13c36afc11cd215f6839ab55893268a167b1d7b229）。これはツールの動作確認で、Rの能力確認ではない。

## 公式dev90比較と暫定採用の記録

両runは同じ90項目を全件採点し、未採点0、正式chat119 callsずつ。
native ON、公式sampler/caps、c32、TP2/MTP4、固定source5e3を維持した。
実runner OCIを自動削除前に取得し、両image設定・RootFS、host76 source pins、
installed Python73 files、配信CIDとreadonly weightsの継続性を照合した。

| 指標 | fresh P | R |
|---|---:|---:|
| 公式4source等重みcode | 73.269841 | 73.626984 |
| HumanEval+ | 30/30 | 29/30 |
| MBPP+ | 23/30 | 24/30 |
| LiveCodeBench | 15/20 | 15/20 |
| SciCode source平均 | 41.412698 | 42.841270 |
| 出力tokens | 318,770 | 322,596 |
| 入力tokens | 94,082 | 94,017 |
| wall秒 | 339.9 | 427.1 |

差は+0.357143 pp、paired95% CI [−2.142857,+2.857143] pp。
小幅でCIは0を含むが、固定した正の改善条件に従って作業モデルを更新した。
MBPP641が0→1、SciCode80が2/7→3/7となり、HumanEval95が1→0となった。
SciCodeの部分点増加をbinary flipへ置き換えない。
全sourceのtruncation／emptyは0。counterの正式119 stopに加わるlength1、
入力21／出力1 tokenは起動確認であり、採点対象の打切りではない。
wallはこの実行窓の観測で、隔離された速度効果とは扱わない。

暫定採用時の次入力はcheckpoints/code90-heretic-public-base-math-mlp-c1-v1、
tree ea3d27ababa4f189fb5058c5055492ddad42180ec619c80d5a427baccea9cbee。
その時点のworking-incumbent.jsonはc0e271c4811d2e60b856e58dd6c7940fef10b6593454ef8554efcb2932a89d49、
作業更新判断はheretic-public-base-math-mlp-c1-v1/working-decision.json
（SHA197caa94f18184f09002e51434e9194a5f1cf074937d09718d86c0c92a18e62b）。
作成時の親P state3bcはhistoryへ保存した。
この暫定採用中はP微分を流用せず、Rで2目的を再観測しR由来S/Uを計算した。R由来S/Uの有限応答・生成評価は実行していない。撤回後は完全なP保存重みの不変性を確認し、既存P観測に基づいて別のP由来S/Uを計算する。
全14領域、known64、DeepSWE、HF適格はこのdev90更新とは別に確認する。
過去のP−original差と今回のR−P差を、異なる実行窓をまたいで加算しない。

最終比較証拠はdev90-final-summary.json
（SHA5472def6705c91eaa4a76fe8e8fa5c8bf85dc0089956f250e81d8e9162d63520）、
r-vs-current-p-evidence.json
（SHAebde5a3a1dedbbb5c8929d3a674f5358b6b7096c9da9c17abbaa2464af00d483）。
本節の相対artifact pathはanalysis/code90-resume-20261106/を起点とする。

## 全14領域の結果とR追加分の撤回

778/778採点、未採点0。全体はP78.475099→R76.382319、差−2.092780 pp、paired95% CI [−4.265873,−0.096748] pp。比較Pは歴史runであり同時実行対照ではないが、実runner RootFS16layers・Image.Config・全installed82 source/resource bytes・保存runtime設定の一致を確認した。共有GPU/CPU/NVMe作業との重複があり、wall差をモデルだけの速度効果とはしない。

| 領域 | P | R | 差pp |
|---|---:|---:|---:|
| 全体 | 78.475099 | 76.382319 | −2.092780 |
| code（26項目、SciCodeなし） | 87.777778 | 84.444444 | −3.333333 |
| 数学 | 95.625000 | 98.750000 | +3.125000 |
| 推論 | 76.956522 | 64.782609 | −12.173913 |
| 長文 | 87.627943 | 81.377943 | −6.250000 |
| 常識 | 63.000000 | 64.000000 | +1.000000 |
| XSTest | 21/36 | 20/36 | −2.777778 |

knowledge−3.333333、multilingual−3.055556、facts−2.5 pp、expert・instruction・tools・agents・visionは純差0。変更25項目は9 gain（HMMTの半点増を含む）／16 loss。codeのlossはMBPP792、推論はBBEH disambiguation119とMuSR murder205/object187、長文はRULER fwe32768:013。RULERとTriviaQAにtruncated/emptyが各1件あり、Pは0件だった。logprob等の非適用・未計測nullを0へ置き換えない。Rの正式chat557／logprob3062 calls、wall1481.9秒（load105.2秒は内数）。通信error／abort0と答案の打切りは別の観測である。

dev90の小幅なcode利得に対して、codingにも関係する推論・長文・別code cohortの損失が大きいと判断し、R追加分を退けた。これは全領域無損失や有意差の新しいgateではなく、目的に沿う得失判断である。originalへ戻す判断ではなく、既に採用したPの58 attention変更を維持する。

現在のP stateは2666c54246852bf328be8b6293ff3402f011403fc27ad28113c5bcddaf923c99、tree3dfb3bd1db0ab2ba4fff843616bb0d3109e0b97a6c9349658620908f2a24bebc。Pの全9file statを既存attestation87975545…と再照合した。R state c0e…と全結果・仮説はhistoryに保存し、R計算済みS/UをPへ貼り替えない。既存P観測からP用S/Uを新規計算する。R有限応答の起動は自動承認審査で拒否され、model/service操作0。R撤回後にこの不要となったrunを再試行していない。

証拠：incumbent-r-retention-full14-v1/final-summary.json（SHAdb693d71de3180fe49910d9a89cf07c494771a1f2e37dfc4c41fbd2750969095）、同historical-parent-evidence.json（d70b61f07ef45776427cbf0f2c582364cb86007458a0b7eed3c59be9fc190eda）、heretic-public-base-math-mlp-c1-v1/working-retention-decision.json（1619f5487e92716434641797eb93971a4ad88e181ccc9f105d96f5d788b1e528）。

## 保存重みの作成と独立検証

現Pからprivate checkpointを1回作成し、既存CPU recipeによる独立verifyを1回実行した。
作成は **385.406秒**、verifyは **269.462秒**。
実行はUID1003、CPU0–7、BLAS4 threads、既存af9 image/runc、300GiB reserveで、
GPU・配信・生成はこの作成手順には含まない。並行したPのnative観測（GPU4／CPU32–39）
とCPU/NVMe資源を共有しており、この時間を隔離した速度比較とはしない。

| 保存後に確認した量 | R |
|---|---:|
| 完全に検証したtext mapping | 410行列 |
| 既存FP32演算で適用・再生成したMLP | 180行列、60 complete triplets |
| 実際にstored BF16が変わった行列 | 180 |
| attentionの明示0係数・raw byte保持 | 230行列 |
| stored BF16変更要素 | 5,458,871,089 |
| 親Pに対する実ΔL2 | 2.487264934266907 |
| source／candidateの全file hash・stat範囲 | 各9ファイル |

attentionの係数0には加算・丸めを通さず、native normalization/scalar、vision、
embedding、その他対象外spanとprotocol metadataもbyte単位で照合した。
[CPU freeze bridge](../wsp-dense-checkpoint-freeze.md)はvalidated410／regenerated180／zero230を分け、
`all_410_matrices_cpu_regenerated=false`を記録する。全410を再計算したという証跡ではない。
元Pの全9file stat、parent state、数値source/kernel pinは作成・verify前後と
終了後のmetadata監査で一致した。checkpointは
`checkpoints/code90-heretic-public-base-math-mlp-c1-v1`、
保存treeは `ea3d27ababa4f189fb5058c5055492ddad42180ec619c80d5a427baccea9cbee`。

追加GEMMやpayload再走査をせず、過去Iの保存manifestでも構成を照合した。
P−Oの既存exact deltaが全180MLPのraw byte一致を示し、
IとRは同じmath factor・係数1・128-row FP32演算・NumPy/BLAS/kernel条件を用いる。
Rの180行列のsource SHA、完成target SHA、変更数はIのMLP180と全て一致した。
この照合はRのMLP構成を確認するもので、Iのattentionや能力scoreをRへ引き継ぐ証拠ではない。
RのattentionはPの保存値を保持する。

original→Pの58 attention行列とP→Rの180 MLP行列は重ならないため、既存のexact差分からRの累積変更は238 tensor、6,473,994,931要素、original比ΔL2 25.695807972794と算出できる。これは保存済み証拠の合成で、新たな全payload走査や能力効果の推定ではない。変更要素数の小ささと数値差分normの小ささは別の尺度として扱う。

gemma-3だけをRへ置き換え、N／P／original／embeddingの設定・identityを保持した。
配信後の固定24問authored smokeは完了し、**R20/24対current parent P21/24**。
Rはtrain13/16・transfer7/8、
gain0／loss1（lexicographic-pair-rankingのruntime失敗）だった。
全24件stop、unknown0、新規構造失敗0、usable final無しのcap0で、
事前の構造停止条件は発火しなかった。これをsemantic非退行とは呼ばない。

記録output tokensはR43,536／P41,241。
両者は同一case・template・sampling・sandbox seed/trialsだが、Pは保存済みsmoke参照で
fresh同時比較ではない。これはWSP scoreでも採用判断でもなく、公式R−fresh P評価は別工程である。
上記は構造化numeric summary／receiptだけから記録し、モデル出力本文やreasoningは読んでいない。

## 仮説と変更セット

[公開math GRPO方向I](2026-11-06-math-grpo-c1-review.md)のうち、
全60層のMLP gate/up/downを**3つ一組で180行列**適用し、
親のattention 230行列とnative normalization/scalarをbyte単位で保持する。
個々の層・channelを選ぶ探索ではなく、機構上つながるMLP入力変換・gate・出力変換を
全層にわたり固定した構造subsetである。

| 固定する内容 | 計画 |
|---|---|
| donor | aestudio/gemma-4-31B-it-GRPO-math-lora |
| 公開revision | 8ec9097a26697380f9e1546fdc1e485f23bd3b61 |
| factor形式 | factor_format v2、F32、rank128、alpha256、ordinary scale2 |
| 明示係数 | 1（適用MLP内の相対方向・符号を保持） |
| mapping | 元の全410対応を検証し、MLP180をactive、attention230を明示0 |
| 保持 | attention・native・vision・embedding等の対象外weightsとprotocol metadata |
| 演算 | 既存FP32 primitiveと最後の1回BF16丸め。新しい数値kernelは作らない |

係数0のattentionは浮動小数加算を通さず親のraw bytesを保持する。
mapping数410、active再計算180、丸め後の実変更tensor数は別々に記録する。
検証が「全410行列を変更／数値再生成」と誤表示しないことを実装レビュー対象とする。
既定の全410適用は変えず、module/native/joint/RAM wrapperのfactor v2拒否も維持する。
対象外byte、全file hash、入力継続性と1回の独立recipe再生成は既存の保証を使う。

Rが測るのは、親のattention変更を残したまま数学方向のMLP成分を足すことで、
科学・algorithmの実行結果が変わるかである。
数学問題のGRPOは科学プログラムの訓練と同じではない。
donorはattentionとMLPを一緒に学習しており、Rはその学習済みadapter全体を再現せず、
共同適応を切り離した新しい構成になる。失敗してもdonor全体を否定しない。

旧Iの公式dev90利得はHE151／95だけで、SciCode・LCBには利得がなかった。
M比較時のPではその2件とも成功したが、直近N比較時のfresh PはHE95で失敗した。同じ重みでも結果は揺れており、過去の成功を安定解決や新候補の固有利得とは数えない。
保存IのΔnorm²の67.18%がMLPという観測は変更範囲の大きさを示すだけで、
科学能力の局在やRの効果を証明しない。
attentionの保存bytesが同じでもMLP出力が変わるため、attention依存の動作保持も未保証である。

## Qとの区別とOpusレビュー

Qはagent/instruction系donorの全方向を小さく加える案だった。
過去Hのlong-context／reasoning利得は現在Pの実測と重なっており、
そのまま新しい改善余地とは言えないため、Qを無条件の次候補とはせず保留する。
QとRはdonor自体も異なるので、Q対Rを「全方向対MLPのみ」の純粋なablationとは呼ばない。

匿名の公開一般方法Q/Rを1回送信した。
actual assistantとmodel usageはclaude-opus-5-5のみ、labels QR一致、
verified_requested_model／review_usable=true、tools0、fallbackなし。
52.994秒、記録cost $0.119424。
私的score・norm・座標・model/checkpoint識別・問題・回答・reasoningを送っていない。
raw stream／thinking本文は保存せず、公開finalだけを方法上の意見として利用した。

受理する助言は、親とfactorの固定、complete tripleの検証、親比の実BF16変更量、
対象外byte保持、fresh parentとの実行比較、得失・長さ・打切り・不確実性の別報告である。
不完全なtripleや曖昧なmappingは黙って一部適用しない。
任意のcosine追加計算、巨大proxy replay、追加GEMM、greedyへの変更、
振幅grid、有意差・全領域非退行の新gateは必須工程にしない。
逐次BF16保存では順序がbyteへ影響し得るため、連続差分の加算性と区別する。

## 親Pの観測と次に判別すること

Pの全14領域dev10は778/778採点・未採点0、overall78.47510、
code87.77778だった。code26項目にはSciCodeがなく、code90や最終既知64と混ぜない。
Pのtools80、agents50、instructions92.0833、long-context87.6279、
reasoning76.9565、math95.625、safety58.3333を保持確認の文脈として残す。

歴史originalとの既存比較ではoverall−1.68906 pp
（95% CI [−4.24263,+0.92584]）、code同点。
expert+12.5／long-context+6.25／reasoning+2.1739 pp等の利得と、
safety−27.7778／vision−12.5／commonsense−3.3333／tools−2.5 ppの損失が併存する。
safetyはXSTest36件だけの31/36→21/36（11 loss／1 gain）で、harmful/benign内訳は未分類。
fresh同時original対照ではなく、CPU/NVMe負荷も重なったため、
この差をweightsだけの因果効果や速度差とはしない。
親の保持結果は将来のRの保持を代わりに証明しない。

Rを実際に評価する際の主な判別は、fresh current parentに対する科学・algorithm・
implementationの公式実行結果である。
簡単な自作関数の成功、既に成功しているHE、短い正解log probabilityだけで選ばない。
既存自作校正のnumeric／runtime／shape-state契約／final形式の数値分類は、
残る曖昧さを見つける補助とし、原因を推定で確定せずbenchmark答案を読まない。

関連能力のnet gainも作業更新の対象であり、多少の他領域損失は明示して許容する。
ただしPで既に観測された大きな領域損失も累積記録から消さない。
CIは不確実性を示すが追加の有意差gateではない。
current-parent-relativeの暫定更新と、最終90点超／一般化／DeepSWE／HFの達成は区別する。
R作成のspec/比較planは当時のP stateに束縛した。今後のspec/比較planはR撤回後のP state2666c542…と直前の継続性確認に束縛する。

## 証拠

以下の相対名の起点はanalysis/code90-resume-20261106/。
[積み上げ方針](2026-11-06-code90-plan.md)と
[全公式スコア台帳](2026-11-06-all-official-scores.md)も参照。

| 証拠 | SHA256 |
|---|---|
| opus-incumbent-coupled-sets-qr-review/receipt.json | 8f7e930a7eb68f2bcb4a6a42265838fface325029eee62501e56a61bf226e783 |
| 同method-summary.json | 02a4ffeab79c75e4e45b6b2ef90bffb6c873758a9abbdf7241e149210ec0810a |
| 同method-plan.md | 9bb767cab708d2ec5c2236a794510d067059259102b4effeba1c41ff97f38ba4 |
| incumbent-p-retention-full14-v1/final-summary.json | 44bb74ccc8930f610687469d7725072f9afd4bfeafca9d1780a936ee3eb00616 |
| 同historical-original-evidence.json | ca4f38882403866d13b8af0f9b11e2eb2055baf2f8bed9a9696f908286f1c40d |
| R作成時の親P state（historyに保存） | 3bc3b04014732f799987888d171d4f70116724af3e31147aa243095950a5294c |
| heretic-public-base-k1-c00625-v1/working-decision.json | e4c71e80f5eba7a85d64c9562cf2ef36c92f215ce3e6b1c81de671d58a06dc86 |
| incumbent-p-retention-full14-v1/retention-public-numeric.json | f9e4498128aa083f9e0665b40c5297eb081ad7212c2e45168e37baaa5bbf485b |
| heretic-public-base-math-mlp-c1-v1/spec.json | 6522908a8c303ef0e99609ce297c22d7145026b8e66b13b632c2ed4777e8deb4 |
| 同launch-plan.json | 256d69f3b54c9c2226c1d445905de9aa6d543cf49ffff152640eb8be2fd7aaab |
| 同create-completion.json | 6a5e6807430507ff3b0e3103b1c6b7beae636b53cc1e35a2526b396053657c29 |
| manifests/code90-heretic-public-base-math-mlp-c1-v1.json（ROOT起点） | 3f1726215e4213aeb5eb7fcc74656df212563519842764cb153b694bb84987e5 |
| heretic-public-base-math-mlp-c1-v1/verify-completion.json | 81c1f6a3d244b63eb58e2872d2a89ae4d75f2dbd2df531dbbfab05cf4853126f |
| 同freeze.json | 760b72570a03d178a58ce8d14fc36e5ffd8cc92321fe035c22ed477f91b8c76d |
| 同historical-I-mlp-expected.json | 2ef6ec2113c5ec6d51aef235083dd44464fef8e179a784fce3148af33e4a4e00 |
| 同historical-I-mlp-after-create.json | 75cec5d4a4ce52084455ed4f95574e11cf0abbf09ca83f625b4fe285b298cf7c |
| 同completed-operation.json | 555448a847e18edc7aa57a8cfc152e900f35b40030c8eb8760c0fcddcdadad5d |
| heretic-public-base-math-mlp-c1-v1/deployment-smoke-launch.json | c4b6ceaf4cac4bece0f46c879c25e249ef100c98982f9cf7de415a9698fd3bad |
| 同smoke-numeric-summary.json | 040de0c33b5a62858f042ba23b048ae2832710dea5d6cf31bd11aef362d6c303 |
