# R: 現作業モデルのattentionを保持するMLP数学方向（2026-11-06）

**範囲指定ツールの実装・独立レビュー・CPU検証は完了。Rの作成・配信・smoke・公式scoreはまだない。**
[公式N−P](2026-11-06-incumbent-coherent-step-n.md)はfresh P74.242063に対しN71.603175、
差−2.638889 pp（95% CI [−7.083333,+1.111111]）で、Nを採らずPを保持する判断となった。
Rの親はPに決まり、更新state SHAは3bc3b04014732f799987888d171d4f70116724af3e31147aa243095950a5294c。
sourceはcheckpoints/code90-heretic-public-base-official-protocol-v1、treeは
3dfb3bd1db0ab2ba4fff843616bb0d3109e0b97a6c9349658620908f2a24bebc。
作成直前にもSHA/path/treeを確認し、original Oを作成元へ戻さない。
Pの73.8254→74.2421は重み不変の再測定差であり、新しい重み改善とは数えない。

ツールのCPU suiteは507件収集、455 pass／52 optional skip、失敗0。前版506件から、r128/F32の範囲指定作成→独立verify→freezeと対象外byte保持を検証する1件を追加した。既定の410行列経路も維持し、Ruff／mypy73もPASS。証跡はanalysis/code90-resume-20261106/cpu-integration-structured-mlp-scope-v1/ci-results.json（SHA256 5a47f8fe58ec82f525b62f13c36afc11cd215f6839ab55893268a167b1d7b229）。これはツールの動作確認で、Rの能力確認ではない。

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

## 親の現在の観測と次に判別すること

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
具体的なspec/比較planは、更新済みP stateと直前の継続性確認に束縛する。

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
| 更新後working-incumbent.json | 3bc3b04014732f799987888d171d4f70116724af3e31147aa243095950a5294c |
| heretic-public-base-k1-c00625-v1/working-decision.json | e4c71e80f5eba7a85d64c9562cf2ef36c92f215ce3e6b1c81de671d58a06dc86 |
| incumbent-p-retention-full14-v1/retention-public-numeric.json | f9e4498128aa083f9e0665b40c5297eb081ad7212c2e45168e37baaa5bbf485b |
