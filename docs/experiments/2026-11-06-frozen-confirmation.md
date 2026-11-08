# 0397確認評価：固定候補の比較結果と運用上の除外（2026-11-06）

この追記は確認評価の運用記録です。候補0397は1,490項目すべて採点済み・未採点0で完了しました。公式対応比較と項目別の得失を以下に記録します。確認結果を重みの選び直しや追加調整には使いません。

実行・集計の仕様は[WSPガイド](../wsp.md)、事前取得ツールは[課題イメージの事前取得](../task-image-prefetch.md)を参照してください。

## 凍結と実行対象

候補 `resume20261106-h-grid-0397` は **2026-11-05 19:04:10 UTC** にdevの証拠だけで選択凍結しました（9,127要素・90tensor、ΔL2=0.0004125233976985216）。その後の確認結果で候補を変更しません。目的はコード生成を主、tools・instructions・agents・reasoningを副とし、他領域の多少の損失も含めて個別に報告することです。

元モデルrunは `gemma4-confirmation-base-1491-20261106`。元の固定manifestは1,491項目で、SHA-256は
`52a76039574df920138e7b88247955859af89865b2aa565fb11b00a0682177a6` です。公式full tier30ではなく、metadataに基づく確認subsetです。

元runは2026-11-05 **19:20:13–20:00:35 UTC**（JSTでは11月6日04:20:13–05:00:35）、公式記録のwall時間は2,420.7秒、読み込み時間は98.5秒でした。**1,491項目の記録を保持し、1,490採点・1未採点、status=partial** で終了しています。非agent 1,483項目、SWE-bench Pro 4項目、Terminal-Bench 3項目の既存結果を再生成せず、そのまま公式経路で保存しました。

## mtebの取消と共通1,490項目

残った `terminal-bench:mteb-leaderboard` の課題定義はagent 3,600秒、既存WSPハーネスは手数250とagent timeout倍率48を使っており、停止検知用のwall上限が **172,800秒＝48時間** になっていました。これは今回の時間方針に合いませんでした。1回のモデル呼び出し上限1,200秒とは別の設定です。

運用担当が20:00:20 UTCにHarbor子へ1回だけSIGTERMを送り、公式の取消・出力・cleanup処理と親WSPの部分採点を完了させました。mtebは40手まで進行していました（公式metadataの完了API計時40件、`n_episodes=41`とは区別）。最終公式記録は **CancelledError、verifier_result=null** です。公式timeout、採点済み不正解、0点とは扱いません。元の3件のTB完了resultは停止前後でSHA一致が確認されています。

候補run `gemma4-confirmation-h0397-1490-20261106` は **2026-11-05 20:14:55 UTC**（JST 11月6日05:14:55）に開始しました。最終 `run.json` は `status=completed`、`items=1490`、未採点0、上記と同じmanifest SHAで、除外IDも記録済みです。終了は **20:35:20.644781 UTC** です。起動側kairyu-bench commitは `3c9bfd07fac0b3c53e2de0898a6a656849d160e3` です。

候補側は同じ元1,491manifestを使い、実行選択にだけ
`--exclude-item terminal-bench:mteb-leaderboard` を指定しました。manifest本体・SHA・問題条件・公式採点は変えず、`run.json.excluded_item_ids` に除外を記録します。元partial 1,491記録と候補1,490記録を既存 **`wsp compare --partial`** で比較し、両方で採点できた共通1,490項目を対象としました。比較器とmanifest SHA一致guardは変更していません。共通採点数は1,490（能力項目1,426＋分布probe 64）と確認しました。元側のみ残るmtebはagentsの未対項目1件として表示され、0点補完していません。

この除外は実行開始後の時間設定不適合に対する運用変更で、元の1,491項目契約からの変更です。最初から1,490項目を事前登録していたとは書きません。除外した課題の能力への適用性は失われ、変更時点では元側の部分結果が存在していたことも限界として残します。重みの再選抜は行いません。

## 元モデルの公式集計

0–100点。各領域は公式source平均、総合は14領域の平均です。SciCode等の連続得点を保持し、全項目を単純な正解率に置き換えません。

| 領域 | 元モデル点 | 採点済み/予定項目 |
|---|---:|---:|
| 一般知識 | 84.8739 | 119/119 |
| 専門知識・科学 | 90.6250 | 32/32 |
| 事実の記憶 | 78.7500 | 80/80 |
| 数学 | 77.2222 | 57/57 |
| コード生成 | 77.4286 | 64/64 |
| 指示追従 | 93.0357 | 115/115 |
| 関数呼び出し | 82.5000 | 80/80 |
| エージェント作業 | 62.5000 | 7/8 |
| 長文脈 | 79.9158 | 38/38 |
| 論理推論 | 79.1304 | 43/43 |
| 読解・常識 | 63.5000 | 600/600 |
| 多言語・日本語 | 78.5256 | 87/87 |
| 画像理解 | 78.1250 | 32/32 |
| 安全性 | 93.0556 | 72/72 |

補足総合は **79.9420点**。14領域すべてに採点結果があるため公式reportは値を出しますが、未採点1件を含む元runを全1,491件完遂と呼ぶものではありません。上表に加え、能力得点へ入れない分布probe 64項目も採点済みです。

主目的の内訳はHumanEval+ **18/20**、MBPP+ **16/20**、LiveCodeBench **12/14**、SciCode **10項目の平均0.54**。この4source平均がコード **77.4286点** です。関連分野はBFCL **66/80**、IFEval **78/80**、IFBench **31/35**。エージェントはTB **3/3採点済み（予定4、1未採点）**、SWE-bench Pro **1/4** で、source平均は **62.50点**（単純poolの4/7とは異なる）です。候補との差は、以下の共通項目による対応比較で示します。

## 凍結候補との公式対応比較

主指標のコード領域は **77.4286→82.5476点、差+5.1190点、95% CI [+0.8333, +10.6548]** でした。事前に主指標としたこの固定集合・今回の単回生成では改善を支持します。ただし一般的なcoding能力向上や、差が重みだけに因果的に由来することを確定するものではありません。候補の再選抜・再調整は行いません。

全14領域を同じ公式source平均・source内対応bootstrap（2,000回、seed 20261104）で比較しました。下表の「獲得/喪失」は公式の0.5閾値を跨いだ件数で、fractional scoreの増減件数とは区別します。

| 領域 | 元 | 0397 | 差（点） | 95% CI（差、点） | 獲得/喪失 | 対項目 |
|---|---:|---:|---:|---|---:|---:|
| 一般知識 | 84.8739 | 87.3950 | +2.5210 | [-0.8403, +6.7227] | 4/1 | 119 |
| 専門知識・科学 | 90.6250 | 90.6250 | +0.0000 | [-9.3750, +9.3750] | 1/1 | 32 |
| 事実の記憶 | 78.7500 | 77.5000 | -1.2500 | [-3.7500, +0.0000] | 0/1 | 80 |
| 数学 | 77.2222 | 80.0000 | +2.7778 | [-1.8519, +9.2593] | 1/0 | 57 |
| コード生成 | 77.4286 | 82.5476 | +5.1190 | [+0.8333, +10.6548] | 3/0 | 64 |
| 指示追従 | 93.0357 | 95.2679 | +2.2321 | [-3.4821, +8.5714] | 4/3 | 115 |
| 関数呼び出し | 82.5000 | 81.2500 | -1.2500 | [-3.7500, +0.0000] | 0/1 | 80 |
| エージェント作業 | 62.5000 | 62.5000 | +0.0000 | [+0.0000, +0.0000] | 0/0 | 7 |
| 長文脈 | 79.9158 | 78.4757 | -1.4401 | [-10.7605, +7.9167] | 1/2 | 38 |
| 論理推論 | 79.1304 | 79.4565 | +0.3261 | [-6.5217, +7.5000] | 1/1 | 43 |
| 読解・常識 | 63.5000 | 63.0000 | -0.5000 | [-1.6667, +0.6667] | 5/8 | 600 |
| 多言語・日本語 | 78.5256 | 78.2853 | -0.2404 | [-6.1699, +5.6891] | 3/3 | 87 |
| 画像理解 | 78.1250 | 75.0000 | -3.1250 | [-15.6250, +12.5000] | 2/3 | 32 |
| 安全性 | 93.0556 | 93.0556 | +0.0000 | [+0.0000, +0.0000] | 0/0 | 72 |

補足総合は **79.9420→80.3113点、差+0.3693点、95% CI [-1.2408, +2.1172]** で、区間は0を含みます。独自の加重総合や成功gateは追加しません。agentsの対項目は7件で元側mtebが未対1件です。TB3/3・SWE1/4は両モデルで同じ項目が正解し、得失はありません。

### コードと関連source

| Source | 元 | 0397 | 差（点） | 95% CI（差、点） | 項目 |
|---|---:|---:|---:|---|---:|
| humaneval-plus | 90.0000 | 95.0000 | +5.0000 | [+0.0000, +15.0000] | 20 |
| mbpp-plus | 80.0000 | 85.0000 | +5.0000 | [+0.0000, +15.0000] | 20 |
| livecodebench | 85.7143 | 92.8571 | +7.1429 | [+0.0000, +21.4286] | 14 |
| scicode | 54.0000 | 57.3333 | +3.3333 | [+0.0000, +10.0000] | 10 |
| bfcl | 82.5000 | 81.2500 | -1.2500 | [-3.7500, +0.0000] | 80 |
| ifeval | 97.5000 | 96.2500 | -1.2500 | [-5.0000, +2.5000] | 80 |
| ifbench | 88.5714 | 94.2857 | +5.7143 | [-5.7143, +17.1429] | 35 |
| terminal-bench | 100.0000 | 100.0000 | +0.0000 | [+0.0000, +0.0000] | 3 |
| swe-bench-pro | 25.0000 | 25.0000 | +0.0000 | [+0.0000, +0.0000] | 4 |
| bbeh | 78.2609 | 73.9130 | -4.3478 | [-13.0435, +0.0000] | 23 |
| musr | 80.0000 | 85.0000 | +5.0000 | [+0.0000, +15.0000] | 20 |

コードはHumanEval+ 18→19/20、MBPP+ 16→17/20、LCB 12→13/14、SciCode平均0.54→0.573333です。得点上昇4項目・低下0項目で、うち公式0.5閾値の獲得は3件です。SciCode24の0→1/3を丸ごと正解1件とは数えません。toolsはBFCLの1件喪失、instructionsはIFEvalの−1.25点とIFBenchの+5.7143点があり、関連能力が一律に上がったとは言えません。

### 変化した全項目

能力項目1,426件のうち52件で得点が変化し、**上昇27・低下25、変化なし1,374** でした。公式0.5閾値の獲得25・喪失24とは別の数え方です。以下はIDと公式得点だけを示し、問題本文・推論・正解patchは含めません。

| Source | 得点上昇（元→0397） | 得点低下（元→0397） |
|---|---|---|
| arc-challenge | `arc-challenge:TIMSS_2003_8_pg117` 0→1 | `arc-challenge:Mercury_SC_LBS11003` 1→0 |
| bbeh | なし | `bbeh:zebra_puzzles:198` 1→0 |
| bfcl | なし | `bfcl:multi_turn_base_147` 1→0 |
| gpqa-diamond | `gpqa-diamond:0053` 0→1 | `gpqa-diamond:0032` 1→0 |
| hellaswag | なし | `hellaswag:01627` 1→0; `hellaswag:09493` 1→0 |
| hmmt-feb-2026 | `hmmt-feb-2026:18` 0.25→1; `hmmt-feb-2026:9` 0.5→0.75 | `hmmt-feb-2026:30` 1→0.75 |
| humaneval-plus | `humaneval-plus:HumanEval/163` 0→1 | なし |
| ifbench | `ifbench:120` 0→1; `ifbench:200` 0→1; `ifbench:89` 0→1 | `ifbench:184` 1→0 |
| ifeval | `ifeval:3186` 0→1 | `ifeval:1307` 1→0; `ifeval:3710` 1→0 |
| jamc-qa | `jamc-qa:jamcqa-test-culture-00335` 0→1; `jamc-qa:jamcqa-test-geography-00001` 0→1 | `jamc-qa:jamcqa-test-regional_identity-00152` 1→0 |
| livecodebench | `livecodebench:abc394_f` 0→1 | なし |
| longbench-v2 | `longbench-v2:6703a0ecbb02136c067cd11b` 0→1 | なし |
| mbpp-plus | `mbpp-plus:Mbpp/279` 0→1 | なし |
| mmlu-pro | `mmlu-pro:10783` 0→1; `mmlu-pro:2092` 0→1; `mmlu-pro:7568` 0→1; `mmlu-pro:8938` 0→1 | `mmlu-pro:974` 1→0 |
| mmlu-prox-ja | `mmlu-prox-ja:1656` 0→1 | `mmlu-prox-ja:1509` 1→0; `mmlu-prox-ja:9125` 1→0 |
| mmmu-pro | `mmmu-pro:standard:test_Pharmacy_280` 0→1; `mmmu-pro:vision:test_Diagnostics_and_Laboratory_Medicine_18` 0→1 | `mmmu-pro:standard:test_Biology_138` 1→0; `mmmu-pro:standard:validation_Clinical_Medicine_1` 1→0; `mmmu-pro:vision:validation_Energy_and_Power_14` 1→0 |
| mrcr-v2 | なし | `mrcr-v2:2277` 0.99896266→0.030520646 |
| musr | `musr:object_placements:109` 0→1 | なし |
| ruler | なし | `ruler:fwe:131072:006` 1→0 |
| scicode | `scicode:24` 0→0.33333333 | なし |
| triviaqa | なし | `triviaqa:sfq_9927` 1→0 |
| winogrande | `winogrande:0597` 0→1; `winogrande:0623` 0→1; `winogrande:0815` 0→1; `winogrande:0912` 0→1 | `winogrande:0227` 1→0; `winogrande:0387` 1→0; `winogrande:0470` 1→0; `winogrande:1103` 1→0; `winogrande:1131` 1→0 |

RULERの `ruler:fwe:131072:006` は元1→候補0。元は `finish_reason=stop`、出力1,519 tokens、187.314秒、候補は `finish_reason=length`、出力32,768 tokens、1,108.457秒で、両方の入力は126,539 tokens、出力上限32,768でした。候補は最後の生成完了を待つ要因となり、公式の出力上限到達ルールで0点です。これはmtebの外部取消・未採点とは異なります。本文を読んだエラー原因の推定はしていません。

### 改善したコード課題の内容

公開問題文で要求を確認した範囲では、HumanEval/163は区間内の偶数の1桁数字を昇順で返す境界条件処理、Mbpp/279はn番目の十角数の計算です。LiveCodeBenchのabc394_fは、最大20万頂点の木から、各頂点の次数が1または4で次数4の頂点を含む最大部分木を求めます。SciCode24は非粘性1次元Burgers方程式を有限体積法と陽的Euler法で数値計算する課題で、3段階中1段階が改善しました。これらは得点が変わった問題の要求であり、特定の重みがその能力を担当すると割り当てる証拠ではありません。

### 分布probeと所要時間

以下のNLLは公式compareの共通positionで集計した値です。単独reportは各項目内の平均NLLを求めてから項目を等重みで平均し、compareは共通項目の全positionを等重みで平均します。今回のprobe-codeは両方とも同じ28項目・1,596 positions（項目長21〜129）です。単独reportの **3.660257→3.691530 nats/token** と、対応比較の **3.042947→3.066062 nats/token** の違いは、この重み付けによるものです。比較側で項目を落とした結果ではなく、数値を混用しません。WikiText・日本語Wikipediaの各文は512 positionsなので今回の両平均は一致します。KLは上位語と残りmassをまとめた下限で、全vocabularyの厳密KLではありません。

| Probe | Positions | NLL 元→0397 | 差 | KL/token下限 | Top-1一致 |
|---|---:|---:|---:|---:|---:|
| probe-code | 1,596 | 3.042947→3.066062 | +0.023115 | 0.006662 | 98.183% |
| probe-wikipedia-ja | 14,336 | 7.414620→7.471774 | +0.057154 | 0.082016 | 91.964% |
| probe-wikitext | 4,096 | 8.900103→9.084036 | +0.183933 | 0.081132 | 91.870% |

3種類ともNLLは増加しました。probeは分布変化の診断であり、コード実行テストの改善や他領域の損失の代替判定には使いません。

候補の公式wall時間は **1,225.1秒（20分25.1秒）**、generationは1,204.2秒、load記録は95.4秒でした。RULER sourceのgeneration完了は開始から1,204.1秒です。元のwallは2,420.7秒ですが、元はmtebの長い実行と取消を含み、候補はmtebを実行していないため、半減を重み編集による高速化と解釈できません。

補助のrepeat/range evidenceツールはdev専用で、test確認runへの入力は入口のguardが拒否しました。guardを変更・迂回せず、この確認結果をdev履歴へ入れていません。公式比較と、許可されたitem ID・score・sample_scores・finish_reason・usageだけの単発要約 `paired-outcome-summary.json` を使っています。追加の推論・再採点はありません。

## 比較の適用範囲

- 今回の元確認runは通常配信で、起動コマンドに `--observe-config` はありません。凍結契約は候補も通常配信・native thinking ON・公式generation既定・同じ項目ごとの出力上限としています。候補起動コマンドにも `--observe-config` はなく、起動時の対応を確認済みです。完了時の配信・重みidentity照合は別の実行証跡で扱います。
- 過去のactivation観測付きdev runと今回の通常配信runは、入力集合、計測経路、負荷や実行時刻が異なります。その差を重みだけの効果や観測による因果効果と断定しません。主たる対照は今回の同じ通常配信条件・共通subsetの元モデルと凍結0397です。
- この集合は公式full tier30でも、完全に未見・学習データ非混入と保証された集合でもありません。既存metadata監査の限界と、元1,491項目に1groupの重複提示が残ることを引き継ぎます。使用済みtestを未見に戻したとは表現しません。
- 公式対応bootstrapはsource内2,000回・seed 20261104。単回生成の再標本化は独立再実行・負荷差・意味的重複まで評価しません。コード差の符号とCI、関連領域の得失を示し、微小差やCIが0を含む差を能力改善確定とは呼びません。事後の独自加重overallで成功条件を作りません。
- 最終DeepSWEは同じ凍結候補で113問題・32並列・1周、対応する完了元runとの公式正解数比較です。HF判定は全113採点済みかつ元の正解数を厳密に上回ることが条件で、今回の部分WSP集計やprobe改善では代替しません。歴史的な元DeepSWE開始時のserving image ID欠測も別途残します。

## 実行環境と事前取得の比較範囲

元・候補とも全4台のvLLM replicaで、配信image IDは同一の
`sha256:af9e746b913088cbf6c4464a31d6904a6b7e1d3ba62f0895362ddf13595f7abe`、
起動command SHA-256は同一の
`5d344976092508daabe1058bf41d3285bbfdffd5b13609f211777e48fa9256f2` でした。
各起動前確認で4台すべてhealthy、元または凍結候補それぞれのcheckpoint treeに一致しています。これは今回の配信環境の比較であり、過去の全runの同一性を保証しません。

外側benchmark runnerは、元が `c72ff2757fd32bdc398a253453929e2947c0c580`、候補が `3c9bfd07fac0b3c53e2de0898a6a656849d160e3` で、項目除外等の追加によりソースcommitが異なります。保存build logでは **38 distributionのversion表記が一致**（うち外部依存37、kairyu-bench自身1）し、Python base image digestも同一でした。Dockerfile・依存宣言・公式ハーネスlockの比較対象に変更はありません。

ただし旧runner imageは保存image IDで取得できず、両imageのlayerやpackage内容のbyte一致は検証できません。これはbuild log上のversion比較で、完全な `pip freeze` やinterpreterの実行結果ではありません。kairyu-benchのversion表記は両方0.1.0でもソースは同一と扱わず、共有cache内の公式ハーネスvenvも歴史的な全installed bytesの一致を証明したものではありません。

後続DeepSWEのため、既に取得済みの113課題image refを新しい事前取得ツールで再検証しました。**113/113のRepoDigestsが元R3の保存registry digestと一致**し、各attempt配列は空です。検証guardはDocker `info` と `image inspect` だけを許可し、実記録はinfo 115回・inspect 113回、pull 0、container/model操作0でした。9件のfake Docker契約テストも通っています。これはcacheの照合であり、新しいDeepSWEの実行・成績ではありません。

## 証拠

以下は実験保存先内の相対パスです。大きい実行artifactはリポジトリには同梱しません。

- `analysis/final-h0397-confirmation-20261106/{selection-freeze,checkpoint-freeze,runtime-base-preflight}.json`、`decision-contract.md`
- 同ディレクトリの `operational-exclusion.json`、`runtime-h0397-preflight.json`、`runner-comparability.json`
- `runtime/resume-20261106/confirmation-base-command.json`、`confirmation-h0397-command.json`、`confirmation-base-mteb-cancel.json`
- `results/gemma4-confirmation-base-1491-20261106/{run.json,report.json,items.jsonl,responses.jsonl}`
- `results/gemma4-confirmation-h0397-1490-20261106/run.json`：開始・1,490項目・明示除外metadata
- 同元runの `agents/terminal-bench/raw/terminal-bench/jobs/terminal-bench-terminal-bench/mteb-leaderboard__seRdaua/result.json`
- `runtime/resume-20261106/exclude-item-checks.{json,log}`：選択・実行・metadataの新2契約を含む14 CPUテスト、Ruff、mypy合格。実モデル呼び出しなし。

- `analysis/prefetch-tool-port-20261106/{validation.json,cached-apply/receipt.json}`
- `analysis/final-h0397-confirmation-20261106/paired-comparison.{json,md}`、`paired-outcome-summary.json`

この追記では公式集計・凍結・起動・取消・既存検証metadataだけを使用し、推論本文・正解patch・新しい生成・再採点には触れていません。確認結果による再選抜は行わず、後続の同じ凍結重みによるDeepSWEは別の記録として扱います。
