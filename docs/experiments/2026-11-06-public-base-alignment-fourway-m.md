# M: 公開改変baseと固定adapterの4条件比較・事前契約（2026-11-06）

**固定4条件の公式dev90は全360項目の採点が完了した。Pを次の作業モデルとして保持し、Mには更新しない。事前契約と途中記録は保持し、最終結果・方針変更を末尾に追補した。**
[Lの公式dev90](2026-11-06-k1-exported-text-c1-review.md)はfresh original比−3.517857 pp、
出力token約1.47倍で不採用だった。次は係数を探索せず、同じexported-unit text LoRAの
効果が公開改変baseに依存するかを4条件で区別する。
Lの失敗をbase不一致の証明とせず、公開baseへの移植が成功するとも仮定しない。

## 公開baseの固定と分かっていないこと

対象Pは [coder3101/gemma-4-31B-it-hereticの固定revision](https://huggingface.co/coder3101/gemma-4-31B-it-heretic/tree/ea846e2b6d58fc9f943c462c19a1b60891407bf6)、
ea846e2b6d58fc9f943c462c19a1b60891407bf6（公開履歴2026-04-09）。
これはK1 adapter upload（2026-04-12）より前の公開revisionだが、
adapter configのtraining revisionはnullであり、真の学習weightsやローカル変更は不明である。
最新公開headのtemplate変更を学習時templateの証拠にしない。

既存の公開metadata＋全header監査では、1188 stored tensorの名前・BF16 dtype・shape・
サイズがoriginalと一致し、tensor payload合計は62,546,177,752 Bだった。
shard分割とindex weight mapは異なるため、公開Pのindexを保持する。
header照合は値の同一性や能力の互換性を証明しない。
pre-adapter revisionと監査時headの公開LFS OIDは一致するが、
それも実download後のlocal fullhash検証の代わりではない。

公開configの構造差はtext_config.moe_intermediate_size=nullの有無だけ。
tokenizer.jsonとgeneration_config.jsonはoriginalとbyte同一だが、
chat_template.jinjaとtokenizer_config.jsonは異なる。
「tokenizer全設定が同一」「著者protocolを再現」とは述べない。

| 公開P weight file | Bytes | 公開LFS SHA256（local検証前のpin） |
|---|---:|---|
| model-00001-of-00002.safetensors | 49,923,154,850 | 57338b5e4a55441e20853ce7aedb3d93b6fee5f5095ac67e2610d6cbd3206f38 |
| model-00002-of-00002.safetensors | 12,623,183,414 | 607d371a9e209fecbd12d53256fe8052f340c6ea8ce3218553c62ac95a678b2e |

## 固定する4条件と推論protocol

| Arm | 保存weights | 意味 |
|---|---|---|
| O | 検証済みoriginal BF16 | 基準 |
| O+D | 既存Lと同じ固定text D、exported coefficient1 | originalへのcross-base transfer |
| P | 公開固定BF16 baseのみ | 公開baseの効果 |
| P+D | 同じtext DをPから一度merge | baseによるadapter効果の変化 |

DはLでlossless抽出したF32 text820 factors／410 pairsで、vision378 factorsを明示除外したもの。
rank32、alpha32、普通LoRA scale1、係数1を維持し、再fitや振幅gridは行わない。
P+DはPを直接基点に作り、丸め済みO+Dへ後からbase差分を加えない。
同じ因子でもbaseのBF16格子が異なるため、保存後の実deltaは各baseに条件付けられる。

全条件でoriginalのconfig、generation_config、tokenizer.json、tokenizer_config、
chat_template、processor_configをbyte同一で使用する。
公開Pの元metadataはprovenanceとして別に保存し、書き換えて著者の配布物と偽らない。
これは**元protocolに固定した公開base view**であり、Heretic/K1著者のdeployment再現ではない。
native thinking ON、公式sampler、caps、scorer、direct TP2と各c32の比較条件を揃える。
4armは同じ比較窓で測り、過去Lのscoreを新しい4条件のO+D実測として代用しない。
正式run ID・endpoint・tree・counter bindingは実作成／配信後の実証跡で固定する。

## 事前に固定する比較と解釈

主比較は **P+D − O**。公式codeの4-source等重み平均を維持する。
同じ固定90項目（HE30／MBPP30／LCB20／SciCode10）を全armで採点し、
新しい未加重overallや成功宣言用の重みを作らない。
known64の90超という目標をこのdev90で置き換えない。

副比較はP−O、(O+D)−O、(P+D)−P。
相互作用は、各source内の同じpaired項目について
[(P+D)−P]−[(O+D)−O]を求め、4-source等重みで記述する。
不確実性を示す場合もpaired/source構造を保ち、
独立性の根拠なしに4armのvarianceを単純加算しない。
公式paired CIを優先し、相互作用は機序の証明や自動採用gateにしない。

SciCode部分点、binary flip、成功集合のunionを区別する。
best soloを上回る累積改善に、union外の新成功を必須としない。
全arm単回stochastic出力での個別得失を因果的なmodule機能へ帰属しない。
正の相互作用も真のtraining base一致や「DにはPが必要」の証明ではない。
Pの修復・悪化、template条件との相互作用はこの設計だけでは分離できない。

既存authored24は構造smokeであり目的指標ではない。
長さ、cap、unknown、形式、実usage、wallをofficial scoreと併記する。
保存callと起動discovery probe、独立HTTP回数、思考tokenを混同しない。
追加logit probe、独立GPU oracle、seed/grid、著者templateの追加armをこの計画だけで必須化しない。
新しい承認待ちや全領域非退行基準を設けず、coding優先・関連能力と多少の他分野損失を個別に報告する。

## 最小のbytes/provenance契約と現在のplan

base-view planはheretic-public-base-official-protocol-v1/plan.json、SHA
00b3d5c5f2016ac9d9ade5c2ddf88b367c2ea9581418d3191013c9bb79e41182。
plan作成時のstatusはprepared_not_executedで、base viewの取得・検証用入力を固定したもの。
rootはこのSHAの取得planを承認済み。これはdownload完了や4armの実配信／公式run完了を意味しない。

公開Pの2weight filesを指定LFS SHAでlocal検証し、独立ファイル1コピーと公開indexを保持する。
上記6つの元protocol metadataをコピーする計画で、original weightsのhardlink/copyは行わない。
公開weight filesは計62,546,338,264 B。viewは2shards＋公開index＋6元metadataの9files、
予定checkpoint容量62,578,656,419 B、300 GiB reserve込み必要空き384,701,203,619 B。
部分downloadは完成品とせず失敗receiptと共に保持する。

後続mergeは同じDの元／派生hash・ordinary scaling・numerical code/kernelを固定し、
baseごとに保存BF16の実変更数/norm、非対象span、全file hashを記録する。
target数と実変更tensor数は丸めno-opがあり得るため同一と仮定しない。
既存の全CPU recipe verificationとfreeze/stat継続を再利用し、
余分な第三GEMMや新しいfingerprint測定を追加しない。
この文書作成ではweight download、候補作成、配信、モデルcallは実行していない。

## Opus 5.5方法レビュー

匿名の公開一般機構だけを1回送信し、actual assistantとusageはclaude-opus-5-5のみ。
json_reviewsのM一致、verified_requested_model/review_usableともtrue、tools0。
67.5981秒、$0.135252。privateスコア・norm・候補ID・座標・課題データは未送信。
公開finalをreceiptに保存し、raw stream／thinking blocks／環境は未保存。

4条件によるbase主効果とadapter/base相互作用の切分けを支持する意見だった。
input/output provenance、同じoriginal protocol、paired得失・不確実性は採用する。
変更tensor数=target数、追加logit指紋必須、追加revision/precision/seedや
未加重item平均への置換は、そのまま新要件にはしない。
方法上の妥当性は能力を実証したという意味ではない。

artifact名はanalysis/code90-resume-20261106/からの相対名。

| 証拠 | SHA256 |
|---|---|
| k1-training-base-feasibility-v1/summary.json | 8a070417f39d340cd2ab6fe26514bc0f9a1467a0e6cb10af4d09979627c24702 |
| 同pre-k1-tree.json | 5335b48c3495dd6d8c20c8984f83fc44e045a864ec81d29e5c39f96ca9f8a0e3 |
| 公開P index | 390b7f9a0a502e4526cf20862d22954c911cc318d8aba2cdb745c6fbd98264f6 |
| 公開P pre-K1 template | 2dfbfc7d538912f4ea11d29d85b4e25d7bc26386e53f57529f1d707c28b5828c |
| opus-base-alignment-fourway-m-review/receipt.json | 046a57848316a3bf2ad33d803538ee0de047056eaf71497071798b3f7db1dd26 |
| 同independent-summary.json | 745ba2e5731580aa5dcb13941c06cc4db700141a00b52c31ae8b20d56e2cb45d |
| 同REVIEW.md | 7892de96991b79454f39977e85d2b9167afa84867b4435933e1086aafa30b550 |
| 匿名prompt | 9e1a89412ba35c2f4d7377d975ba0c6296f2bd81b87239897ce0bf6952380de1 |

## 追補：P取得検証と自作24ケースの観察（2026-11-06）

公開Pの取得・検証は1123.742841秒（18分43.74秒）で完了した。
両weight filesの独立local SHA照合、1188 BF16 logical layout、
9file合計62,578,656,419 Bを確認した。元の公開metadataを別途保存したうえで、
事前指定の6つのoriginal protocol filesを使っている。
確定treeは3dfb3bd1db0ab2ba4fff843616bb0d3109e0b97a6c9349658620908f2a24bebc。
freezeは取得時のfull SHA証拠とstat継続を再利用し、freeze発行時に重みを再hashしたとはしない。
これは真のadapter training baseや著者protocolの再現証明ではない。

| 固定advanced-v2 smoke | 歴史original参照 | 公開P・元protocol |
|---|---:|---:|
| train | 15/16 | 14/16 |
| transfer | 8/8 | 7/8 |
| 合計 | 23/24 | 21/24 |
| evaluated／unknown | 24／0 | 24／0 |
| finish_reason=stop | 24/24 | 24/24 |
| 記録input tokens | 2,864 | 2,864 |
| 記録output tokens | 43,599 | 41,241 |
| parsed finalあり | 24/24 | 24/24 |

新規成功は0、追加失敗はcategorical-without-replacementとwhiten-recolor-rowsの2件。
前者は8試験中7件のruntime失敗、後者は8試験すべてのnumeric失敗である。
reflecting-wall-flightのnumeric失敗1/8はoriginalとPで共通だった。
公開回答や例外本文を読まず、既存sandboxのID・数値・失敗kindだけを確認した。
evaluatedで判定されたruntime失敗は、実行基盤のunknownへ付け替えない。

全24件でparse済みfinalがあり、追加format/syntax/missing/execution-unknownは0、
cap到達かつusable final無しも0だったため、事前の構造停止条件は発火しない。
これは意味的な2件の悪化を打ち消さず、公式WSPへ進める判断や能力改善を意味しない。
case SHA・ID順序・generation設定（期待されたcheckpoint tree差を除く）・
generator source/concurrency/timeout/retry・sandbox seed/trialsを元参照と照合した。
参照は既実施のoriginalであり、freshな因果比較ではない。
smoke pipelineの56.477秒はgenerationとexecutionを含み、benchmark throughputとは比較しない。
thought channelの開閉や独立reasoning token数は未計測のままである。

この追補時点ではP/O数値weight diffとM smokeは未完了で、正式4条件評価の判断も未発行。
重み差のmodule集中を機能や正誤の原因と結び付けない。

| 追加証拠（analysis/code90-resume-20261106/からの相対名） | SHA256 |
|---|---|
| heretic-public-base-official-protocol-v1/download-completion.json | 287023a320e82786cd093b1ae1a69d2fa054ca3aeece27801015f2cac23de357 |
| 同freeze.json | 104de676cd416ada1842649cf3f2d8e35d81bfac353238b530885f3f25645c45 |
| 同smoke-numeric-summary.json | 0152092e5081fabf05706e032325d0c9c8e5b81bea7ed57f21192ef0f0b8d842 |
| 同smoke-independent-review.json | bb17ee48b199b8c401142bab8f992c2d3bdba1f1a7e2e350b96ff074f2137497 |

## 追補：M smokeと正式4条件評価の開始（2026-11-06）

P+DのMは固定advanced-v2で22/24（train14/16、transfer8/8）だった。
歴史original23/24からはreset-running-statsが追加numeric失敗（8試験中5件）となり、
reflecting-wall-flightのnumeric失敗1/8は続いた。元参照への新規成功は0。
P単体21/24との記録上の比較ではcategorical-without-replacementとwhiten-recolor-rowsが回復し、
reset-running-statsを失った。これは自作検証集合内の観察で、WSPの改善ではない。

全24件がevaluated／stop／parsed finalあり、unknown0、追加構造失敗0、
capかつusable final無し0。固定構造停止条件は非該当だった。
記録inputは2,864、outputは88,693（歴史original43,599の2.03429倍）。
smoke pipelineは162.701秒で、生成と実行を含む。
全caseのsafe数値と11個の入力hashを既存report/receiptに照合し、
case/設定/source/seedのparityを確認した。回答本文や隠れた思考は読んでいない。
独立HTTP回数やreasoning token数の実測には置き換えない。

Mのtreeはa09697f0f7cd8f3b569499b8a40a5ca31393007d32fdec3c873b7544c6699361、
freeze SHAは41cf936187c47dbc2dab1786932aaf387abf1b5c38e996fc3e416a9019d4d9f4。
rootはP/Mの構造条件と数値を確認して固定4条件dev90へ進むと判断した。
開始時刻は2026-11-06T12:13:20.407725+00:00、観測時statusはrunning。
この追補には正式scoreを先取りして記載せず、weight geometryと能力の因果関係も推論しない。

| 追加証拠（同analysis起点） | SHA256 |
|---|---|
| heretic-public-base-k1-c1-v1/smoke-numeric-summary.json | b358969f8ff517ad1d7462fe5b7bf6372b878d6b6a1cc22ec30effe9af265c2b |
| 同smoke-independent-review.json | b5094a050ac99deadd933c91deb309ead5acfd72a13df1e70cea1e8fbc92cce8 |
| base-alignment-fourway-m-v1/structural-smoke-decision.json | 3fc10a834ac15a6dc306210fc7ebaa15c975f6fab8620ffa81069e230d2b1c55 |
| 同dev90-four-launch.json（開始時） | e27c3ac5427f7fd70ab789b90e29c1d43f79427733fd3bd1463ea6cde2145c2d |

## 追補：保存weightsで観測したP−Oの位置と大きさ（2026-11-06）

[必要統計に絞ったBF16差分ツール](../checkpoint-delta-footprint.md)の出力を、
tensor→family／layer→globalの順に独立再集計した。これは既存数値JSONの確認であり、
レビュー担当による追加weight scanやmodel callはない。

| P−Oの保存値観察 | 結果 |
|---|---:|
| 比較対象 | 全1188 BF16 tensors／31,273,088,876保存要素 |
| bitが変わったtensor | 58（残り1130はbit同一） |
| bit差／有限数値差のある要素 | 両方1,015,123,842 |
| 全保存要素に対する変更割合 | 3.245998% |
| 非有限値／非有限比較pair | 0 |
| ΔL2（FP64平方和から算出） | 25.575145366575246 |
| 変更familyと層 | text self_attn.o_projのみ、層1〜58 |
| 層48のΔL2² share | 42.6086% |
| 上位10層のΔL2² share | 72.5887% |

層0・59のo_proj、および他の保存tensorはbit同一だった。
したがってこのPの差を「全層のMLP変更」や「全parameterの一様変更」と説明しない。
全保存要素を分母にした3.245998%は変更数の割合であり、relative L2ではない。

Dの既存factor-space観測はK1を右側に固定した410 text projectionの連続unit差分である。
そのΔL2² shareはMLP（gate/up/down）80.7117%、全o_proj5.8536%、
Pが変えた58tensor内5.7323%だった。Pの変更58個はすべてDのtarget410に含まれるが、
それぞれをnormで並べた上位20moduleの集合の共通要素は0だった。
Pの保存BF16差分とDの連続factor差分という異なる量を比較している。
normの集中やtargetの包含から、方向cosine、相殺、機能の補完、科学コード能力を推論しない。
M−Oの累積normもこの2つのnormの加算や直交仮定から作らない。

この観察に基づく具体的な仮説は「attention出力だけにあるPの変更が、主にMLPへ広がる
固定Dの効果を元protocol下で変えるか」であり、事前指定4条件の相互作用で確認する。
58tensorでは範囲が重なるのでdisjointな編集とは呼ばない。
この追補で追加のlayer選択、係数探索、candidate作成は行っていない。

当初の包括的full-statistics scanは715.305秒時点で停止され、完成した数値結果はない。
全moments・quantiles・cosine等も計算する処理の進捗から長時間が見込まれたためである。
新しいfocused scanは125,092,355,504 Bのpaired payloadを読み、
数値処理83.239秒／外側wall83.388秒で完了した。
必要統計が異なるうえ旧処理は未完了なので、これを同等処理の速度倍率として示さない。
新scanは正式4条件runと時間が重なっており、独立したbenchmark latency測定でもない。

freshなのは全tensorの差分統計である。payload SHAの追加全scanは行わず、
既存の全file SHA証拠と全inventory/statの前後一致を用いた。
非有限値とsigned-zeroを区別する契約を維持し、今回は全要素有限でbit差数と数値差数が一致した。
base norm／relative norm／direction cosineはこのfocused観測では未計測である。

| 追加証拠（同analysis起点） | SHA256 |
|---|---|
| heretic-public-base-official-protocol-v1/fast-delta-footprint.json | a15a986cd376d738410e5ff6a1d9170e3e7ae0f75d0fbdcee74d8fc531cb5b7f |
| 同original-versus-public-summary.json | f54297e99764964a9236d64df8ade2a4fae12b491bb1df3df545f96f479ca7db |
| 同original-versus-public-D-scope-supplement.json | 507d170cb2a969279cd9e9d100e763d8bf9fa4cfc658ac991a8c9cab4a6a48f3 |
| 同numeric-independent-review.json | 22c800e26fa287bc3a76c05ef7f898258f97057a55a0f300503d0994eec0960c |

## 最終追補：4条件の公式結果と作業モデル更新（2026-11-06）

全4条件がcompleted、各90項目採点・未採点0・記録chat API calls119。
HE30／MBPP30／LCB20／SciCode10を同じmanifestで比較した。
以下は公式source等重みのcodeであり、項目数による加重平均や全領域overallではない。
SciCodeの部分点をそのまま保持した。

| 条件 | HE | MBPP | LCB | SciCode | code | report wall秒 | 記録output tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| O: original | 93.333333 | 73.333333 | 80.000000 | 38.079365 | 71.186508 | 336.5 | 343,128 |
| L: O+D | 96.666667 | 80.000000 | 70.000000 | 25.634921 | 68.075397 | 831.4 | 614,190 |
| P: 公開base | 100.000000 | 80.000000 | 75.000000 | 40.301587 | 73.825397 | 324.6 | 316,109 |
| M: P+D | 90.000000 | 86.666667 | 65.000000 | 29.968254 | 67.908730 | 841.4 | 524,575 |

| 比較 | 差（pp） | 95% CI（pp） |
|---|---:|---:|
| P−O | +2.638889 | [-1.666667, +6.944444] |
| M−O（事前指定primary） | -3.277778 | [-9.416667, +2.500000] |
| M−P（事前指定secondary／作業更新判断） | -5.916667 | [-12.500000, +0.250000] |
| (M−P)−(L−O)（探索的interaction） | -2.805556 | [-11.000000, +4.833333] |

公式pairは既存比較器のsource層化paired bootstrap（2,000回、seed20261104）を使った。
interactionも同じ全90行について各source内のrow抽出を4条件で共有し、
独立pairの分散を足していない。単回生成に条件付けた探索的比較であり、
反復生成の揺れや真の学習base、moduleの因果的寄与を証明しない。
CIは判断に付随する不確実性であり、今回の作業モデル更新に有意差を追加要求しない。

P−OではHE151・HE95・MBPP641・MBPP801が成功へ変わり、LCB abc399_cを失った。
SciCode72は5/9から7/9へ改善した。
M−PではMBPP103・126とLCB abc399_cを得たが、HE116・151・95、
LCB3777・abc390_g・arc195_bを失い、SciCode14・35・58の部分点も低下した。
部分点の正負と公式0.5閾値flipの件数は同じ指標として扱わない。

ユーザーは「改善したら改善後から積み上げる」と明示したため、
P（73.825397）を現在の作業モデルへ更新した。
M（67.908730）はP比−5.916667 ppなので更新せず、以後はPの保存weightsから積み上げる。
事前指定の科学的primary M−Oは書き換えず、既にsecondaryとして指定していたM−Pを
作業モデル更新の比較にも使う。多少の他領域損失を許容する方針を維持し、
全領域非退行やCI下限が正という新しい条件は課さない。

時系列は分けて残す。Mのreport終了は12:27:26.463233 UTCで、
方針監査12:28:24／M score初回閲覧12:28:35より前に結果は存在していた。
ユーザー指示はその初回閲覧より前だったが、恒久的な方針・working stateの保存は
全4report完成後である。「M結果が存在する前の事前登録」とは述べない。
Pのknown64・full14・DeepSWE確認は未実施で、90超達成やHF適格を意味しない。

今後の作成元はworking-incumbent.jsonのSHAとcheckpoint path/treeで固定する。
現在のtreeは3dfb3bd1db0ab2ba4fff843616bb0d3109e0b97a6c9349658620908f2a24bebc、
freeze SHAは104de676cd416ada1842649cf3f2d8e35d81bfac353238b530885f3f25645c45。
original Oは不変の参照として保持する。全期間の条件別スコアは
[全公式スコア台帳](2026-11-06-all-official-scores.md)、運用方針は
[コード90計画の追補](2026-11-06-code90-plan.md)を参照する。

4条件のrunnerは取得済み範囲でRootFSの16 layersとImage.Configが一致した。
wrapper build由来のimage ID差はそのまま記録し、取得していなかった全HostConfigや
重複environment keyの事後確認を推測で補っていない。
CPU weight scanとCIが実行時間に重なったため、wall差をweightsだけによる速度差としない。
serving counterは各120完了で、公式119 callsとの差1件・入力21／出力1 tokenは
別のwrapper readiness requestに対応する。counterだけから全itemへの帰属や他client不在を証明しない。

対象run ID：

- gemma4-code90-original-base-c32-dev30-code-pairm-20261106
- gemma4-code90-k1-exported-text-c1-c32-dev30-code-pairm-20261106
- gemma4-code90-heretic-public-base-c32-dev30-code-pairm-20261106
- gemma4-code90-heretic-public-base-k1-c1-c32-dev30-code-pairm-20261106

| 最終証拠（同analysis起点） | SHA256 |
|---|---|
| base-alignment-fourway-m-v1/official-fourway-summary.json | 51fade1ed59521995d748bc0bb219e303d323fee62778ced518daad4b0647073 |
| working-incumbent-policy-v1.json | 068993d7db2d7d5c44c4d7d955da2123daaa91cb580c872be8534cf2439cc51d |
| working-incumbent.json | 609190932c606e5b9219be072344f68fd2b8e2fdb695109eec63581ed8a75f21 |
