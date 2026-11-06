# Code90：dev90の利得と既知64問への転移確認（2026-11-06）

**native案Bのdev90での利得は、既知64問では確認できませんでした。code >90は未達です。**
4条件のdev90と、B・0397対照の既知64問は全て完了し、各runの未採点・API errorは0でした。その後の元モデルとのfull14対応比較ではBに純損失があり、B・F・Gはいずれも採用していません。

[再開計画](2026-11-06-code90-plan.md)、
[科学コード校正と行列pilot](2026-11-06-science-matrix-pilot.md)、
[前回0397の固定確認](2026-11-06-frozen-confirmation.md)を併読してください。

## 比較条件と分母

Bはscience-native-contracts-v1、Fはscience-matrix-balanced-v2、
Gはscience-joint-matrix-numerics-native-contracts-v2。対照は0397です。
自作校正での微分・実BF16 finite測定から選んだ共同編集を、公式WSPの実行成功で測りました。自作16学習case・8転移caseの点数をWSPに加算していません。

| 集合 | split / tier | HE+ | MBPP+ | LiveCodeBench | SciCode | 各条件の採点数 |
|---|---|---:|---:|---:|---:|---:|
| dev90 | dev / 30 | 30 | 30 | 20 | 10 | 90/90 |
| 既知64 | test / 30、固定確認subsetのコード部分 | 20 | 20 | 14 | 10 | 64/64 |

dev90のmanifest SHA256はa3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b、
既知64は52a76039574df920138e7b88247955859af89865b2aa565fb11b00a0682177a6。
各比較内で全ID・分母・manifest SHAが一致します。既知64は以前に成績を見た参照集合で、独立した未見holdoutではありません。保存上のsplit=testは変更せず、dev専用の反復証拠ツールも使用していません。

各条件は **1つのTP2 replicaへ直接接続、chat concurrency32**。dev90は4条件を同時に、既知64はB・0397を同時に測定しました。native thinking ON、checkpointの公式generation設定を使い、今回だけのsampling変更はありません。起動元は共通commit 5e3f2df0c7f455013af3fa2d7abb3719858f808f。dev90の実outer containerは4つとも同じimage sha256:4fd52f9993e5d5a51918dd091958b6b408129f9b4d7c2f257ec4b116c53231f7で、runtime configと16 filesystem layersの一致をinspect証跡で確認しました。異なるbuild attestationログや事前buildのimage IDから推定したものではありません。

前回0397の既知64 **82.5476** は4 replica配信経由のfull14評価内の値で、今回の1 replica・コードだけの条件と異なります。今回の対照は **77.0119**。旧82.5476を新しいpaired baselineに置き換えたり、その差をbatch構成または重みだけの因果効果と決めつけたりしません。

## dev90：nativeの3問利得、科学コードは不変

codeは4 sourceの等重み平均です。SciCodeは各問題の成功段階率を10問で平均し、全段階の単純成功率や完全正解問題数とは区別します。数値は0–100点、差とCIはpercentage pointsです。

| 条件 | code | 対照差 [paired 95% CI] | HE+ | MBPP+ | LCB | SciCode |
|---|---:|---|---:|---:|---:|---:|
| 0397対照 | 70.4921 | — | 29/30 | 22/30 | 15/20 | 36.9683 |
| B native | 73.4087 | +2.9167 [0.0000, +6.6667] | 30/30 | 23/30 | 16/20 | 36.9683 |
| F matrix | 67.5754 | −2.9167 [−7.5000, +1.6667] | 28/30 | 23/30 | 12/20 | 40.3016 |
| G joint | 70.7698 | +0.2778 [−1.6667, +2.5000] | 29/30 | 22/30 | 15/20 | 38.0794 |

以下は公式scoreの変化です。0.5閾値を跨ぐ「flip」だけに縮約しません。

| 条件 | 利得 | 損失 |
|---|---|---|
| B | HE151、MBPP641、LCB abc390_g：各0→1 | なし |
| F | MBPP801：0→1、Sci35：1/3→2/3 | HE95、LCB3777・abc399_c・arc195_b：各1→0 |
| G | Sci35：1/3→2/3 | Sci72：7/9→5/9 |

Bの科学10問・39段階の合否は対照と全て同じでした。F・Gは35.2を回復（対照の例外hintはValueError）。Gは72.7・72.8を失い、両方のhintはAssertionErrorです。Gの公式threshold flipは獲得1・喪失0ですが、Sci72が0.5以上に残るためであり、退行なしを意味しません。

科学の合計段階成功数は対照18/39、B18/39、F19/39、G17/39。これは診断値で、上表の公式source平均とは別です。完全に解けた科学問題は各条件1/10、段階timeoutは全条件0でした。

## 既知64：利得と損失が入れ替わり、科学平均も増えず

| 条件 | code | 対照差 [paired 95% CI] | HE+ | MBPP+ | LCB | SciCode |
|---|---:|---|---:|---:|---:|---:|
| 0397対照 | 77.0119 | — | 18/20 | 16/20 | 12/14 | 52.3333 |
| B native | 76.4762 | −0.5357 [−8.5119, +7.3810] | 19/20 | 16/20 | 11/14 | 52.3333 |

| 種類 | 変化した項目 |
|---|---|
| 利得 | HE163、LCB abc399_d、MBPP279：各0→1。Sci21：0→1/3 |
| 損失 | LCB abc397_f・abc388_e、MBPP635：各1→0。Sci24：1/3→0 |

実scoreは4利得・4損失。公式threshold flipの3/3には、0.5未満のSci21・24の変化が入りません。科学の平均は同じでも、21.2回復と24.2喪失があります。

Sci15.2は **両条件とも公式300秒timeout**、0397対照の26.3もtimeoutでした。Bの26.3はValueError hintで不合格です。これらは公式採点器が失敗とした段階で、未採点API errorではありません。生成完了後の約300秒の待ちと整合しますが、sanitized metadataだけで環境・数値・アルゴリズムのどれが原因かは確定しません。正解・stderr本文・隠れたreasoningを使った再採点はしていません。

## 所要時間と判断の範囲

| 集合・条件 | generation秒 | wall秒 | leaf API calls | input / output tokens |
|---|---:|---:|---:|---:|
| dev90・0397 | 300.7 | 355.3 | 119 | 97,086 / 348,578 |
| dev90・B | 291.9 | 363.5 | 119 | 95,420 / 335,524 |
| dev90・F | 283.4 | 352.6 | 119 | 96,759 / 330,645 |
| dev90・G | 277.6 | 351.4 | 119 | 96,935 / 342,286 |
| 既知64・0397 | 255.5 | 560.8 | 87 | 64,916 / 280,427 |
| 既知64・B | 255.2 | 557.9 | 87 | 65,249 / 280,086 |

call数・token数は保存済みleaf response metadataに基づき、coverageは全件。SciCodeのaggregate sampleを二重加算していません。全callのfinish_reasonはstopです。呼出時間の単純和は並列wall時間とは別の値です。

公式paired bootstrapは2,000再標本化、seed=20261104。各条件1回の生成なので、CIはこの問題集合の不確実性を表し、生成の反復変動までは測りません。候補探索で複数の条件を見たこと、既知64の過去露出も残ります。全比較でCIは0を含み、一般的な能力向上・重みだけの因果効果・目標達成を断定しません。

このcode-only測定後、Bを元モデルとのfull14対応比較へ進めました。結果は後節に記載し、最終的にBは採用していません。事前の15–25分という目安に対し、実際のBは約28分でした。tier名はhard wall上限ではなく、既存agent timeout倍率48の長時間化リスクも別途管理が必要です。

ここまでのdev90/既知64はcodeのみの測定です。後述のdev10 full14にはSciCodeがなく、その広域確認だけで4 sourceのcode >90を達成扱いしません。

## 証跡

実験artifact rootからの相対名です。各JSONにrun ID・全選択ID・manifest SHA・入力hash/stat・数値metadataを保持しています。

- analysis/code90-resume-20261106/matched-code90-final-analysis.{json,md}
- analysis/code90-resume-20261106/matched-code90-comparisons/*-official.{json,md}
- analysis/code90-resume-20261106/matched-code90-comparisons/*-evidence.json（devのみ）
- analysis/code90-resume-20261106/matched-code90-runner-image-proof.json
- analysis/code90-resume-20261106/known64-first-pair-final-analysis.{json,md}
- analysis/code90-resume-20261106/known64-first-pair-official.{json,md}
- analysis/code90-resume-20261106/full14-runtime-risk.json

比較は既存公式wsp compare --partialを使用しました。partialはコードsourceのみの選択を表し、90問または64問の中に欠測があるという意味ではありません。

## Bと元モデルのfull14 dev10比較

暫定Bを元モデルと同条件の各1 TP2、chat32/logprob8/agent1で測定した。
両方とも778項目を完了、未採点・通信エラー0。実runner image、source 5e3f2df、
manifest、選択IDは一致した。元の総合80.4638に対してBは79.2920、
差−1.1718pt（paired 95% CI −2.9890〜+0.6440）。指示追従は上がる一方、
コード、長文、推論が下がり、先のdev90利得を広い改善とみなす証拠にはならなかった。

| 領域 | 元 % | B % | 差 pt | paired 95% CI pt |
|---|---:|---:|---:|---|
| knowledge | 88.333 | 90.000 | +1.667 | +0.000〜+5.000 |
| expert | 81.250 | 81.250 | +0.000 | +0.000〜+0.000 |
| facts | 70.000 | 67.500 | -2.500 | -7.500〜+0.000 |
| math | 98.750 | 97.188 | -1.562 | -4.688〜+0.000 |
| code | 87.778 | 84.444 | -3.333 | -10.000〜+0.000 |
| instructions | 88.750 | 92.083 | +3.333 | +0.000〜+10.000 |
| tools | 85.000 | 85.000 | +0.000 | +0.000〜+0.000 |
| agents | 50.000 | 50.000 | +0.000 | +0.000〜+0.000 |
| long-context | 81.378 | 68.878 | -12.500 | -31.250〜+0.000 |
| reasoning | 79.130 | 71.957 | -7.174 | -19.348〜+0.000 |
| commonsense | 66.667 | 66.333 | -0.333 | -1.667〜+0.667 |
| multilingual | 82.096 | 81.843 | -0.253 | -7.323〜+6.566 |
| vision | 81.250 | 87.500 | +6.250 | +0.000〜+18.750 |
| safety | 86.111 | 86.111 | +0.000 | +0.000〜+0.000 |

codeはHumanEval10/10→10/10、MBPP8/10→7/10、LCB5/6→5/6で、
失点はMBPP801。ここにはSciCodeがないので、既知64の90目標やdev90と直接比較しない。
toolsは34/40で同一、agent2項目もportfolio成功/install-windows失敗で同一だった。
指示gainはIFBench182。推論lossはBBEH disambiguation_qa119/MuSR object_placements187、
長文lossはRULERの32768二項目で、Bはいずれも16,384token上限に達した。
MGSM Bengali2件も公式失点だが、この比較では数字表記の同値性を診断していない。

数値が変わった19項目は7gain/12lossで、HMMT6の1→0.75も含む。
連続値の変化を0.5閾値の得失へ丸めず、実scoreを別に保存した。
gainとlossの実ID・scoreはbroad14-first-pair-final-analysis.jsonに残る。

| 計測 | 元 | B |
|---|---:|---:|
| run経過秒 | 1553.80 | 1694.68 |
| chat leaf calls | 561 | 561 |
| chat入力tokens（agent別） | 1,401,087 | 1,403,952 |
| chat出力tokens（agent別） | 1,013,842 | 1,055,536 |
| length終了 | 1 | 3 |
| TB portfolio episodes / recorded request times | 5 / 5 | 6 / 6 |
| TB install-windows episodes / recorded request times | 12 / 12 | 10 / 10 |

agentのepisodesとrequest-times件数はTerminus自身の記録であり、独立HTTP telemetryや
全shell command数ではない。元のlengthはRULER fwe:131072:005、Bはそれに加え
fwe:32768:013、cwe:32768:009。最後の完了項目はBのBBEH sarc_triples141だった。
同じchat32でも各1 TP2の長い生成は28分程度かかり、tier10は今回の実行時間上限ではない。

probe-codeの対応位置NLLは2.711→2.739、KL下界0.0094、top1一致98.2%。
wikitext NLL8.496→8.704、Wikipedia-ja7.152→7.285。これらは公式の対応位置集約で、
全生成分布が安定したという保証ではない。1回の生成比較とbootstrap CIだけで、
重みの因果効果や安定した能力変化を確定しない。

証拠: broad14-first-pair-{official,evidence,final-analysis}.json、
broad14-first-pair-runner-image-proof.json。推論本文・参照解・hidden testsはこの分析で読んでいない。

## 元モデルのdev90追加対照

元モデルでも同じdev30コード90問を各1 TP2・chat32・公式設定で測定し、
90/90採点、API error・未採点・length終了は0だった。codeは72.5754、
HE28/30、MBPP23/30、LCB16/20、SciCode40.3016。Bの73.4087との差は
+0.8333ptで、0397対照を使った+2.9167ptとは異なる。

| 条件 | code | HE+ | MBPP+ | LCB | SciCode |
|---|---:|---:|---:|---:|---:|
| 元モデル・追加対照 | 72.5754 | 28/30 | 23/30 | 16/20 | 40.3016 |
| B・先行測定 | 73.4087 | 30/30 | 23/30 | 16/20 | 36.9683 |

この元モデルrunはfull14後に実施した追加対照で、先の4条件と同時の反復ではない。
source/IDを揃えても単発の時間差・生成変動は残る。この表だけでBの安定した純増を
確定せず、full14のcode/長文/推論の損失と既知64の未改善を合わせて不採用とした。
参照runはgemma4-code90-original-base-c32-dev30-code-20261106、
証拠は同runのrun.json・report.jsonとoriginal-dev90-control-launch.json。
これは既知64の90目標達成でも、未見確認の成功でもない。

## 広いdonor方向の有限再測定と目的の補正

**後から判明したframe制約:** 以下の自作teacher-forced診断はthinking ONの
assistant開始直後を採点しており、意図した公開final回答の継続を確立していなかった。
NLL/marginの解釈はこの旧frameに限定する。
[6案の停止記録](2026-11-06-positive-correct-donor-amendment.md#retirement-of-the-intended-proxy)を参照。
前節までの公式WSP生成・採点・probeの記録は、この自作frame問題とは別で有効である。

full14の結果も踏まえ、Bは採用しなかった。さらに、公開donorから全410投影の
共同方向を構成して実BF16で再測定したが、4案すべての完走とはしない。
20分予算のv2実行は2案を完了し、TimeoutErrorで全体status=incompleteを保持した。
経過1222.20秒にはfinallyの元重み復帰と確認を含む。attention-interfacesと
late-compensationは未測定であり、失敗した能力候補とも改善候補とも扱わない。

| 実測案 | BF16変更要素 | 実delta L2 | 主目的margin平均差 | 主目的correct NLL平均差 | NLL悪化fold |
|---|---:|---:|---:|---:|---:|
| donor-balanced | 943,283,732 | 0.620958543 | +0.022706 | +0.090879 | 33/34 |
| donor-numerical-algorithms | 1,135,409,924 | 0.710552826 | +0.028762 | +0.108608 | 32/34 |

主目的はcode/implementation/algorithm/science_numerics/science_contractsの10fold。
上表は自作teacher-forced校正のlogprob単位で、実行正答率やaccuracyポイントではない。
correct-minus-foil marginが増えても、correct NLL自体が上がっており、
正しい続きを生成しやすくなったとは言えない。これは誤答の確率だけを下げる方向でも
marginを改善できる目的の弱さを示す診断であり、weightの局在を確定するものではない。
KLはcorrect回答の先頭/末尾位置だけの値で、全tokenの分布不変性を示さない。

元410行列のreadbackはすべてexact、復帰後margin/NLL/KLの差は0、source continuityも
維持された。2案はこの段階で採用せず、大きなcheckpoint保存やWSPへ進めなかった。
その時点では、次にcorrect回答の平均logPを直接最大化する目的と従来marginを分けて計測する計画を立てた。
既存120学習caseを値・順序とも保持し、新advanced16を追加した136case/21目的/42foldを
両目的で共通に使う。最大公式chat frame505tokens、transferは含めない。
この入力準備自体はモデル生成・gradient実測・WSP改善の証拠ではない。

証拠: donor-module-actual-finite-comparison.json、positive-correct-v1/
merge-metadata.json・official-frame-evidence.json。校正文の2件の曖昧さを見つけた後は、
原本を保持してadvanced revision2とpositive-correct-v2へ移行した。trainの変更は
予算caseのpromptのみで、旧120行・計算・foil・oracleは不変。共通136caseの両目的を
新たに測り、旧34foldと新8foldを別実装のまま継ぎ足して同一収集とは扱わない。
[匿名Opusの助言と採用範囲](anonymous-positive-correct-review-2026-11-06.md)も別記した。

revision2の共通136caseではnative/donor×margin/correct-logprobの4収集が全42foldを完了し、positive-correct solverの6案は連続制約内の解を得た。その後にfinal回答frameの欠落を確認したため、6案は有限測定・適用前に停止した。旧入力・数値は保存し、意図したfinal回答proxyとしては採用しない。Bの不採用判断とcode >90未達は変わらない。

## 元モデルの既知64追加対照

元モデルのgemma4-code90-original-base-c32-known64-r1-20261106も64/64採点、
未採点0で完了した。元・B・0397で全IDとmanifest52a760…が一致し、
既存公式wsp compare --partialで比較した。source archiveは共通5e3f2df。
元は後時点の追加測定で、B/0397 firstpairと同時の反復ではない。

| 条件 | code | 元との差 [paired 95% CI] | HE+ | MBPP+ | LCB | SciCode |
|---|---:|---|---:|---:|---:|---:|
| 元モデル | 78.2619 | — | 19/20 | 15/20 | 12/14 | 57.3333 |
| B firstpair | 76.4762 | −1.7857 [−10.1786, +6.3690] | 19/20 | 16/20 | 11/14 | 52.3333 |
| 0397 firstpair | 77.0119 | −1.2500 [−5.0000, +2.5000] | 18/20 | 16/20 | 12/14 | 52.3333 |

元に対しBの実scoreは4gain/5loss。LCB abc399_d、MBPP92/279が0→1、
Sci21が0→1/3。一方LCB abc397_f/abc388_e、MBPP635は1→0、
Sci15は1→1/2、Sci24は1/3→0だった。0.5閾値だけの3gain/3lossでは
科学の部分成功率の変化が落ちるため、実score差を保持する。

0397はMBPP92を0→1とした一方、HE163を1→0、Sci15を1→1/2とした。
Sci15は先行B/0397で同じ段階に公式timeoutが記録され、元の追加対照では全段階成功。
これは観測結果であり、コード・環境・時間変動のいずれかを断定しない。

元のwall581.6秒、生成276.7秒、記録chat API calls87、length0。
bootstrapは2,000回、seed20261104、両CIに0を含む。各条件1回の生成であり、
時間差・生成変動・既知集合への適応を含む。この結果から一般的な悪化の確定も
改善の確定も主張せず、少なくともcode >90の達成を示していない。

証跡: analysis/code90-resume-20261106/known64-original-comparisons/
analysis.json、native-contracts-official.{json,md}、h0397-control-official.{json,md}。
testラベルをdevへ変更せず、dev専用履歴ツールも通していない。
[自作proxyのframe問題](2026-11-06-public-answer-frame-audit.md)とは別の、
有効な公式WSP実行結果として保存した。
