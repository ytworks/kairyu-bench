# Rの再観測と有限未実行のS/U提案（2026-11-06）

> 判定更新：全14領域のnet gain確認を受け、ownerはR増分の暫定昇格を撤回し、積み上げ済みPを再び作業親とした。originalへ戻したわけではない。本ページのR微分とR起点S/Uは履歴として保持する。R S/Uのfinite実行は0で、起動のautomatic approval review拒否後、owner判断で未実行のまま対象外となった。追加GPU観測・再solveで履歴を置換しない。

Rはfresh Pとの公式dev90で73.2698413→73.6269841（+0.3571429 pp、95% CI [−2.142857,+2.857143]）となり、ユーザーの積み上げ方針で作業モデルへ昇格した。[Rの変更記録](2026-11-06-cumulative-mlp-math-step-r.md)と[全公式スコア台帳](2026-11-06-all-official-scores.md)を参照。微小差を能力改善の確定や最終90点超の達成とは呼ばない。CIは不確実性であり追加のworking採用gateではない。

[Pの目的別微分](2026-11-06-current-incumbent-purpose-observation.md)は歴史観測へ移した。新規S/UはRの保存重みと新規R微分を入力とし、P/OのGを流用しない。

## Rの観測条件と復帰

自作train136例／21目的／42fold、native421 tensor／1,331,516座標。correct-answer log probability（correct-LL）とcorrect-minus-foil marginを各1回収集した。同じv2 forced-empty-thought frame、serial／1case microbatch、threads4、seed0、BF16/eager、Torch2.14.0+cu130／Transformers5.20.2。transfer/testを勾配へ入れない。

R state SHA c0e271c4…、tree ea3d27ab…、CPU検証freeze760b7257…の全9ファイルfull hashを現在statへ結び付けた。新しいcheckpoint全payload hash scanは行っていない。保存native値はP/Rの両目的間でbitwise一致したが、他のMLP重みが異なるため微分の同一性を仮定しない。

GPU4のidle O2だけを停止し、R3/P1/N0とembeddingを維持。14:17:59.210–14:22:59.058 UTC、計299.85秒で両収集と同一O2 CIDのhealthy復帰を完了した。computeはcorrect-LL54.3436秒／margin85.2421秒。96出力ファイルのSHA/stat、共有source5、R9file stat、current state、保護サービスの継続一致を確認。R full14とCPU/NVMe利用が重なるため速度の因果比較には使わない。

## 観測された差とproxyの限界

P→Rのcorrect-LLは42fold中38改善／4低下、marginは17改善／25低下。21目的等重み平均はそれぞれ+0.10233844／+0.03372999 nats/token。一方、公式dev90の差は+0.3571429 ppに留まる。**LLだけの順位付けはしない。** これらの単位をaccuracy ppへ換算しない。

forced-empty-thoughtのteacher forcingは、native ONが実際に生成するthinking contextを含まない。短い自作回答の尤度・foil識別・自由生成したコードの実行成功は異なる測定である。21目的にXSTest safety目的はなく、safety改善は主張できない。

既存joint.read_collectionで保存数値だけを読み、CPU8–15／BLAS2で20.55秒集約した。各目的の2foldを等重み平均、グループ内は目的等重み。cosineは保存native座標のEuclidean内積で、raw norm²のrole/layer分布は尺度依存の感度であり能力の因果局在ではない。全42fold／136例／21目的／目的ペアとrole/layer内訳をsummary.jsonへ保存した。

| 仮説群／review label | 目的セット | Δcorrect-LL R−P | Δmargin R−P | RのLL–margin cosine |
|---|---|---:|---:|---:|
| H1 / S | numerics / contracts / state | +0.01249234 | +0.00133063 | -0.2994 |
| H2 / T | simulation / probability | +0.00969543 | -0.00474818 | +0.0426 |
| H3 / U | implementation / algorithm / workflow | +0.01569722 | -0.01661033 | -0.1616 |
| H4 / V | call / no-call / arguments | +0.00678670 | -0.01198792 | -0.4035 |
| H5 / W | lookup / referent / fact retention | +0.21939093 | +0.13223613 | +0.6860 |
| H6 / X | instruction / reasoning / general | +0.19274318 | -0.03384666 | +0.0400 |
| H7 / Y | counting / state | +0.01507906 | -0.00292323 | -0.2411 |

H1–H7は仮説群の記号で、歴史checkpoint Hではない。匿名Opus5.5の7群reviewは[P観測文書](2026-11-06-current-incumbent-purpose-observation.md)に記録済み。private計測値・座標・問題・回答は送っていない。R起点S/Uはその方法助言を踏まえた既定2セットで、R観測の最高値を選んだgridではない。P復帰後は同じ目的・bounds規則をPの保存微分へ適用する別の計算として扱う。

後続のR full14 dev10は778/778採点・未採点0、overall76.3823193／code84.4444444。歴史Pの78.4750988／87.7777778を下回った。math95.625→98.75の利得と、reasoning76.95652→64.78261、long-context87.62794→81.37794、safety58.33333→55.55556の損失が併存する。異時点の固定cohort比較で、code26にはSciCodeがない。LL改善から広範な保持を推測せず、公式結果を別に記録する。公式overall差は−2.0927795 pp、95% CI [−4.26587,−.09675]、746 capability paired items中25件変化（9 gain／16 loss）。code−3.3333、reasoning−12.1739、long-context−6.25 ppに対しmath+3.125／commonsense+1 ppという得失を総合してownerがR増分を退けた。CIだけを新しい必須gateにしたものではない。

## 固定S/Uの一度限りのCPU提案

Sはnumerics/contracts/advanced_state、Uはimplementation/algorithm/workflow。全421 native／1,331,516座標、Euclidean cost1。選択foldのcorrect-LL下限はH=−R mean correct-LLに対しH≤.004なら0、それ以外min(.01,.25H)。今回の選択12foldは全て+.01となった。選択marginは0、他correct-LLは−.01、他marginは−.02。これは明示した探索入力であり新たな採用gateではない。

各案84制約、relative radius.025／absolute radius1e−6／最大16補正、実変更100万座標を超えたら拒否し切り詰めない。source/planを先に固定し、CPU8–15／BLAS2／総budget300秒で2案を一度だけ計算した。grid・再試行0、計91.9035秒。

| 提案 | BF16提案変更数 / tensor | R基準ΔL2 | 選択LL予測範囲 | 選択margin最小 | 全制約判定 |
|---|---:|---:|---:|---:|---|
| S | 7,346 / 88 | 0.000388546825768 | 0.0100011477–0.0101333971 | 7.9180236e-07 | true / 未達0 |
| U | 9,228 / 96 | 0.000518034364889 | 0.0099991309–0.0179112025 | 2.1940888e-06 | false / 最大未達8.6907e−7 |

Uのalgorithm両foldは+.01に僅かに届かず、falseを保持した。rootは当初この既知未達を明示して両案の有限測定を承認したが、R撤回により未実行のまま終了した。R案の再solveはない。BF16丸め方向と連続方向のcosineはS .887864／U .906383だが、実応答や能力の証明ではない。

**予測上の損失も残す。** Sのinstruction margin平均−.0188366（最小−.0196318）、reasoning平均−.0089570／Japanese−.0102883。Uのinstruction margin平均−.0183486（最小−.0198340）、tool_call平均−.0113896、reasoning−.0065311。Uはinstruction correct-LLの片foldも−.0094489で、許容下限−.01に近い。instructionの尤度とfoilとの差を混同せず、有限実測で両方の予測差を確認する。これらは公式instruction/tool accuracyの退行を既に観測した値ではない。

当初はSを科学セット、Uを実装・algorithm・workflowセットとして両方のactual BF16 finiteを先に比較し、その後native ONの公式実行結果で判別する予定だった。R撤回後、このR起点の実行計画は無効となった。予測LLの広い改善、tiny L2、制約trueだけで採用しない一方、追加の全領域zero-loss／有意差gateも設けない。有限測定の結果はまだ本表に含めていない。

計算終了後、全入力SHA/stat・R9file stat・current stateを再確認し共有source pinを解除した。detached launcherのexit codeは保存しておらずunknown。完了report・成功JSON・両spec・プロセス終了と全postcheckを照合した事実をcompletion receiptに明記している。

## 証拠

以下はanalysis/code90-resume-20261106/を起点とする。

| Artifact | SHA256 |
|---|---|
| current-r-purpose-observation-v1/plan.json | 454f6d413c5bbd4e611927bd40823eda96a403bdc7fcd2797b7dcd3a32d9438b |
| current-r-purpose-observation-v1/checkpoint-attestation.json | 9a65784f062f4bc8ba52a39171806ec96c140ef23b924c4f772304ff9cf8c14d |
| current-r-purpose-observation-v1/completion.json | 919f5cdbbddc2d511019c203a7e9b5635544c548c3f0d3da983b97fc85f5c1f3 |
| current-r-purpose-observation-v1/postchecks.json | ff69950c2fb87e24f2e2216980fbde4d946571c676caf76e72414133c3570724 |
| current-r-purpose-observation-v1/collector-output-hashes.json | eb819663238c4895e68ce4205111386624c872c8d5cef2fe420c8612d404545b |
| current-r-purpose-observation-v1/numeric-analysis-v1/summary.json | 1a0881f20bc2b1494b081907b8729a4716e66810db5a7521f958e76e76fe25df |
| current-r-native-su-proposals-v1/plan.json | c0d467ee8ae882ae10fcb150a5df84f1ac0b7e911077af732fa93c1bad3c62b1 |
| current-r-native-su-proposals-v1/proposals/report.json | e662cd34616f0f6b17aefcaa733acfd4a5107c59f4ee672d563ded5e5eebb88c |
| current-r-native-su-proposals-v1/solve-completion.json | 2e4fe3a66cbaa59ce2e2b5ce9016d8424b72a635746adcb9a2d393c450f4fc61 |
| current-r-native-su-proposals-v1/proposals/current-r-science-coupled-s.json | 7df289505375a1194b8cc8eac2f773cd8bc31153192bd5b469c9a161cb6b74f4 |
| current-r-native-su-proposals-v1/proposals/current-r-implementation-coupled-u.json | 72be34c918989d729bc45be818140f9980a96d2d7206c740afd2b29d8f37bf32 |
| R full14 report（results/gemma4-code90-incumbent-r-c32-dev10-full14-20261106/report.json; ROOT起点） | c0a78d4488a5df88afb4108153bd7adb42decf795e5893485614639e8ae32ccc |
| R full14 closure incumbent-r-retention-full14-v1/final-summary.json | db693d71de3180fe49910d9a89cf07c494771a1f2e37dfc4c41fbd2750969095 |
| R finite未起動 current-r-native-su-finite-v1/launch-not-started-review-hold.json | 204afed62ac7bc3c50dcb53130ec98f89eaa00235b7730e0b6e44f2e96a95c9d |

撤回の正式記録：heretic-public-base-math-mlp-c1-v1/working-retention-decision.json SHA 1619f5487e92716434641797eb93971a4ad88e181ccc9f105d96f5d788b1e528。復帰P state SHA 2666c54246852bf328be8b6293ff3402f011403fc27ad28113c5bcddaf923c99、P tree3dfb3bd1…／同一保存checkpoint・9file statが一致。RのG・spec・完了証跡は変更しない。
