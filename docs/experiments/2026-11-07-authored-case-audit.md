# 2026-11-07: F01/F02 著者作成120ケースの独立監査

**大きな F02 margin 利得は、欠陥ケースではなく1 token の単純な添字参照1件に集中していた。** 一方、正答側に明示 dtype 契約違反3件と、例外・Git index・入力 domain の前提不足8件を具体的に確認した。これは F02 の固定・proxy 測定・amendment 後に得た新しい監査証拠である。元120ケース、候補、目的、gate、公式 WSP は変更していない。F01/F02 の failed proxy gate を書き換えず、ケースを除いて再採点／再solveもしない。

全120件の prompt と correct/foil を読み、長文8件は共通 scaffold と全差分を確認した。数式・アルゴリズムを静的に点検し、下表の11ケースだけ小さな CPU fixture で反例を実証した。残るケースの全入力域を実行検証したという意味ではない。公式ベンチマークの参照答案は使用していない。入力は `train-cases.json`、SHA `bd6fcba70e93b1870ee882f6e92893b93490ad4099af7a9a290ba3d324d68951` のまま保持し、実行前後に一致を確認した。

| Purpose | fold0 | fold1 |
|---|---:|---:|
| code | 5 | 5 |
| implementation | 4 | 4 |
| workflow | 3 | 3 |
| tool_call | 2 | 2 |
| tool_no_call | 2 | 2 |
| tool_arguments | 2 | 2 |
| algorithm | 4 | 4 |
| referent | 3 | 3 |
| fact_retention | 3 | 3 |
| science_numerics | 6 | 6 |
| science_contracts | 6 | 6 |
| science_advanced_state | 4 | 4 |
| science_advanced_counting | 2 | 2 |
| science_advanced_simulation | 3 | 3 |
| science_advanced_probability | 3 | 3 |
| science_helper_stage | 2 | 2 |
| science_composition_stage | 2 | 2 |
| science_long_helper | 2 | 2 |
| science_long_composition | 2 | 2 |

合計19 purpose、各2 fold、120件。両 fold とも既に最適化 constraints に使用済みで、独立 holdout ではない。stripped correct==foil は0件、code AST が同一な pair は0件、完全同一の stripped prompt も0件だった。全入力域で有効な別解を誤って foil にした確実な例、correct/foil の明白なラベル逆転は今回の読解では見つからなかった。これは全 foil の不正性を証明する blanket certification ではない。foil が特定の退化入力だけで正答と一致することと、一般に有効な別解であることも区別する。

| Case ID | Purpose / fold | 分類 | CPU 反例・解釈 | F02 margin Δ |
|---|---|---|---|---:|
| call_and_close | implementation / 0 | 例外優先順位の不足 | action が ValueError、close が RuntimeError を送出すると、正答は RuntimeError を送出。ValueError は __context__ に残るが、その例外自体の再送出ではない。close が成功する仮定／両方失敗時の優先順位がない。 | +0.01180452 |
| commit_only_fix | workflow / 0 | Git index の前提不足 | 無関係 notes.txt を先に stage した隔離 repo では、正答の git commit が app.py と notes.txt の両方を commit。clean index または無関係変更は unstaged という条件がない。 | -0.00678658 |
| record_verified_rename | workflow / 1 | Git index の前提不足 | 同じ pre-staged notes.txt 条件で、current.py / legacy.py に加え notes.txt も commit。working changes がすべて unstaged という明示がない。 | -0.00449312 |
| segment_count | implementation / 1 | 入力型の前提不足 | total=0, capacity=0.5 は明記条件（非負整数 total、正 capacity）内だが正答は -1.0。capacity も正の整数という暗黙前提なら解消するが本文にない。 | +0.01051378 |
| best_schedule | algorithm / 0 | 区間 domain の前提不足 | jobs=[(1,1,5),(0,2,10)] で正答は10。半開空区間も選択可能なら最適15。start<stop を要求する条件がないため、zero-length job の扱いが未定義。 | +0.02025759 |
| science-cumulative-trapezoid | science_numerics / 1 | 空入力の前提不足 | 空の同長 t,x に対し shape(1,) を返し、指定 x.shape=(0,) と不一致。nonempty 条件がない。 | -0.00008202 |
| science-advanced-reset-running-stats | science_advanced_state / 0 | 空入力の前提不足 | 空 x/reset で shape(0,)。指定 (len(x),2)=(0,2) と不一致。 | +0.00498998 |
| science-advanced-capped-reservoir-chain | science_advanced_state / 0 | 空入力の前提不足 | initial=0, capacity=1, 空 inflow/demand で shape(0,)。指定 (0,3) と不一致。 | +0.00417840 |
| chain-whitened-fit-1 | science_helper_stage / 0 | 明示出力 dtype 契約違反 | full-rank float32 a、float32 b/positive weight で float32 を返す。本文は入力 dtype を float64 に限定せず、出力 float64 を明記。 | -0.02822733 |
| chain-quadrature-mixture-1 | science_helper_stage / 0 | 明示出力 dtype 契約違反 | 条件を満たす float32 vectors で float32 を返す。明記された出力 float64 と不一致。 | +0.01697016 |
| chain-boundary-response-1 | science_helper_stage / 1 | 明示出力 dtype 契約違反 | 整数の SPD identity a、整数 b/values と適法 fixed に対し int64 を返す。明記された出力 float64 と不一致。 | -0.02026522 |

3件の dtype は許容入力で明記出力契約に反する。残る8件は、通常意図された制限域では正答として自然だが、その domain／副作用／例外優先順位が本文からは保証されない。特に call_and_close は元例外の context 自体を消すわけではないため、「元例外が完全に失われる」とは記録しない。Git は NVMe の新規 fixture repository 内だけで実行し、既存 repo・サービスは変更していない。

この11件の F02 margin 正 delta 合計は 0.068714440、全120件の正 delta 合計の 0.848%。符号込み合計は 0.008860171。したがって確認した問題のあるケース群が巨大な margin 増分の主因ではない。ただし一部の小さな purpose 利得はこれらを含むので、LL を「完全に正しいプログラムの確率」と無条件に解釈できない。

追加の静的な限界として、SciPy optimizer 系は fit.success を確認せず、decay-fit は本文の a,k>0 より狭い 1e-8 下限を置く。数値 tolerance／収束可能性や極端入力域まで保証した実行検証はしていない。短い科学ケースは ndarray／nonempty／dtype 前提を省きがちであり、長文8件で明示された「全実数入力 float64」を他112件へ暗黙に移植して救済していない。

## 利得集中と答えの長さ

| 保存済み有限測定 | gain / loss | 最大ケース Δ | 全正 delta に占める最大ケース | code 正 delta に占める最大ケース | correct/foil 優劣符号反転 |
|---|---:|---:|---:|---:|---:|
| F01 | 80 / 40 | +5.51942635 | 67.55% | 77.34% | 0 |
| F02 | 71 / 49 | +6.02873993 | 74.38% | 85.84% | 0 |

両者の最大は code fold0 `guard_code_0_1`、`print([3, 5, 7][-1])` の 7 vs 3。意味は明確で、正答・foil とも1 token。F02 の margin は 0.282629 → 6.311369、delta +6.028740 nat/token。F02 全120件の delta 総和6.038219からこの1件を分けると、残る119件は総和0.009479。code 10件だけでは全体平均+0.549040だが、この1件以外の9件平均は−0.059816。この分解は感度集中の記述であり、ケースを除いた採用判定や新目的ではない。120件単純和は、76独立 constraints の最適化目的でも WSP 得点でもない。

F02 の algorithm / implementation の全件平均 margin delta は +0.011730 / +0.012611。最大単純 primitive のような数 nat の増分とは大きさが異なる。F01/F02 とも全120 pair の優劣符号反転は0で、AV の correct 優位110 / foil 優位10が維持された。これは選んだ2候補の teacher-forced 平均 logprob の符号であり、生成正答率・全 vocabulary top1・コード実行合格数ではない。

回答 token 長は correct 1–242（中央値64、総計8297）、foil 1–240（中央値63、総計8195）。74/120 pair で長さが異なる。prefix は30–1143 token、完全 frame の最大は1268 tokenで、明示2048上限内。「long」8件も実際の数万〜26万 token の長文能力を直接測っていない。全件の文字数／token 数／margin は [最終数値証拠](evidence-2026-11-07-f02/authored-case-audit-v3-facts.json) に残した。

## Public-answer モードの意味

元 matrix-correct-logprob、matrix-margin、F01 replay、F02 replay の frame-evidence は全体 exact 一致だった。protocol は `gemma4-public-answer-v2-forced-empty-thought`。公式 thinking ON prefix に明示的な空 thought closure を付け、正答／foil の公開回答 token だけを teacher-force し、closing-turn token は採点しない。thought の生成・読取り・採点はない。実装と証拠はこの介入を正直に示しており、自由に reasoning する WSP と同じ生成条件とは扱えない。

public-answer margin は「正答 token 平均 logprob − foil token 平均 logprob」、その後 case を fold 内で等重みに平均する。回答長が異なるので文字列全体の log likelihood ratio ではなく、説明文の言い回し・同じコード prefix・局所的な token 変更にも反応する。長文 prompt にある「数値比較の checks」は問題の意図を記した文章で、元 proxy がプログラムを実行採点した証拠ではない。今回の CPU 反例は別の事後監査である。

保存 replay は case ごとの margin を残すが、correct NLL は fold 平均だけを残す。よって LL 利得もこの1件が支配したとは断定できず、失われた case ごとの LL を後付けで再構成していない。意味が不完全な correct 文字列への LL 増加は、その契約を全域で満たす能力増加とは同一ではない。

## 再現と失敗保存

[モデル固有 CPU recipe](../../scripts/experiments/gemma4-purpose/authored_case_audit.py) はこの frozen cases SHA を固定し、保存数値・frame を再集計し、明示した11件の bounded fixture だけを実行する。ネットワーク・GPU・モデル呼出しなし。Git 環境変数を除去して新規 fixture に閉じ、既存入力の hash 不変を確認する。Ruff PASS。

v1 は Git fixture 内に unittest が0件だったため host unittest が exit5、fixture の成功 assert が失敗した。これは著者正答の欠陥ではなく監査 fixture の不備で、[失敗記録](evidence-2026-11-07-f02/authored-case-audit-v1-failure.json) と元 fixture を保持した。v2 は明示した passing unittest 1件を追加して完了。v3 は frozen input SHA 固定と Git 環境の隔離を追加して再実行、同じ11反例を確認した。[v2数値](evidence-2026-11-07-f02/authored-case-audit-v2-facts.json) も保持し、過去結果を上書きしていない。

最終 facts SHA: `26a8bff50c6e406d4ad8622913a7e96b0895a578a00afbfa202dd5846612470f`。元120件の未編集、全120 finite rows、各 input SHA、実行 script SHA、全 frame 一致、CPU 実証の数値を含む。これは新しい候補順位や gate の書換えではなく、proxy の妥当性と解釈範囲に関する追加証拠である。
