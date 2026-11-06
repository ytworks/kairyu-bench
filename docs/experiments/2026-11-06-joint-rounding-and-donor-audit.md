# JOINT の丸め後実測と一様 donor の既存評価（2026-11-06）

JOINT は正解文の teacher-forced 対数尤度を module-only balanced より広く改善したが、公式 dev90 は元モデル72.575397%に対して70.492063%（−2.083333pp）となり、不採用とした。balanced に続き、自作校正の改善が公式コード成績へ移らなかった。本記録は数値メタデータと自作校正だけを扱い、benchmark の生成コード・隠れた推論・正解実装を読んでいない。

## JOINT の実体と同じ元重みに対する診断

元 checkpoint から native と donor の両成分を適用した一つの共同解である。既存 balanced checkpoint に native を足した実験ではなく、donor 係数も異なる。したがって以下の差だけから native 成分の因果的効果や機能の加算性は断定できない。

136 自作校正例、21目的×2 fold、同一 public-final frame・serial answer batching で比較した。正解 LL は回答 token 当たりの平均 log probability、正の差が改善を表す。margin は正解文と foil の差であり、どちらも正解率の percentage point ではない。

| 指標 | module-only balanced | JOINT |
|---|---:|---:|
| 実 BF16 変更要素 | 718,765,573 | 84,142,113 |
| 実 ΔL2 | 0.338364145 | 0.020911997 |
| 正解 LL が改善 / 悪化した fold | 29 / 13 | 38 / 4 |
| fold 等重みの平均 ΔLL | +0.030343 | +0.042018 |
| fold 等重みの平均 Δmargin | +0.029308 | +0.043391 |
| 最大の正解 NLL 増加 | 0.120481 | 0.078294 |
| 事前の線形制約を実測で満たした数 | 46 / 84 | 59 / 84 |

JOINT の内訳は module 84,133,744要素・ΔL2 0.020906424、native 8,369要素・ΔL2 0.000482758。両者は非重複なので全体 ΔL2 は二乗和平方根で求める。これは変更要素率とは異なる費用軸であり、少ない変更が能力改善を保証するわけではない。

donor の連続予測 ΔL2 は 0.059170802、native を含む予測は 0.059172772 だった。実 module ΔL2 は連続値の約35.3%であり、丸め前の大きさを保存済み重みの大きさとして引用しない。比だけでは実方向の cosine は求まらない。

module は410行列の CPU target SHA と全要素 device-int16 equality、native は92 tensor の選択座標を D2H で照合した。module の証明を D2H hash と呼ばない。終了時は両成分を元へ復元し、410行列の byte equality と再測定 margin / NLL / 診断 KL の差0、source continuity を確認した。監査側の新しいモデル呼出しは0である。

## 有限測定で残った弱点

JOINT の主要18 fold（code、implementation、algorithm、科学6目的）は全て正解 LL が上がった。科学12 fold の ΔLL は +0.002570〜+0.029931、平均 +0.011541。しかし科学 margin は12中6 foldで低下し、tool_call と tool_arguments も各2 foldとも margin が低下した。正解文を読みやすくする方向と、誤答を排除する方向は同じではない。

正解 LL の悪化は次の4 foldに集中した。

| 目的 / fold | Δ正解 LL | Δmargin | balanced の Δ正解 LL |
|---|---:|---:|---:|
| instruction / 0 | −0.060742 | −0.061741 | −0.063686 |
| reasoning / 0 | −0.022851 | −0.017990 | −0.026867 |
| lookup / 0 | −0.078294 | +0.114876 | +0.034083 |
| japanese / 1 | −0.067059 | +0.025391 | +0.239005 |

instruction と reasoning の同じ fold は balanced にも残った。lookup と japanese では正解 LL と margin の符号が違う。全42 fold の ΔLL 中央値は +0.013445で、平均 +0.042018だけを見ると大きな一部の改善を強調する。また code fold1 の診断 KL は0.141604と最大だった。KL は正解文の最初・最後の位置の全語彙分布だけであり、全生成列の安定性を示さない。per-case の正解 NLL は保存されていないため、fold の悪化を特定の自作例へ割り当てない。

[public-final frame の監査](2026-11-06-public-answer-frame-audit.md)で修正した明示的な空 thought 境界を使っている。自由生成でモデルが実際に考えた文脈とは異なる。84制約の未達25件は予測と有限差の不一致として報告し、ユーザーが許容した能力間 tradeoff を否定する新しい全非退行 gate にはしない。

この有限測定後、同じ選択90問・生成設定・runtimeでJOINTの公式dev90を1回測定した。結果は次節のとおりで、元モデルを上回らなかった。dev90は、既に成績を見たknown64でのcode >90判定とも、新しい未見集合への確認とも異なる。有限測定の数値と原本は保持し、公式失点に合わせて予測値を修正していない。

## JOINT の公式 dev90 結果

run は `gemma4-code90-public-final-joint-c32-dev30-code-20261106`。
元モデル `gemma4-code90-original-base-c32-dev30-code-20261106` と完全に同じ90 ID、
manifest・tier30/dev・model IDで、全90採点・未採点0・応答エラー0。
公式codeは4 sourceを等重みにした平均で、全項目を一括した正解率ではない。

| Source | 問数 | 元モデル | JOINT |
|---|---:|---:|---:|
| HumanEval+ | 30 | 28 / 30 | 28 / 30 |
| MBPP+ | 30 | 23 / 30 | 22 / 30 |
| LiveCodeBench | 20 | 16 / 20 | 15 / 20 |
| SciCode | 10 | 40.301587% | 40.301587% |
| **公式code（4 source平均）** | **90** | **72.575397%** | **70.492063%** |

差は−2.083333pp、公式paired bootstrap 95% CIは **[−5.416667, 0.000000]pp**
（2,000 resamples、seed=20261104）。新gain0、loss2で、
`livecodebench:arc195_b` と `mbpp-plus:Mbpp/641` が1→0。他88項目の点は同じだった。
SciCodeは10問それぞれの部分点が同値であり、全stepを一括した率へ置き換えない。

| 実行量 | 元モデル | JOINT |
|---|---:|---:|
| 記録API呼出し | 119 | 119 |
| 入力token | 97,703 | 96,166 |
| 出力token | 336,625 | 345,689 |
| stop / length終了 | 119 / 0 | 119 / 0 |
| run開始→完了のwall秒 | 394.1 | 340.153 |

JOINTは07:08:29.853→07:14:10.006 UTCに完了した。wallはload・生成・採点を含む全体時間であり、
各段階を別に足し合わせない。SciCode10項目は39個のAPI呼出しを持つため、
aggregate sample数を119回のleaf API呼出し数と混同しない。これらはagent tool手数ではない。

実runnerのimage IDは `9aca43ac343d0b2c15012d8bdd249452fb5a20e68ee2c8fe89054855bdae3e43`。
`kairyu-bench:code90-5e3f2df` のruntime Config SHAと16層RootFS SHAは既存5e3 runner証拠に一致した。
元dev90固有のCID記録は残っておらず、同archive/CLIと既知の元full14 image証拠までが確認範囲。
現在の測定は単一TP2の直接endpointで、mixed gateway経由ではない。
他replicaの停止はownerが事前に記録した予定状態であり、このrunの障害と扱わない。

失点の具体的な構文・アルゴリズム原因は未分類。追加のcanonical itemから診断field名だけを
確認する操作も、自動承認reviewが生成内容を含むitemの読取として実行前に拒否した。
その後は既存の数値集計だけを使い、本文読取・迂回・再採点は行っていない。
**balancedのSciCode35.3 timeoutをJOINTにも起きたものとして転記しない。**

証拠は `positive-final-v1/joint-dev90-{evidence,final-summary,progress,runner-evidence,parity,diagnostic-access-block}.json`。
公式evidence SHA256は `217dc675d5cf1ecef69cdddfd0a6267ce2fb54397a51473c6de852c86db21506`。
known64での>90達成や全14領域overallは、このcodeのみの測定から主張できない。

## balanced で実際に起きたこと

balanced の元モデル比較は同じ90問・item sampling metadataで、全90採点・未採点0だった。公式 code は4 source の平均で72.575397→71.325397（−1.25pp、paired 95% CI −5.00〜+1.6667pp）。

| Source | 問数 | 元モデル | balanced |
|---|---:|---:|---:|
| HumanEval+ | 30 | 28 / 30 | 28 / 30 |
| MBPP+ | 30 | 23 / 30 | 23 / 30 |
| LiveCodeBench | 20 | 16 / 20 | 15 / 20 |
| SciCode | 10 | 40.301587% | 40.301587% |

得失は LCB `abc395_c` と MBPP `641` の失点、MBPP `801` の得点。SciCode は10問の得点が全て同じだが、保存診断では step `58.5` が AssertionError→TypeError、`35.3` が AssertionError→約302.6秒の timeout に変わった。いずれも元から不合格だった step であり、原因は例外クラスだけでは分からない。生成は両 run とも空回答・length終了0、119 API callsだった。

この公式失点と JOINT の4つの負校正 fold の間に共通の問題IDはない。dev90 はcodeのみなので、instruction / lookup / Japanese の退行と相関したと述べる根拠もない。数値だけから「科学数学」「実装保全」など特定能力が原因と分類しない。既存 balanced の結果は、良い proxy が実行成功にならない直接の反例として扱う。

## 自作 advanced v2 の元モデル生成ベースライン

同じ公式元重みの通常の native thinking ON 生成を、自作 advanced-v2 の24例で確認した。生成receiptはtrain16 / transfer8ともcompleted、received=cases、errors=0、retry=0。samplingはserver公式generation config、max_tokens=32768、seed=0で、hidden reasoningは保存していない。別のCPU sandboxが各例8 trial（seed=20261107）を実行した。

| Split | 生成応答 | 実行合格 | 実行不合格 | 未評価 |
|---|---:|---:|---:|---:|
| train | 16 / 16 | 15 | 1 | 0 |
| transfer | 8 / 8 | 8 | 0 | 0 |
| 合計 | 24 / 24 | 23 | 1 | 0 |

唯一の不合格IDは `science-advanced-reflecting-wall-flight`。この追記では応答・log本文を読まず、保存済みの構造化ID・status・集計だけを確認したため、失敗機構は分類していない。transfer8例はgradient入力から引き続き除外する。

v1の21/24にはbudget入力順とSchur寸法の曖昧なpromptが含まれ、v2ではその2つを明確化した。異なるprompt版の21→23を重み変更の改善と呼ばない。これは自作校正の有限テスト合格数でありWSP scoreではない。元モデルで既に23/24を通ることは、この小集合でproxyを改善しても、より難しい公式科学コードへ移るとは限らないという限界でもある。

証拠は `advanced-science-original-generation-v2/{train,transfer}/receipt.json`（SHA256順に `42b1debc666636186b1e2ecb6f5ad63095668c41016b0fa12874f61480fac2b3`、`2ffecec35b9af20492a123b0911830ce95e239ab0cde3bd5d62125fd2c4efc5b`）と `advanced-science-original-execution-v2/{train,transfer}/report.json`（`1b2034b73fc9be908721f38792ebd1065381f655292af0733d262507b7f6e867`、`48cf49e1f60cd49ebb96160652fe450203250df0bd13a241945880a1ba7a2771`）。本監査では新しい生成・実行を行っていない。

## 一様 donor は何が既に試されたか

同じ公開 donor `Tsunamayo7/gemma4-31b-ja-agent-coder` の固定revision `b07b0f2a87a0ff83802e6dc6b7228bab5af55aea`、同じ公式元重みから、text410行列に `ΔW = 2cBA` を加えた `c=1` と `c=.25` は、2026-11-05に各3本の公式WSPを完了した。tokenizer / template / vision を変更したモデルではない。

| 一様係数 | 95 screen：code / BFCL | hard30：LCB / SciCode | full14 dev10 の主な差 |
|---|---|---|---|
| 1 | 17→14/20、34→32/40 | 15→15/20、30.3016→36.9683% | code −10pp、tools −5pp、overall −4.6346pp |
| .25 | 17→17/20、34→36/40 | 15→13/20、30.3016→35.3016% | code同値、tools −2.5pp、instructions −3.3333pp、long +9.9937pp、overall −0.0289pp |

各stageは別の実行であり、screen のBFCL利得が full14 で維持されたわけではない。`.25` のhard30 code差は−2.5pp（95% CI −12.5〜+7.5pp）。full14 overall CIは−2.6881〜+2.8727ppだった。旧full10 codeはSciCodeを含まない3 source、hard30はLCB20＋SciCode10で、今のdev90やknown64とは分母が違う。旧条件は4 replica、native thinking ON、公式generation config、追加thinking capなしで、現在の単一TP2 endpointと同一の実行負荷とはしない。

実 BF16 の変更数 / ΔL2 は `c=1` が22,574,585,664 / 34.358892、`.25` が12,569,638,846 / 8.696031。`.05/.10` の公式WSP記録は既存spec・manifest・run一覧・22候補45runの閉鎖監査に見つからない。新しい方向比較にある値は local-G 予測で、`model_calls=0`、`actual_bf16_rounding_measured=false`である。一様方向全体を「未評価」と呼ぶのも、全ての係数を「公式失敗済み」と呼ぶのも誤りである。

cached [固定公開card](https://huggingface.co/Tsunamayo7/gemma4-31b-ja-agent-coder/blob/b07b0f2a87a0ff83802e6dc6b7228bab5af55aea/README.md)はQLoRA NF4・1,546独自訓練例を宣言する。元モデル名は一致するが正確な訓練base revisionは不明で、訓練データ・評価harnessは公開されていない。作者の12カテゴリ9.4→10.0は宣言された自己評価であり、WSP / DeepSWEの改善証拠ではない。独自tool formatと公式Gemma parserの差、vision除外、量子化baseでの学習も、text BF16 mergeへの転移の限界に含める。

## JOINT の公式利得がなかった後の別仮説

大量の符号付き係数を再探索するより、**学習されたtext donor全体の相対係数を保つ、一つの小さい正の一様係数**を対照にする仮説が区別しやすい。これは400以上の係数を自作短文proxyへ合わせる方法と異なる。`.25/1` の再試行ではなく、例えば既存CPU監査のある `1/16` を一つだけ選ぶなら、実 ΔL2 1.888537、変更4,328,720,639要素、丸め前方向とのcosine 0.712784という費用と歪みが既知で、公式WSPは未測定である。新しいspec・重み・実行は本監査では作っていない。

`1/16` は保存checkpoint・spec・manifestが存在する候補ではなく、旧CPU丸め監査だけである。既存の `realized-summary.json`（SHA256 `e3a577115a1c4835063f4bf9fc71ce00e08f345f115d95a1fbade3373ed03e7e`）と当時のC++ / SOは残るが、summary自身にkernel SHAは記録されていない。現在のファイルhashを後から取得して当時の完全なbackend pinと見なさない。旧計測はFP32 NumPy matmul→2倍→係数→元BF16重みへ加算→1回BF16丸めであり、BLASの丸め境界まで現creatorと同じbyteになるかは未検証である。

一様係数でもBF16で完全な方向は保存されない。さらに小さい既監査係数 `1/256` / `1/64` のcosineは0.2163 / 0.4172であり、「より小さいL2」を唯一の選択理由にしない。旧 `.25/1` のSciCode部分点改善は仮説の弱い支持である一方、LCB・MBPP・toolsの損失もある。新係数で科学能力が90へ届くと予測する根拠はない。

[既存の重み・activation観測](../wsp-weight-observation.md)は層・token・TP・採取位置を分けた記録であり、特定headの因果的機能や長い科学プログラムの成功方向を特定したものではない。短い先頭位置のactivationから新しい巨大な係数探索を正当化しない。学習方向全体を比較する目的はこの不確かな局在推定を増やさず、proxy適合と実行能力の転移を切り分けることである。

## 証拠と境界

以下は実験artifact内の相対名。公式run・採点・過去記録は変更していない。

- `positive-final-v1/finite-joint-native-module/final-comparison.json`：SHA256 `edeb700978d138d6df8fdcf7472ba374acc822871adc9f76843b29a027bb98fc`。元重み一致、42 fold、84制約、全復元を含む。
- `positive-final-v1/balanced-dev90-{final-summary,evidence,scicode-diagnostics,parity}.json`：公式90問の比較とsanitized診断。evidence SHA256 `0423e6c8d756553655311600cc5e5bbaee0a4ed0d5d0f6d8d5832f3137bdf27a`。
- `positive-final-v1/direction-alternatives-v1/report.json`：一様係数の線形予測だけ。SHA256 `4b29b38edf392f61065d2c31e61e4ad34efa38b70b03167cc544d60324fa7c30`。
- 旧experiment rootの `reporting/gemma4-donor-text-{c1,c0p25}-{dev10-screen,dev30-hard-code,dev10-full}-20261105-comparison.json`、対応spec / manifest / thinking証跡。
- `analysis/public-adapter-donor-audit/{README.md,public-adapter-metadata-review.json,realized-summary.json}`：固定公開metadataと実BF16監査。card SHA256 `0506ebd49d25b607b7e125280a5200f8159b325dc5c3d428eb8f9c26636d7b41`。

この記録で新規モデル呼出し、GPU操作、benchmarkコード本文読取、download、候補spec生成は行っていない。

## 保存済み共同重みの検証

独立レビュー済みの joint creator で元の BF16 checkpoint から別コピーを作成し、430.45秒で完了した。保存後の実変更は84,142,113要素、ΔL2 0.020911997283708586でRAM実測と一致。candidate treeは `3d45f526266c2a79f74c4792830c6c06e148143b7394ae2976d77b67b71f08ea`、manifest SHAは `178eabccc5d880a1da9838b4f298ad7058429206be759450baaddeaf87a006f6`。元重みの全体hashは作成前後で一致した。

追加GEMMを行わない saved verifier は112.55秒で、410行列のRAM target SHA・92 native tensorの選択byteと全tensor再構築・全保存file hash・source continuityを照合した。全て一致し、freeze SHAは `d31813d65ebd59774ed641a481499f411df3318a50f0c74acb64e813fc6881ad`。これはCPU再構築と記録済みGPU byte証拠による同一性確認であり、新しい精度評価ではない。

全体CPU検証で、新しいfixtureのimport時kernel参照が後続の既存fixtureへ残る問題を発見した。テストのmodule cache隔離だけを修正し、生成・検証の本体や数値kernelは変更していない。修正後は471件中419成功・52任意依存skip、失敗・error0、Ruff/mypy通過、検証中source変更0。失敗した初回のCIも `cpu-integration-joint-save-v1` に保持し、成功結果は `cpu-integration-joint-save-v2` に分けた。
