# 目的微分の費用比較とBF16有限応答（2026-11-06）

codingと関連能力を優先し、他領域の小さい損失は目的全体の得失で判断する方針で、
保存済みの全層native微分から3案をCPUで比較した。
公式thinking ONの公開assistant回答をteacher-forceするchat条件の421 tensor・1,331,516座標・30foldを再利用し、
raw Fの微分や過去holdoutの成績は入力にしていない。
再利用レシピは[gemma4-purpose](../../scripts/experiments/gemma4-purpose/README.md)。

## CPU比較で分かったこと

3案のcoding目標・tool/general保護を同じにして、絶対L2と役割別相対費用を比較した。
相対費用は1/max(abs(weight), role floor)で、floorは役割内の絶対値中央値の0.05倍と0.2倍。
役割は同名tensorのlayer番号を取り除いた分類であり、機能の因果的同定ではない。
小さいfloorでの数値発散を避ける正則化であって、Fisherやactivation摂動の実測ではない。

| 案 | 変更値 / tensor | 実ΔL2 | 連続→実方向cos | 局所G制約 |
|---|---:|---:|---:|---|
| A 絶対L2 | 1,139 / 50 | 8.73829e−5 | 0.74684 | 充足 |
| B 相対floor 0.05 | 739 / 60 | 9.52106e−5 | 0.000885 | 未達 |
| C 相対floor 0.2 | 761 / 60 | 9.66514e−5 | 0.01396 | 充足 |

Aの実変更はpost-attention 943値とpost-FFN 196値だけで、元値中央値は0.00013542。
B/Cでも生存した変更はすべて各役割floor未満だった。
B/Cの連続提案ノルムのうち実変位として残った比は0.110% / 1.758%であり、
相対費用で意図した連続方向はBF16丸めでほぼ消えている。
floor変更による実変位同士のcosは0.949865、共通変更712 / union 788。
この比較は29.01秒、追加forward・実checkpoint作成・WSP実行なしで終えた。

Cの全保存重みに対するΔL2比は9.88466e−10。
分母97,779.1491644232はvisionを含む全保存浮動小数tensorのFrobenius normで、
tiedは保存1回分だけ数える。パラメータ個数比ではない。
変更が小さいことを能力改善と同一視しない。

## 後続の実有限forward

上の時点では全3案とも有限応答が未測定だった。
その後、所有者がA・Cと既測定0397を同じGPU/BF16/chat条件で
base→固定候補→baseの順に測定した。
この後続測定は41.84秒、base復元時のmargin / NLL / KL差はすべて0。
0397対照の前回有限測定との差もすべて0だった。
GPU peak allocatedは60.3653GiB、reservedは62.5430GiB。
Bは有限未測定のまま。

以下は正と対照の平均log probability差の変化であり、正解率の百分点ではない。

| 目的 | A 実有限margin差 | C 実有限margin差 |
|---|---:|---:|
| code | +0.03000 | +0.06841 |
| tools | -0.06545 | -0.00282 |
| instruction | -0.01733 | +0.03651 |
| general | -0.05445 | -0.03369 |
| reasoning | +0.05671 | -0.05694 |
| japanese | -0.12047 | +0.62193 |
| implementation | +0.00700 | +0.00580 |
| algorithm | +0.01669 | +0.01060 |
| tool_call | +0.01835 | -0.00629 |
| tool_no_call | +0.03870 | -0.01751 |
| tool_arguments | -0.00355 | -0.01663 |

局所Gによる予測と実応答は一致しなかった。
例えばAのtools予測は+0.00838、general予測は+0.00354だったが、実際は−0.06545 / −0.05445。
Cのtools予測+0.00653は実際−0.00282、general予測+0.00205は−0.03369、
reasoning予測−0.02461は−0.05694だった。
Cのcode平均は+0.06841だが2foldは−0.00421 / +0.14103で、両foldの改善ではない。
tool call・non-call・引数の個別目的の損失も表に保持した。

したがって「相対費用が意図した連続方向を保持した」とは言えない。
一方、実際にBF16で残ったCの方向には別途有限応答が測定された。
この2つは矛盾せず、丸めの幾何とモデル応答を別の証拠として扱う必要がある。
所有者はCを公式WSPで検証する判断をした。Aを優先せず、0397も比較対照として保持する。
有限marginだけでCを採用・能力改善とは断定しない。

## 証跡と信頼境界

実験artifactはlocal-g-cost-comparison-20261106配下の
plan.json、results各report/spec、analysis-summary.json、
actual-finite/{summary,provenance,base-restoration}.json、
finite-followup-summary.jsonで対応する。

実有限plan SHAはa8bbf41b8d90a4ad14b4886dff0c4d6ca0831d90f5ccf4ea05ab6b45173bd829、
runner SHAは2eb004009d038b3523c1eb71a478696f3dde20f1c3871474481c8a9346d633a8。
実行版にはreader依存hash追加が含まれ、ある旧commitそのものを走らせたとは記録しない。

CPU比較は保存basisの値、対応するcase/frame/source identityとtensor inventoryを信頼する。
元checkpoint payloadを再読・再attestした処理ではない。
初回CPU runの証跡には実依存finite_response.pyのSHAが欠けていた。
後続修正で依存SHAの記録・終了時照合と全profileの制約一致検査を追加した。
当初のartifactは変更せず、dependency-provenance-followup.jsonへ修正時点のhashを記録した。
これを「初回から記録されていた証跡」とは扱わない。

## Cの公式WSP結果：screen95とhard-code30

Cを元checkpointから作成・照合したtreeは
`cc3bf6d50a1a71648d67367e78e5dc9151f92f76f97a641042d9f80e441e5006`、
spec SHAは
`c52daac9975bf06556d0c67afd1b167491527be53f7d63592451ea7cdfcbc078`。
以下はdevだけの比較で、過去holdoutを探索・選抜に使っていない。
標準WSP比較を[再利用集計レシピ](scripts/README.md)から呼び、
共通項目だけを比較した。95問screenを全14領域baselineとは呼ばない。

### screen95

`gemma4-resume-localgc-dev10-screen-20261106` は95/95採点・未採点0、
71.9秒（load10.1秒を含む）、171 chat calls、出力86,070 tokensで完了した。
20 probe項目は下表の正解数に混ぜない。

| 同じ95問に含まれる評価 | 直近の観測付き元モデル | 0397 | 0485 | C |
|---|---:|---:|---:|---:|
| HumanEval＋MBPP | 18/20 | 17/20 | 17/20 | 18/20 |
| BFCL | 35/40 | 35/40 | 34/40 | 34/40 |
| IFBench | 12/15 | 12/15 | 13/15 | 13/15 |

Cと直近元モデルのコード20問は、各問の成否まで同じだった。
旧95問基準や0397/0485に対するMBPP801の回復は、
元モデルの旧dev30・直近観測付きdev10でも既に出ており、新固有利得ではない。
BFCLは無関係なツールを呼ばない `live_irrelevance_287-69-0` を失い、
公開出力では不要な2関数を呼んだ（1 call、2,335 output tokens）。
0485が失ったoptional引数の924-191はCでは正解へ戻ったが、
BFCL全体では直近元モデルを上回らない。
IFBenchは93・259を獲得し182を失った。
93は全単語を改行する形式、182は文ごとに連続する頭韻語列を長くする制約である。
Cの93/259/182は各1 call、出力1,411/2,335/1,906 tokens。
これらは項目単位の交換で、総数だけで一律改善とは記録しない。

### hard-code30

事前条件（code≥16/20、BFCL≥32/40、未採点0）を満たしたため、
同じ重みで `gemma4-resume-localgc-dev30-hard-code-20261106` だけが続いた。
30/30採点・未採点0、265.8秒で終了した。
run.jsonの開始18:32:14.433 UTC、終了18:36:40.210 UTCを根拠とする。
load7.5秒はwallの内数、generation202.1秒とwallは同じ値ではない。
59 calls、入力87,218・出力263,252 tokens、全呼出しがstopでlength打切り0。
並行呼出し時間の総和2,451.94秒はwall時間ではない。
ここでのcall数はAPI呼出し数であり、agentのツール手数ではない。

| 同じ公式項目 | 旧元モデルdev30 | 直近通常配信の元モデル | 0397初回／反復 | 0485 | C |
|---|---:|---:|---:|---:|---:|
| LiveCodeBench 20問 | 15/20 | 14/20 | 16/20・15/20 | 16/20 | 14/20 |
| SciCode 10問の平均step得点 | 30.3016% | 未測定 | 35.3016%・未測定 | 40.3016% | 40.3016% |

CのLCB20問は直近通常配信の元モデルと全問の成否が同じだった。
0397初回・0485に対して `arc195_b` と `abc390_g` を失い、
0397反復に対しては `arc195_b` だけを失った。
SciCode10問は0485と全項目同じで、0397からの増分は14の1/2→2/2だけ。
この14は旧元モデルdev60でも2/2で、過去範囲を超える新固有利得ではない。
旧dev60はこの履歴確認だけに使い、dev30の公式比較・CIへ混ぜない。
旧dev30との共通30問のcode差は+2.5ポイントだが、
95% CIは[-7.5,+15.0]ポイントで、LCB−1問とSciCode14回復の合成である。
0485との差は−5.0ポイント、95% CIは[-12.5,0.0]ポイントだった。
subsetから全領域overallを算出しない。

| 問題 | 要求とCの観測 | Cのcall数 / 出力tokens / API秒合計 |
|---|---|---:|
| arc195_b「Uniform Sum」 | 未知値を非負整数で埋め、要素を重複使用せず全ペアの和を統一する。最終コードはA×Bの組合せ数を対応数と誤認する。A=[1,1,2]、B=[2,2,2]では同じ要素を再利用して和3を6組と数えYesにするが、要件上はNo。 | 1 / 11,429 / 102.21 |
| abc390_g | 1〜Nの全順列を十進連結した整数の総和を求める。保存finalの式をN=2で追うと30になるが、12+21は33。 | 1 / 12,271 / 105.34 |
| SciCode14 | 光ピンセット中のBrown運動のLangevin積分と平均二乗変位を実装する2段階。公式2/2合格。 | 2 / 8,980 / 84.76 |

上の反例は公開最終コードを静的に追ったもので、追加コード実行はしていない。
LCBの保存公式detailは `passed:false` のみであり、
反例が実際の最初の失敗テストだったとは断定しない。
生成されたhidden reasoningは分析・掲載していない。

この結果では有限code margin +0.06841が新しいhard-code正解へ移らず、
tool関連proxyの負側もscreenの無関係呼出し失点と併存した。
ただし、この一致だけでproxyと失点の因果関係は示せない。
C単独を全14領域へ昇格させず、探索所有者が次の判断を行う。
自動driverはscreenとhard30で終了し、追加評価・別候補への自動遷移はない。

証跡は `analysis/resume-localgc/{screen,hard30}/` の
公式比較・項目別usage差分・公開最終コードのSHA付き静的確認に保存した。
一部の起動補助command.jsonにはテンプレート由来の旧pid/started値が残ったため、
時刻・完了状態は実run.jsonとdriver statusを使い、
補助記録の訂正証跡は別に保存する。元記録は改変しない。
