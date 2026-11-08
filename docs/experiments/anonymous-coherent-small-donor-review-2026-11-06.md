# 一様 donor H の匿名手法レビュー・dev90・全14分野（2026-11-06）

H は、固定公開 donor の text410行列を一様係数 `1/16` で元のBF16重みに加える実験である。module別の符号付き係数を短い校正proxyへ適合させたJOINTが公式dev90で改善しなかったため、学習済み方向全体の相対関係を保つ別仮説として選んだ。**Hの保存・CPU検証と公式dev90は完了**。新しい元モデル71.325397%に対しH72.158730%（+0.833333pp、95% CI [−3.750000,+5.416667]pp）で、分野別には利得と損失が混在した。続く全14領域も完了し、総合80.164156→81.972528%（+1.808373pp、95% CI [−0.178234,+4.196766]pp）の一方、code87.777778→84.444444%（−3.333333pp）だった。H単独はコード改善候補として採用せず、将来の合成の可能性だけを未検証仮説として残す。安定したコード利得・コード90達成は主張しない。

## 受理した意見と先行試行

送信した内容は公開一般機構・匿名H仮説・反証方法で、私有の問題文、座標、生成軌跡、候補の測定値は送っていない。v5の公開最終回答を方法論のセカンドオピニオンとして受理した。assistant modelとmodel usageはいずれも `claude-opus-5-5` のみ、tools=0、returncode=0、75.422秒、記録cost $0.1430274。入力2 / 出力6,991 tokenに加えcache read 1,077 / cache creation 373を記録している。これはCLIのusage記録である。

| 試行 | 実際に観測したassistant model | 受理状況 |
|---|---|---|
| v1 | Opus 5.5と4.8 | 指定model以外を含むため不受理 |
| v2 | Opus 5.5と4.8 | 同上。構造化した回答があっても指定model条件を満たさない |
| v3 | Opus 5.5のみ | 当方helperのJSON形式・抽出契約で最終回答を受理できなかった |
| v4 | Opus 5.5のみ | 同じ形式側の問題でusable finalが残らず、不受理 |
| v5 | Opus 5.5のみ | `verified_requested_model=true`、`review_usable=true`、`review_format=public_final_text` |

v3/v4をmodel不一致や「意見がなかった」事例と扱わない。ユーザーが求めたのは意見であり、こちらのJSON外形ではない。helperを単一label向けの明示opt-inで修正し、公開final textを受理できるようにした。assistantの内部block、hidden thinking、raw stream、環境変数は保存していない。各試行の原receiptは変更せず残した。

## 助言と今回の判断

レビューは、固定の正の一様係数で学習済みcross-layer方向を試すことを、低自由度の妥当な実験とした。ただし新しい手法の種類ではなく、通常のtask-vector / LoRA scalingの比較である。NF4量子化baseで学習したdonorの正確なbase revision、訓練・評価データの重複、tool formatとthinking条件の違いは未解決である。小さい係数ではBF16丸めが方向を変え、差が実行変動に埋もれる可能性もある。

採用するのは、元重み・donor・設定を固定した実保存値の検証、同条件の元モデルとHの公式実行比較、得失・長さ・打切り・例外の報告である。既存recipeは410個のA/B mapping、shape、`alpha/r=32/16=2`、FP32積と加算、最後の1回BF16丸めを明示し、vision・tokenizer・templateを除外する。Hの実効LoRA倍率は `2 × 1/16 = 0.125`。

レビュー内の「hard requirement」という分類を、そのまま新しい採用条件や追加承認要求にしない。任意のcosine閾値、追加の大規模GEMM、全teacher-forced replay、runtime adapterへの変更、追加振幅gridは今回の必須手順にしない。現在のCPU検証は既存recipeを一度再生成して保存byteを照合するもので、同じ数式を別実装で独立証明したとは呼ばない。[検証bridge](../wsp-dense-checkpoint-freeze.md)は元・候補の全file hash、未編集領域、spec / code / kernelと前後inventoryを束縛する。GPU/RAMとの比較ではない。

係数は過去の一様 `.25 / 1` の公式結果と既存丸め監査を見た後の選択であり、未見評価から完全に独立した選択と偽らない。Hの結果が悪くてもこの係数での結果として扱い、学習済み方向の全振幅を一括して否定しない。逆に少数の改善やproxy値だけで一般的なコード能力改善を確定しない。

## 保存重みの実測と未測定量

作成manifestは以下を記録した。owner launch全体は415.43秒、manifest内のcreator経過は413.578秒で、計時区間を混同しない。

| 項目 | 実測 / 固定値 |
|---|---:|
| 一様係数 | 0.0625 |
| 変更tensor | 410 |
| 変更BF16要素 | 4,328,720,639 |
| 実 ΔL2 | 1.888537495617679 |
| 全stored浮動重みの元L2に対する比 | 0.00001931431713（0.00193143%） |
| 全stored浮動要素に対する変更率 | 13.841679% |

L2の分母は全1,188 stored浮動tensor・31,273,088,876要素のノルム97,779.1491644232で、visionも含み、tied重みは保存1回分を数える。変更率とは別の指標である。対象text行列内だけの比へ読み替えない。

旧CPU監査のcosine **0.712784** は歴史的測定であり、今回のcreatorのfresh cosineではない。今回の変更数・ΔL2が旧監査の数値と一致しても、旧監査は現kernelのhashを当時pinしていないため、その一致だけで全byteやcosineの再検証済みとはしない。fresh cosineは未測定のまま保持する。

- candidate tree: `0f9775ba6a2406989407d86d365eed69e1684a4400faddf56bb3822fad6b98d3`
- creator manifest SHA256: `b1eb4249f07baebb4796ec12e8d1097ea50b330f46a9e3128c0279dc801fc6fd`
- 数値recipe: `dense_lora_recipe.py` SHA256 `ca84dde8dc3947fccadc8b725fa387aca6c5643910fe79080b8bf0157823081c`
- compiled kernel SHA256: `7796c76da941982bf442de539883e58436d101a64929d2488f2817f0da0af18f`

CPU私有コピーの準備はレビュー取得と並行して進めた。これはユーザーが承認済みの改善作業であり、レビューを事後承認へ置き換えたものではない。usableなOpus 5.5意見は新しいH配信・WSPの前に受領した。CPU検証freezeとCIの成功後、新しい元モデルdev90をHと並行に測定した。単一TP2 endpoint・c32・native thinking ON・公式generation configを揃え、過去の元モデル参照も保持した。結果は末尾に追記する。

## JOINTの結果を利得として再利用しない

[JOINTの公式比較](2026-11-06-joint-rounding-and-donor-audit.md)は全90採点・未採点0で、元72.575397%→JOINT70.492063%、差−2.083333pp、paired 95% CI [−5.416667, 0.000000]ppだった。gain0 / loss2で、失点IDは `livecodebench:arc195_b` と `mbpp-plus:Mbpp/641`。JOINTは不採用であり、校正42fold中38のLL改善を公式コード利得として再利用しない。これはdev90であってknown64の>90判定ではない。

JOINTのexact manifestを役割別に集計すると、K projectionのΔL2は0.016730152、Vは0.008251460だった。非重複tensorなので二乗和で計算し、K/Vは全ΔL2二乗の**79.5736%**を占める一方、変更数は18,978,388 / 84,142,113（22.5552%）だった。これは保存重みの変更分布だけの観測であり、K/Vに特定の科学・推論能力が局在する、またはその変更が失点を起こしたという証拠ではない。

## 証拠

以下は実験artifact rootからの相対名。新しいモデル呼出し・benchmark本文読取・採点変更は本記録作業では行っていない。

| 証拠 | SHA256 |
|---|---|
| `opus-coherent-small-donor-review/receipt.json` | `dc1b23e22828f0dbe922078de3b9ef55b05841811424b3f557ef3e8db236a59b` |
| `opus-coherent-small-donor-review-v2/receipt.json` | `34276a2a42b958ba6de155dc5ac684c22748182983e6ea025f3e83668e6b112c` |
| `opus-coherent-small-donor-review-v3/receipt.json` | `cbf4e8be730eaa63afe045a94c826cd3a85736ca0aa1472e14fe9cc3a624d354` |
| `opus-coherent-small-donor-review-v4/receipt.json` | `78e73990a83d6ef06c81a73dfef1ea6a2bfd61179a1b009b32129eb1d95e5f36` |
| `opus-coherent-small-donor-review-v5/receipt.json` | `272a3b7ac2774c7c4c5ac9b83b3b6f95907b22d16687eb1f466b24cd793365a4` |
| `positive-final-v1/joint-dev90-final-summary.json` | `27aed8b199faa22151e2a32704f48440ceb5e76668ecc780ae04f589830c90ec` |
| `positive-final-v1/joint-materialization/edit-footprint.json` | `20b955bd93074b92aa299ca4a10046f8bc194424a0bae21cdd727b1e04a1d2b7` |

Hのcreator manifestはexperiment rootの `manifests/code90-coherent-donor-c00625-v1.json`、分母は同rootの `analysis/global-weight-norms.json`。過去の丸め監査・公開donorの宣言と限界は[一様donor監査](2026-11-06-joint-rounding-and-donor-audit.md#一様-donor-は何が既に試されたか)を参照する。

## 保存値のCPU再計算検証完了

既存dense recipeによる1回の全410行列再計算、元・候補の全file hash、未編集領域、前後statの検証を完了した。GPU/RAM比較は行っていない。検証本体は289.52秒、freeze SHAは 8941b3271db0d2593871f114a53759d3ad4741741b10f0fd8d089bca6af37a8b。追加したbridgeは数値recipeを変更せず、この既存検証結果を配信用freezeへ接続する。

最新全体CPU検証は475件中423成功・52任意依存skip、失敗/error0。Ruff/mypy通過、検証中source変更0。配信後にHと元モデルのdev90を同時に新規実行し、以下の公式点数を得た。

## 完了した公式 dev90 の対照結果

主対照は今回新しく測った元モデルであり、過去の元モデル点へ差し替えない。
同一90 ID、tier30/dev、manifest・model IDで、双方90採点・未採点0・応答エラー0。
codeは4 sourceの等重み平均である。

| Source | 問数 | fresh original | H |
|---|---:|---:|---:|
| HumanEval+ | 30 | 28 / 30 | 30 / 30 |
| MBPP+ | 30 | 23 / 30 | 22 / 30 |
| LiveCodeBench | 20 | 16 / 20 | 15 / 20 |
| SciCode | 10 | 35.301587% | 40.301587% |
| **公式code** | **90** | **71.325397%** | **72.158730%** |

主比較の差は **+0.833333pp**、公式paired bootstrap 95% CIは
**[−3.750000,+5.416667]pp**（2,000 resamples、seed=20261104）。0をまたぐ区間であり、
この一対だけで安定した能力改善と結論しない。変化は次の5項目だった。

| ID | fresh original → H | 旧originalの点 | 解釈の限界 |
|---|---:|---:|---|
| HumanEval/151 | 0 → 1 | 0 | 今回の新利益候補、反復未確認 |
| HumanEval/95 | 0 → 1 | 0 | 同上 |
| LCB abc390_g | 1 → 0 | 1 | 今回の新退行候補、原因未分類 |
| Mbpp/801 | 1 → 0 | 0 | 既存の元モデル変動範囲内 |
| SciCode/14 | 0.5 → 1 | 1 | 既存点の回復、科学能力の新規獲得とは断定不可 |

公式の二値flipはgain2 / loss2で、SciCodeの+0.5部分点は別に記録する。
旧originalのcode72.575397%に対してHは−0.416667ppだが、これは主比較ではなく歴史的context。
旧→fresh originalの変化はMBPP641 1→0、MBPP801 0→1、SciCode14 1→0.5であり、
元モデル反復の変動も存在した。

| 記録量 | fresh original | H |
|---|---:|---:|
| API呼出し | 119 | 119 |
| 入力token | 97,175 | 93,972 |
| 出力token | 327,614 | 327,353 |
| stop / length終了 | 119 / 0 | 119 / 0 |
| wall秒（run開始→完了） | 299.883 | 369.643 |

両runは07:38:18 UTCに開始し、元モデル07:43:18、H07:44:28 UTCに完了した。
API119回はSciCodeの39個のleaf呼出しを含み、agent tool手数ではない。
wallはload・生成・採点を含む全体時間で、各段階を別に加算しない。

両実runnerのCIDを稼働中に取得し、同一image
`eb2e43c2f08d6f76fbbf71917543607cb92b9a633202fc3cd381687c2a2a73e9`、
runtime Config SHA、16層RootFS SHA、User、host設定の一致を確認した。
sourceは固定archive5e3f2df、測定はそれぞれ直接の単一TP2 endpointで、
mixed gatewayを使用しない。過去の元モデル固有CIDの欠落を今回の対へ持ち込まない。

証拠は `coherent-donor-c00625-v1/dev90-{evidence,final-summary,progress,original-repeat-context}.json` と
`dev90-runner-evidence-{h,base}.json`。主比較evidence SHA256:
`7aeef9d1d5489263d8803d42b859127ef6f7e1951586587e026b8b59b440bcc7`、
final-summary SHA256:
`8d1deea8f0002efcb1923bf73d6b3a9086a786c6ede46b3e95c0aa388eb1dcaf`。

この記録は既知の数値fieldと既存の安全な集計helperだけを使う。生成回答・問題本文・隠れた推論・
追加診断fieldの探索はしておらず、各失点の具体的機構は未分類のまま。
このdev90は全14領域overallでも、known64のcode >90確認でもない。

## 完了した全14分野の対照結果

各778項目（能力746項目＋probe32項目）、tier10/devで双方completed・778採点・未採点0・応答エラー0。source/manifestにfilterを加えず、chat32・logprob8・agent1、archive5e3f2df、直接TP2 endpointを揃えた。主対照は今回のfresh originalで、旧full14は項目別の既存範囲だけに使う。

- H: gemma4-code90-coherent-c00625-c32-dev10-full14-20261106
- fresh original: gemma4-code90-original-base-c32-dev10-full14-pairh-20261106

公式overallは **80.164156→81.972528%（+1.808373pp）**、paired 95% CI **[−0.178234,+4.196766]pp**。2,000 resamples / seed20261104で、区間は0を含む。

| 領域 | fresh original (%) | H (%) | 差 (pp) |
|---|---:|---:|---:|
| knowledge | 90.000000 | 90.000000 | +0.000000 |
| expert | 75.000000 | 81.250000 | +6.250000 |
| facts | 72.500000 | 70.000000 | -2.500000 |
| math | 95.625000 | 97.187500 | +1.562500 |
| code | 87.777778 | 84.444444 | -3.333333 |
| instructions | 92.083333 | 92.083333 | +0.000000 |
| tools | 82.500000 | 82.500000 | +0.000000 |
| agents | 50.000000 | 50.000000 | +0.000000 |
| long-context | 81.377943 | 87.627943 | +6.250000 |
| reasoning | 74.782609 | 76.956522 | +2.173913 |
| commonsense | 66.333333 | 67.333333 | +1.000000 |
| multilingual | 80.707071 | 79.343434 | -1.363636 |
| vision | 87.500000 | 100.000000 | +12.500000 |
| safety | 86.111111 | 88.888889 | +2.777778 |

このtier10のcodeはHE10・MBPP10・LCB6の3 source等重み平均で、SciCodeは入らない。HE10/10とLCB5/6は同じ、MBPP8/10→7/10で、code差の95% CIは[−10.000000,0.000000]pp。dev90のHE2件獲得はこの26問には含まれず、4 source平均72.158730%と混同しない。

点変化は25件（正16・負9）。公式二値flipはgain15 / loss9で、HMMTの0.5→0.75は別の部分点変化。主なtradeoffは次のとおり。

- code: Mbpp/801を喪失。fresh/旧full14の両originalが1、Hが0。dev90で観測した元モデルの変動も別に保持する。
- tools: live_multiple_924-191-12を喪失、multi_turn_long_context_15を獲得し33/40のまま。後者は旧originalにも成功歴があり、新規能力獲得とは断定しない。
- instructions: IFBench182を喪失、259を獲得。IFEval39/40・IFBench13/15は同点。
- multilingual: MGSM bn:024 / bn:243を喪失、JAMC geography-00039喪失 / geography-00050獲得、MMLU-ProX-JA4820獲得。領域全体は−1.363636pp。
- long-context: RULER fwe:131072:005を獲得し7/8→8/8、MRCR75.255885%は同じ。expert GPQA0101、visionの2件などは総合に寄与したが、コード目標の達成とは別である。

全loss ID（原因の本文診断なし）:

- bfcl:live_multiple_924-191-12
- ifbench:182
- jamc-qa:jamcqa-test-geography-00039
- mbpp-plus:Mbpp/801
- mgsm:bn:024
- mgsm:bn:243
- mmlu-pro:3789
- triviaqa:qz_6077
- winogrande:0398

| 記録量 | fresh original | H |
|---|---:|---:|
| 認識済みsample呼出し | 559 | 560 |
| 入力token（記録分） | 1,394,418 | 1,396,754 |
| 出力token（記録分） | 1,052,878 | 1,015,179 |
| stop / tool_calls / length終了 | 471 / 85 / 3 | 474 / 86 / 0 |
| wall秒（run開始→完了） | 1,677.427 | 1,349.816 |

双方TerminalBench2件のcall/agent-step fieldは既存集計で欠測であり、上のcall数は完全なHTTP総数ではない。欠測手数を0とせず、logprob probe/選択肢処理とも合算しない。lengthは終了記録で、失点機構やscoreへの寄与を推定していない。wallは全体時間でload等を二重加算しない。tier10はwall10分を意味せず、今回は元27分57秒、H22分30秒だった。

両実runnerを稼働中に記録し、image SHA256 50e5a080ea01e9396ff0fe9cc3279a4df4090667a89f543edf2ea94e63e5b107、runtime Config、16層RootFS、User・host設定・mountが一致した（mountは同一集合、列挙順だけ異なる）。CIDは別で、H 36658ac54fb9… / 元07c3d2016aed…。これは実runner証拠であり、checkpoint payloadをこの集計時に再hashしたという意味ではない。

結論は**総合に利得と退行が混在し、H単独のコード改善は未確認**。許容するtradeoffを一律ゼロへ変更せず、現コード目標に対しては単独採用しない。別方向との将来の合成は未検証であり、今回の総合差をその成功証拠にしない。

証拠: coherent-donor-c00625-v1/full14-{evidence,final-summary,progress}.json と full14-runner-evidence-{h,base}.json。

- evidence SHA256: a5e8a582d5f8c008305bf3cc88b70d3f848e44d86995dc8b52e737b73b9baf3e
- final-summary SHA256: e5db193e9d54fa8a322018ae0e7837b8b87351152eafd4d4d097dd6ef208488a
