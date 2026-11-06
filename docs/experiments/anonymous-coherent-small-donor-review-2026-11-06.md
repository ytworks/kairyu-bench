# 一様 donor H の匿名手法レビューと実行前記録（2026-11-06）

H は、固定公開 donor の text410行列を一様係数 `1/16` で元のBF16重みに加える実験である。module別の符号付き係数を短い校正proxyへ適合させたJOINTが公式dev90で改善しなかったため、学習済み方向全体の相対関係を保つ別仮説として選んだ。**本記録時点でHの保存重み作成は完了、CPU再検証は進行中、Hの公式成績は未測定**。コード90達成や能力改善は主張しない。

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

CPU私有コピーの準備はレビュー取得と並行して進めた。これはユーザーが承認済みの改善作業であり、レビューを事後承認へ置き換えたものではない。usableなOpus 5.5意見は新しいH配信・WSPの前に受領した。配信・評価の開始はCPU検証freezeとCIの成功も確認してからとする。新しい元モデルdev90をHと並行に測り、単一TP2 endpoint・c32・native thinking ON・公式generation configを揃える計画で、過去の元モデル参照も消さない。この段落は実行計画であり、完了済み測定ではない。

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

最新全体CPU検証は475件中423成功・52任意依存skip、失敗/error0。Ruff/mypy通過、検証中source変更0。配信後にはHと元モデルのdev90を同時に新規実行する。この節の時点では、その公式点数はまだ得られていない。
