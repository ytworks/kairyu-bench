# helper / long 全16件の既存隔離採点への接続可能性（読取りのみ）

**既存の数値fixtureを全16件に適用するなら、小さな offline 入力adapterで可能。新oracle・worker変更は不要。公開契約の全項目を完全検証する機能は現在ない。** 現48件の凍結計画と、未対応72という分類・分母は変更していない。実装・生成・reference計算・fixture実行・Docker/GPU操作は0回。

helperは whitened-fit / boundary-response / quadrature-mixture / log-mixture-score の4 family×2 stage、longは weighted-constraint / conditional-covariance / affine-state-history / affine-mixture-moments の4×2。全16の現 `row(stage)` が元120の **prompt/correct/foilを含む全フィールドと完全一致**した。[16 ID・全hash・公開context同一性](records/2026-11-07-helper16-offline-feasibility/read-only-feasibility.json)を保存した。元120 SHAは `bd6fcba70e93b1870ee882f6e92893b93490ad4099af7a9a290ba3d324d68951`、未変更。

既存APIの対応は次の通り（実行せずsourceだけ確認）。

| 元の課題 | 将来のadapterが渡す code | 既存API |
|---|---|---|
| short helper stage1（4件） | 抽出した公開回答だけ | `helper_chain_cases.score(chain, 0, code, run=docker.run, seed=20261107, trials=8)` |
| short composition stage2（4件） | **promptに既に載る固定helper**＋公開回答 | 同 `score(chain, 1, ...)` |
| long helper stage1（4件） | 公開回答だけ | `long_helper_chain_cases.score(chain, 0, ...)` が公開supportを1回前置する |
| long composition stage2（4件） | **promptの固定helper**＋公開回答 | 同 `score(chain, 1, ...)` が公開supportを1回前置する |

公開helperは `short.as_helper(chain.helper.code())` と prompt 内の提示コードの一致を確認済み。これは課題で既に与えられたcontextであり、非公開の正解プログラムを追加することではない。stage1の生成に失敗しても、固定helperを与えられたstage2は別課題として採点可能。生成されたhelperへの差替えやstage2の省略はしない。[evaluate()](../../scripts/experiments/gemma4-purpose/helper_chain_cases.py#L489) は別条件の generated-helper 全chain測定も追加し、`evaluate_requests()` は追加生成するので、そのまま使わない。

安全境界は既存のまま維持できる。[short payload/score](../../scripts/experiments/gemma4-purpose/helper_chain_cases.py#L467) と [long score](../../scripts/experiments/gemma4-purpose/long_helper_chain_cases.py#L786) は、既存 `samples()` の期待値をhost側に残し、隔離 `Docker.run(payload)` へ **code / parameters / trials（入力だけ）** を渡す。期待値・非公開の参照答案は渡さず、比較はhostの `sandbox.compare()` が行う。longの公開supportをadapter側でも足すと二重になるため、既存scoreに任せる。生成codeは [WORKER](../../scripts/experiments/gemma4-purpose/science_sandbox.py#L45) 内だけで実行し、host mountなし・network none・uid1003・cap-drop ALL・no GPUの既存Docker入口を使う。`trusted_run()` は組込み著者コード専用のhost subprocessであり、生成codeへの流用は禁止。

不足は **保存済み回答のreaderと全16 dispatch/status**。[現read_inputs](../../scripts/experiments/gemma4-purpose/science_sandbox.py#L451) は既存5 catalogの完全一致split専用で、chain IDや全72応答の入口がない。最小adapterは元の **全72 case envelope＋全72 response** の `cases_sha256`・ID重複/欠落・公開回答schemaを検証し、事前固定した全16だけを上表へ対応させる。他56の記録も保持する。8件用へ応答を切り出してcase SHAを書き換えない。各IDのprompt/context/source/response hash、format error・生成失敗・timeout・unknownを残す。既存5 catalogの受理条件を緩めず、別の明示入口にすれば変更範囲は限定できる。将来実装時の主要検証点はidentity不一致、全16漏れ/重複、固定helperとgenerated helperの混同、expectedがpayloadに入らないこと。

契約上は [既存11反例](2026-11-07-authored-case-audit.md) のうち `chain-whitened-fit-1`、`chain-quadrature-mixture-1`、`chain-boundary-response-1` が含まれる。元correctは許容されたfloat32/整数入力でfloat64出力契約に反する。既存short samplerはこれらのreal入力をfloat64で作るため、この反例を新たに検出する採点にはならない。3件を削除・救済せずraw結果と契約疑義stratumを併記する。longの公開promptはreal入力float64を明示しており、その条件をshortへ移植しない。また現workerは数値・shape・dtype・入力mutationを検査するが、helper再定義禁止・指定helper再利用などの全指示遵守までは保証しない。

したがってこれは **既存の有限数値fixtureへの限定的接続案**であり、能力得点や全契約保証の追加ではない。現在の48を優先し、追加16の実装・採点・分母変更は別の明示判断を待つ。参照した3 sourceのSHAと全16照合結果は上記JSON、[そのSHA](records/2026-11-07-helper16-offline-feasibility/SHA256SUMS)に固定した。`.catalog()` / `.row()` とASTによる公開helper整形だけを使い、`.samples()` / `.score()` / `.trusted_run()` は呼んでいない。
