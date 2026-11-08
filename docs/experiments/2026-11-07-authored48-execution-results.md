# Authored48 CPU execution after AV native generation v2

全48件を指定の既存採点器で1回ずつ実行し、**46合格・2失敗・未採点0**となった。
各caseの固定8試験は合計375/384合格、採点全体のwallは49.8836秒。
これは既知の自作校正問題の実行結果であり、公式WSP得点、未見性能、重み改善ではない。

| Catalog | 完了 | 合格 | 失敗 | 未採点 | subprocess秒 |
|---|---:|---:|---:|---:|---:|
| basic | 16/16 | 15 | 1 | 0 | 12.2495 |
| advanced-v2 | 16/16 | 16 | 0 | 0 | 12.2801 |
| library-train-v1 | 16/16 | 15 | 1 | 0 | 24.2758 |

既知の契約疑義3件（cumulative-trapezoid / reset-running-stats /
capped-reservoir-chain）は3/3、24/24試験合格。それ以外45件は43合格・2失敗、
351/360試験合格。疑義3件を削除・救済せず全48分母にも含める。
今回の固定samplerで合格しても、空入力等の未測定契約の疑義が解消したとは言えない。
この採点段階では残72件は未採点。後続の別executorを本結果へ混在させない。

元120件のHTTP400失敗とinactive v1採点計画は不変。
v2通常生成のowner正常終了（120公開回答、エラー0、全stop、133.1371秒）と
PID消滅を確認し、rootの明示指示に従って採点した。
新しいモデル呼出し、サービス変更、再採点、重み変更は0。
全case/responseのhashは採点前後で一致し、3 catalogすべてcompletedとなった。

## 固定した実装と実行条件

新計画 `analysis/fresh-purpose-20261107-av-authored48-execution-plan-v2/plan.json`
のSHAは `5991b28285825fcb12dba32b9daf886adaf9c6d26b1b45b34c076d10b512771b`。
[PR版計画](records/2026-11-07-authored48-execution-v2/plan.json)に
実v2全response SHA、case SHA、generation完了・owner sourceの証拠、
全コマンド、3 previewと条件を保存した。準備0.8581秒、
preview各16/16成功、preview時のcontainer実行0。

repoで並行変更中の採点器を使わず、commit `4e088df`から
sandbox＋3つのcase sourceを新NVMe runtimeへ複製し、read-only fileとして固定した。
sandbox SHAは `cf26e77e90d5789422ad2ebd3ac43599ee0a2ebc781b7a1c0c8ae834953a817e`。
[凍結した4ソースと操作原文](records/2026-11-07-authored48-execution-v2/SHA256SUMS)
をPRへ含める。既存worker/oracle/抽出/採点条件は変更していない。

host Pythonは `/mnt/nvme/kairyu/dev-logs/wsp/venv-scicode/bin/python`。
Docker socketは `/var/run/docker.sock`、daemonは
`3e518d2b-6972-43f8-b325-f7f33697a2d2`、rootは `/mnt/nvme/docker`。
immutable imageは
`sha256:c4ab50223f03fca17bf871e4d05f77001e76525a1c574b6c6ee7024db06cfdc2`。
seed20261107、8 trials/case、startup込み15秒/case、512MiB、
CPU1、network none、UID1003、read-only、cap-drop ALL、runc、
no-new-privileges、pull never、GPUなし、host mountなし。
生成コードはcontainer内だけで実行し、期待値はhostで計算・比較した。
終了後に採点PID消滅とscience-sandboxラベルの残存container0を読み取り確認。
既存サービス・他実行への停止/削除操作は行っていない。

## 失敗2件の具体的な意味

`science-batched-linear-system` は0/8。試験0,2,3,5,6がValueError、
1,4,7は出力shape不一致。公開promptはa=(batch,n,n)、b=(batch,n)から
各系を独立に解きb.shapeを返す契約で、生成回答は
`np.linalg.solve(a, b).astype(np.float64)` を直接返した。
固定入力はbatch=3、n=2/3/4を変えるため、数学的な線形方程式の知識だけでなく、
ライブラリのバッチ軸/右辺形状の契約を正しく扱う必要がある。
記録は例外種類までで、完全な例外本文やtracebackは保存していない。

`science-library-symbolic-numeric-variable-order` は7/8。
trial0だけValueErrorで、固定samplerではこの試験のa=0。
生成回答はSymPyで微分後aを代入し、lambdifyの2つの戻り値を
`np.column_stack((res_dx, res_dy))` に渡す。
a=0では最初の微分が定数ゼロへ縮約され、もう一方はベクトルなので、
scalar/vectorのbroadcastを明示しない組立てが失敗に整合する。
これは保存codeとsamplerからの原因解釈であり、追加の再実行はしていない。
残り7試験合格を保ったまま記録し、全面的な記号微分不能とは解釈しない。

両件とも科学ライブラリ同士の境界とshapeの扱いに具体的な弱点を示すが、
どの重みが原因かはこの採点だけでは分からない。
46合格の高い飽和と既知校正への適応を考慮し、
この48件の平均だけを次の重み編集目的にする根拠にはしない。

[全48 raw reportとsummary](records/2026-11-07-authored48-execution-v2/summary.json)、
[全120件のこの段階の採点状態](records/2026-11-07-authored48-execution-v2/all120-fixed48-stage.json)、
[実行receipt](records/2026-11-07-authored48-execution-v2/execution-receipt.json)、
[終了確認](records/2026-11-07-authored48-execution-v2/closure.json)を保存。
summary SHAは `5adaf4f7a7c223bbbaf936df898683c035fa9aadf5e25542cb8eff03fa8c93da`。
新しい採点器やoracleは作らず、実行した準備・呼出し・集計原文を同ディレクトリへ残した。
