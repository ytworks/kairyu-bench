# AV behavior train12: native-answer execution, 2026-11-07

**10/12合格、2失敗、unknown0。固定数値試験164/192合格。**
全12件を通常生成ownerの正常closure後に同じDocker workerで一回だけ採点し、
10.3009秒、12 CPU containers、12 raw worker出力を保存した。
これは新しい既知校正のbaselineであり、公式WSP得点や編集モデル改善ではない。
transfer6の生成/採点、追加モデル送信、再採点は行っていない。

| Case | 数値試験 | 結果 |
|---|---:|---|
| A1 | 0/16 | ValueError10、出力shape6 |
| A2 | 16/16 | 合格 |
| B1 | 16/16 | 数値合格、方法要件は別確認 |
| B2 | 16/16 | 数値合格、方法要件は別確認 |
| C1 | 16/16 | 要求名/単一定義ASTと数値が合格 |
| C2 | 16/16 | 要求名/単一定義ASTと数値が合格 |
| D1 | 16/16 | 合格 |
| D2 | 16/16 | 合格 |
| E1 | 16/16 | 合格 |
| E2 | 16/16 | 合格 |
| F1 | 16/16 | 合格 |
| F2 | 4/16 | 数値不一致12 |

A1の公開回答は2D RHSをそのまま `np.linalg.solve(A_arr,rhs)` に渡した。
正答のbatched RHS shape契約に対応する失敗が通常生成でも確認できた。
F2はzeroを削らず、`arr[0::k]`を選んだ。要求は元event番号1でk番ごと、
正答は`[k-1::k]`。固定例values=[0,4,0,8],k=2では正答12、
事前のzero除去foil8、実回答0である。
**実際の失敗と事前foilは同じではない。** foilを差替えず、pair marginの
変化だけをこの行動改善と同一視しない。
B1/B2の`method_requirement_checked=false`を保持した。
SymPyのdiff/lambdify使用は数値fixtureから推定せず、親の別公開code監査と分ける。

入力は事前固定train SHA`b42990b0caac24ccc7843e40718c0c5e7ad61dd5baa131ed6852c126bf37f66d`、
通常生成全response SHA`5ea0e169922048a2e5bf649ceb6b1add7bb81db5ee19bcf48c620cb00f3de3db`。
生成は12/12公開回答、error/empty/length0、owner18.6203秒、正常lock解放を確認済み。
採点はgeneration ownerのprotected23 snapshot、daemon、image、全source/input SHAを
照合した新plan/previewの後、rootの条件付き許可に従って実施した。

最初のscore-v1 preflightは、停止済みcontainerにHealth keyがないため
read-only Docker inspect templateが失敗した。採点container/モデル呼出し0のまま停止し、
[失敗記録](records/2026-11-07-av-behavior12-score-v1/failure.json)と原操作を保持。
optional keyを安全に読むだけのscore-v2へ移り、同じ23 snapshotが完全一致した。
これは数値再測定ではない。

[新plan](records/2026-11-07-av-behavior12-score-v2/plan.json) SHA
`71e7d6bcd3757131c9efb83aa2f351b7ca19eedd500072c5ba5a11f7ff48816c`。
凍結済みv4 runtimeのcase `10ba3e48…` / adapter `30715a30…` /
sandbox `4a65378e…`を変更せず使用した。
raw保存用の小さなDocker subclassは`super().run`の返却値を記録するだけで、
WORKER、payload、実行条件、比較処理を変えない。生成コードのhost execは0。

実image `sha256:c4ab50223f03fca17bf871e4d05f77001e76525a1c574b6c6ee7024db06cfdc2`、
host daemon `3e518d2b-6972-43f8-b325-f7f33697a2d2`、NVMe root`/mnt/nvme/docker`、
CPU1、512MiB、15秒、UID1003、network none、read-only、cap-drop ALL、
runc、no-new-privileges、pull never、GPU/host mountなし。
全12のseed202611071、8固定境界＋8seed試験をそのまま使った。
終了後protected23不変、全pin一致、sandbox残存0を確認した。

一次reportは`analysis/fresh-purpose-20261107-av-behavior12-score-v2/evaluation/report.json`、
SHA`a327b3fc0cd5b5c21c633bd0cb09ae691f96b260afada7810391a284c79284cf`。
[全12/全192 trial report](records/2026-11-07-av-behavior12-score-v2/report.json)、
[全raw worker](records/2026-11-07-av-behavior12-score-v2/raw-worker-outputs.json)、
[summary](records/2026-11-07-av-behavior12-score-v2/summary.json)、
[実行receipt](records/2026-11-07-av-behavior12-score-v2/execution-receipt.json)、
[前後closure](records/2026-11-07-av-behavior12-score-v2/after.json)と
全操作原文/SHAを同じPRへ保存した。
