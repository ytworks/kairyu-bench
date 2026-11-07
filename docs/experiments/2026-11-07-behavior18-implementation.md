# Behavior18 implementation and authored validation, 2026-11-07

**全18件の著者正答が288/288数値試験に合格し、全18foilが事前の固定反例または公開入口契約で失敗した。**
初回モデル生成はtrain12だけ。transfer6は候補spec/幅を凍結後、AVと候補を各1回測る。
この実装・検証ではモデル呼出し、GPU、重み編集、serving操作は0。
著者問題の正しさとfoilの識別性を確認した結果であり、AVの能力得点ではない。

[case recipe](../../scripts/experiments/gemma4-purpose/behavior_cases.py)は
A=batched RHS、B=symbolic broadcast、C=関数入口、D=状態保持、
E=長方形の座標対応、F=zero/indexの6群×3件。
各群1/2がtrain、3がtransfer。trainは6目的に2つのsingleton foldを持ち、
全caseを独立に記録する。問題の意味・軸・空入力・dtype・出力契約は公開promptに明記した。
E1の方向は短いinteger encoding 0=N,1=E,2=S,3=Wとした。
詳細は[事前設計](2026-11-07-behavior-calibration-proposal.md)。

## 入力と実装を固定

remote bundleは
`analysis/fresh-purpose-20261107-behavior18-inputs-v1`。
[train12](records/2026-11-07-behavior18-inputs-v1/train-cases.json) SHA
`b42990b0caac24ccc7843e40718c0c5e7ad61dd5baa131ed6852c126bf37f66d`、
[transfer6](records/2026-11-07-behavior18-inputs-v1/transfer-cases.json) SHA
`4951c70a686ebcd7135919c48d4038c6aa6d359abc018730aee73c1afe9f4b55`。
transferは`gradient_allowed=false`の別envelopeで、既存collectorは受理しない。
モデル出力を見てから除外・split変更・foil交換は行わない。
完全frameのtoken計数は別担当のtokenizer-only検査へ渡した。
このbundle metadataは計数前の`token_count_measured=false`を保持し、
後続証拠で結果を追記する。1024超過時に切り詰めやcase削除はしない。

[freeze](records/2026-11-07-behavior18-inputs-v1/freeze.json)に全source/入力/検証hashを結んだ。
case source SHAは`10ba3e480c8685933904060629b351758cf7b61362756179e55779435d01a310`、
[offline adapter](../../scripts/experiments/gemma4-purpose/behavior_offline.py)は
`30715a30e02c7fab8ec4e1ffad28e57c4925b16fb45048680d6f5735bd3af9e5`、
共通sandboxは`4a65378e493fb2f109afdaf0d98b5b93dc067d6d3f48d2a4cf17017f1aadcb78`。

## 実行境界と測定の意味

新しい任意code実行基盤を作らず、同じ`science_sandbox.Docker.run`とWORKER、
既存extract/prepare/compareを再利用する。hostは自作sampler/期待値だけを計算し、
著者correct/foilを含む回答プログラムのexecはcontainer内だけで行った。
CではASTが正確な要求名1個、他のtop-level関数なし、top-levelはimport/その定義のみと
確認できた場合だけ`solve=要求名`を追記する。欠落名・重複定義を救済しない。
元public answer SHAと実行code SHAを別々に記録する。

数値合格は全公開契約の証明ではない。
`entrypoint_contract_passed`はC限定（他はnull）で、
Bの`method_requirement_checked=false`を保持する。
SymPyのdiff/lambdifyを実際に使ったかは数値fixtureでは保証できず、
生成された公開codeの呼出経路を別に確認する。単なる文字列出現を実行証明にはしない。
E3の「回転配列を作らずlookup」等の方法要件も数値だけでは証明できない。
数値、入口、方法の結果を同じ合否として読まない。

score modeは元のsplit全件payloadの完全一致とcase/responseのSHAを要求し、
unknown/duplicate response ID、reasoning本文、改変caseを拒否する。
missing/HTTP error/timeoutはunknown。過去の48/16や公式WSPと分母を混ぜない。

## 著者検証と全試行

各caseは8個の固定境界入力＋8個のseed付き入力。
固定半分は入力seedに依存しないsource固定の形状/値・witness、
後半はtrain202611071/transfer202611072にcase名hashを加えた既存prepare RNGで作る。
correct/foilは同一のfixture SHA。float32/64、zero係数、空列、
n/B/qの一致・不一致、非可換更新、長方形両方向を保持。
正答期待値は独立loop/式/DP等で計算する。

全18 correctは16/16。C1/C2/C3のfoilはAST入口契約違反で実行前に拒否されるので、
「数値16試験を実行した」とは数えない。残15foilは各16試験=240試験を実行し、
全foilが固定半分の少なくとも1反例で失敗した。各foilが他の入力で通る場合も全て残す。
最終は18 correct + 15 numeric foil = **33 CPU containers**。

| 版 | 著者検証 | 秒 | raw worker出力 | 内容 |
|---|---|---:|---:|---|
| v1 | 15/18成立 | 28.8151 | 未保存 | B2/B3/E2のcorrectが空(0,2)のwire形状で2trialずつ失敗。全foilは反例失敗。 |
| v2 | 18/18成立 | 28.5266 | 未保存 | adapter内で空shapeを限定復元。 |
| v3 | 18/18成立 | 29.0382 | 33保存 | 根本原因の共通compareへ移動。adapter比較複製を撤去。 |
| v4 | 18/18成立 | 29.0856 | 33保存 | 共通最小差分、入口/方法metadataを明確化した最終source。 |

初回RuffはE701/E702のformat指摘37件で失敗し、formatter後に通過した。
fixtureの定義・correct/foil本文はv1から変更していない。
v1/v2のraw worker未保存を後から復元したとは称しない。
全版のsource、preview、実コマンド、receipt、report、失敗を
[全試行の数値summary](records/2026-11-07-behavior18-inputs-v1/validation-summary.json)と
各`records/2026-11-07-behavior18-validation-v{1,2,3,4}/`へ保存した。
最終report SHAは`39adbafa7884b2abd2ba58b1408a6ad9a0cdcf8ccd4ad0534ef34dfc253ed300`。

共通compareの修正は11行。workerは正しくshape=[0,2],data=[]を返すが、
JSON[]から単にnp.asarrayすると(0,)となる。
期待と宣言shape一致、期待size0、dataが正確にlist[]の場合だけreshapeする。
raw結果を変更せず、誤shape、欠測、非空値、dtype不一致、入力変更は救済しない。
同境界の回帰testを既存sandbox suiteへ追加した。
旧48/16は当時のsourceを既に凍結しており、過去結果を書換えない。

既存case suite7件（新4件を含む）0.317s、sandbox suite8件（新1件）0.569s、
focused Ruff/diff check通過。共通境界と新adapterの全repo CIは親の統合検証へ渡す。
host scientific Pythonは`/mnt/nvme/kairyu/dev-logs/wsp/venv-scicode/bin/python`、
実image `sha256:c4ab50223f03fca17bf871e4d05f77001e76525a1c574b6c6ee7024db06cfdc2`、
daemon`3e518d2b-6972-43f8-b325-f7f33697a2d2`、NVMe root`/mnt/nvme/docker`。
CPU1/512MiB/15秒、network none、UID1003、cap-drop ALL、runc、
no-new-privileges、read-only、pull never、host mount/GPUなし。
終了後sandboxラベルcontainer0を確認し、サービス停止・pruneはしていない。

## CLI

~~~sh
# 全18著者検証。既定previewはDocker実行0。新outputに --run で明示実行。
<SCIENTIFIC_PYTHON> scripts/experiments/gemma4-purpose/behavior_offline.py   --mode validate-authored --docker-socket /var/run/docker.sock   --daemon-id <EXACT_ID> --image sha256:<EXISTING_IMAGE> --output <NEW_OUTPUT>

# 固定splitの保存済み公開回答だけを採点。モデルAPI呼出しなし。
<SCIENTIFIC_PYTHON> scripts/experiments/gemma4-purpose/behavior_offline.py   --mode score --split train --cases <TRAIN_JSON> --cases-sha256 <SHA>   --responses <COMPLETE_RESPONSE_JSON> --responses-sha256 <SHA>   --docker-socket /var/run/docker.sock --daemon-id <EXACT_ID>   --image sha256:<EXISTING_IMAGE> --output <NEW_OUTPUT>
~~~

transferのscore実行は候補凍結後の別planでのみ行う。CLIは自動採用・再実行・
候補作成・token切り詰めをしない。実観測したtrainの行動失敗と全領域WSPを見て、
次の目的・重みの組合せを定めるための道具であり、既存F02の目的関数を継承しない。
