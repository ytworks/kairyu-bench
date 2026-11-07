# 保存済みmodule編集とRAM検証の一致確認

[verify_saved_checkpoint.py](../scripts/experiments/gemma4_lora/verify_saved_checkpoint.py) は、
作成したチェックポイントが、完了済みRAM有限測定で使ったBF16重みと同じかをCPUだけで確認する。
LoRAの行列積を再計算せず、モデル・GPU・Docker・APIを起動しない。採点や候補採用は行わない。

この経路は `finite_module_replay.py --device-proof staged-int16` の**完了済み**証跡専用。
CPUで計算した目標バイト列のSHAと、GPU上の全要素のint16一致を根拠にする。
D2H読出しのSHAではない。未完了runや復帰検証不成立からfreezeを作らない。

## 入力と実行

全JSONのSHAを明示する。replayディレクトリから読むのは
`summary.json`、`provenance.json`、`ram-weight-receipts.json`だけ。
旧resultやprivate runtimeをimportしない。数値kernelと4つのrecipe実装も、
RAM測定・creatorのfingerprintに一致することを確認するが、load/実行はしない。

```sh
PYTHONPATH=src python scripts/experiments/gemma4_lora/verify_saved_checkpoint.py CANDIDATE \
  --manifest MANIFEST.json --manifest-sha256 MANIFEST_SHA \
  --spec SPEC.json --spec-sha256 SPEC_SHA \
  --replay FINAL_REPLAY_DIRECTORY \
  --summary-sha256 SUMMARY_SHA --provenance-sha256 PROVENANCE_SHA --ram-sha256 RAM_SHA \
  --plan PLAN.json --plan-sha256 PLAN_SHA \
  --source-attestation SOURCE_IDENTITY.json --source-attestation-sha256 IDENTITY_SHA \
  --kernel PINNED_KERNEL.so --kernel-sha256 KERNEL_SHA \
  --output FRESH_FREEZE.json
```

既定はmetadataのpreviewで、freezeも重みも書かない。`--apply`を付けると、
保存されたtensorと全checkpointファイルを実際にhashし、新しいfreezeだけを排他的に公開する。
重み・manifest・RAM証跡は変更しない。`--implementation-dir`は通常このrecipeのディレクトリ。
RAMのprovenanceが記録した絶対ファイルとSHAに一致する必要がある。

## 検証範囲

- 同一候補ID・specのSHA/内容・全410係数・source・plan・数値build・compiled kernelを照合する。
- 全60層の410行列を、実safetensors headerとRAMのshape/dtype/spanへ照合する。
  非ゼロ係数はcreatorの編集行・変更数・tensor SHAも比較する。
  creatorが省略したゼロ係数も必ず検証し、source SHA＝RAM target SHA＝保存tensor SHAを要求する。
- 保存tensorのraw BF16を最大8MiBずつ読み、RAM目標SHAと比較する。
  続いて既存Checkpoint readerで全認識ファイルをhashし、残る小metadataもhashして
  creatorの**全**file inventoryと照合する（実checkpointでは12ファイル）。
- candidate全ファイルの前後stat、sourceの先行full-hash証跡と認識ファイルのstat継続性、
  sourceの追加小metadata、入力証跡と数値実装の継続性を確認する。
- 出力の `checkpoint_path`、`candidate_inventory_stat`、
  `all_checkpoint_files`、`control_tree_sha256` は既存
  [redeploy-replica.py](experiments/scripts/redeploy-replica.py) の `--freeze` が受理する形式。
  この検証自体に配信切替は含まれない。

候補のpayloadはtensor照合と全file照合の2回読む。大きなsource payloadは再hashしない。
元のhashとstat継続性はfresh full hashと同義ではなく、同一刻みの書換えや特権的改変を
すべて検出できる保証はない。検証後の消費側も保存statを再確認する。
同じバイト列でも、teacher-forced proxyからコード能力への転移は別途確認が必要である。

## 検証

6つのCPU契約テストでは、410個の小さい実safetensors行列と、非ゼロ1・ゼロ409の
manifest/RAM fixtureを使用する。保存tensorと追加metadataの実不一致、ゼロ行の欠落、
変更数・証跡種別・kernel・SHA・未完了・source stat不一致を拒否し、
生成freezeが既存redeployのidentity検査に通ることを確認する。
これらは小fixtureでの検証であり、実モデル保存物の検証完了を意味しない。

## 実保存物での確認（2026-11-06）

ownerが作成したdonor-positive-balanced保存物で、このverifierを実行し成功した。
全410行列と全12ファイルを照合し、所要111.681秒、追加GEMM/GPU/モデル呼出し0。
creatorの所要時間は430.15秒。実BF16変更は718,765,573要素、
ΔL2=0.3383641445583714で、RAM測定と保存物の値が一致した。
candidate treeはf5137ab7e45d3458057e9163350720613247b6dc5a00a741b9b4b402a12d2e4a。
元payloadのfresh full hashは行わず、既存full hashとstat継続性を使用した。

証跡はexperiment artifact rootから
analysis/code90-resume-20261106/positive-final-v1/balanced-materialization/
のverification-launch.jsonおよびfreeze.json。
freeze SHA256はc565a707765b6f8f89ed71300e5067c858f6e7a92e9a3f4108163e2029851835。
この確認は保存バイトの一致であり、開始されたdev90の得点や能力改善を意味しない。


## nativeとmoduleの共同保存物

schema4 `joint_native_module_recipe_v1` のmanifestには、同じdonor specに加え、
元baseからのnative scalar/vector編集とその共同reportのSHAが含まれる。
上のCLIへ `--native-spec NATIVE.json --native-spec-sha256 NATIVE_SHA` を追加する。
module-onlyのschema3経路と数値実装5ファイルは変更しない。共同creator/editor/checkpointの
別fingerprintも現在のsourceと照合する。

- 全410行列は従来のstaged-int16証拠で確認する。nativeは**選択座標をGPUから
  読み戻した**別証拠であり、両方を同じ証明方式と呼ばない。
- native spec、RAMの選択indices/before/afterバイト、元checkpointの選択値、
  復帰後の元バイトが一致することを要求する。元native全tensorにその編集だけを
  当てた期待値を作り、creatorのfull tensor SHAと実保存tensorを照合する。
  これにより、選択外の座標や負ゼロのバイト変更も検出する。
- native/moduleのtensor名と物理spanの重複を拒否する。合計変更数は和、実ΔL2は
  二乗和の平方根。連続donor係数の予測コストと実BF16の合計ΔL2を混同しない。
- freezeは従来のredeploy契約を保持し、native spec SHA、両componentの実変更数/ΔL2、
  native全tensor検証の件数を追加する。GEMM・モデル呼出し・重み書込みは0。

追加2つのCPUテストは、410行列＋2native vectorの合成保存物を用いる。
正しい合計コストとfreeze、復帰証拠欠落・spec不一致・重複/未知entry・合計値不一致、
選択外の負ゼロ破損を検証する。実共同保存物の完了は別のowner実行証跡で確認する。
