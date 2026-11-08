# LoRA 因子のCPU比較

`scripts/experiments/compare_lora_directions.py` は二つの通常LoRAについて、
保存された因子の未丸め更新 `sBA` を比較する。NumPyと既存safetensors readerだけを使い、
元モデルの巨大なpayload、dense更新、GPU、推論APIは使わない。
Gemma専用のrank16監査とは異なり、任意の正rankとF32/BF16因子を受け付ける。

## 入力と実行

SHA固定したJSON planには次を明示する。

- `schema_version: 1`。
- `source`: 元checkpointの `path`、既存 `Checkpoint.identity(hash_files=False)` の
  `identity`、`declared_tree_sha256`、その `source_tree_sha256` を記録した
  `receipt_path / receipt_sha256`。現在のheaderと全file statを照合する。
  元payloadのfull hashはこのtoolで再計算しない。
- `left / right`: 単一safetensorsの `path / sha256`、
  `config_path / config_sha256`、任意の `coefficient`（既定1）。
  repo/revision等はprovenanceとして同じobjectへ追加できる。
  選択外の因子は `excluded_tensors: {tensor_name: reason}` へ全て列挙する。
- `mapping`: 各行に元の `tensor`、`shape: [out,in]`、
  `left: {A: name, B: name}`、`right: {A: name, B: name}`、
  `groups: {layer: label, family: label}` 等。名前の曖昧な自動照合はしない。

```sh
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
PYTHONPATH=src python scripts/experiments/compare_lora_directions.py \
  --plan PLAN.json --plan-sha256 PLAN_SHA --output NEW_REPORT.json
# 上はpreview。header/config/planだけを読み、factor値も出力fileも作らない。
PYTHONPATH=src python scripts/experiments/compare_lora_directions.py \
  --plan PLAN.json --plan-sha256 PLAN_SHA --output NEW_REPORT.json --run
```

出力先の親directoryは事前に作り、出力fileは未存在にする。
既存source・数値recipeを編集せず、結果はexclusiveに公開する。
SHA不一致、途中変更、非有限値、未対応のLoRA形式、欠落/重複mappingは
例外として可視化し、成功reportを残さない。失敗時に入力や過去結果を削除しない。

`s = coefficient × lora_alpha/r` をconfigから計算する。
DoRA、rsLoRA、QALoRA、fan-in/fan-out、rank/alpha patterns、bias、
modules-to-save等の追加forward変換は拒否する。
adapterの学習時dropoutはeval方向の計算へ適用しない。
選択外vision等は明示して除外できるが、追加の非LoRA保存パラメータを
黙って通常LoRAとして扱わない。source形状一致は学習base revisionの一致を保証しない。

## 数学と費用

各moduleでA/BをFP64へ変換し、小さいrank空間で次を計算する。

```text
||sBA||_F² = s² sum((B.T @ B) * (A @ A.T))
dot(s1B1A1, s2B2A2) = s1s2 sum((B1.T @ B2) * (A1 @ A2.T))
cosine = dot / sqrt(norm1² norm2²)
```

moduleは非重複の元tensorなので、全体・groupのdotとnorm²を足してからcosineを求める。
module cosineの平均ではない。零normのcosineはnull。負の小さい丸め誤差を
許容範囲で0へ補正した場合やcosineの丸めclipはflagに残す。
大きな矛盾・非有限結果は成功値へ丸めず拒否する。
「厳密」と言えるのは全因子を使った代数的恒等式であり、実演算はFP64の有限精度である。

読み取りchunkは最大8MiB。一度に1module分の因子とrank²のGramだけを保持する。
previewは選択数、配列メモリ見積り、Gram FLOPs、I/O範囲を返す。
`--max-module-mib` は見積り配列の上限（既定256MiB）で、
BLAS workspaceを含む実RSSの厳密上限ではない。
`--seconds`（既定600秒）はchunk・module境界の協調的予算で、
実行中の一つのBLAS呼出しを途中でkillする期限ではない。
全adapter SHAを前後に検証し、config/plan/receipt/code SHAとstatを記録する。

初版にはspectrumを含めない。出力は未丸め数学方向の幾何であり、
実BF16 merge後のcosine、activation、正答率、科学能力や機能局在を示さない。
低cosineは別方向を観察した根拠にはなるが、「能力が補完する」という証明ではない。

## 検証

3件の小さいCPU契約fixtureで、異rank・F32/BF16・異scaleの全値をdense参照と照合し、
零norm、全体/group集約、previewのfactor非読取、full input SHA、
未対応config・曖昧mapping・非有限値・途中変更の拒否を検証する。
大規模adapterでの費用と観測値は別の実行receiptに残す。
