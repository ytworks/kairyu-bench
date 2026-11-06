# Dense donorのCPU検証から配信用freezeを作る

`scripts/experiments/gemma4_lora/verify_dense_checkpoint.py` は、既存の
`dense_lora_recipe.verify(manifest)` を一度呼び、その戻り値を
`redeploy-replica.py --freeze` が受け取る証跡へ変換する。対象はschema2の
一様text410 donorと明示MLP scopeで、native/module/joint recipeは対象外。

数値計算を新たに実装せず、元の検証器の対象行列CPU再生成（既定410、MLP scopeは180）・保存BF16 byte比較・
非変更領域照合・source/candidate全ファイルhash確認を使う。GPUのlikelihood測定や
RAM receiptは不要である。これはCPU recipeと保存重みの一致であり、GPU上のbyte一致・
能力改善・採用を証明しない。生成済みcandidateとmanifestが先に必要。

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONPATH=src python \
  scripts/experiments/gemma4_lora/verify_dense_checkpoint.py \
  --source /nvme/original --candidate /nvme/candidate \
  --manifest /nvme/candidate-manifest.json --manifest-sha256 MANIFEST_SHA \
  --spec /nvme/candidate-spec.json --spec-sha256 SPEC_SHA \
  --kernel /nvme/pinned-bf16-kernel.so --kernel-sha256 KERNEL_SHA \
  --output /nvme/fresh-freeze.json
```

既定previewはmanifest/spec・inventory/stat・コード/kernel pinを確認するだけで、
数値recipeをimportせず、GEMM・checkpoint全hash・output書込みを行わない。
実検証には同じ引数へ `--apply` を追加する。kernelはcreatorの証拠と一致するものを
指定し、bridgeと同じdirectoryの数値実装もcreatorに記録されたSHAとの一致が必要。

実行前後の全inventory/statとmanifest/spec/code/kernel SHAを照合し、検証器の戻り値が
最初に固定したmanifestと同じ場合だけfreezeを排他的に公開する。既存outputや入力に
重なるoutputは拒否する。candidate・source・specを書き換えず、配信操作もしない。
freezeには `CPU dense recipe regenerated ... no RAM/GPU comparison` と明示し、
`all_checkpoint_files`、`candidate_inventory_stat`、`control_tree_sha256`を保存する。
その後のredeployは全payload再hashではなく、保存hashとstat継続性を使う。

tiny CPU fixtureは既存410行列recipeを再利用し、previewでverifyを呼ばないこと、
applyで一度呼び配信identityへ接続できること、重み/非変更metadataの不一致、
検証直後のinput変更とoutput重複を検査する。実checkpointでの使用前に得られる証拠は
このfixtureの範囲であり、候補の性能値は含まない。

## Explicit MLP scope

The standalone factor-format-v2 recipe additionally supports
`structured_scope=mlp_triplets_all_layers` with all 410 coefficients explicit:
180 MLP gate/up/down matrices use the same scalar and 230 attention matrices
use zero. The existing verifier regenerates only the 180 active matrices;
all attention, native and other untouched spans are compared byte for byte.
The numerical kernel and one-verify workflow are unchanged.

For this scope, freeze `verified_scope` distinguishes validated 410,
regenerated 180 and zero-coefficient 230. The legacy
`all_410_matrices_cpu_regenerated` flag is false rather than overstating the
proof. With no structured scope the original 410-matrix path and true flag
remain. Both paths retain full source/candidate file hashes, stat continuity
and the same redeploy identity contract. Module/native/joint wrappers are not
extended by this option.
