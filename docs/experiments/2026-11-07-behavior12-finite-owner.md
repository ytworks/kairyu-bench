# 全12 case・保存native候補1件の有限GPU4 owner

既存 [GPU4 owner](scripts/fresh-av-native-gpu4-20261107.py) に明示 `mode=finite` を追加した。native既定のcorrect-logprob→marginは維持し、Docker設定、global/GPU4 lock、全27snapshot、他26保護、exact owned collector停止、client終了後のfresh CID再確認、元gemma-2 sameCID復帰を共有する。別の大型ownerは作成していない。

**現在はCPU fixture検証だけ。実candidate/specは未作成、実finite planは未bind、実モデル・API・GPU・service変更は0。** 完全なplanのCPU previewとroot・独立reviewを経た承認SHAが必要で、追加したmode自体は実行許可ではない。

## 実測済み版の保全

実行済みnative owner SHA `9a8d9255312a97ae2762ec97c34c9701ac5d64dcb0dc273e01f8d7f6700b7183` は変更前の原byteを、N2 `analysis/fresh-purpose-20261107-av-behavior12-native-gpu4-v2/owner-executed-9a8d9255.py` と [PR記録](records/2026-11-07-native-gpu4-owner/measurement-v2/owner-executed-9a8d9255.py.txt)へ保存した。[保全receipt](records/2026-11-07-native-gpu4-owner/executed-owner-preservation.json)が両copyの完全一致を記録する。既存source snapshot、plan、全測定、失敗は変更していない。

この版のowner SHAは `39707865f4cc4851290d218d0f4cb6a3fa8768a940a124a9f0ea953ee0d2c5d7`。新しい実行では既存native planのowner/source SHAを無断で書き換えず、新しいsource snapshotと新しいplanを作る。

## planとCPU preview

共通planのcases/frame/attestation/native_behavior_result/source_root/source_pins/target/protected/output_rootはnativeと同じ意味。全train12はraw A..F各2 singleton、behavior-A1..F2、全row.split=trainを要求する。transferを混ぜない。新planでは現在のgemma-2 StartedAtを含むtargetと全26protected snapshotを取り直す。`approved_for_execution=false`から始める。

finite分岐は次の5件のpath/SHA pinを要求する。これは構造説明で、実行用planではない。

```json
{
  "mode": "finite",
  "finite": {
    "plan": {"path": "<finite-replay-plan.json>", "sha256": "<sha256>"},
    "spec": {"path": "<saved-candidate-edit.json>", "sha256": "<sha256>"},
    "manifest": {"path": "<editor-create-manifest.json>", "sha256": "<sha256>"},
    "verify": {"path": "<editor-full-verify.json>", "sha256": "<sha256>"},
    "candidate_identity": {"path": "<candidate-full-nine-file-identity.json>", "sha256": "<sha256>"}
  }
}
```

内側finite planは既存 `schema_version=1, architecture=gemma4_native_bf16`、候補をexact ID/spec/SHAの1件に限定。cases/frame_evidence/checkpoint_attestationが外側pinと一致し、base_tree_sha256はAV attestationと一致する。observation_policyはserial/1case/1024/text SDPA/88GiB/longest_frame_firstを完全一致で要求する。native-onlyのscalar/vector exact BF16非noop specを既存load_planで検証し、tensor数の上限を新設しない。

CPU previewは以下のowner入口だけを使う。**有限CLIには `--run` gateがないので、その通常CLIをpreviewとして呼ばない。** このownerはpreviewでsubprocess/Docker/Torchモデルを開始せず、固定sourceのCheckpoint/load_plan/editor.validate_manifestと少数の指定座標読取りのみを使う。

```text
python docs/experiments/scripts/fresh-av-native-gpu4-20261107.py --plan <新規owner-plan.json>
```

全9個のAV・保存candidate full content SHA receiptをfresh size/device/inode/mtime/ctimeへ照合し、各tree SHAもreceipt内file列から再計算する。manifest/spec/parent一致、verify.verified=true・legacy_full_hashes_missing=false・全9fileのsource/candidate SHA一致、実current candidate inventoryを確認する。保存candidateの選択BF16値をexact specへ直接照合し、ロード後RAM proofと比較するbyte hashをCPUで作る。全checkpointをこのpreviewで再hashしたとは表示しない。全量verify/full hashは保存時の別処理である。

フレームは既存の全train12 tokenizer-only実証とSHAを固定し、全input位置・長さは実finite開始前に既存runnerがもう一度完全再構成して照合する。新しいtokenizerやframeが必要になったときは、既存のCPU-only frame検査を先に実施する。native用cpu_previews二目的はfinite modeでは不要である。全src・recipe・owner/helperのsource pinsは引き続き必要。

## 実行条件と完了の意味

rootが保存candidate・全plan・CPU preview・独立reviewを確認した後だけ、ownerの `--run --approved-plan-sha256 <exact SHA>` で実行する。実containerはaf9 image、UID1003、network none、GPU4のみ、CPU32-39、memory/swap各128GiB、入力とモデルreadonly、NVMeだけのcache/tmp/output。collectorは1回ロードでAV→固定candidate→AV、全12correct/foilを各sweepで保持する（24 forward×3=72、生成0）。modelはAVをロードしRAMへexact specを適用するため、保存candidateとの同一性は上記保存証拠とRAM selected-byte proofを結んで確認する。

finite内 `--maximum-seconds 600` はloadを含む診断の時間予算で、通常sweepは600−60秒で中断判定しrestored sweepへ移る。recovery sweepは内側deadlineを外す。ownerはモデルloadを含むcollector全体に900秒を適用する。stopと同CIDのhealth復帰は別時間で、600秒を全作業のhard capとは呼ばない。有限処理はinference_mode/勾配無効なのでsaved-activation-storageは適用対象外。88GiBはCUDA caching allocator上限、128GiBはcgroup制限でありピーク測定値ではない。

完了はsummaryの文字列だけで決めない。全12labelsと全3sweepのA..F/2fold/singleton/finite数値、case順means、correct NLL、endpoint KL、1候補のexact BF16差分、candidate−baseのcase別delta、source/provenance hash、実text SDPA/88GiB/no fallbackを確認する。base/candidate/restoredのselected-byte-proofsを保存candidateの実BF16から作ったtensor/count/index/hashへ照合する。restorationのattempted/forward_completed/matched/storage_bytes_exactを要求し、実数値からmargin/NLL<=1e-4、endpoint KL<=1e-6を再計算する。元sameCID healthy/idle・他26不変とowner完了はさらに別の必須条件である。

未完了、OOM、timeout、write failure、入力変更、不一致は全て失敗として保存し、retryやcase削減はしない。selected byte proofは変更座標のunionであり全model hashではない。NLL減少の符号反転がcorrect-LL利得、KLはcorrect最初/最後tokenだけである。有限値は自由生成利得・WSP採用判定ではない。

## CPU検証と失敗記録

[CPU検証](records/2026-11-07-native-gpu4-owner/cpu-validation-finite-v1.json)はfocused22tests（既存native13＋finite9）、Ruffとも成功。実Checkpoint/editor/create/verify/load_planを小型9file BF16 fixtureへ適用し、追加候補・policy変更・manifest/verify/spec/filehash不一致・保存BF16不一致・stat変更・partial sweep・偽RAM proof・数値復元失敗・source/実attention不一致を拒否した。finite collectorのlate-createも同じcleanupで停止してからsameCID復帰するmockを確認した。実GPU未検証で、全suiteはこの段階で繰り返していない。

途中の21tests/Ruff成功と、最初の編集commandがローカルshell parseでSSH実行前に失敗した記録も同receiptへ残した。後者はremote mutation0。fixtureで使った小型候補は一時NVMe test directoryのみで、実Gemma候補ではない。
