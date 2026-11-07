# AV train12 native感度：GPU4単独の未実行計画

**準備・CPU検証だけを実施した。配信停止、GPU/model呼出し、候補作成は0。** 全native scalar/vectorを収集し、変更するtensor数は事前に固定しない。範囲は新caseの共同感度、BF16刻み、機能上の組合せと有限変更後の得失から判断する。

## 確認済みの環境と既存実装

[読み取り証拠](records/2026-11-07-native-gpu4-owner/planning-evidence.json)は27containerの安全なsnapshot（Config/HostConfigはSHAだけ）、GPU、host RAM、空き容量、依存versionを保存した。gemma-2はhealthy/queue0、CID `fd4ec2eb9d7deba2ce3e790add6b50c0ee79a6390495320d8697873b32957bcf`、image `sha256:af9e746b913088cbf6c4464a31d6904a6b7e1d3ba62f0895362ddf13595f7abe`。GPU4/5、通常CPUset32-47,96-111である。AV checkpointは `/mnt/nvme/kairyu/wsp/gemma4-coding-20261104/checkpoints/code90-current-ad-helper-av-v1` を `/checkpoint` へreadonly mountしている。通常配信のTP2/MTP4/native BF16設定は変更しない。

同imageのCPU package metadataを既存container内の `importlib.metadata` だけで確認した。Python3.12.3、Torch2.14.0+cu130、Transformers5.20.2、NumPy2.2.6、safetensors0.8.0、accelerate1.15.0。importだけの調査でTorch/modelはロードしていない。GPUは全8枚が配信中で空きなし。host availableは約900GiB、NVMe空きは証拠採取時66.25GiB。今回checkpointコピーは行わず、出力はnative全421tensor×1,331,516座標×12fold×2目的のFP32配列（圧縮前約122MiB）とsource値・metadata程度。FP32 CPU活性退避はRAMでありNVMeへ書き出す機能ではない。

`observe-saved-parent-purpose.py` は939行、native-onlyを拒否しGPU4/5・旧helper・1800秒・plan-local lockに固定される。runtimeの135行 `run-current-r-native-v1.py` はGPU4に2目的を順次収集しgemma-2を同CID復帰する直接先例だが、旧plan/42fold固定でglobal lock/timeoutがない。どちらもそのまま実行しない。[新しい小さいowner](scripts/fresh-av-native-gpu4-20261107.py)は後者の順次構造を使い、既存 `fresh-f01-finite.py` のsnapshot/metrics/実flock証明だけをsource SHA付きで再利用する。過去候補や採用判断を再利用するものではない。

## 実行の入力契約と順序

未確定なのは新train12のケースSHA、CPU tokenizer/frame証拠、native通常生成12件の結果、source freezeと最終plan。これらが揃わない限り実行不可。transfer6件はcollectorへ渡さない。

1. `source_root` へ現行srcとrecipeをfreezeし、owner/helper・全src Python・native.py/recipe_common.py/objective_variants.pyを `source_pins` に束縛する。`cases` は固定raw mapping A..F→各2 singleton fold（train envelopeも受理）で、全row.split=train、IDはbehavior-A1,A2〜F1,F2に固定。transfer wrapperや行を拒否する。`frame` は同AV tokenizerから作った12件だけのprepare_frames戻り値で、correct/foil双方の完全prefix+answerが1024以下、切詰めなし。全18のCPU長さ検査とtrain12のframe識別を分ける。
2. `attestation` は既存N30 `analysis/fresh-purpose-20261107-av-dev30-dp4-v1/http-trace/capture.json`（SHA `8e4b57eacaeca88260281c4d39a0f347b55ed0a7f1cabcac3ff0507d1e6d9e74`）。checkpoint全9fileのcontent SHAを持ち、tree `5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf`。今回全9fileのsize/device/inode/mtime/ctime一致を確認した。これは既存全量hash＋freshstatの再利用で、今回の再hashとは記載しない。
3. `native_args(plan, objective)` を生成し、host CPU Pythonから `--run` 無しでcorrect-logprob/margin各previewを1回行う。CUDA指定はpreviewで政策/入力を検証するための文字列であり、collectorはpreview時Torchをimportしない。receiptは `returncode=0, model_calls=0, collector_arguments=<exact list>` を保持し、planの `cpu_previews` にpath/SHAを入れる。`native_behavior_result` には先行する12件native自由生成の完了結果をpinする。rootが結果と完全planをレビューした後だけ `approved_for_execution=true`。
4. detached ownerがglobal `candidate-evaluation.lock` と `gpu4-native.lock` をnonblocking取得し、plan/source/attestation/全27snapshot/target queue0を再確認する。gemma-2のexact CIDだけstop（最大90秒猶予）。GPU4 free>=92GiB（88GiB allocator＋4GiB以上reserve）を確認する。他26へstop/start/killを出さない。GPU5は使用しない。
5. 同image、UID1003、network none、GPU4だけ露出、CPUset32-39、memory128g/memory-swap128g、read-only rootfs、NVMe cache/tmp/output、入力はreadonly mount。correct-logprob→margin各1回を順次実行。共通引数は下記。全2collector subprocessの合計deadlineは900秒で、**各プロセスのモデルloadを含む**。事前stopと最後のhealth復帰（最大600秒）は別に記録する。collector自身のsummary.secondsはモデルload後の計測時間なので、owner時間とは区別する。
6. summary/provenance/frame/coverage/全12fold/source不変を確認。途中fail/OOM/timeoutでも部分gradient、progress、failure、stdout/stderr、owner completionを残し、自動retry/長さ短縮/別backendへのfallbackはしない。GPU peak allocated/reservedとoffload copy bytesは既存summaryで得られるが、CPU RAM peakはこのownerでは未測定。128GiBはcgroup上限であり、ピーク実測ではない。
7. finallyではまずcollectorのCID/name/imageとplan SHA・owner root labelsを照合し、stop後の停止/消滅を再確認する。止まらない**自分のexact collectorだけ**kill可能。inspect失敗だけを「消滅」とみなさずDocker inventoryでも確認する。所有不明、Docker不通、host client未終了ならoriginalを衝突再起動せず `restoration_blocked` を保存して親へ報告する。inactive証明後だけ元gemma-2 exact CIDをstartし、image/config/hostconfig/devices/CPUset・healthy/queue0と他26不変を確認してlock解放する。Compose再作成は使わない。

```text
--mode chat-public-answer --objective correct-logprob  # 続いてmargin
--answer-batching serial --max-cases-per-microbatch 1
--saved-activation-storage cpu-fp32 --max-frame-tokens 1024
--text-attention-backend sdpa --cuda-memory-cap-gib 88
--execution-order longest_frame_first --device cuda:0
--threads 4 --seed 0 --gpu-exclusive-approved
```

previewは `python docs/experiments/scripts/fresh-av-native-gpu4-20261107.py --plan <絶対plan.json>`。実行時にのみ `--run --approved-plan-sha256 <確定SHA>` を追加する。現時点では完全な実行planがないため、成功したGPU preview/測定と称さない。

## メモリ・分母・解釈の限界

31B全モデルBF16を1GPUへロードする。88GiBはPyTorch caching allocatorだけの上限であり、非allocator確保まで制御しない。text SDPAはPyTorch自動kernel選択（mathを含む）で、Flash/GQA節約を保証しない。CPU退避は非leaf CUDA FP32・ndim>=2のsaved activationだけで、BF16 graphはGPUに残る。1case/answer即VJP、最長caseから実行し、メモリ不足なら失敗を保存して復帰する。correct-LLは12forward、marginはcorrect/foil各12で計24forward、合計36forwardを予定する（生成ではない）。

collectorのfilterはtext language modelのfloating parameter ndim<=1とlayer_scalarで、全native421tensor/1,331,516座標を収集する。solver側の `native_tensors` は**観測後**のeligible maskであり、collectorに先入観で2〜4tensorの上限を置かない。[native scaleの全量観測](2026-11-07-native-scale-observation.md)の通り全native L2約70,135は大きいinput/pre-FFW normに支配される。候補は保存BF16へ丸めた実値のtensor/family別差分、変更座標・eligible全体それぞれの分母、全checkpoint分母を混同せず報告する。correct-LL/marginの意味を照合する既存2目的solverを使う。forced-empty-thought teacher forcingはnative自由生成能力ではなく、有限BF16反応・native実行利得・全問公式WSPを別々に確認する。

[CPU検証v1](records/2026-11-07-native-gpu4-owner/cpu-validation-v1.json)は初期7tests。[v2](records/2026-11-07-native-gpu4-owner/cpu-validation-v2.json)はroot reviewで見つかった「cleanup失敗後の衝突再起動」を修正し、owned-hang・unknown-owner・Docker不通・hostclient未終了のmockを追加した10tests。GPU未検証である。最終readonly入力guardを含む[v3](records/2026-11-07-native-gpu4-owner/cpu-validation-v3.json)は11testsとRuffが成功した。

実入力bind後のroot reviewで、初期ownerがtrain envelopeだけを受理し、固定raw A..F形式を拒否する問題が判明した。原入力は変更せずownerをraw/envelope両対応・全row.train・exact behavior-A1..F2へ修正した。[v4 CPU検証](records/2026-11-07-native-gpu4-owner/cpu-validation-v4.json)は12tests/Ruff成功。実際のtrain12/frame/source/2つのnative metadata previewは通過し、実native生成結果が未完成のため最後の入力gateで意図どおり停止した。

未承認planは `analysis/fresh-purpose-20261107-av-behavior12-native-gpu4-v1/plan.json`、SHA `f47f6a6c23184d06980cb3c5791ac2126e9df3be183cb96c6c8b30dd2d1ee9d3`。79 source fileをpinし、correct-LL/marginのmetadata previewは0.365/0.346秒で成功。`approved_for_execution=false`、native_behavior_resultは未存在/null SHAであり、owner全previewの成功・GPU実行可能とは表示しない。[planとpreviewのPR記録](records/2026-11-07-native-gpu4-owner/plan.json)を保持する。


## 生成・採点後の未承認v2

生成は全12公開回答、APIerror/empty/length=0、owner18.620285秒で正常lock解放。CPU採点は12件/192trialを完了し、10件合格・2件不合格・unknown0、164/192trial合格だった。A1=0/16、F2=4/16、残り10件=16/16を全て保持した。BのSymPy微分/lambdify方法要件は数値fixtureだけでは検証できず、未検証のままである。F2の実生成失敗は開始index off-by-oneで事前foilのzero削除とは異なるため、correct-LLとそのmarginを同じ修正指標と見なさない。

[結果identity](records/2026-11-07-native-gpu4-owner/plan-v2/native-behavior-result-identity.json)は原生成と採点reportをID/public-answer SHAで結合し、全12caseと全192trial/failureを残した。SHAは `463e9ff5cf4f8183527bc1080f1fab36a5214a63891345cc07f254d2ee9ff099`。追加モデル呼出し・コード実行を行わず、既存完了artifactを読む操作だけである。

独立reviewでc70版に「初回不存在確認→client.wait中にdaemon create→client終了→original再起動」というraceが見つかった。ownerはclient終了後にも新cidfileを読み直し、両objectiveのexact所有/不存在/停止を再確認するよう修正した。最終source SHA `9a8d9255312a97ae2762ec97c34c9701ac5d64dcb0dc273e01f8d7f6700b7183`、[v5 CPU検証](records/2026-11-07-native-gpu4-owner/cpu-validation-v5.json)はその具体mockを含む13testsとRuffが成功した。r7全suiteは旧sourceに対する正しい記録として保持し、同suiteを繰り返していない。

新出力rootは `analysis/fresh-purpose-20261107-av-behavior12-native-gpu4-v2`。[plan v2](records/2026-11-07-native-gpu4-owner/plan-v2/plan.json) SHA `8edba22f5aa38ec5eebf443a3b4b2eab2794b23c9968fcd0c09d13d003f20316`、`approved_for_execution=false`。新source snapshot/79pins/2つのnative metadata previewに更新し、[owner full CPU preview](records/2026-11-07-native-gpu4-owner/plan-v2/owner-full-preview.json)も成功した（SHA `d6f2d62c6f10527c387ac47ae3464290f200c83a2b8636f382e96a673a4cec27`）。27containerはv1 snapshotと一致、gemma-2 queue0。GPU計算・配信stop/start・global lock取得は0で、rootの目的/計画承認を待つ。

[再現用のCPU-only preparation操作](records/2026-11-07-native-gpu4-owner/prepare-v2.py)と[実行receipt](records/2026-11-07-native-gpu4-owner/prepare-v2-operation-receipt.json)を保存した。元f47plan/v1 source、途中の検証記録、失敗と未実行状態は上書きしていない。
