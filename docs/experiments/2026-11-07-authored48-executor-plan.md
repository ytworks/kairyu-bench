# 全48件の既存 CPU executor 計画（2026-11-07、未実行）

**3 catalog 全48件の採点条件と入力を固定し、既存 CLI の preview だけ成功した。実際の fixture 実行は0回。** 通常生成v1は全120件が HTTPError、公開回答0のまま終了したため、このv1応答に対して採点containerを起動しない。生成エラーをコード不正解とは扱わず、元の全120失敗を保持する。

固定 plan は `analysis/fresh-purpose-20261107-av-authored48-execution-plan-v1/plan.json`、SHA256 `b38ff6752ccd17c706670792f486ac78d2296044e89e8043756a514a660e46c8`。[PR版 plan](records/2026-11-07-authored48-executor-plan/plan.json) は `approved_for_execution=false`、`frozen_preview_only_no_public_answers`。対応する16件ずつの元 case envelope と、完了済み `train-responses.json` の全bytes SHA・stat・generation receipt・元120 owner完了記録を結び、ケースSHAの書換えや回答の選択抽出は行っていない。

| Catalog | 対応行 | 公開回答 | CLI preview | fixture実行 |
|---|---:|---:|---|---:|
| basic/train | 16 | 0 | PASS | 0 |
| advanced-v2/train | 16 | 0 | PASS | 0 |
| library-train-v1/train | 16 | 0 | PASS | 0 |

`responses_received=16` はエラー行を含む受理件数であり、生成成功件数ではない。既存 [science_sandbox](../../scripts/experiments/gemma4-purpose/science_sandbox.py#L451) は全catalog行・正確なcase hash・公開回答のみのschemaを検査する。今回の各previewは受理とplan表示だけで、`results=[]` のまま保存されている。

実行条件は seed **20261107**、各case **8 trials**、wall limit **15秒**、memory/swap **512 MiB**、CPU **1**。全48件に実行可能な回答がある場合は最大384 trialsで、3 catalogを順次処理する。採点用 host Python は `/mnt/nvme/kairyu/dev-logs/wsp/venv-scicode/bin/python`（NumPy 2.2.6 / SciPy 1.13.1 / SymPy 1.14.0、[import実測](records/2026-11-07-authored120-science-environment/host-science-python.json)）。今回のpreviewも同環境を使用した。

Dockerは `/var/run/docker.sock`、実 daemon ID `3e518d2b-6972-43f8-b325-f7f33697a2d2`、`DockerRootDir=/mnt/nvme/docker` を読取り確認。ローカル exact image は `sha256:c4ab50223f03fca17bf871e4d05f77001e76525a1c574b6c6ee7024db06cfdc2`、9,777,059,118 bytes。image既定のvLLM entrypointは使わず、既存 [arguments()](../../scripts/experiments/gemma4-purpose/science_sandbox.py#L338) が `python3` に明示上書きする。

同実装で `--runtime runc --network none --read-only --user 1003:1003 --cap-drop ALL --security-opt no-new-privileges --pull never`、pids64、16 MiB `/tmp` tmpfs、BLAS1 thread、CUDA/NVIDIA非公開を固定。host mount・GPU・privileged指定はない。実際のargv templateとworker SHAはplan、daemon/imageの読取り結果は [Docker証拠](records/2026-11-07-authored48-executor-plan/docker-readonly-evidence.json) に保存した。container生成・Docker変更・GPU/API・サービス・lock操作は0回。

報告では **生成分母120** と **既存fixtureの対象分母48** を分ける。48のraw pass/fail/unknown・未回答・例外を全件保持し、既知の契約疑義3件（cumulative-trapezoid、reset-running-stats、capped-reservoir-chain）と残る45件を別stratumとしても表示する。3件を削除・救済せず、未対応72件は未採点/unknownのままにする。48件の合格率や48/120の対応率を120全能力得点、公式WSP、未見能力改善とは呼ばない。

**実行には生成完了後のroot明示指示が必要。現在の公開回答0では起動しない。** 別の通常生成が明示許可された場合は新しい全件 response/receipt に対して新planを作り、この失敗v1のplan・入力・出力・履歴を残す。実行直前にはsource/case/response SHA、socket、daemon ID、NVMe root、local imageを再確認する。自動再試行なし。

[実行したCPU準備処理原文](records/2026-11-07-authored48-executor-plan/operation.py.txt)、[全出力SHA](records/2026-11-07-authored48-executor-plan/SHA256SUMS)を同じPRに含めた。準備0.774秒、CLI preview 3/3成功、fixture実行0。現sourceのコマンドをそのまま呼び、新oracleや新executorは実装していない。
