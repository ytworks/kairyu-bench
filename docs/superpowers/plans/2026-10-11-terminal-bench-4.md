# Terminal-Bench 4.0 フルラン追加計画（`terminal-bench-4`）

## Context

- 現状の `terminal-bench` は Terminal-Bench **2.1** を Harbor 0.21.0（`ea2fee7`）で実行している。その仕組みは次のとおり。
  - 1試行固定（`--n-attempts 1`）。
  - データセットのrevisionは実際にはHarborへ渡しておらず、`latest` に解決される。
  - `_harbor` 正規化器（`src/kairyu_bench/official.py:137-189`）は重複task名を拒否する。このため複数試行は扱えない。
  - 進捗ファイル、DinD手順、`KAIRYU_BENCH_CLEAN_TASK_IMAGES` 対応、AGENTS.md のフルラン手順書はいずれも無い。
- 新モデルの発表値はTerminal-Bench **4.0** へ移行済み（`docs/model-comparison.md:13`）。
- 目的：LiveCodeBench Pro / SWE-bench Pro / DeepSWE と同じ流儀で、4.0の公式条件フルランを `./kairyu-bench` から実行できるようにする。同じ流儀とは次の要素を指す。
  - ピン留めしたソースとデータセット
  - NVMe上の専用DinD
  - 毎分の進捗
  - 厳密な完了条件
  - 手順書

### ユーザー決定事項（2026-10-11）

1. 新規ベンチ `terminal-bench-4` を追加する。2.1の `terminal-bench` は変更しない。
2. 比較用フルランは **66タスク×5試行=330 trial**（公式リーダーボードと同条件）。既定は1試行で、フルランのときだけ5を指定する。
3. H100が必要な3タスク（fp8-rmsnorm-gemm, jax-speedrun-gpu, math-eval-grader）も実行する。
   - GPU未提供で起動できない試行は、公式規則どおり**0点**とする。
   - 理由ラベル `gpu-unavailable` を付けて明示する。
   - これらはrunを `partial` にしない。
4. 比較用フルランのエージェントは **claude-code**。`--harbor-agent` で terminus-2 / codex も選択可能にする。

## 検証済みの前提（調査結果）

**Terminal-Bench 4.0**
- tag `v4.0.0` = harbor-framework/terminal-bench `452bf305c6daa62fc59061d22133a7cbc7c1572e`（2026-08-26）。タスク数は66。
- Hub上のデータセット：`terminal-bench/terminal-bench`。リーダーボードCIのピンは `sha256:39d9f44b40420cde8fdcc087579c0d72a7e14fa3656d603c3f0d22fb35e27732`。

**公式規則（leaderboard @ `209679e34327a78ce8e9bf5300b2f317992863c9`）**
- 条件
  - 66タスク×各5試行以上。
  - タイムアウト・リソースのoverrideは全て禁止。`agent_setup_timeout_multiplier` も禁止。
  - errored trialは0点。
- 指標
  - accuracy = reward>0 の試行数 ÷ 全試行。
  - CI = 1.96·s。ここで `s² = (1/n²)Σ p_i(1−p_i)/(k_i−1)`。
  - pass@2〜5 は不偏推定量。
- `n_concurrent` と retry は検査対象外。
- 公式claude-code runの設定：`max_retries: 3`。exclude は次の7種。
  - `RewardFileEmptyError`, `AgentSafetyRefusalError`, `VerifierOutputParseError`, `RewardFileNotFoundError`, `AgentTimeoutError`, `VerifierTimeoutError`, `ApiUsageLimitError`

**Harbor**
- ピンは **v0.24.0**：`b53b8134e1241686dca7759af188f987ecc48e8b`、PyPI wheel sha256 `23b7ba61…877c`。
- 0.22/0.23を使わない理由：Docker環境がGPU非対応と判定する。GPUタスクの `_validate_gpu_support()` が例外を出し、**job全体が中断**する。
- 0.24 では compose up の時点でその試行だけが失敗する。
- 試行の並び順は attempt-major（全66タスクを1巡してから2巡目に入る）。
- retry時は失敗試行のディレクトリが `rmtree` される。ただし削除はtrial hookの後に行われる。
- `--plugin` でjobにhookを付けられる。

**タスク特性**
- agent timeout：全タスク8時間。
- verifier：全タスクで別コンテナ。うち2タスクはverifierがoffline（nftables egress sidecarが必要）。
- リソース：CPU 2〜16、メモリ4〜32GB。11タスクがcompose構成。
- `storage` 指定はDockerでは無視される。

**エージェント**
- claude-code：`ANTHROPIC_BASE_URL` を使う（runnerが既に設定済み）。Kairyu APIは `/v1/messages` に対応済み（2.1のclaude-code実績あり）。
- terminus-2：`openai/<id>` と `--ak api_base=` で接続できる。

**その他**
- Harborはtelemetryを送信する（`HARBOR_TELEMETRY=0` で停止）。
- プロセス環境由来のAPIキーをHarbor自身はscrubしない。

## アーキテクチャ

方式は **公式 `harbor run` CLI ＋ Kairyu job plugin** とする。

```
harbor run -c raw/terminal-bench-4/job-config.json \
  --plugin kairyu_bench.terminal_bench4_plugin:KairyuTB4Plugin --pk raw_dir=<raw> --yes --quiet
```

- 公式のCLI経路（設定組み立て・検証・SIGTERM処理）を保ったまま、hookで次の処理を行う。
  - 試行計画の記録
  - retry証跡の退避（`rmtree` より前）
  - 進捗の出力
  - imageの掃除
- 別プロセスでのpollingは採用しない。retryディレクトリ削除と競合するため。
- DeepSWE式のJob API直呼びも採用しない。CLI経路を迂回することになり、テストもHarbor導入時しか動かないため。
- 新規2モジュールはロード時にharborをimportしない。通常の `unittest` で検証できるようにするため。
- trial hookは決して例外を投げない。全て捕捉し、エラーファイルに記録する。
- `on_job_start` だけは fail-closed で例外を投げてよい。
- Docker操作は全て非同期subprocessで行う。

## 環境変数契約

wrapperは `${VAR-default}` で転送する。空文字は拒否する。

| 変数 | 既定 | 内容 |
|---|---|---|
| `KAIRYU_BENCH_TERMINAL_BENCH_4_WORKERS` | 4 | 1–16。`n_concurrent_trials`。即時補充はHarborのキューが行う |
| `KAIRYU_BENCH_TERMINAL_BENCH_4_ATTEMPTS` | 1 | 1 または 5。比較フルランは5（330スロット） |
| `KAIRYU_BENCH_TERMINAL_BENCH_4_RETRIES` | 3 | 0–16。`retry.max_retries`。excludeは常に公式の7種 |
| `KAIRYU_BENCH_TERMINAL_BENCH_4_REASONING_EFFORT` | 未設定 | エージェントごとに検証する。未設定ならサーバ既定 |
| `KAIRYU_BENCH_CLEAN_TASK_IMAGES` | 0 | 1のとき、各タスクの**最終試行が確定した時点**でそのタスクのimageを削除する |

- setup-timeout multiplier は**提供しない**（公式規則で禁止）。
- 正規化結果の `conditions` に次を記録する。runnerの `_validate_identity` がこれを照合する。
  - 版・digest：dataset digest、Harbor版とcommit、lockのsha256、エージェント名と版
  - 実行パラメータ：非秘密のagent env、effort、attempts、workers、retries、exclude
  - 方針：全multiplierとoverrideが `null` であること、`gpu_policy`、`error_policy`

## 採点・集計ルール（`terminal_bench4.py`、stdlibのみ）

各スロット（task×attempt）の最終試行を、次の順で分類する。

1. 結果が無い、または `finished_at` が無い → **unscored**。
2. 同一性違反 → **例外**（採点しない）。照合項目は次のとおり。
   - task名が選択に含まれる
   - trial名が計画に含まれる
   - `task.ref` がピンのdigestと一致する
   - agentとmodelが一致する
   - 公式static checkに合格する
3. 数値rewardがある → **scored**（reward>0で成功）。ただし、インフラ系のAPI例外があり、かつ成功していない場合は unscored とする。
4. rewardが無く、`gpu-unavailable` の判定条件を**全て**満たす → **scored 0（gpu-unavailable）**。判定は純関数で行う。
   - `gpu_tasks` に含まれ、task.tomlで `gpus>0` である。
   - preflightで `gpu_available=false` である。
   - `RuntimeError` かつ nvidia device driver のメッセージに一致する。実メッセージはTask 0で採取し、正規表現を固定する。
   - agent_setup / agent_execution / verifier が全て空である。
5. rewardが無く、例外がモデル起因 → **scored 0**。対象は `AgentTimeoutError`, `ContextWindowExceededError`, `OutputTokenExceededError`, `AgentSafetyRefusalError`。
6. 上記以外 → **unscored**。型を記録する。例：`VerifierTimeoutError`, reward-file系, `AgentSetupTimeoutError`, `EnvironmentStartTimeoutError`, Docker系。

**ステータス**
- `completed`：全スロットscored、66タスク評価済み、cleanup/runtimeエラーなし。
- `partial`：一部がunscored、欠損、またはcleanupエラーあり。
- `failed`：scoredが0件。

**指標**
- 主スコア：accuracy% = 成功 ÷ scored。completedのときは公式式と完全に一致する。
- 併記する指標：
  - CI、pass@2〜5（公式式を移植。attempts=1のときはnull）
  - リーダーボード同等値（全計画スロットを分母とし、unscoredは失敗扱い）
  - 非GPUタスクのみのaccuracy
  - gpu-unavailableのスロット数（フルランでは15）
  - `published_leaderboard_comparable: false` とその理由（ローカルDocker、GPU未提供）

## 成果物ツリー（`results/<run-id>/raw/terminal-bench-4/`）

- `selected.json`, `instance-ids.txt`, `trial-plan.json`：330スロット
- `job-config.json`：秘密情報なし
- `host-gateway.yaml`：runnerで解決したIPを書く。DinD内の `host-gateway` はホストを指さないため
- `runtime.json`：Harbor・lock・docker info（NCPU / Mem / Runtimes）、`gpu_available`、egress probe、static check結果
- `task-definitions.json`：digest、image ref、gpus / cpus / mem、timeout
- `jobs/<run-id>-terminal-bench-4/`：Harborの公式job/trialディレクトリ
- `retries/<trial>/<n>/`：非最終試行。redact済み
- `cleanup-errors/`, `runtime-error.json`, `progress.json`, `summary.json`

## 変更ファイル

**新規**

- `src/kairyu_bench/artifacts.py`
  - `deepswe.py` / `deepswe_runtime.py` から次を共通化する：`write_json`, `read_object`, `sha256_file`, `redact_artifacts`, `env_integer`。
  - DeepSWE側は再exportで互換を保つ。
- `src/kairyu_bench/terminal_bench4.py`
  - 設定、conditions、選択（`--limit` はソート済み先頭N件を `-i` で渡す）、`build_job_config`。
  - 公式static checkと指標の移植、分類器、`summarize`。
  - `prepare` / `finalize` サブコマンド。
- `src/kairyu_bench/terminal_bench4_plugin.py`：Harbor job plugin。
  - job開始時：同一性検証と `trial-plan.json` の書き出し（`job._trial_configs` を使う。private属性なのでruntimeテストで固定する）。
  - 最終試行かretryかの判定（`_should_retry_exception` と試行回数を写す）。
  - retry証跡の退避。
  - 参照カウント付きのimage削除。
  - 60秒ごとと試行確定ごとの `progress.json`、ログ行 `Terminal-Bench 4.0 progress: scored=…/330 passed=… gpu_unavailable=… unscored=… tasks=…/66 active=… elapsed=… root_free=… cache_free=…`。進捗には `/v1/models` のhealthも含める。
- `src/kairyu_bench/data/terminal-bench-4.json`
  - 66タスクのdigest表、`gpu_tasks`、offline verifierのタスク、公式retry exclude。
  - エージェントのピン：claude-codeは公式run設定の版（調査では2.1.231）、codexも同様。2.1で必要だったKairyu互換env。
- `adapters/terminal-bench-4/{run.sh,normalize.py}`：既存のboilerplateと同じ形。
- `scripts/harnesses/terminal-bench-4.sh`：`deepswe.sh` の構造に倣う。処理順は次のとおり。
  1. `ensure_venv`：hash付きlockからHarborを導入する。
  2. `prepare`
  3. `HARBOR_TELEMETRY=0 env -u HF_TOKEN harbor run …` をバックグラウンドで起動する。TERM/INTを転送して `wait` する。
  4. `set +e` で終了コードを保持する。
  5. `finalize`：redactとsummary。
  6. `normalize_official`
- `scripts/locks/terminal-bench-4-harbor.txt`
  - `uv pip compile` で生成する（`harbor==0.24.0`、py3.12、linux、`--generate-hashes`）。
  - あわせて `scripts/locks/README.md` に追記する。
- テスト：
  - `tests/test_terminal_bench4.py`
  - `tests/test_terminal_bench4_plugin.py`
  - `tests/fixtures/terminal-bench-4/`：trial結果、ローカルGPUタスク、offline verifierタスク
- ドキュメント：
  - `docs/terminal-bench-4.md`
  - `docs/superpowers/plans/2026-10-11-terminal-bench-4.md`：本計画を既存のplan形式で残す。

**変更**

- `src/kairyu_bench/benchmarks.py` と `data/benchmarks.json`
  - 14番目に `terminal-bench-4` を追加する。
  - source：harbor @ `b53b8134…`
  - dataset：`{"id":"terminal-bench/terminal-bench","revision":"sha256:39d9…","version":"4.0.0","tasks":66}`
  - secondary_sources：TB `452bf305…`（タスク）と `209679e3…`（leaderboard）
  - scoring：`terminal-bench-4.0-leaderboard-accuracy`
- `src/kairyu_bench/runner.py`
  - `_selected_agent` がTB4の採点方式でも `--harbor-agent` を返すようにする。
  - DeepSWE専用の分岐（`runner.py:176-180,225-226`）を、ベンチ名をキーにした小さなconditions resolver表に置き換える。表はモデル発見の前に評価する。
- `src/kairyu_bench/cli.py`：dry-runでTB4設定を検証する（`cli.py:103-106` と同じ形）。
- `src/kairyu_bench/official.py`：`NORMALIZERS["terminal-bench-4"]` と同一性検証を追加する。
- `src/kairyu_bench/reporting.py`
  - 試行数行をベンチ別の文言にする（現在は全て "DeepSWE trials"。`:90-96`）。
  - `trial_metrics` をTB4にも出す（`:148-150`）。
  - `_compatibility` でruntimeの由来（Harbor版、task digest、image digest）の一致を要求する。
- `kairyu-bench` wrapper：新しい変数の `-e` 行を追加する（`:31-39`）。
- `.github/workflows/ci.yml`：`terminal-bench-4-runtime` ジョブを追加する。
- `AGENTS.md`：冒頭の並列度節を更新し、「Terminal-Bench 4.0 full run」節を新設する。
- `README.md`：ベンチ数を13→14にし、表の行と節を追加する。
- 既存テスト：`test_cli`, `test_manifest`, `test_host_entrypoint`, `test_runner`, `test_reporting`, `test_official_normalizers`, `test_official_shell`。

## タスク（TDD順）

**0. 事前検証**（scratch上のみ）
- Harbor 0.24 で `@sha256:39d9…` を解決し、66件のdigestを記録する。
- Hubのtask.tomlの `docker_image` がdigestでピンされていることを確認する。
- 公式trialのlockから、Harbor版とclaude-code版・envを読む。
- 66タスクの全imageの圧縮サイズを、registryのmanifestから合計する（ピークのストレージ見積り）。
- hash付きlockを生成する。
- GPUなしのdockerdで `-a nop -i terminal-bench/math-eval-grader` を実行し、実際のエラーメッセージを採取する。

**1. 共通化**
- `artifacts.py` を抽出する。既存のDeepSWEテストが通ることを確認する。

**2. 登録**
- manifestと名前を追加する。`test_cli` の件数（14）、`test_manifest` を更新する。

**3. 設定と静的検査**
- settingsを実装する。
- `build_job_config` を実装する。期待JSONと完全一致させる。multiplier/overrideキーが無いこと、公式retry、digest ref、`--limit` 時のみ `-i` を付けることを確認する。
- 公式static checkを移植する。

**4. 指標・分類・集計**
- 指標を固定ベクトルで検証する。例：[1,1,0,0,0]+[1,1,1,1,1] → 70.00%、±24.00、pass@2 = 0.85。
- 分類器を表駆動テストで検証する。GPU判定の各条件について否定ケースを用意する。
- `summarize` の completed / partial / failed を検証する。cleanupエラーでcompletedにならないこと、計画外trialで例外になることも確認する。

**5. plugin**
- fakeのjob/eventで次を検証する。
  - 計画の書き出しとfail-closed
  - 最終試行かretryかの判定
  - 証跡の退避
  - fake `docker` による参照カウント削除
  - hookが例外を投げないこと
  - 進捗の出力周期

**6. harness**
- fakeの `harbor` / `docker` で次を検証する。
  - 引数とconfigを捕捉し、`--plugin` と `--yes` があること。`--agent-setup-timeout-multiplier` が無いこと。
  - `HF_TOKEN` が無く、`HARBOR_TELEMETRY=0` が設定されていること。
  - 非0終了でも部分結果を正規化すること。
  - TERMを転送すること。
  - 引数・configに秘密情報が無いこと。
- wrapperの転送テストを追加する。

**7. 統合**
- runner / CLI / 正規化器 / reportingのテストを追加する。

**8. CI runtimeジョブ**
- pinしたlockで次を実行する。
  - `harbor run --print-config -c` でスキーマを照合する。
  - ローカルfixtureでoracle agent＋pluginを実行する（GPUエラーの分類、offline verifierでのreward）。

**9. ドキュメント**
- `docs/terminal-bench-4.md`：日本語。`docs/deepswe.md` と同じ4節構成（概要とsmoke、環境変数、専用DinD、採点・進捗・成果物と検証状況）。
- `AGENTS.md`：フルラン節（下記）。
- `README.md`
- plan doc

## 手順書（AGENTS.md「Terminal-Bench 4.0 full run」要旨）

**前提と承認**
- 特権runner、特権DinD daemon、共有socketの使用について、明示的な承認を得る。
- 複数日にわたりAPIを占有することについても承認を得る。

**専用DinD daemon**
- 構成：
  - `docker:27.5.1-dind`
  - data rootとsocketは `/mnt/nvme/kairyu/docker/terminal-bench-4-<tag>/` 配下
  - cacheは `/mnt/nvme/kairyu/bench-cache/terminal-bench-4-<tag>`
  - `results/` とcacheは同一パスでmountする
- 起動前に、`nproc` / `free -g` を確認する。
  - daemonから見えるCPUは**16以上**必要（16 CPUのタスクがあるため）。
  - API用のコアは `--cpuset-cpus` で外す。メモリにも上限を付ける。
- 起動後、`docker info` の NCPU / MemTotal / Runtimes を確認する。

**実行**
- 環境変数：`KAIRYU_BENCH_CACHE_DIR`, `KAIRYU_BENCH_DOCKER_SOCKET`, `KAIRYU_BENCH_CLEAN_TASK_IMAGES=1`, `..._TERMINAL_BENCH_4_WORKERS=4`, `..._RETRIES=3`
- smoke：`--only terminal-bench-4 --harbor-agent claude-code --limit 1`
- 本番：`..._ATTEMPTS=5` で、`--limit` なしで実行する。
- run IDは一意にする。`/bin/bash -ic` で実行し、`HF_TOKEN` はタスクへ渡さない。

**起動直後の確認**
- `run.json=running` であること。
- `model_id` が `/v1/models` と一致すること。
- 66タスクが選択され、330スロットが計画されていること。
- 最初の試行で次を確認する：claude-codeの版が導入されたこと、API到達、別verifierのreward。

**監視**（毎分報告）
- 報告項目：scored/330、passed、gpu-unavailable、unscored、評価済みタスク数/66、実行中のタスク、経過時間、API/Dockerのhealth、root/NVMeの空き容量。
- retryとcleanupの失敗は即時に報告する。
- 遅いだけで健全なrunは止めない。
- 停止手順：runnerコンテナ内で `pkill -TERM -f 'harbor run'` を実行する。

**完了条件**
- 330スロット全てscored（gpu-unavailableの15を含む）。
- 66タスク評価済み。
- cleanupエラーなし。
- runと正規化結果の両方が `completed`。

**報告内容**
- accuracy（分子/分母）、CI、pass@2〜5、リーダーボード同等値、非GPUのみの値。

**後片付け**
- 結果は保全する。
- cacheとDinDのdata rootは、パスを再確認したうえで削除する。

## 検証

1. `PYTHONPATH=src python3 -m unittest discover -s tests -v`、`ruff`、`mypy`、`sh -n scripts/harnesses/terminal-bench-4.sh`、`scripts/update_model_comparison.py --check`、CIのruntimeジョブ。
2. `./kairyu-bench run <URL> --only terminal-bench-4 --dry-run` で、選択・設定の検証と表示を確認する。
3. 実ホストでのインフラsmoke（モデル不要）：nop agentでGPUタスクを実行し、oracle agentで `batched-eval-parity` を実行する。
4. `--limit 1`、attempts 1、claude-code。続けて `--limit 2`（offline verifierのタスクを含む）。
5. 承認後にフルラン（66×5）を実行する。

## リスク

- **所要時間**：(330 × 平均試行時間) ÷ 4。
  - 平均45分なら約2.6日、平均2時間なら約6.9日。
  - smokeの実測でETAを更新する。resume機能は無い。
- **image pull**：attempt-major順で、各タスクの最終試行は5巡目になる。
  - そのため、CLEAN=1でもピーク時には全タスクのimageが共存する。
  - 容量はTask 0で実測する。NVMeの空きを超える場合は、計画を見直してからフルランに進む。
  - Docker Hubのpull制限にも注意する。
- **リソース**：4並列で計最大約40 CPU・64GB以上の制限値になる。
  - sidecarには制限が無い。
  - 競合すると性能系verifierがタイムアウトする恐れがある。
- **claude-codeの導入**：setup multiplierが禁止のため、360秒で終わらないと `AgentSetupTimeoutError` になる。retryで吸収する想定。
- **GPUの15スロット**：方針として0点にする。公開値との直接比較性は失われ、その旨を明記する。
- **offline verifier**：nftablesのegress sidecarにはカーネル対応が必要。probeに失敗した場合、`prepare` は `unsupported` として停止する。

## スコープ外（後続）

- 公開値カタログ `public_results.json` と `docs/model-comparison.md` への「Terminal-Bench 4.0」行の追加。
  - effort違いの行が多い。表示方針も含めて、別のdocs PRで扱う（PR #25/#26 と同じ扱い）。
- resume機能。
- 2.1 harnessへの `HARBOR_TELEMETRY=0` の追加。2.1は変更しない方針のため。

## 実装PR時点の状態（2026-10-11）

**計画からの差分**
- モジュールは小さく分割しました。
  - `terminal_bench4.py`：設定、job config、公式指標、static check、試行計画
  - `terminal_bench4_accounting.py`：分類、同一性検証、集計
  - `terminal_bench4_cli.py`：`prepare` / `finalize`
  - `terminal_bench4_plugin.py`：Harbor job plugin
- Harborの版は`harbor --version`の出力を`prepare --harbor-version`へ渡し、lockの版と一致しなければ停止します。
- 公式claude-code configの`CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`は、Claudeモデル固有の値として採用していません。
  - `CLAUDE_CODE_SIMPLE=1`と版2.1.231は公式どおりです。
  - 2.1で検証済みのKairyu互換envも渡します。
- retry対象外の例外はagentごとに公式configに合わせました。
  - claude-codeは7種、codexは6種です。
  - terminus-2には公式runがないため、claude-codeと同じ7種にしています。
- 毎分のprogress snapshotが失敗してもrunは`completed`を妨げません。Harborは`result.json`を非atomicに書くため、読み取りが競合しうるためです。最終正規化ではすべての結果を再読込し、fail closedで判定します。
- `plugin-errors/`と、Harborの非0終了時の`runtime-error.json`は、`completed`を妨げます。

**検証済み**
- 単体・統合テストを追加しました。fakeの`harbor`/`docker`を使ったharness、runner、正規化、reportを含みます。
- 固定Harbor 0.24.0で次を確認しました。
  - `--print-config`がconfigを受け付けること。
  - registryの66件のdigestがlockと一致すること。
  - 実Jobへのplugin接続で計画を書き出せること。
- task image 157件の圧縮サイズは約97.7GBでした（Docker Hubのtag API）。
- `live-database-cutover`（`redis:7-alpine`）と`payments-pipeline-fix`（`apache/kafka-native:4.3.1`）の2つのsidecarはtag指定のままです。これは上流のタスク定義そのものです。

**未実施**
- GPU分類の実Docker検証。CIの`terminal-bench-4-runtime`ジョブで初めて実行されます。
- 実ホストでのインフラsmoke（nop/oracle agent）。
- `--limit 1`と`--limit 2`のclaude-code smoke。
- 330試行のフルラン。privileged実行とAPI占有の承認が必要です。
- 公開値カタログへの「Terminal-Bench 4.0」行の追加。別のdocs PRで扱います。
