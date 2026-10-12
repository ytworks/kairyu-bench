# Terminal-Bench 4.0

Terminal-Bench 4.0（`terminal-bench/terminal-bench@4.0.0`、66タスク）を、既存ベンチマークと同じwrapper、API URL、model自動検出、結果・レポートの形式で実行します。既存の`terminal-bench`（2.1）とは別のベンチマーク`terminal-bench-4`です。

```bash
./kairyu-bench run http://host.docker.internal:8003/v1 \
  --only terminal-bench-4 --harbor-agent claude-code --limit 1 \
  --run-id terminal-bench-4-smoke-20261011
```

Linux / x86_64、Docker Composeを推奨します。66タスクをタスク名の昇順に選択してから`--limit`を適用し、全件は`--limit`なしです。

**公式harnessとピン留め**
- 実行は公式の`harbor run --config`です。Harbor 0.24.0は`harbor-framework/harbor@b53b8134…`のPyPI wheelを使い、`scripts/locks/terminal-bench-4-harbor.txt`のhash付きlockで導入します。
- datasetは公式リーダーボードの`DATASET_REF`（`sha256:39d9f44b…7732`）でdigest固定し、66タスクのdigestとtask imageのdigestを`src/kairyu_bench/data/terminal-bench-4.json`に保存しています。
- Harborが解決したtask digestが1つでも異なれば、trial開始前に停止します。

**エージェント**
- `--harbor-agent`で`claude-code`（比較用フルランの既定方針）、`terminus-2`、`codex`を選べます。
- agentの版・環境変数とretry設定は、公式リーダーボードの実行config（terminal-bench@`209679e3…`の`leaderboard/runs/`）に合わせています。
- claude-codeは2.1.231です。ローカルAPI向けに、2.1で使っている互換envも渡します。
- codexは、モデルIDに`/`を含む場合は実行前に`unsupported`で停止します。固定HarborのCodex agentは最後の`/`より後ろだけをモデル名として送るため、記録したIDと実際の指定が一致しなくなるからです（2.1と同じ扱い）。claude-codeとterminus-2は完全なIDを渡します。

| 環境変数 | 既定 | 設定範囲・意味 |
| --- | --- | --- |
| `KAIRYU_BENCH_TERMINAL_BENCH_4_WORKERS` | 4 | 1–16。Harborの同時trial数。完了した枠に即補充 |
| `KAIRYU_BENCH_TERMINAL_BENCH_4_ATTEMPTS` | 1 | 1または5。5は公式リーダーボードと同じ66タスク×5試行（330 slot） |
| `KAIRYU_BENCH_TERMINAL_BENCH_4_RETRIES` | 3 | 0–16。Harborの`retry.max_retries`。対象外の例外は常に公式と同じ |
| `KAIRYU_BENCH_TERMINAL_BENCH_4_REASONING_EFFORT` | 未指定 | agentが受け付ける値のみ（claude-code: low/medium/high/xhigh/max）。未指定はAPI既定 |
| `KAIRYU_BENCH_CLEAN_TASK_IMAGES` | 0 | 専用daemonで1に設定。各タスクの全試行が確定した時点で、そのタスクのimageを削除。失敗は記録し、後続タスクの完了時とjob終了時に再試行 |

**タイムアウトとリソース**
- 公式規則に合わせ、タイムアウト倍率とCPU・メモリ・GPU・ストレージのoverrideは一切設定しません。
- 2.1の`KAIRYU_BENCH_TERMINAL_BENCH_SETUP_TIMEOUT_MULTIPLIER`は4.0には適用されません。agent setupはHarbor既定の360秒です。

**APIとの接続**
- `/models`で検出したmodel IDを変えずに渡します。
- claude-codeはtask container内から`ANTHROPIC_BASE_URL`（`/v1/messages`）を使います。terminus-2は`openai/<model>`と`api_base`で`/v1/chat/completions`を使います。codexは`OPENAI_BASE_URL`（`/v1/responses`）を使います。
- 専用Docker daemon内では`host-gateway`がホストを指さないため、runnerが解決した`host.docker.internal`のIPをtaskの`main` serviceへ渡します。
- APIキーは環境変数で渡し、configやコマンドライン引数に含めません。
- Harborには`HF_TOKEN`と`KAIRYU_API_KEY`を渡さず、`HARBOR_TELEMETRY=0`で実行します。

## 全件測定用の専用Docker daemon

以下はLinuxのrepository rootから実行する手順です。`AGENTS.md`に従い、**privileged runner、privileged Docker-in-Docker daemon、Docker socket共有、APIを数日占有することへの明示的承認を得てから**実行します。API/Open WebUI用のhost Dockerをpruneしません。既存パスやcontainer名が使われていたら日付・suffixを変えてください。

**事前に確認すること**
- **CPU**：TB4には16 CPUを要求するタスクが2件あり、daemonから見えるCPUが16未満だとrunnerは`unsupported`で停止します。`nproc`と`free -g`を確認し、`--cpuset-cpus`でAPIが使うコアを外してください。dockerdは割り当てたcpusetをCPU数として報告します。
- **メモリ**：daemonに`--memory`上限を付けます。4並列なら、CPU制限値の合計は最大約40、メモリは64GB以上になり得ます。sidecarには制限がありません。
- **ディスク**：task imageは66タスク・157件で、圧縮時に約98GBあります。試行はattempt-major順（全タスクを1巡してから次の巡）なので、`KAIRYU_BENCH_CLEAN_TASK_IMAGES=1`でもピーク時には全imageが共存します。NVMeの空きは400GB以上を確保してください。Harborがダウンロードするtask定義（約0.5GB）は`$KAIRYU_BENCH_CACHE_DIR/harbor-tasks`に保存します。

```bash
set -eu
REPO=$(pwd -P)
TAG=20261011
TB4_CACHE=/mnt/nvme/kairyu/bench-cache/terminal-bench-4-$TAG
TB4_DOCKER=/mnt/nvme/kairyu/docker/terminal-bench-4-$TAG
TB4_DAEMON=kairyu-tb4-dind-$TAG
# APIが使っていないコアを16以上。例: TB4_CPUS=32-63
TB4_CPUS=<cores>
TB4_MEMORY=<memory>   # 例: 160g

test ! -e "$TB4_CACHE"
test ! -e "$TB4_DOCKER"
test ! -e "$REPO/results/terminal-bench-4-smoke-$TAG"
test ! -e "$REPO/results/terminal-bench-4-comparison-$TAG"
if docker container inspect "$TB4_DAEMON" >/dev/null 2>&1; then exit 1; fi
mkdir -p "$REPO/results" "$TB4_CACHE" "$TB4_DOCKER/data" "$TB4_DOCKER/run"
docker run -d --privileged --name "$TB4_DAEMON" \
  --cpuset-cpus "$TB4_CPUS" --memory "$TB4_MEMORY" \
  -e DOCKER_TLS_CERTDIR= \
  -v "$TB4_DOCKER/data:/var/lib/docker" \
  -v "$TB4_DOCKER/run:/var/run" \
  -v "$REPO/results:$REPO/results" \
  -v "$TB4_CACHE:$TB4_CACHE" \
  docker:27.5.1-dind dockerd --host=unix:///var/run/docker.sock

export KAIRYU_BENCH_CACHE_DIR="$TB4_CACHE"
export KAIRYU_BENCH_DOCKER_SOCKET="$TB4_DOCKER/run/docker.sock"
export KAIRYU_BENCH_CLEAN_TASK_IMAGES=1
export KAIRYU_BENCH_TERMINAL_BENCH_4_WORKERS=4
export KAIRYU_BENCH_TERMINAL_BENCH_4_RETRIES=3
# NCPUが16以上、RuntimesにnvidiaがないことをGPU方針の前提として確認する。
docker --host "unix://$KAIRYU_BENCH_DOCKER_SOCKET" info \
  --format '{{.NCPU}} {{.MemTotal}} {{json .Runtimes}}'

./kairyu-bench run http://host.docker.internal:8003/v1 \
  --only terminal-bench-4 --harbor-agent claude-code --limit 1 \
  --run-id terminal-bench-4-smoke-$TAG

# smokeの公式trial、claude-codeの版、API到達、別verifierのrewardと
# normalized結果を確認してから全件へ進む。
export KAIRYU_BENCH_TERMINAL_BENCH_4_ATTEMPTS=5
./kairyu-bench run http://host.docker.internal:8003/v1 \
  --only terminal-bench-4 --harbor-agent claude-code \
  --run-id terminal-bench-4-comparison-$TAG
```

**補足**
- 認証が必要なら、既存runnerと同じ`KAIRYU_API_KEY`を設定します。
- Terminal-Bench 4.0自体にHF tokenは不要です。
- offline verifierのタスク（`batched-eval-parity`, `lake-temp-glm`）を含む選択では、Harborのnftables egress制御がdaemonのkernelで使えるかを事前に検査します。使えなければ`unsupported`で停止します。

## 採点・進捗・成果物

`results/<run-id>/`以下に`run.json`、`context/terminal-bench-4.json`、`logs/terminal-bench-4.log`、`normalized/terminal-bench-4.json`、`report.json`、`report.md`を保存します。`raw/terminal-bench-4/`には以下を保持します。

- `selected.json` / `instance-ids.txt`：選択タスク、試行数、agent、model ID、job名。
- `job-config.json` / `host-gateway.json`：Harborへ渡した実際のconfigとcompose overlay。秘密情報は含みません。
- `runtime.json`：Harborの版、dataset digest、Docker daemonのCPU数・runtime、GPU有無、egress検査、conditions。
- `trial-plan.json`：Harborが作ったtrial名と、task×attemptの対応。job開始時にpluginが記録します。
- `jobs/<run-id>-terminal-bench-4/`：Harbor公式のjob/trialディレクトリ。config、lock、result、trajectory、verifierログを含みます。
- `retries/<trial>/<n>/`：Harborが再試行前に削除する、失敗した試行の証跡を退避したもの。
- `progress.json` / `summary.json`：現在の集計、実行中trial、経過時間、API/Dockerの状態、root/cacheの空き容量。
- `cleanup-errors/`、`plugin-errors/`、`runtime-error.json`：これらがあるrunは`completed`になりません。`cleanup-errors/`の記録は、その後の再試行でimageを削除できた時点で消えます。`docker image rm`が「No such image」以外で失敗した場合（daemon接続エラーやtimeoutを含む）は削除済みとみなしません。

**試行の分類**

各slot（task×attempt）の最終試行を、次の規則で分類します。

| 条件 | 扱い |
| --- | --- |
| verifier rewardがある | 採点。reward>0なら成功 |
| 上記で失敗、かつAPI障害系の例外（`Api*`、`NetworkConnectionError`など）がある | 採点対象外 |
| agent timeout、context超過、出力上限超過、安全拒否 | 0点 |
| GPUタスクで、daemonにNVIDIA runtimeがなく、環境起動時にdockerdのGPU driverエラーで失敗し、agentもverifierも実行されていない | `gpu-unavailable`として0点。runを`partial`にしない |
| verifier timeout、reward fileエラー、agent setup timeout、環境起動失敗、欠けたtrialなど | 採点対象外 |

- GPUタスク（`fp8-rmsnorm-gemm`, `jax-speedrun-gpu`, `math-eval-grader`）のフルランでは15 slotがこの扱いになります。
- 採点対象外があれば`partial`、採点が0件なら`failed`です。
- identityの不一致があれば、採点せずに正規化を失敗させます。照合する項目は次のとおりです。
  - task digest
  - agent、model、agent版
  - 計画外のtrial
  - 公式static checkで禁止されたtimeout・resourceのoverride

**指標**

公式リーダーボードの式（terminal-bench@`209679e3…`の`core/metrics.py`）をそのまま移植しています。

| 指標 | 算出方法 |
| --- | --- |
| 主指標accuracy | 成功試行 ÷ 採点試行。全slotが採点済みなら公式式と一致 |
| 95% CI | 1.96 × 公式のper-task standard error。試行数が5のときのみ |
| pass@2〜5 | 試行数が5のときのみ |
| リーダーボード同等値 | 全計画slotを分母とし、採点対象外を失敗として数える |
| 非GPUタスクのみのaccuracy | GPUタスクを除いたaccuracy |
| `gpu-unavailable` slot数 | GPU未提供で0点にしたslotの数 |

ローカルAPIの費用は不明とします。

**起動後の確認と監視**
- 起動直後に次を照合してください。
  - `run.json=running`であること
  - APIのmodel IDが一致すること
  - 選択が66件、計画が330 slotであること
  - 最初のtrialでclaude-codeの版、API到達、別verifierのrewardを確認できること
- pluginは各フェーズ・retry・cleanup失敗時と、実行中は毎分`Terminal-Bench 4.0 progress: …`を出力します。
- 遅いだけで正常な全件実行は止めないでください。各タスクのagent timeoutは8時間です。

**完了時の確認**
- 330 slotがすべて採点されていること（`gpu-unavailable`の15を含む）。
- 66タスクが評価済みであること。
- cleanupエラーがないこと。
- runとnormalizedの両方が`completed`であること。

**停止と後片付け**
- 計画的に停止するときは、runner container内で`pkill -TERM -f 'harbor run'`を実行します。Harborの終了処理を経由し、その後に部分結果を正規化します。
- 強制killやホスト障害時のcleanupは保証できません。再開機能はないため、新しいrun IDでやり直します。
- 完了後も`results/`を保存します。cache/data rootの削除は、共有されていない正確なパスを再確認してから行います。

## 検証状況と公開比較

**CIの`terminal-bench-4-runtime`ジョブで検証する項目**
- 固定したHarbor 0.24.0がこのconfigをそのまま受け付けること。
- registryの66タスクのdigestがlockと一致すること。
- pluginが実際のHarbor Jobに接続して試行計画を作れること。
- NVIDIA runtimeのないDockerでGPUタスクの環境起動が`gpu-unavailable`に分類されること。

前の3項目は開発時にも固定Harborで確認済みです。GPU分類はDockerのあるCIで初めて実行されます。

**未実施**
- **実モデルでのsmoke、および330試行の全件評価は別途必要です。**

**公開値との比較**
- 公開リーダーボードの値はModal上のrunで、GPUタスクも実行されています。そのため本runnerの値は`published_leaderboard_comparable: false`として記録し、公開値の実測平均やdeltaには入れません。
- ローカルrun同士は、source、選択ID、agent、採点、conditions（Harbor版、dataset digest、agent版、試行数を含む）、daemonのGPU有無がすべて一致する場合に限り比較します。
