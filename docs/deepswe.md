# DeepSWE v1.1

DatacurveのDeepSWE v1.1を、既存ベンチマークと同じwrapper、API URL、model自動検出、結果・レポートの形式で実行します。AgenticaのDeepSWEモデルとは別です。

```bash
./kairyu-bench run http://host.docker.internal:8003/v1 \
  --only deepswe --limit 1 --run-id deepswe-smoke-20260910
```

Linux / x86_64、Docker Composeを推奨します。113タスクの定義を検証し、ディレクトリ名の昇順に選択してから`--limit`を適用します。全件は`--limit`なしです。モデルが使うコマンド、prompt、commit済みpatchの取得、独立verifierによる採点は固定した公式Pier / mini-swe-agentを使用します。`--harbor-agent`はDeepSWEには適用されません。

| 環境変数 | 既定 | 設定範囲・意味 |
| --- | --- | --- |
| `KAIRYU_BENCH_DEEPSWE_WORKERS` | 4 | 1–16。1反復内の同時問題数、完了した枠に即補充 |
| `KAIRYU_BENCH_DEEPSWE_ATTEMPTS` | 1 | 1または4。4の場合は同じ113問を4反復、並列数は合計workers以内 |
| `KAIRYU_BENCH_DEEPSWE_RETRIES` | 3 | 0–16。一時的API・環境エラーのPier再試行上限。mini自身のAPI retryも適用 |
| `KAIRYU_BENCH_DEEPSWE_REASONING_EFFORT` | 未指定 | none/minimal/low/medium/high/xhigh/max。未指定はAPI既定 |
| `KAIRYU_BENCH_CLEAN_TASK_IMAGES` | 0 | 専用daemonで1に設定。評価後に当該taskの残存imageを削除 |

APIへは`/v1/chat/completions`を使い、`/models`で検出したmodel IDを変えずに渡します。対応していないeffortを自動で弱めません。APIキーは環境変数から渡し、configの引数に埋め込みません。APIホストとポートだけをPierの認証付きproxyへ許可し、taskの内部ネットワークと独立verifierのネットワーク遮断を保持します。`host.docker.internal`はrunnerが解決したIPをproxy containerへ渡します。

## 全件測定用の専用Docker daemon

以下はLinuxのrepository rootから実行する手順です。`AGENTS.md`に従い、**privileged runner、privileged Docker-in-Docker daemon、Docker socket共有への明示的承認を得てから**実行します。API/Open WebUI用のhost Dockerをpruneしません。既存パスやcontainer名が使われていたら日付・suffixを変えてください。

```bash
set -eu
REPO=$(pwd -P)
DEEPSWE_CACHE=/mnt/nvme/kairyu/bench-cache/deepswe-comparison-20260910
DEEPSWE_DOCKER=/mnt/nvme/kairyu/docker/deepswe-comparison-20260910
DEEPSWE_DAEMON=kairyu-deepswe-dind-20260910

test ! -e "$DEEPSWE_CACHE"
test ! -e "$DEEPSWE_DOCKER"
test ! -e "$REPO/results/deepswe-smoke-20260910"
test ! -e "$REPO/results/deepswe-comparison-20260910"
if docker container inspect "$DEEPSWE_DAEMON" >/dev/null 2>&1; then exit 1; fi
mkdir -p "$REPO/results" "$DEEPSWE_CACHE" "$DEEPSWE_DOCKER/data" "$DEEPSWE_DOCKER/run"
docker run -d --privileged --name "$DEEPSWE_DAEMON" \
  -e DOCKER_TLS_CERTDIR= \
  -v "$DEEPSWE_DOCKER/data:/var/lib/docker" \
  -v "$DEEPSWE_DOCKER/run:/var/run" \
  -v "$REPO/results:$REPO/results" \
  -v "$DEEPSWE_CACHE:$DEEPSWE_CACHE" \
  docker:27.5.1-dind dockerd --host=unix:///var/run/docker.sock

export KAIRYU_BENCH_CACHE_DIR="$DEEPSWE_CACHE"
export KAIRYU_BENCH_DOCKER_SOCKET="$DEEPSWE_DOCKER/run/docker.sock"
export KAIRYU_BENCH_CLEAN_TASK_IMAGES=1
export KAIRYU_BENCH_DEEPSWE_WORKERS=4
export KAIRYU_BENCH_DEEPSWE_RETRIES=3
# daemon起動後、Unix socket経由でServer情報が返ることを確認する。
docker --host "unix://$KAIRYU_BENCH_DOCKER_SOCKET" info

./kairyu-bench run http://host.docker.internal:8003/v1 \
  --only deepswe --limit 1 --run-id deepswe-smoke-20260910

# smokeの公式trial/patch/verifier/rewardとnormalized結果を確認してから全件へ進む。
export KAIRYU_BENCH_DEEPSWE_ATTEMPTS=4
./kairyu-bench run http://host.docker.internal:8003/v1 \
  --only deepswe --run-id deepswe-comparison-20260910
```

認証が必要なら既存runnerと同じ`KAIRYU_API_KEY`を設定します。DeepSWE自体にHF tokenは不要です。`HF_TOKEN`をtask containerへ渡しません。

## 採点・進捗・成果物

`results/<run-id>/`以下に`run.json`、`context/deepswe.json`、`logs/deepswe.log`、`normalized/deepswe.json`、`report.json`、`report.md`を保存します。`raw/deepswe/`には以下を保持します。

- `selected.json` / `instance-ids.txt`: 選択ID、model ID、dataset revision。
- `trial-plan.json`: task × repeatと、正規化に使うtrial結果の対応。
- `jobs/repeat-*/`: 公式config、lock、trial result、trajectory、commit済みpatch、verifierログ。
- `trials/repeat-*/<task>/`: 最終trial結果と使用imageの情報。
- `retries/`: Pierが再試行時に消す失敗trialの証跡を退避。
- `runtime.json`: パッケージ版、source lock、設定。各trialには実際のimage ID/digestも保存。
- `progress.json` / `summary.json`: 現在の集計、実行中task、経過時間、root/cache空き容量。

4反復では**113問題・452試行**です。主指標pass@1は成功試行/採点対象試行で、4回のbest-ofではありません。全反復が揃った場合だけpass@4と4反復のrun-to-run 95% CIを算出します。agent timeoutとcontext超過は失敗、API・verifier・network障害や欠けたtrialは採点対象外として記録し、全選択IDを維持します。未採点があれば`partial`（採点対象が0なら`failed`）です。cleanup失敗も完了扱いにしません。ローカルAPIの費用は不明とします。

起動直後に`run.json=running`とAPIのmodel ID、選択113件・計画452slotを照合してください。runtimeは各フェーズ・完了・エラー時と、実行中少なくとも毎分進捗を出力します。運用時はAPIと専用Dockerのhealthも毎分確認し、遅いだけで正常な全件実行を止めないでください。完了時は452試行が採点され、run/normalizedが`completed`であり、公式trialの成功数とreportの分子・分母が一致することを確認します。

SIGTERM/SIGINTはPierの終了処理を経由します。強制killやホスト障害時のcleanupは保証できないため、再開時は専用daemonの残存container/imageを確認します。完了後も`results/`を保存し、cache/data root削除は共有されていない正確なパスを再確認してから行います。

## 検証状況と公開比較

固定した公式ライブラリで、API transport、ジョブ設定、並列キュー、retry、4反復、公式task定義113件を検証しています。CIではこれらをDocker/有料モデルなしで再検証します。**実モデル＋Dockerのsmokeおよび452試行の全件評価は別途必要です。** 実測を行っていない段階で公開値を自モデルのスコアに流用しません。

[モデル比較](model-comparison.md)に28モデル・70設定の公式公開値と出典を保存しています。公開側のModal環境・agent版などが一致すると確認できないため、生成レポートの`public_references`は実測平均・deltaに入りません。ローカルrun同士もsource、選択ID、agent、採点、設定、実際のtask image provenanceの一致が必要です。
