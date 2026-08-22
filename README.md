# kairyu-bench

Kairyu互換APIを、公式実装に基づく12種類のベンチマークで評価する独立ランナーです。必要な引数はAPIのURLだけです。生成、採点、集計、比較はすべてDocker内で行い、`kairyu` 本体のコードはimport・コピーしません。

[他モデルの公開ベンチマーク比較（出典付き）](docs/model-comparison.md)

```sh
./kairyu-bench run https://kairyu.example/v1
```

モデル名の指定はありません。`GET /v1/models` が返すIDを順にchat completionで確認し、最初に応答できたモデルを自動選択します。

## 使い方

必要なホスト依存はDockerだけです。初回実行時にはrunner imageをbuildし、公式sourceと個別Python環境は `.cache/` に固定revision別で保存します。Docker layer cacheを使うため、2回目以降のbuild確認は短時間です。

```sh
# 全12種類
./kairyu-bench run https://kairyu.example/v1

# 指定したベンチマークだけを、公式順の先頭10問で実行
./kairyu-bench run https://kairyu.example/v1 \
  --only gpqa-diamond,hle,mrcr-v2 \
  --limit 10

# 名前の一覧
./kairyu-bench list

# 同一条件の行だけを比較
./kairyu-bench compare results/<baseline-run> results/<candidate-run>
```

認証が必要なAPIでは環境変数を使います。

```sh
KAIRYU_API_KEY=secret ./kairyu-bench run https://kairyu.example/v1
```

### Terminal-Benchのagent

Harborで実行する `terminal-bench` は `terminus-2`（既定）、`claude-code`、`codex` からagentを選べます。

```sh
# Claude Code + KairyuのAnthropic Messages互換API
KAIRYU_API_KEY=secret ./kairyu-bench run https://kairyu.example/v1 \
  --only terminal-bench --harbor-agent claude-code --limit 1

# Codex + KairyuのOpenAI互換API
KAIRYU_API_KEY=secret ./kairyu-bench run https://kairyu.example/v1 \
  --only terminal-bench --harbor-agent codex --limit 1
```

Terminus-2とCodexには `OPENAI_BASE_URL=<endpoint>/v1`、Claude Codeには `ANTHROPIC_BASE_URL=<endpoint>` を設定し、自動検出した同じmodel IDを渡します。Claude Codeでは初期互換範囲を安定させるためadaptive thinking、experimental beta、attribution headerを無効化します。API keyは対応するagent環境へ渡しますが、Harborのコマンドラインや結果には保存しません。Claude Codeを使うKairyuサーバーには `POST /v1/messages` が必要です（実装要件は [kairyu#508](https://github.com/ytworks/kairyu/issues/508)）。

agent本体はHarborがtask container内へ導入して起動するため、ホストのClaude Code/Codexのログイン状態や設定は引き継ぎません。

`localhost` 上のAPIをDocker Desktop/Linuxから評価する場合は、URLのhostに `host.docker.internal` を指定してください。runnerとHarborのnested task containerは、そのhost aliasを自動設定します。

## ベンチマーク

| 名前 | 実行・採点 | 補足 |
| --- | --- | --- |
| `swe-bench-pro` | mini-SWE-agent + SWE-bench公式harness | nested Docker、x86_64推奨 |
| `swe-bench-verified` | mini-SWE-agent + SWE-bench公式harness | nested Docker、x86_64推奨 |
| `terminal-bench` | Harbor公式harness、Terminal-Bench 2.1 | nested Docker |
| `livecodebench` | LiveCodeBench v6公式prompt・code tests・pass@1 | API transportだけ薄いshim |
| `livecodebench-pro` | 公式LightCPVerifier | gatedデータとnested Dockerが必要 |
| `hle` | CAIS `simple-evals` HLE | 対象モデル自身をjudgeに使用 |
| `charxiv-reasoning` | CharXiv reasoning validation・公式grading prompt/statistics | 対象モデル自身をjudgeに使用、vision必須 |
| `gpqa-diamond` | GPQA Diamond、exact choice match | gatedデータへの同意が必要 |
| `scicode` | SciCode公式Inspect task・numeric tests | 公式HDF5を別途配置 |
| `tau-bench-banking` | tau2 `banking_knowledge/alltools`、4 trials | 対象モデル自身をuser simulatorに使用 |
| `long-context-reasoning` | LongBench v2 choice accuracy | private Fugu行の代替として明示 |
| `mrcr-v2` | OpenAI MRCR v2、公式token bins/SequenceMatcher | 4K–128K、8 needles |

source、dataset、補助checkerのrevisionは [`src/kairyu_bench/data/benchmarks.json`](src/kairyu_bench/data/benchmarks.json) に固定しています。`--limit N` は各公式データ順を確定した後の先頭N問を選び、選択した問題IDをレポートへ残します。

## 追加データが必要な項目

GPQAとLiveCodeBench ProはHugging Face側で利用条件への同意が必要です。同意済みtokenを渡します。

```sh
HF_TOKEN=hf_... ./kairyu-bench run https://kairyu.example/v1 \
  --only gpqa-diamond,livecodebench-pro
```

SciCodeの数値test artifactは公式repositoryの案内に従って取得し、次へ配置してください。

```text
.cache/scicode/test_data.h5
```

これらが無い場合は0点を生成せず `unsupported` を記録します。API障害や公式harnessの異常終了も、推測したスコアへ置き換えず `failed` または `partial` になります。

## 結果

各実行は `results/<run-id>/` に保存されます。

```text
run.json                 実行状態、自動検出したmodel ID、Harbor agent
report.json              機械可読スコアレポート
report.md                表形式スコアレポート
normalized/<name>.json   共通schemaへ正規化した結果
raw/<name>/...           公式harnessの成果物
logs/<name>.log          stdout/stderr
context/<name>.json      adapterへ渡した固定条件
```

`report.md` のmacro averageは、完了したpercent指標の単純平均です。公式leaderboardの総合指標ではありません。HLE/CharXivは `self_judged`、tauは `self_simulated` として常に表示されます。

比較でdeltaを出す条件は、benchmark名、agent、source/dataset revision、問題ID、採点法、自己採点方針、score unitがすべて一致し、両方が `completed` であることです。それ以外は理由を表示しますが数値差は出しません。

## API契約

対象サーバーには次が必要です。

- `GET /v1/models`
- `POST /v1/chat/completions`
- OpenAI互換の `model`、`messages`、`max_tokens`、`temperature`
- CharXivでは `image_url` のdata URL入力
- Claude Codeを使う場合はAnthropic互換の `POST /v1/messages`

公開レポートにはendpointのSHA-256 fingerprintだけを記録します。API URLはローカルのadapter contextに保存され、API keyは保存しません。

## 注意

SWE-bench、Terminal-Bench、LiveCodeBench Proはtask containerを作るため、runnerを `--privileged` で起動しDocker socketをmountします。信頼できるホスト上で実行してください。`KAIRYU_BENCH_DOCKER_SOCKET`を指定すると、nested task用に別のDocker daemonを使用できます。LiveCodeBench Proを別daemonで動かす場合は、そのdaemon containerのrunnerから到達可能なIPを`KAIRYU_LIGHTCPVERIFIER_HOST`へ指定してください。LiveCodeBench Proは`KAIRYU_BENCH_LIVECODEBENCH_PRO_WORKERS`（既定1）で問題単位の並列数を指定し、完了した枠へ次の問題を即補充します。一時的な生成失敗は`KAIRYU_BENCH_LIVECODEBENCH_PRO_RETRIES`（既定3）まで再試行し、全試行失敗時はその問題を不正解として記録して全体を続行します。SWE-bench Proを専用daemonで実行する場合は、`KAIRYU_BENCH_CLEAN_TASK_IMAGES=1`で公式評価済みtaskのcontainerとimageを1問ごとに削除できます。`KAIRYU_BENCH_SWEBENCH_PRO_WORKERS`（既定1）で問題単位の並列数を指定でき、同時保持するtask imageもその件数以下に抑えます。全件実行は長時間・大容量・高コストになるため、疎通確認にはまず `--only ... --limit 1` を推奨します。
