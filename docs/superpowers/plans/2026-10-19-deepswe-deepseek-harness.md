# DeepSWE DeepSeek Harness（Minimal）対応 実装計画

**Goal:** DeepSWE v1.1 を、既定の mini-swe-agent に加えて DeepSeek 公式の agent harness（DeepSeek Harness、`dsh`）でも実行できるようにする。同じローカル API のモデルを両方の harness で測れるようにする。

**Architecture:** 既存の経路（`./kairyu-bench` → 共通 runner → DeepSWE adapter → 固定した公式 Pier → 公式 verifier → 共通の結果・比較レポート）はそのまま使う。Pier の installed agent を1つ追加し、agent だけを切り替える。dataset、instruction、trial の実行、collect hook による commit 済み patch の取得、独立 verifier、採点規則は共通にする。

**Tech Stack:** 既存の POSIX shell、Python 3.12、固定 Pier（`0c802fc`）、`deepseek-harness-sdk==0.2.1rc2` と native runtime、標準 `unittest`、Ruff、mypy。

**Spec:** ユーザー依頼「deep sweのベンチマーク実行でdeep seek hernesでも動作できるように対応してprを作ってね その計画」と、`AGENTS.md`。

## 決定事項（ユーザー回答）

| 項目 | 決定 |
| --- | --- |
| agent の選択 | 環境変数 `KAIRYU_BENCH_DEEPSWE_AGENT=mini-swe-agent\|deepseek-harness`。既定は mini-swe-agent で、既存の実行は変わらない |
| dsh の構成 | 同梱の `sdk-minimal` profile だけを使う。DeepSeek が公開 coding benchmark に使っている Minimal mode |
| API 接続 | dsh 公式の `deepseek-official` provider を使い、Anthropic Messages 互換の `POST <endpoint>/v1/messages`（SSE）で接続する。Claude Code と同じ endpoint |

## 調査で確認した根拠

| 対象 | 確認した事項 | 一次資料 |
| --- | --- | --- |
| dsh の配布 | PyPI の `deepseek-harness-sdk` / `deepseek-harness-runtime-bin` `0.2.1rc2` は tag `dsh-v0.2.1-rc.2`（`687f282e`）に対応する。runtime は `manylinux_2_28_x86_64` の単一実行ファイル（Node 24 を同梱）で、実行時に Node もネットワーク取得も要らない | [Python SDK guide](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/docs/user/guide/python-sdk.md) |
| benchmark 用の推奨 | SDK の minimal 構成を、独立した workspace と session で実行する | [BENCHMARK.md](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/BENCHMARK.md) |
| sdk-minimal | tool は永続 bash・read_image・working_directory。`session-log-deepseek` と `plugin-package-inventory-deepseek` は既定で有効で、どちらも `enabled: false` で止められる | [bundle](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/bundle/sdk-minimal/cordis.patch.yml) |
| 通信方式 | `<base>/v1/messages` に SSE で送る。`thinking` と `output_config` を常に送る | [wire extensions](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/docs/deepseek-llm-api-wire-extensions.md) |
| proxy | `HTTP(S)_PROXY` の URL に含めた認証情報を使い、loopback 宛てだけ直接接続する | [network proxy](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/docs/user/guide/network-proxy.md) |
| compaction | threshold=floor(min(0.8W, W−O−65536))、retain=floor(0.16(W−O))。retain≥threshold なら自動 compaction が無効になる | [compaction-basic](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/compaction/compaction-basic/src/config.ts) |
| Pier | agent の install は image build 時に行う。実行時の通信は、許可した host だけを通す認証付き squid proxy を経由する。採点するのは task の collect hook が取った `git diff base HEAD`。雛形は Antigravity SDK agent | [Pier 0c802fc](https://github.com/datacurve-ai/pier/tree/0c802fc067a425345b24d1c69411aa98acf61a1d/src/pier/agents/installed) |
| leaderboard | 公開値はすべて Pier の mini-swe-agent によるもの | [deep-swe README](https://github.com/datacurve-ai/deep-swe/blob/0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea/README.md) |

## 設計・実行契約

### 選択と実行条件（`deepswe.py`）

**mini-swe-agent の実行条件**
- key は増やさない。
- `adapter_sha256` だけは `deepswe_runtime.py` の変更で値が変わる。PR #29・#30 と同じ扱い。

**dsh の実行条件**：次を記録する。
- agent・版・source の revision・profile・provider・transport
- lock・runtime adapter・runner・patch の sha256
- 無効にした profile 行
- uv の版、context window、`max_tokens`、effort

**dsh の設定の検証**
- `KAIRYU_BENCH_DEEPSWE_CONTEXT_WINDOW` は必須。dsh の compaction 検査と同じ式で、事前にエラーにする。
- `KAIRYU_BENCH_DEEPSWE_MAX_TOKENS` の既定は 49152（SDK guide の値）。
- effort に指定できるのは `off/low/high/max` だけ。未指定は `agent_default`（dsh の既定値は high）。
- API timeout の変数は dsh ではエラーにする。
- dsh 専用の変数を mini-swe-agent で指定してもエラーにする。

### Pier agent（`deepswe_dsh.py`）と container 内 runner（`deepswe_dsh_runner.py`）

**install（root）**
- sha256 で検証した uv（Pier の Antigravity installer と同じ版）を入れる。
- `/installed-agent/python` に Python 3.12 を置き、venv を作る。
- lock から `--require-hashes --no-deps --only-binary :all:` で導入する。
- runtime と ripgrep sidecar に実行権限を付ける。
- 空の runtime cwd を作る（task repo の `.env` を読ませないため）。

**install（agent user）**
- `dsh --version` で runtime を実際に起動し、glibc・実行権限・起動を確認する。

**setup**：runner と patch を upload する。

**run**
- instruction は加工せず、runner に渡す。
- API の URL、key、model、`max_tokens`、`DSH_CONTEXT_WINDOW`、effort は環境変数で渡す。
- `DSH_TELEMETRY_MODE=DISABLED`。
- 許可する host は API の host だけ（mini と共通の関数を使う）。

**timeout**
- mini と同じで、Pier の `AgentTimeoutError` は 0 点。
- 強制 kill は入れない（score が変わらず、失敗要因が増えるだけのため）。

**runner**
- `DeepSeekHarness(profile="sdk-minimal", provider="deepseek-official", cwd=/app, runtime_cwd, dsh_home=/logs/agent/dsh-home, patches=(patch,))` で 1 回だけ実行する。
- event は届いた順に `dsh-events.jsonl` へ追記し、最後に `dsh-result.json` を書く。
- git identity の設定も自動 commit もしない。

**終了状態の扱い**

| 終了状態 | 扱い |
| --- | --- |
| `completed`・`max-tokens` | 採点する |
| `ContextWindowExceededError` | 0 点 |
| `provider:RATE_LIMIT/SERVER/TRANSPORT/TIMEOUT/EMPTY_RESPONSE` | `TransientAgentError` として Pier が再試行する。それでも失敗すれば採点対象外 |
| その他の provider エラー・runtime 異常・結果の欠落 | 採点対象外 |

### 集計と結果

- `selected.json` に `agent` を記録する。
- 集計は agent ごとに次を照合する：name / provider（`openai` または `deepseek-official`）と、exit status の key。
- agent キーのない旧 `selected.json` は mini-swe-agent として扱う。
- normalized の `agent`、`context.agent`、report の Agent 列には実際の agent を入れる。
- agent が違う run 同士では delta を出さない（既存の比較ルールのまま）。

## 検証

**単体テスト**（Docker や有料 model は使わない）
- settings・conditions・採点・旧 selection の互換・normalizer・runner の context
- Pier を使うもの：job config・install spec・allowlist・setup・run の環境変数、実 Pier queue での記録

**実 dsh runtime を使う wire テスト**（`tests/test_deepswe_dsh.py`、CI では別 venv）
- 構成：`127.0.0.1` に basic 認証付きの偽 proxy を置き、Anthropic SSE を返させる。API の URL は、解決できない `host.docker.internal` のまま。
- 確認すること：
  - 認証付きの absolute-form `POST /v1/messages` が届く
  - model ID が変わっていない
  - `max_tokens`、`output_config`、tool 3 種
  - DeepSeek 専用の field を送らない（patch を空にすると送ることも対照として確認する）
  - bash の commit が成功する
  - `completed` になる

**実行結果（2026-10-19、この branch）**
- `python -m unittest discover -s tests`：212 tests OK（skip 24）
- locked Pier 環境の `test_deepswe*.py`：38 tests OK
- `KAIRYU_BENCH_DSH_PYTHON` を指定した `test_deepswe_dsh.py`：10 tests OK
- ruff、mypy、`update_model_comparison.py --check`、`sh -n`：OK
- 非 editable の wheel を導入した状態で dsh の conditions を解決できること：OK

**未実施**
- 実 Docker と実 API での smoke（`--limit 1`）。
- 実施時は `AGENTS.md` に従い、privileged runner・専用 DinD・socket の共有について承認を得る。
- 確認項目：
  - dsh 0.2.1rc2 の導入
  - proxy 経由で API に到達すること
  - commit 済み patch と、別 verifier の reward
  - normalized の agent が `deepseek-harness` であること
  - Kairyu の `/v1/messages` が dsh のリクエスト（署名なしの thinking の再送、`output_config`）を受け付けること

## リスク

- Kairyu `/v1/messages` との互換は smoke で確認する。不足があれば Kairyu 側の issue として扱い、この変更では回避策を入れない。
- dsh は pre-release なので、版と hash を固定する。更新は lock の再生成と CI の再実行を伴う別の変更で行う。
- agent image は task ごとに約 330MB 増える。`KAIRYU_BENCH_CLEAN_TASK_IMAGES=1` で削除する。
