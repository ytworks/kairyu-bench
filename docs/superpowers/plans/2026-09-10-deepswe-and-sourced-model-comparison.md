# DeepSWE and Sourced Model Comparison Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task by task in the current task. Steps use checkbox syntax. Do not delegate without a subsequent user instruction authorizing agents.

**Goal:** DeepSWE を既存と同じ Docker ランナーから実行可能にし、実 API での全件評価と、出典・測定条件を追跡できる他モデル比較まで完成させる。

**Architecture:** `./kairyu-bench` → 共通 runner → DeepSWE adapter → 固定した公式 Pier／mini-swe-agent → 公式 verifier → 共通の結果・比較レポートという既存構成を踏襲する。公開スコアは実測結果とは別の参照カタログに保持し、同じカタログから比較ドキュメントと実行レポートの参考表を生成する。

**Tech Stack:** 既存の POSIX shell、Python 3.12 runner、Docker Compose、公式 DeepSWE v1.1、Pier、mini-swe-agent、標準 `unittest`、Ruff、mypy。

**Spec:** このファイルの「設計・実行契約」、リポジトリの `AGENTS.md`、ユーザー依頼「DeepSWE benchも動作するようにして、他モデルの結果も出典付きでちゃんとまとめるところまでやり切る」「動かし方は他のベンチマークを踏襲」。

## 状態と対象範囲

- 2026-09-10 作成。計画段階であり、ベンチマーク実装・Docker 起動・本番 API への評価リクエストはまだ行っていない。
- 調査時の本体 HEAD は `9252292`。既存は12ベンチで、DeepSWE の登録・adapter・normalizer はない。
- 対象は Datacurve の **DeepSWE v1.1**。Agentica の同名モデル DeepSWE-Preview とは別である。
- 比較表の範囲は、回答があるまで **DeepSWE＋既存12ベンチの公開値・出典の全体再確認**を採用する。既存18モデル列を最低限の確認対象とし、DeepSWE は公式の全モデル・全 reasoning effort を収録する。既存12ベンチの実測を全件やり直すことは、この計画の必須作業には含めない。
- 他モデルについては公開結果の収集・検証を行う。他社の有料 API を新たに実行することは含めない。
- 実装後は1問の疎通確認で終わらせず、113問の全件評価と最終比較レポートまで進める。比較用の最終測定は同じ条件で4反復、計452試行とする。

## Global Constraints

- 必須の実行引数は API URL のみ。モデルは既存の `/v1/models` と chat completion の疎通で自動検出する。
- 公式実行は `./kairyu-bench` を使う。ホストに Pier、Python 環境、Modal アカウントを要求しない。
- `kairyu` 本体を import・コピーしない。既存12ベンチの agent、既定値、採点条件を変更しない。
- API 契約は `/v1/chat/completions`。DeepSWE のためだけに `/v1/responses` を必須にしない。
- `--limit N` は固定した問題順の先頭N問。試行数の制限やランダムな部分集合へ読み替えない。
- 既定4並列、1〜16で指定可能。完了した枠へ直ちに次を補充する。
- cache と専用 Docker daemon の data root/socket は `/mnt/nvme/kairyu/` 配下。`results/` はリポジトリに保存し、同じ絶対パスで必要なコンテナに mount する。
- 現行 Kairyu API／Open WebUI の Docker daemon・image・container を掃除しない。
- API key／HF token を引数、設定ファイル、log、trajectory、report に保存しない。公開レポートの endpoint は fingerprint のみ。
- 公式の問題、prompt、verifier、reference solution、agent の問題解決ロジックを評価都合で改変しない。
- infrastructure error を正解・不正解へ推測変換しない。不足結果を落として `completed` にしない。
- privileged runner、専用 privileged Docker-in-Docker、socket mount の利用は、実行時に `AGENTS.md` に従って明示的承認を確認する。計画への同意をこれらの承認とみなさない。

## 調査で確認済みの根拠

| 対象 | 確認事項 | 一次資料 |
| --- | --- | --- |
| DeepSWE | v1.1、113 task、agent と別の verifier environment、commit 済み patch を採点 | [固定 README](https://github.com/datacurve-ai/deep-swe/blob/0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea/README.md) |
| task source | `0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea`。Git tree 内の `tasks/*/task.toml` を数えて113件 | [固定 tree](https://github.com/datacurve-ai/deep-swe/tree/0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea/tasks) |
| Pier | `0c802fc067a425345b24d1c69411aa98acf61a1d`、package version 0.3.1、Python >=3.12 | [pyproject](https://github.com/datacurve-ai/pier/blob/0c802fc067a425345b24d1c69411aa98acf61a1d/pyproject.toml) |
| API transport | mini-swe-agent の `model_class=auto` は `openai/` を `litellm_response` にする。明示設定が必要 | [agent 実装](https://github.com/datacurve-ai/pier/blob/0c802fc067a425345b24d1c69411aa98acf61a1d/src/pier/agents/installed/mini_swe_agent.py) |
| 実行条件 | Pier の job に task 配列、並列数、反復数、retry policy を指定可能 | [JobConfig](https://github.com/datacurve-ai/pier/blob/0c802fc067a425345b24d1c69411aa98acf61a1d/src/pier/models/job/config.py) |
| 公開比較 | 2026-09-03 更新の JSON に28モデル・70設定。全行 `n_runs=4`。成功数、採点対象数、pass@1、pass@4、CI を保持 | [公式 leaderboard JSON](https://deepswe.datacurve.ai/artifacts/v1.1/leaderboard-live.json) |
| 採点ポリシー | pass@1 は採点対象試行の成功率。context 超過・agent timeout は失敗、provider/verifier/network error は除外 | [同 JSON の unit 定義](https://deepswe.datacurve.ai/artifacts/v1.1/leaderboard-live.json) |
| v1との違い | v1.1 の検証環境と結果は v1 から区別する | [v1.1 技術説明](https://deepswe.datacurve.ai/blog/deepswe-v1-1) |

調査時の例: GPT-6 Astra xhigh は335/452=74.12%、Claude Opus 5 max は327/444=73.65%。後者の分母を452に置き換えたり、未採点8試行を0点にしたりしない。これらは確認用の例であり、実装時に取得日付き snapshot を作り、最終成果物の数値はその snapshot に統一する。

## 設計・実行契約

### 1. 採用する構成

**推奨・採用:** 専用 `deepswe` adapter から公式 Pier を利用する。Terminal-Bench と同じ shell／venv／Docker／成果物構造を使い、Pier 固有の API 接続・network・結果読取りだけを薄く補う。

検討した代替案:

- 既存 Harbor adapter へ task path だけ渡す案は変更量が少ないが、公式 Pier の air-gapped agent 接続、v1.1 の別 verifier、結果条件との差が残るため採用しない。
- 独自の patch generator／採点ループを作る案は API 対応を制御しやすいが、公式 agent と評価条件を再実装する必要があるため採用しない。

新しい公開名は **`deepswe`** に一本化し、13番目として追加する。既存 benchmark の相対順序は保持する。`--harbor-agent` は引き続き Terminal-Bench 専用で、DeepSWE の agent は `mini-swe-agent` 固定とする。

### 2. 利用方法

```sh
# 既存と同じ使い方。既定は1 taskあたり1試行。
./kairyu-bench run http://host.docker.internal:8003/v1 \
  --only deepswe --limit 1 --run-id deepswe-smoke-20260910

# 全113問、1試行ずつ
./kairyu-bench run http://host.docker.internal:8003/v1 \
  --only deepswe --run-id deepswe-full-20260910

# 公開比較の最終測定: 113問×4反復、同時実行は合計4まで
KAIRYU_BENCH_DEEPSWE_ATTEMPTS=4 \
  ./kairyu-bench run http://host.docker.internal:8003/v1 \
  --only deepswe --run-id deepswe-comparison-20260910

./kairyu-bench compare results/deepswe-run-a results/deepswe-run-b
```

上記は adapter 実装後の利用例。実機ではさらに後述の NVMe cache と専用 socket を指定し、使用済み run ID は避ける。

| 環境変数 | 既定 | 契約 |
| --- | --- | --- |
| `KAIRYU_BENCH_DEEPSWE_WORKERS` | `4` | 整数1〜16、全反復を合わせた並列上限 |
| `KAIRYU_BENCH_DEEPSWE_ATTEMPTS` | `1` | `1` または `4`。最終比較は4 |
| `KAIRYU_BENCH_DEEPSWE_RETRIES` | `3` | 整数0〜16、明示した一時的 infrastructure error の再試行上限 |
| `KAIRYU_BENCH_DEEPSWE_REASONING_EFFORT` | 未設定 | 指定時だけ送信。未設定は `server_default` と結果へ記録し、暗黙に high 等へ読み替えない |

timeout は固定 task の定義を尊重する。agent setup の倍率は4に固定して記録する。モデル実行時間の倍率は1のままとし、setup の延長と混同しない。API が明示指定した reasoning effort を拒否した場合は、黙って別設定に落とさず理由を返す。

### 3. 再現性と接続

- manifest の `source` は Pier の固定 commit、`dataset` は DeepSWE の固定 commit と v1.1、`generator` は採用する mini-swe-agent の固定 source/package を表す。Pier の依存 lock、agent version、agent 設定の hash、task image digest も記録する。
- mini-swe-agent の正確な version は Task 1 の公式 trajectory/config 確認で決定する。その版が特定不能な場合、現行の互換版を固定し、公開値と agent version が一致したとは主張しない。浮動 `latest` による完了は認めない。
- 上流の `uv.lock` に基づく Pier 専用環境を作る。既存 Harbor の環境を上書きしない。動的に更新される LiteLLM の model map も hash／取得元を記録し、コスト不明の自動検出モデルを0円や既知モデルへ偽装しない。
- `model_class` は固定した mini-swe-agent の Chat Completions 実装を明示する。送信された URL と request body をテストし、`openai/` は transport prefix として付け、API payload の model ID は検出した文字列を完全に維持する。
- `OPENAI_BASE_URL`／`OPENAI_API_BASE`／`OPENAI_API_KEY` は既存 runner の環境を使う。token を含む `--agent-env KEY=value` を組み立てない。
- `host.docker.internal` は runner だけでなく、Pier の API egress proxy から解決・到達できることを確認する。必要な Docker 差分は専用 environment subclass の compose override で与え、task の network 制限を解除しない。専用 daemon での gateway は通常の host daemon と同じと決めつけない。
- agent には API 接続だけを許可する。verifier は別環境・network 無効を維持する。source checkout、`tests/`、`solution/`、Docker socket、HF token を agent へ mount・伝播しない。

### 4. 選択・試行・採点

1. 固定 checkout の有効な `tasks/*/task.toml` を列挙し、task directory 名の昇順をこのadapterの canonical order として固定する。上流のdirectory列挙順に依存せず、順序決定後に `--limit` を適用する。113件の全体件数検証は、この選択関数の前のdataset検証で行う。
2. `selected.json` に dataset revision と問題IDの順序を保存する。113件との不一致、重複、空集合は実行前に失敗させる。
3. `trial-plan.json` に各 `(task_id, repeat_index)` の slot と Pier の job/trial ID の対応を保存する。retry は同じ slot の再実行として履歴を残す。通常の不正解を再試行して良い方を採用しない。
4. 4反復は113問の同じ選択順に対する repeat 0〜3 として扱う。各反復の集計を復元できるようにする。Pier の実行順やランダムな trial suffix から反復番号を推測しない。
5. 主指標は `verifier_result.rewards.reward` の binary 0/1 と、公式に明記された agent timeout/context failure の分類から求める。補助の test pass fraction を主指標に代入しない。
6. `pass@1 = 100 * passed_attempts / scored_attempts`。4反復時の `pass@4` は少なくとも1回成功した task の割合で、主指標にはしない。CI は反復単位の率から公式方式を採用する。
7. `counts.requested` は選択 task 数、`counts.evaluated` は必要な反復すべてに採点対象の結果がある task 数。試行側は `requested_attempts`、`scored_attempts`、`passed_attempts`、`excluded_attempts`、`retry_count` を metrics に保存する。
8. 一部の slot に infrastructure error が残ると `partial`。採点対象が0なら主指標を作らず `failed`。環境の必須機能が利用不能なら `unsupported`。未知の error type は推測せず原因を保存する。
9. 全113問×4反復の452 slot が採点済みのときだけ最終比較実測を `completed` とする。公式の除外ポリシーに基づく途中スコアは表示できるが、未解消の除外を隠して全件完了としない。

### 5. 成果物と比較条件

```text
results/<run-id>/
  run.json
  context/deepswe.json
  logs/deepswe.log
  raw/deepswe/selected.json
  raw/deepswe/trial-plan.json
  raw/deepswe/provenance.json
  raw/deepswe/progress.json
  raw/deepswe/jobs/...          # 公式 result/config/lock/trajectory/verifier/patch
  raw/deepswe/summary.json
  normalized/deepswe.json
  report.json
  report.md
```

`conditions` に benchmark version、Pier/agent revision、agent config hash、transport、reasoning effort、反復数、timeout/setup、retry/error policy、image digest の集合を保存し、既存 `_compatibility()` の条件一致判定へ渡す。task順を保持して正規化し、並列完了順の違いは比較を妨げない。worker数・host環境も provenance に残す。

公開参考値は `BenchmarkResult` として偽装しない。`report.json` の `public_references` と `report.md` の「公開参考値」に載せ、出典・取得日・設定・分母を添える。自前 Docker と公開 Modal、transport、agent version、provider、reasoning effort 等の差が残る場合は理由を表示し、自動 delta や同条件順位を作らない。未完了の実測と公開値も同条件の勝敗に使わない。

## 変更ファイルと責務

| ファイル | 変更・責務 |
| --- | --- |
| `src/kairyu_bench/benchmarks.py`、`data/benchmarks.json` | `deepswe`、固定 source/dataset/generator、採点法の登録 |
| `adapters/deepswe/run.sh`、`scripts/harnesses/deepswe.sh` | 既存と同形式の dispatch、source checkout、専用環境、終了時の正規化 |
| `src/kairyu_bench/deepswe.py` | 設定検証、選択・trial plan、公式成果物の集計。Pier への依存は import しない |
| `src/kairyu_bench/deepswe_runtime.py` | Pier 環境での公式 Job 呼出し、hook、接続用 environment 差分、retry・cleanup の記録 |
| `deepswe_runtime.py`内のcompose/proxy調整 | 接続に必要な compose 設定。公式の隔離設定を保持 |
| `src/kairyu_bench/runner.py` | DeepSWE の固定 agent と実効 conditions の context 伝播 |
| `src/kairyu_bench/official.py` | DeepSWE summary の厳密な読み取りと既存 result schema への変換 |
| `kairyu-bench` | 新しい環境変数の container への伝播 |
| `src/kairyu_bench/public_results.py`、`data/model-references.json` | 公開値・出典・条件のカタログ検証と読み取り |
| `scripts/update_model_comparison.py` | 保存済みカタログから比較 Markdown を決定的に生成／`--check` |
| `src/kairyu_bench/reporting.py` | 試行側の件数と公開参考値の表示。既存の実測 delta 判定を保持 |
| `docs/model-comparison.md`、`docs/model-comparison-details.md` | 全13ベンチの一覧、DeepSWE全設定、条件差・出典の詳細 |
| `docs/sources/` | DeepSWE公開集計 JSON と取得URL・日時・SHA-256。第三者記事本文の丸ごと保存はしない |
| `README.md`、`AGENTS.md`、`.github/workflows/ci.yml` | 13ベンチへの更新、実行手順、監視・完了条件、軽量CI |
| `tests/test_deepswe.py`、`tests/test_deepswe_harness.py`、`tests/test_public_results.py` | 新しい選択・API接続・採点・出典の契約テスト |
| 既存 `tests/test_manifest.py`、`test_runner.py`、`test_official_normalizers.py`、`test_reporting.py`、`test_host_entrypoint.py` | 登録・identity・表示・env伝播の回帰検証 |

## Task 1: 公式条件と依存関係を固定する

**Consumes:** 上記の公式 repository と公開結果資料。

**Produces:** manifest に入れる固定値、選択順、mini-swe-agent の実効設定、公式 artifact schema の小さな fixture。

- [ ] 固定した2 repository を cache 内に取得し、`git rev-parse HEAD` が上表と一致すること、113件の task があることを確認する。実装途中で勝手に main を追わない。
- [ ] 公開 trajectory/config と Pier の install spec を読み、mini-swe-agent version、Chat Completions の model class 名、prompt/config、retry の error 分類を固定する。特定できない公開条件は `unknown` と明記し、比較可にしない。
- [ ] Pier の固定 lock を使った依存導入を実装する。`ensure_venv` の既存キャッシュ契約を使い、cache key に依存 lock hash を含める。task側の mini-swe-agent と関連 package も固定する。
- [ ] 成功・不正解・agent timeout・context failure・provider error・verifier error の公式出力構造を確認し、テスト用 fixture は小さな合成データとして作る。promptやreference solutionの内容をfixtureへ転記しない。
- [ ] task画像の architecture、disk／memory／CPU、agent・verifierの同時生存数を確認し、最初の smoke の資源上限と Linux 実行環境を決める。

**完了条件:** task/source/agent/dependencies/画像の固定方針と、公式との既知の条件差を機械可読に記録できる。追加の前提調査を実装者へ丸投げした状態で次へ進まない。

## Task 2: 共通 runner に DeepSWE を登録する

**Consumes:** Task 1 の固定値。

**Produces:** 次の公開 interface と adapter context。

| interface | 戻り値 |
| --- | --- |
| `resolve_deepswe_conditions(entry: dict, env: dict[str, str])` | `dict`: 固定source条件と検証済みの実効設定 |
| `select_deepswe_tasks(task_root: Path, limit: int \| None)` | `list[str]`: directory名昇順の選択task ID |
| `write_trial_plan(raw_dir: Path, task_ids: list[str], attempts: int)` | `Path`: atomicに保存した `trial-plan.json` |

- [ ] `test_manifest.py`／`test_cli.py` に `deepswe` が `list`／`--only`／既定全件に含まれ、既存順序を維持するテストを追加し、未実装で失敗することを確認する。
- [ ] `test_deepswe.py` に workers の0・17・小数・空文字、attempts の2、retries の負数を拒否するテストを追加する。既定の `(workers, attempts, retries) == (4, 1, 3)` も確認する。
- [ ] `benchmarks.py` と manifest へ登録し、wrapper で環境変数を渡す。DeepSWE 用に新しい必須 CLI 引数は追加しない。
- [ ] `runner.py` で `agent=mini-swe-agent` と実効 conditions を作り、identity 検証と失敗結果にも同じ条件を残す。Harbor agent 設定が DeepSWE に流れ込まないテストを追加する。
- [ ] task 選択を確定してから限度を適用し、selected/trial plan を atomic write する。並列実行の前に保存を完了させる。
- [ ] 対象テストを実行し、登録・設定の変更単位をレビューする。

選択順とlimitの契約を確認する具体例:

```python
import tempfile
import unittest
from pathlib import Path

from kairyu_bench.deepswe import select_deepswe_tasks


class DeepSWESelectionTest(unittest.TestCase):
    def test_limit_is_applied_after_stable_task_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("task-z", "task-a", "task-m"):
                task = root / name
                (task / "tests").mkdir(parents=True)
                (task / "environment").mkdir()
                (task / "task.toml").write_text(
                    'schema_version = "1.3"\n'
                    f'[task]\nname = "datacurve/{name}"\n',
                    encoding="utf-8",
                )
                (task / "instruction.md").write_text("Synthetic fixture\n")
                (task / "tests/test.sh").write_text("#!/bin/sh\nexit 0\n")
                (task / "environment/Dockerfile").write_text("FROM scratch\n")
            self.assertEqual(select_deepswe_tasks(root, 2), ["task-a", "task-m"])
            self.assertEqual(
                select_deepswe_tasks(root, None), ["task-a", "task-m", "task-z"]
            )
```

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_manifest.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_deepswe.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_runner.py' -v
```

**完了条件:** `--dry-run --only deepswe` と mock adapter で、正しい model ID、選択、条件を含む標準成果物が生成される。

## Task 3: 公式 Pier の実行・接続・監視をつなぐ

**Consumes:** context、selected/trial plan、固定 Pier 環境。

**Produces:** `raw/deepswe/jobs/` 以下の公式成果物と `progress.json`。公開 benchmark の問題解決を行うのは公式 agent のまま。

- [ ] adapter は既存と同じ最小 shell にする。

```sh
#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
exec "$ROOT/scripts/harnesses/deepswe.sh"
```

- [ ] runtime は次の形の公式設定を、確定済み task 配列と実効値で構成する。`JobConfig.model_validate()` で schema を検証してから実行する。
- [ ] runtime はPier専用venvのPythonで起動し、`PYTHONPATH=/app/src`を指定してこのrepositoryの補助moduleを参照する。runner全体のPython依存へPierを追加しない。

```python
job_config = {
    "job_name": job_name,
    "jobs_dir": str(raw_dir / "jobs"),
    "n_attempts": 1,  # repeatごとに対応を固定。4反復でも全体並列上限を共有する
    "n_concurrent_trials": workers,
    "agent_setup_timeout_multiplier": 4,
    "agents": [{
        "name": "mini-swe-agent",
        "model_name": "openai/" + model_id,
        "kwargs": agent_kwargs,  # Task 1で固定したversion、chat model_class、config
    }],
    "tasks": [{"path": str(task_root / task_id)} for task_id in selected_ids],
    "environment": environment_config,
    "retry": retry_config,
}
```

- [ ] repeat ごとに公式 Job を順に実行し、各 Job は4並列の pool を使用する。4つの Job をそれぞれ4並列で同時起動して16並列にしない。結果を repeat ごとに保存すれば CI と pass@4 の対応も確定する。
- [ ] Pier の `on_trial_started`／`on_verification_started`／`on_trial_ended` hook から、slot・stage・成功数・retry・時刻を含む進捗JSONと `DeepSWE completed` logを出す。retry履歴と最終結果を別に管理する。
- [ ] wrapper／harness は異常終了時も raw を残し、可能なら partial の正規化を行う。`set -e` で正規化前に抜けないよう return code を保持する。
- [ ] Chat Completions だけを提供する偽 API を使い、slash を含む model ID、base URL の `/v1`、認証有無を検証する。`/responses` 呼出しや本来のモデル名の切断を失敗にする。
- [ ] agent と API egress proxy の到達性を container 内で確認する。接続補助が必要なら固定 Pier の DockerEnvironment を薄く拡張し、公式 task checksum と network 隔離が保たれることをテストする。
- [ ] fake secret を使って設定・lock・log・trajectory・例外メッセージへの漏出を検証する。上流の log 出力も確認し、必要な redaction は出力前に行う。
- [ ] 専用 daemon 内でのみ cleanup する。公式が cleanup 失敗を warning にする場合も hook 後の実体確認で検出する。同じ image を使う別試行や verifier が生きている間は削除せず、最後の利用終了後に削除する。

**完了条件:** 実行開始・採点・終了・一時エラー・cleanup が追跡でき、Docker内の API 接続と公式隔離を両立する。

## Task 4: 採点・不完全実行・比較の正確さを保証する

**Consumes:** selected/trial plan と公式 trial results。

**Produces:** `summary.json`、標準 `normalized/deepswe.json`、試行数付き report。

interface は `summarize_deepswe(raw_dir: Path) -> dict`。返すsummaryは `problem_ids`、`attempts_per_task`、`evaluated_tasks`、`requested_attempts`、`scored_attempts`、`passed_attempts`、`excluded_attempts`、`pass_at_1_percent`、`pass_at_4_percent`、`repeat_scores`、`errors`、`agent` を持つ。

- [ ] 「2 task×4反復、A=[1,0,0,0]、B=[0,1,0,0]」のfixtureを作り、主指標25%、pass@4=100%、task evaluated=2、scored attempts=8を検証する。
- [ ] 同fixtureで補助 reward を0.9にしてもbinary主指標が変わらないことを確認する。reward=0.5、NaN、文字列、未知task、重複slot、agent/model不一致は拒否する。
- [ ] 結果ファイルがないslot、未起動task、途中で切れたJSON、余分なretryディレクトリ、agent timeoutとverifier timeoutの相違を検証する。ファイルの数だけで全件完了にしない。
- [ ] `summarize_deepswe()` が selected の順序を保つようにし、`official.py` の専用normalizerへ登録する。Terminal-Bench用 `_harbor()` は転用しない。
- [ ] 部分実行の結果を既存 schema に収める。採点対象0のときは数値を作らない。元ログと公式rewardへのパスを保持する。
- [ ] `reporting.py` に task数と試行数を併記し、4反復なのに113件だけ採点したように見せない。条件一致時のみ既存 compare のdeltaを出す。
- [ ] compare のテストに「1反復 vs 4反復」「v1 vs v1.1」「reasoning effort差」「agent revision差」「選択は同じ・完了順だけ違う」を追加する。

**完了条件:** 独立したfixture計算とnormalizer/reportが一致し、欠測・再試行・補助点によりスコアや完了数が水増しされない。

## Task 5: 出典付き公開結果カタログと比較資料を整える

**Consumes:** 公式 DeepSWE JSON、既存 `docs/model-comparison.md` の全数値・出典。

**Produces:** 検証済み `model-references.json`、snapshot、一覧・詳細表、実測reportの公開参考値。

- [ ] DeepSWEの元JSONを取得日時とSHA-256付きで保存する。`generated_at`、version、各model/config/effort、分子・分母、4反復、CI方式を保持する。丸めは表示時だけ行う。
- [ ] 既存18モデル×12ベンチのセルを順に一次資料まで追う。数字がないセルも「未公表」「別指標のみ」「出典確認不能」のいずれかを記録し、単なる未調査を未公表扱いしない。
- [ ] 一次資料に値がなく公式第三者leaderboardだけにある場合は、提供元と第三者測定であることを記録する。検索snippet、他モデルのカードに転載された値、LLMの記憶だけを確定出典にしない。
- [ ] model version/variant、benchmark version/split、agent、tools有無、reasoning effort、context長、試行数、評価日、単位、分母の違いを各行に保持する。根拠のないalias統合をしない。
- [ ] カタログは次の形を基準にし、未確認値は `null` と理由を持つ。モデル名と設定を複合キーにする。

```json
{
  "benchmark": "deepswe",
  "benchmark_version": "1.1",
  "model_id": "gpt-6-astra",
  "agent": "mini-swe-agent",
  "reasoning_effort": "xhigh",
  "metric": "pass@1",
  "unit": "percent",
  "value": 74.11504424778761,
  "numerator": 335,
  "denominator": 452,
  "task_count": 113,
  "repeats": 4,
  "source_id": "deepswe-v1.1-20260910",
  "source_locator": "rows[config=mini_swe_agent_gpt_6_astra_xhigh]",
  "verification_status": "verified",
  "comparison_class": "reference_only"
}
```

- [ ] source定義に URL、発行主体、一次/第三者、公開/更新日、取得日、該当表・ページ・JSON locatorを持たせる。画像/PDFは該当箇所を目視照合し、複数モデル統合列を個別モデルへ誤帰属させない。
- [ ] 一覧は現行の比較表を更新し、DeepSWE行を追加する。設定別の詳細表は28モデル・70設定を全収録し、一覧でモデルごとの最高公開設定を採用する場合は選択ルールとeffortを明記する。調査時点以降に行数が増減した場合はsnapshotの実数に従う。
- [ ] HLE tools有無、MRCR context、Terminal-Bench version、LongBench代替、SWE-bench split等の条件差を維持する。異なる単位・別ベンチの値を空欄補完に使わない。
- [ ] `public_results.py` はsource参照切れ、数値範囲、分母0、重複行、不明値の0点化を検証する。生成器に `--check` を実装し、カタログとMarkdownの差分をCIで検出する。
- [ ] `write_score_report()` はnetworkを使わず、同梱カタログから実行したベンチの公開参考値を添付する。参照値をmacro averageへ混ぜない。既存reportを読む処理との互換性を保持する。

**完了条件:** 比較表の全数値から根拠の箇所へ辿れる。採用不能の旧値は理由付きで修正・除去され、DeepSWEの全設定と条件差が再生成可能な形で残る。

## Task 6: CI・実機smoke・全件評価を完走させる

**Consumes:** Tasks 1〜5の実装とテスト。

**Produces:** 実機で検証した正式な `results/<run-id>/` と運用手順。

- [ ] repository CI相当の確認を通す。

```sh
python -m unittest discover -s tests -v
ruff check .
mypy --ignore-missing-imports src
python scripts/update_model_comparison.py --check
sh -n kairyu-bench
sh -n scripts/harnesses/deepswe.sh
sh -n adapters/deepswe/run.sh
```

- [ ] Docker build とcontainer内 `list`／DeepSWE dry-runを確認する。CIでは実modelの高コスト全件評価を自動起動しない。既存12adapterの回帰テストも通す。
- [ ] Linux実行ホスト、APIの `/v1/models`、chat接続、4並列処理、root/NVMe空き、imageのarchitectureを確認する。task定義のCPU/memoryと同時生存するagent/verifier/proxyを足して資源計画を作る。
- [ ] 専用daemon作成用の具体的なcontainer名、data root、socket、bind mount、予定コマンドを用意してから、`AGENTS.md` のprivileged runner・privileged DinD・socket mountの承認状況を確認する。未承認ならその3能力だけについて実行直前に承認を得る。
- [ ] `--limit 1`／1反復で、生成→commit済みpatchの収集→別環境の公式採点→normalization→reportまで確認する。正解率そのものをsmoke成功条件にせず、公式が不正解を返した場合も正しく保存・集計できれば通す。
- [ ] `--limit 4`で4並列、slot即補充、重複採点なし、API接続、別verifier、cleanupを確認する。`--limit 1`／4反復も確認してtrial対応とCI集計を検証する。
- [ ] 問題があれば原因を修正し、影響したテストとsmokeを再実行する。評価条件を変えた後は古い実測と混ぜない。
- [ ] 最終比較用に unique run ID と専用cache/socketを確定し、`--limit`なしの113問×4反復を開始する。下記の変数は実行直前に空きパスへ確定してから使う。

```bash
/bin/bash -ic '
set -eu
export KAIRYU_BENCH_CACHE_DIR=/mnt/nvme/kairyu/bench-cache/deepswe-comparison-20260910
export KAIRYU_BENCH_DOCKER_SOCKET=/mnt/nvme/kairyu/docker/deepswe-comparison-20260910/run/docker.sock
export KAIRYU_BENCH_CLEAN_TASK_IMAGES=1
export KAIRYU_BENCH_DEEPSWE_WORKERS=4
export KAIRYU_BENCH_DEEPSWE_ATTEMPTS=4
export KAIRYU_BENCH_DEEPSWE_RETRIES=3
exec ./kairyu-bench run http://host.docker.internal:8003/v1 \
  --only deepswe --run-id deepswe-comparison-20260910
'
```

- [ ] 起動直後に `run.json=running`、APIと同じmodel ID、selected113件、trial-plan452slot、最初の公式trial結果を確認する。
- [ ] 監視中は少なくとも毎分、採点済み試行数/452、全反復完了task数/113、成功数、実行中task、elapsed、API/Docker状態、root/NVMe空きを報告する。retry・採点異常・cleanup失敗は即時報告する。健康な実行を遅さだけで停止しない。
- [ ] log上の完了数をslot単位の最終result数と突き合わせる。実行中の一時errorは同条件・同slotで最大3回retryする。上限後に除外errorが残ればpartialとして原因を直し、unique run IDで全件を再実行する。既存runnerは同じrun IDで再開できないため、未実装のresumeを前提にせず、旧runを保存する。別runの成功結果だけを寄せ集めない。

**完了条件:** 全113問・452採点対象slotが揃い、`run.json`／normalized／reportがすべてcompleted。公式rawと集計の分子・分母・スコアが一致する。

## Task 7: 最終比較・文書・証跡を仕上げる

- [ ] `README.md`を13ベンチへ更新し、通常・smoke・全件・4反復比較のコマンド、環境変数、出力、所要資源を記載する。
- [ ] `AGENTS.md`にDeepSWEの113件/452試行の定義、専用daemon、承認対象、毎分監視、完了判定、cleanupを追加する。
- [ ] 最終実測reportに対象model、固定revision、条件、成功/採点数、pass@1、pass@4、CI、除外数、公式rawへの参照、公開比較のsource snapshotを載せる。
- [ ] 公式公開値との差は、同条件が確認できない限り参考比較として説明する。4反復を揃えただけで、Modal/Docker・model transport等も一致したことにしない。
- [ ] 公開比較表の全数値・注記・出典・取得日を再確認し、生成器 `--check` と変更に対応するテストを通す。
- [ ] `git diff --check` と最終diffレビューを行い、結果ディレクトリにsecretがなく、必要なrawが欠けていないことを検証する。
- [ ] 完全なresultsを保存する。cache／専用daemonのdata rootを片付ける場合は、成功reportを再確認した後、exact pathと他run未使用を再確認してから対象だけを削除する。

## 最終受け入れチェック

- [ ] ユーザーが既存と同じwrapper・API URL・`--only deepswe`で実行できる。
- [ ] URLだけの既定全体実行・`list`には13ベンチが含まれる。
- [ ] 単体テスト、lint、型、shell、Docker、API・nested verifierの実機smokeが通る。
- [ ] 最終実測は113問×4反復で完走し、成功数/452とtask数/113を区別して報告できる。
- [ ] published pass@1、pass@4、CI、除外試行を取り違えていない。
- [ ] 既存12ベンチの公開値も再確認され、確認不能・別条件は理由付きで扱われる。
- [ ] DeepSWEの全公開model/configの比較表があり、各値に直接出典と取得日がある。
- [ ] 実測report・公開比較・raw・revision・実行手順の場所が最終回答から辿れる。
- [ ] 未解消のAPI互換性、未採点slot、source不明の確定値、cleanup障害を残したまま「完了」としない。

完了報告は「実装した」だけで終えず、実行コマンド、実測pass@1と成功/採点数、113問/4反復の確認、検証結果、比較表と結果ディレクトリへのリンク、残る測定条件差を示す。


## 実装PR時点の状態（2026-09-10）

実装、source/agent依存固定、公式Pierジョブ接続、ネットワーク調整、並列/再試行/4反復、失敗証跡保存、正規化、レポート参照値分離、70設定の公開原本照合、README/運用手順/CIを追加した。

- [x] 公式Pierで113タスクの名前、独立verifier、通信制限、task imageの一意性を検証。
- [x] 実際のmini-swe-agent/LiteLLMから偽HTTP APIへのChat Completions通信を検証。
- [x] 公式PierのJob/TrialQueueで同時実行上限、完了枠補充、再試行、4反復の集計を検証（問題の実行のみfixture）。
- [x] 公開原本70設定の値、model ID、effort、分子/分母/CIを検査。Opus 4.8 / maxは111問・429試行のため別途明記。
- [x] 既存12行・19モデルの出典、取得日、欠測、条件差を構造化。最新mainのGPT-6 Astra列も保持。
- [x] 既存公開値102件中100件を2026-09-10に照合。3値を更新し、条件・旧値を保存。残る2件は旧確認日と理由を表・照合記録に明記。提供元による追加DeepSWE値も公式原本と分けて掲載。
- [ ] 実モデル＋Dockerのsmokeと113問×4反復を完走。Linux SSH接続先未確定・privileged実行の明示承認未取得。

実装検証と実モデルの採点結果を区別する。実測未実施の状態で計画全体や全件評価が完了したとは報告しない。具体的なNVMe/cache/socket/run IDと実行コマンドは`docs/deepswe.md`に保存。
