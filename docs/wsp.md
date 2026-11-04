# Weight-space programming ベンチマーク（`wsp`）

重みを直接編集したモデルが「何を得て何を失ったか」を、どの編集手法にも同じ物差しで測る共通ベンチマークです。
基準環境（Gemma 4 31B・8GPU）で **10分・30分・60分** に収まる3つのパターンがあり、どのパターンも14の能力すべてを測ります。

計画とその理由: [`docs/superpowers/plans/2026-11-04-wsp-benchmark.md`](superpowers/plans/2026-11-04-wsp-benchmark.md)

## 原則

1. **どのパターンも全14能力を測る。** パターンの違いは問題数（精度）と1問の最大長（深さ）。10分版 ⊂ 30分版 ⊂ 60分版 の入れ子で、1問の条件はどのパターンでも同じ。
2. **採点に LLM を使わない。** 正解照合・テスト実行・規則チェック・対数確率だけ（編集したモデル自身に採点させると採点も壊れるため）。
3. **元モデルとの比較が主役。** 同じ問題で元モデルと編集版を並べ、能力ごとの差・95% 区間・「正解→不正解」「不正解→正解」の数を出す。
4. **スコアは出力だけで決まる。** 時間切れで採点を変えない。1問ごとの出力上限トークンが固定で、上限到達は0点（到達率は別に報告）。
5. **層別の無作為抽出（固定シード）。** データセットの先頭 N 問は使わない。調整用（dev）と最終確認用（test）は同じ構成で重ならず、同じ問題の翻訳・別形式（MMLU-Pro と MMLU-ProX、GSM8K-Platinum と MGSM、MMMU-Pro の標準版と画像版）は必ず同じ組に入る。

## 基準環境

| 項目 | 値 |
|---|---|
| モデル | `google/gemma-4-31B-it@842da379`（BF16） |
| 構成 | kairyu `examples/gemma-4-31b-8gpu`: 2GPU×4台、MTP 4、vLLM v0.33.0 SM120 overlay |
| サンプリング | チェックポイントの `generation_config`（temperature 1.0 / top_p 0.95 / top_k 64）、思考あり（サーバ既定）。ランナーはサンプリング項目を送らない |
| 時間の範囲 | 起動完了後の評価と採点（起動時間は含めない） |

編集した重みは同じ構成で起動します（kairyu PR #669）:

```sh
CHAT_CHECKPOINT_DIR=/mnt/nvme/<編集済みチェックポイント> ./run.sh up   # kairyu/examples/gemma-4-31b-8gpu
```

ディレクトリは完全なチェックポイント（`config.json`・safetensors・トークナイザ・チャットテンプレート・`generation_config.json`）である必要があります。重みだけを比べたい場合は、元モデルの重み以外のファイルをそのままコピー（同じ NVMe 上ならハードリンク）してください。

## 能力と出典

| 能力 | 出典 | 採点 |
|---|---|---|
| 一般知識 | MMLU-Pro | 公式 API 用プロンプトを例題なしで、apiX の抽出（最後の一致、抽出なしは不正解） |
| 専門知識・科学推論 | GPQA Diamond | simple-evals のプロンプトと `Answer: X` 抽出、選択肢順は kairyu-bench の固定順 |
| 事実の記憶 | TriviaQA（rc.nocontext validation、質問 ID で重複除去） | OpenCompass のプロンプト、公式 `triviaqa_evaluation.py` の完全一致 |
| 数学 | GSM8K-Platinum、MATH-500、AIME 2026（4回）、HMMT Feb 2026（4回） | Platinum 公式（推論モデル用）、lighteval の MATH-500 プロンプト、MathArena の指示、Math-Verify |
| コード生成 | HumanEval+ / MBPP+（EvalPlus 0.3.1）、LiveCodeBench v6（2025年の問題）、SciCode | 各公式テスト（実行は別プロセス） |
| 指示追従 | IFEval、IFBench | 公式チェッカー（IFEval は prompt-level strict、IFBench は prompt-level loose） |
| 関数呼び出し | BFCL v4（非 live・live・不要時に呼ばない・複数ターンの4群を同数） | 公式 FC 処理と公式チェッカー |
| エージェント作業 | Terminal-Bench 2.1（terminus-2）、SWE-bench Pro（30分版から） | 公式 verifier / 公式評価 |
| 長文脈 | RULER（32K・128K の6課題）、MRCR v2、LongBench v2（30分版から） | RULER `string_match_all`、kairyu-bench の MRCR / LongBench 採点 |
| 論理推論 | BBEH、MuSR | BBEH `evaluate_correctness`、MuSR 公式解析 |
| 読解・常識 | HellaSwag、ARC-Challenge、WinoGrande | lm-evaluation-harness 定義の対数尤度（BOS 付き）、acc_norm / acc |
| 多言語・日本語 | MGSM（11言語）、MMLU-ProX 日本語、JamC-QA | simple-evals、lm-eval の mmlu_prox、flexeval の jamcqa_chat |
| 画像理解 | MMMU-Pro（標準10択と画像版） | 公式 CoT プロンプトと解析 |
| 安全性 | XSTest | 公式文字列一致: 安全な質問は答えたら1、危険な質問は断ったら1 |

総合スコアは14能力の単純平均、能力のスコアはその出典スコアの平均です。

スコアに入れない診断（すべてのパターン）:

- **次トークン分布**: WikiText-2・日本語 Wikipedia・コード（MBPP+ の解答）の先頭512トークンで、元モデルとの KL（上位20語による下限近似）、最有力トークンの一致率、負の対数尤度。生成なしで安く、重みの変化を最も早く検出します。
- **生成の健全性**: 上限到達率、空回答の率、平均出力長。
- **確信度の正しさ**: 選択式の対数確率からの校正誤差。

使わないもの: HLE・CharXiv・τ-bench（採点か利用者役を対象モデル自身が行う）、Arena-Hard・MT-Bench・SimpleQA（LLM 審判が必要）、DeepSWE・Terminal-Bench 4.0（1問で60分を超える）、SWE-bench Verified（kairyu-bench で完走実績なし）、HarmBench・StrongREJECT（採点用モデルが別に必要）、JMMLU・Multi-IF（改変禁止・非商用ライセンス）。AIME / HMMT は CC-BY-NC-SA なので研究用途として記録しています。

## 使い方

```sh
# 1回だけ: データ・公式ソース・採点環境・RULER データ・SciCode の test_data.h5 を用意
./kairyu-bench wsp prepare

# 評価（--logprob-endpoint は prompt_logprobs を返す vLLM レプリカ）
KAIRYU_BENCH_DOCKER_NETWORK=gemma-4-31b-8gpu_default ./kairyu-bench wsp run \
  http://host.docker.internal:8020/v1 --tier 10 --split dev --label <重みのハッシュなど> \
  --logprob-endpoint http://gemma-0:8000/v1 --logprob-endpoint http://gemma-1:8000/v1 \
  --logprob-endpoint http://gemma-2:8000/v1 --logprob-endpoint http://gemma-3:8000/v1

# 元モデルとの比較
./kairyu-bench wsp compare results/<元モデルの run> results/<編集版の run>
```

Kairyu の API は `prompt_logprobs` を上流 vLLM へ渡さないため、対数確率を使う項目だけ vLLM レプリカへ直接送ります。`KAIRYU_BENCH_DOCKER_NETWORK` はランナーを例のスタックの Docker ネットワークへ参加させます。
エージェント作業には Docker が必要で、SWE-bench Pro は専用の Docker デーモン（`KAIRYU_BENCH_DOCKER_SOCKET`）を推奨します（[AGENTS.md](../AGENTS.md)）。

調整用（dev）は編集の比較・選択に、最終確認用（test）は最終報告だけに使ってください。test を見ながら編集を選ぶと、その結果は最終確認になりません。

### 結果

`results/<run-id>/` に `run.json`（状態、時間、未採点の項目）、`items.jsonl`（問題ごとの応答・トークン数・得点・抽出結果）、`responses.jsonl`（到着順の生の応答）、`report.json` / `report.md` を保存します。エージェント作業は `agents/<出典>/` に通常の kairyu-bench の結果として残ります。

`compare` は同じ問題リスト（manifest の SHA-256）・パターン・組の2回分だけを比べます。区間は出典ごとに問題を復元抽出する対応のあるブートストラップ（2,000回、固定シード）で、同じ入力からは同じ値が出ます。

API の通信障害で採点できなかった項目は `unscored` として記録され、状態は `partial` になります（推測の0点にはしません）。

## 問題リストの作り方（校正）

1. `wsp pool --output pool.json`: 各出典の dev・test から、60分版の目標数の1.5倍ずつを層別に取ります（エージェント作業は過去の実行で短かった課題が候補: `src/kairyu_bench/data/wsp/agent-candidates.json`）。
2. `wsp run ... --manifest pool.json --tier 60`: 元モデルで1回、長い上限で実行して問題ごとの出力長を測ります。
3. `wsp freeze --pool pool.json --calibration results/<run> --output src/kairyu_bench/data/wsp/manifest.json`: 元モデルがそのパターンの上限の `margin`（0.6）倍以内で答え終えた問題だけを、そのパターンに入れます。上限と目標数は `src/kairyu_bench/data/wsp/targets.json` です。10分版は短く解ける問題が中心になり、長く考える問題は30・60分版で増えます。

## 公式手順との違い

- MMLU-Pro・MMLU-ProX は公式が5-shot のため、例題を外した0-shot。
- MuSR・MMMU-Pro で答えが読み取れない場合、公式はランダムに選びますが、ここでは不正解（判定が実行順に依存しないように）。
- AIME・HMMT は MathArena のプロンプトに Math-Verify の判定（MathArena 自身の解析器ではない）。
- BBEH は問題文と指示文を空行で区切る（論文に区切りの記載なし）。採点は問題ごとの正誤の平均。
- IFEval は言語判定（langdetect）にシードを固定。
- EvalPlus は最小時間制限 4 秒（0.3.1 は1秒、後の版で4秒に変更）。0.3.1 で正解解も必ず不合格になる HumanEval/32 は除外。
- SciCode は背景なし設定（kairyu-bench の既定）。1ステップのテスト上限は 300 秒（公式 1,800 秒）。問題ごとの得点は通ったステップの割合。
- BFCL はエージェント系（web_search・memory）を除外（SerpAPI とモデルのダウンロードが必要なため）。
- RULER は外部コーパス不要の6課題（niah_single_1・niah_multikey_2/3・vt・cwe・fwe）。補完モデル用の回答書き出し（answer prefix）は送らない。
- HellaSwag・ARC・WinoGrande は指示調整版モデルを生の文章で測るため、一般に公表される事前学習モデルの値より低めになります。元モデルとの差で読んでください。
- コードを実行する採点（EvalPlus・LiveCodeBench・SciCode）はテストに時間制限があり、採点機の負荷が極端だと判定が変わりえます。採点は別プロセスで並列数を抑えて行います。

## ロックの更新

`scripts/locks/README.md` を参照してください。
