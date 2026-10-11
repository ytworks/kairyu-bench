# 2026年10月 公開ベンチマーク値の更新計画（README・比較ページ・Wiki）

> **For agentic workers:** この計画はタスク順に実行する。チェックボックスで進捗を管理する。ユーザーの追加指示がない限り、外部への push・PR作成・Wiki への push は行わない。

**Goal:** 本 runner が現在実行できる13ベンチマークについて、2026-09-30 以降に公開された他モデルの最新状況を一次資料で確認する。そのうえで、リポジトリの `README.md`・`docs/model-comparison.md` と GitHub Wiki（`Home` / `Benchmark-Results` / `DeepSWE-Comparison`）の両方を同じ根拠で更新する。

**Spec:** ユーザー依頼「今実行できるベンチマークの最新の他モデルの状況をwikiとreadme両方更新する計画を立てて」「WEBで調べて一次情報をできるだけ集めて」。

## 状態と前提

- **実施状況（2026-10-11）**: ユーザーが全項目を承認したため、Task 2〜6を実施した。ただしネットワーク許可はこのセッションに反映されず、区分Bの一次確認（Task 1）はできなかった。このため、A区分と、作業中に一次資料で確認できた値だけを反映した。反映したのは、Haiku 5.5・Atria-Dawn-Preview・Hy4 previewの列、Gemini Flash各版とGLM-5.3 Flashの空欄、DeepSWEの注記と自己報告、SWE-Bench Pro V2の注記。B区分は [`docs/sources/public-results-audit-20261011.md`](../../sources/public-results-audit-20261011.md) の「保留した値」に残しており、許可が反映された新しいセッションでTask 1から再開する。
- 2026-10-11 作成。計画作成の時点では、リポジトリ・Wiki とも未変更だった。調査時の HEAD はリポジトリ `84560af`、Wiki `629a522`（いずれも 2026-09-30/10-01 の9月分更新の直後）。
- 「今実行できるベンチマーク」は `src/kairyu_bench/data/benchmarks.json` に登録済みの13種と解釈する。対象は SWE-bench Pro、SWE-bench Verified、Terminal-Bench 2.1、LiveCodeBench、LiveCodeBench Pro、HLE、CharXiv Reasoning、GPQA Diamond、SciCode、τ-bench Banking、Long Context Reasoning、MRCR v2、DeepSWE v1.1。
- Wiki は既存3ページだけを更新し、新しいページは作らない（Wiki 履歴 `6e201d9`「remove new pages and retain updates to existing wiki pages」の方針を踏襲）。
- 9月分（PR #26）と同じく、追加値は Markdown にだけ記載する。参照カタログ `src/kairyu_bench/data/public_results.json` と、`<!-- BEGIN/END public-comparison -->`・`<!-- BEGIN/END deepswe-comparison -->` の生成表は変更しない。CI が `python scripts/update_model_comparison.py --check` で生成表の一致を検査しているためである。
- 調査の生ノート（5系統、検証レベル付き）は scratchpad の `research/01〜05-*.md` に置いている（コンテナ終了で消える一時ファイル）。確定した内容は Task 3 で照合記録としてリポジトリに残す。

## 調査環境の制約（計画全体に影響）

- このセッションのネットワークポリシーでは、WebFetch もシェルも次の範囲にしか届かなかった: `www.anthropic.com`、`github.com`、`raw.githubusercontent.com`、`storage.googleapis.com`。
- 名前解決またはプロキシで失敗したのは次のとおり: `huggingface.co`、`openai.com` / `cdn.openai.com`、`www-cdn.anthropic.com`（システムカードPDF）、`artificialanalysis.ai`、`www.tbench.ai`、`deepswe.datacurve.ai`、`labs.scale.com` / `scale.com`、`epoch.ai`、`x.ai` / `media.x.ai`、`research.meta.ai` / `developer.meta.com`、`reflection.ai`、`mistral.ai`、`sakana.ai`、`z.ai`、`qwen.ai`、`api-docs.deepseek.com`、`mimo.xiaomi.com`、`deepmind.google`、`livecodebenchpro.com`、`contextarena.ai`、`arxiv.org`、`*.github.io`。
- WebSearch はターン内の上限（全エージェント共有で200回）に達した。
- **したがって、下表の「B」は一次ページそのものを未確認の値である。** 環境の Allowed domains に上記ホストを追加すれば（クラウド環境メニュー → Edit → Network access）、Task 1 で直接確認できる。AA など JS で描画されるページは、プリインストール済みの Chromium（Playwright）で描画して目視確認する。

## 採用基準

| 区分 | 定義 | 扱い |
| --- | --- | --- |
| **A** | 開発元・主催者の一次資料を直接読んだ値。HTML本文、PDFの表画像・本文、または開発元自身の GitHub README・図 | 転記可 |
| **B** | 一次資料の第三者コピーで読んだ値。HFカードのGitHubミラー（ハッシュ照合済み）、ページの機械取得コピー、AAデータのミラー、Issue上の報告など | **Task 1 で一次ページを確認できたものだけ転記** |
| **C** | 検索要約のみ、またはアグリゲーター経由の値 | 転記しない。手がかりとして記録のみ |

既存ルールは維持する。

- TB は 2.1 だけを使う（4.0 / 3.0 は不採用）。
- HLE は no tools の full set を優先する。tools 値は `(tools)` と明記する。text-only・HLE-Verified・HLE-Diamond・HLE-Rolling は別セットとして扱う。
- LCB Pro は % だけ（Elo は不採用）。
- Long Context Reasoning は AA-LCR v1 だけ（v1.1 は比較不可）。
- MRCR v2 は 8-needle で、コンテキスト長のビンを明記する。
- DeepSWE は公式原本だけを主表に載せ、提供元の自己報告は別表にする。

今回から次を追加する。

- **SWE-Bench Pro V2（642問、2026-09-22公開）と HARD-51 の値は、SWE-Bench Pro 行（v1 public 731問）へ入れない。**
- 他社が測定した他社モデルの値（比較列の値）は、その他社モデルの一次値として扱わない。

## 調査結果（2026-10-11時点）

### 1. 主催者リーダーボード・ベンチマーク本体の状況

| ベンチマーク | 状況 | 区分・根拠 | 反映先 |
| --- | --- | --- | --- |
| DeepSWE v1.1 | 公式JSONは直接取得できなかった。第三者トラッカーが10/09に取得したコピー（`lprhodes/deepswe-tracker`）を保存原本と機械照合した結果、70設定の採点値はすべて一致した（差分はGPT-6 Astraのcost欄のみ）。`generated_at` は `2026-09-22T06:27:15`。9/03のGPT-6 Astra以降、新しいモデルは掲載されていない | B（ミラー）。Task 1 で公式JSONを再取得する | 比較ページのDeepSWE節、Wiki DeepSWE |
| DeepSWE v1.1 の採点問題 | 次の2件はGitHub上で確認済み（A）。いずれもDatacurveの返信はない。<br>・[#105](https://github.com/datacurve-ai/deep-swe/issues/105)（10/02, Tokenless）: `tests/conftest.py` だけを変えたパッチがreward 1になるなど、26件のbypassを報告<br>・[#107](https://github.com/datacurve-ai/deep-swe/issues/107)（10/05）: `claude-code-by-agents-recursive-delegation` でvitestのworkerがクラッシュすると合格と採点される。公開出力のある合格134件のうち95件が該当し、うち47件以上は実際には不合格<br>また、Epoch AI がDeepSWE v1.1を「Flawed」と評価したと報じられている（113問中23問以上で偽陰性）。これはC（要約のみ）で、日付も09-07説と09-17説で一致しない | Issue: A、Epoch: C → Task 1 で確認 | 比較ページの注意、Wiki DeepSWE（**本runnerも同じ verifier `0b9fabb` を使うため、ローカル実測にも影響し得る**） |
| Terminal-Bench 2.1 | 2026-08-11からコミュニティ提出を停止（`harbor-framework/terminal-bench-2-1` README、A）。9/03のGPT-6 Astra以降、公式の新規エントリは確認できない。公式ボード自体は未到達 | A（提出停止）、B/C（新規なし） | 比較ページのS22再確認日 |
| SWE-bench Pro | **2026-09-22にV2を公開**（642問、HARD-51、locked protocol）。HFの `default` configがV2に変わり、v1は config `v1` / tag `v1.0` として残っている（`scaleapi/SWE-bench_Pro-os` の README・`v2/README.md`・`v2/GATE.md`、A）。本runnerはrevision `7ab5114…`（v1、731問）に固定済みなので影響はない。V2ボードの数値は要約のみ（C）で、内容にも矛盾があるため転記しない | A | README のベンチマーク表、比較ページの注意、Wiki Benchmark-Results |
| SWE-bench Verified | 公式データの最新エントリは2026-02-26（`SWE-bench/swe-bench.github.io` の `data/leaderboards.json`、A）。表のモデルの掲載はない | A | 変更なし（注記のみ） |
| LiveCodeBench | 公式サイトの最終更新は2025-08-01で、最新リリースはv6（2023-05〜2025-04）。表のモデルの掲載はない | A（GitHub） | 変更なし |
| LiveCodeBench Pro | 公式サイトに到達できず、状況は不明 | — | Task 1 で確認 |
| HLE | CAISのGitHubに2026-07-27の問題改訂記録がある（A）。HLE-Diamond（9/22、固定1,000問）とScaleのHLE-Rolling（9/17）は別セット（要約・一部A） | A/C | 比較ページの注意（別セットを転記しない旨） |
| SciCode（AA） | 「Under review」が続いている。9/07〜10/10のミラー値は不変。SciCode-Verified（64問・287 subproblems）はAAの288 subproblemsと比較できない | B | S10の注記を更新 |
| τ-bench Banking | AA τ³-Bankingは9月にIndexから外れた（単独評価としては継続）。9/08以降に公開されたフロンティアモデルには値がない。Sierra公式の banking_knowledge トラックはAAと条件が異なる（user simulator、retrieval、反復数） | B/A | S11の注記を更新 |
| AA-LCR | AAはv1.1だけを表示しており、v1の値は再確認できない（既存どおり） | B | 変更なし |
| CharXiv / LongBench v2 | 公式ボードは2025年で停止しており、表のモデルの掲載はない | A（GitHub） | 変更なし |
| MRCR v2 / GPQA | 主催者のボードはない | — | 変更なし |

### 2. 新しい列の候補（2026-09-01〜10-11公開）

| モデル | 公開 | 主表行に入る値 | 区分 | 推奨 |
| --- | --- | --- | --- | --- |
| Claude Haiku 5.5 | 10-07 | HLE **45.9**（no tools、Max）。ツールありは57.4で行には入れない。effort別ではLow 30.9〜Max 45.9 | A（[発表ページ](https://www.anthropic.com/claude-haiku-5-5)のHTML表） | **追加** |
| 同上 | | SWE-Bench Pro 64.8 | C（システムカードPDFに未到達） | Task 1 で[システムカード](https://www.anthropic.com/claude-haiku-5-5-system-card)のTable 8.1.Aを確認してから |
| 同上 | | SciCode 55.0（AA、Max） | B（AAミラー） | Task 1 でAAを確認してから |
| MiMo-V2.6-Pro（Xiaomi、open-weight、MIT） | 09-21 | Terminal-Bench 2.1 **89.9**（ハーネス・試行数の記載なし） | B+（HFカードのミラー。技術報告PDFのハッシュはHFのLFS oidと一致） | HFで直接確認できれば追加 |
| MiMo-V2.6-Flash | 09-21 | Terminal-Bench 2.1 **87.6** | B+ | 同上 |
| Atria-Dawn-Preview（上海AI Lab、GLM-5.2ベース、MIT） | 09-11 | SWE-Bench Pro **59.6**（OpenHands v1.23.0、ネットワーク無効）、Terminal-Bench 2.1 **78.3**（Claude Code v2.1.224、4回平均） | A（[開発元GitHub README](https://github.com/atria-asi/Atria-Dawn-Preview)の表） | **追加**。同表のτ³ 41.2は自前ハーネスのため不採用 |
| Gemini 4 Argon | 09-30（限定提供） | 主表行に入る値なし。AAのHLE 57.1・SciCode 61.8はB | A（[評価PDF](https://storage.googleapis.com/deepmind-media/gemini/gemini_4_argon_model_evaluation.pdf)） | AAを確認できれば追加。DeepSWE自己報告77.9は別表へ |
| Muse Spark 1.3（Meta） | 09-02 | TB2.1 88.8、MRCR 98.5（256K–512K）/ 98.1（512K–1M） | B（スコアカード画像の第三者転記。ハーネス記載が転記元で食い違う） | Task 1 で確認してから |
| Reflection Beam（open-weight予定） | 10-05 | SWE-Bench Pro 65.5、Verified 80.9、TB2.1 80.1、HLE 36.2（no tools）、GPQA 90.5 | B（ページの機械取得コピー） | Task 1 で確認してから |
| Mistral Large 4 Preview | 10-06 | 主表行に入る値なし（AAのHLE 35.0・SciCode 54.2はB） | B | 保留（Mistral自身が追加ベンチマークの公表を予告） |
| Cognition SWE-2 / Fireworks Ember-1（いずれもKimi K3派生） | 09-10 / 09-23 | SWE-2: TB2.1 92.8（effort間の最高値）。Ember-1: TB2.1 82.0、Verified 92.2 | B | 優先度低。確認できれば追加 |
| JetBrains Mellum2.1-12B-A2.5B-Thinking | 10-08 | LCB v6 82.0、Verified 47.0、SWE-Bench Pro 28.0、TB2.1 17.4、GPQA 64.6（Piハーネス、114K） | B（HF model-indexのコピー） | 優先度低 |
| Tencent Hy4 preview（期間直前） | 08-27 | SWE-Bench Pro 65.7（swe-agent）、TB2.1 85.4（Claude Code）、GPQA 92.3 | A（[開発元GitHub](https://github.com/Tencent-Hunyuan/Hy4-preview)の図） | 任意（open-weightで掲載値が多い） |
| Step 5 Preview | 09-18 | — | C | 不採用 |

調査の範囲では、次のモデル・版は見つからなかった。

- 存在しないもの: Gemini 3.8 Pro、Gemma 4.1/5、Grok 4.8/5、GPT-6 mini/nano、新しいGPT-6 Astraのsnapshot。
- 新モデルではないもの:
  - OpenAIの「GPT-6 Sol/Luna (October)」はChatGPT限定のsnapshotで、主表行に入る値はない。
  - GPT-6.1 Sol Ultrafast は提供速度の区分であり、別モデルではない。

### 3. 既存列の空欄を埋める候補

| 列 | 行 | 値 | 条件 | 区分・出典 |
| --- | --- | --- | --- | --- |
| Gemini 3.7 Flash | TB2.1 / CharXiv / MRCR v2 | **85.8** / **84.5** / **97.0** | Terminus 2・自己計測 / no tools（with tools 88.7）/ GDM-MRCR v2 8-needle 128k average | A（[3.7 Flash評価PDF](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-7_flash_model_evaluation.pdf)の表画像を目視確認） |
| Gemini 3.6 Flash | SWE-Bench Pro / TB2.1 / CharXiv / MRCR v2 | **58.7** / **78.0** / **85.2** / **91.8** | Public・社内版Antigravityハーネス / Terminus-2 / no tools / 8-needle 128k avg | A（[3.6 Flash評価PDF](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-6_flash_model_evaluation.pdf)） |
| Gemini 3.5 Flash | SWE-Bench Pro / TB2.1 / HLE / CharXiv / MRCR v2 | **53.9** / **76.2** / **40.2** / **84.2** / **77.3** | Public・5回平均（3.6 PDFでの再掲は55.1で不一致）/ Terminus-2 / full set text+MM（no toolsの明記なし）/ no tools / 8-needle 128k avg | A（[3.5 Flash評価PDF](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-5_flash_model_evaluation.pdf)） |
| Gemini 3.1 Pro | CharXiv | **83.3** | no tools、Google自己計測（3.5・3.6の両PDFで一致） | A |
| GLM-5.3 Flash | TB2.1 / HLE | **84.3** / **55.3 (tools)** | Claude Code 2.1.207 / tools・full set・judge GPT-5.6-luna medium | A（[Z.ai GitHubの図](https://github.com/zai-org/GLM-5/blob/main/resources/bench_53.png)を目視確認）。条件の脚注はHFカード（B） |
| Grok 4.5 | SWE-Bench Pro / TB2.1 | 64.7 / 83.3 | effortの記載なし | B（x.ai記事のコピー。GoogleのPDFも同値を「各社の自己報告値」として引用） |
| Muse Spark 1.2 / 1.1 | TB2.1・MRCR・SWE-Bench Pro・HLE ほか | 1.2: TB2.1 82.9。1.1: HLE 52.2、TB2.1 80.0、SWE-Bench Pro 61.5 | ハーネスは転記元によって異なる | B（画像の第三者転記） |
| DeepSeek-V4-Flash-0731 | GPQA | 89.9 | V4.1技術報告の「DS-V4-Flash」列（0731であることは推定） | B・推定 → 保留 |
| 各列（AA） | τ³ / SciCode / HLE | 例: τ³ Fable 5 38.1、Opus 5 42.1、GPT-5.6 Sol 44.3、Grok 4.6 (High) 50.7。SciCode Opus 5 56.4、GPT-5.6 Sol 57.1、GLM-5.3 59.0。HLE GPT-5.6 Terra 42.9 など | AAのconfig表記を併記 | B（AAミラー2系統で一致）→ Task 1 でAAの `?models=<slug>` を目視確認。全候補は `research/05` の §1.5 |

### 4. 既存値・注記の修正候補

1. **GLM-5.2 の HLE 40.5 は text-only subset の値である可能性が高い。**
   - 根拠: HFカードの脚注「By default, we report the text-only subset; results marked with * are from the full set.」があり、40.5 には * が付いていない（Bのミラー）。Z.aiのGitHubの図には text-only の明記がない。
   - Task 1 でHFカードを直接確認する。確定すれば、2026-08-27のFable 5修正と同じ形式で「修正」節に記録し、セルに注記するか `—` に戻す。tools値54.7の注記も同様に直す。
2. **Gemini 3.1 Pro の注記を直す。**
   - 「TB2.1はTB2.0のみ公表」は古い。Googleの後続PDFに70.3（5月）と73.8（7月）があるが、どちらもGoogle自身の計測ではない（公式ボード/AAからの引用）うえ値が食い違うため、値は採用せず注記だけ更新する。
   - Google の評価PDFは「all run with ... model-id gemini-3.1-pro-preview」と明記している。Preview 列を分けた理由の記述を直す。
   - SciCode 59 は「sourced from Artificial Analysis」と明記されている。
3. **Mythos 5.1 の TB4.0 の表記を「61%」から「60.9%」へ直す**（発表ページの値。A）。
4. **GPT-6 Sol / Luna 列は9月のAPI版であると注記する**（ChatGPTでは10/07から10月版snapshotを提供。OpenAIの要約情報なのでC → 確認後に注記）。
5. **Fugu / Fugu Ultra 列は v1.0 の値であると注記する**（v1.1が7/24、v2.0とMaxが9/11に公開済み）。
6. **DeepSeek-V4.1-Flash の GPQA/HLE の条件が資料間で食い違う。** 技術報告ではtop_p 1.0、カードでは0.95。注記を追加する。
7. **AAの表記と slug の変更を注記する。**
   - Anthropicのconfig名が「(Adaptive Reasoning, Max Effort, Default Fallback)」から「(Max, Default Fallback)」に変わった（値は不変）。
   - Qwen3.8 Max の slug `qwen3-8-max` は後継の「(0902)」を指すようになった。表の値（0803版）は `qwen3-8-max-0803` で確認する。
8. **DeepSeek-V4-Flash 列と V4-Flash-0731 列が同一モデルかもしれない。** DeepSeekは自己報告で、DeepSWEのpreview版を7.3、0731版を54.4としている。Datacurveの53.32は0731版に近い。ただ証明はないため、注記の候補にとどめる。

### 5. DeepSWE 自己報告の別表へ追加する候補

| モデル | 値 | 条件 | 区分 |
| --- | ---: | --- | --- |
| Gemini 4 Argon | 77.9 | Google自己計測、mini-swe agent、最高thinking | A（評価PDF） |
| Muse Spark 1.3 | 75.4 | max、ハーネスは転記元によって異なる | B |
| Fireworks Ember-1 | 75.2 | 条件の記載なし | B |
| Fugu Ultra v2 | 74.3 | 本文のみ | C |
| Cognition SWE-2 | 73.0 | effort間の最高値 | B |
| MiMo-V2.6-Pro / Flash | 71.9 / 67.9 | ハーネスの記載なし | B+ |
| Kimi K3 | 67.5 | Kimi Code（mini-SWE-agentでは67.3） | A（Moonshot GitHub） |
| Gemini 3.7 Flash | 65.3 | 自社のmini swe agent・high（公式は65.49 medium） | A |
| GLM-5.3 Flash | 63.4 | mini-swe-agent、temperature 0.95、6h、400K（公式は63.39 max） | A（図）＋B（脚注） |
| DeepSeek V4-Pro / V4-Flash | 62.7 / 54.4 | 「DeepSWE v1.1」と明記。0813版/0731版と推定 | B+ |
| Mistral Large 4 | 61.7 | AA実行の可能性あり | B |
| DeepSeek-V4-Flash-Vision-Exp | 59.3 | v1.1の明示なし | B |
| Reflection Beam | 44.4 | 条件の記載なし | B |

- 参考: 第三者の独立測定として、Mercor APEX「OSS DeepSWE Leaderboard」がある（mini-swe-agent、3パス。例: Opus 5.5 72.3、DeepSeek V4.1 Flash 71.7、Haiku 5.5 59.9）。主催者の値ではなく到達もできていないため、主表・別表には入れない。Wikiの「参考」に入れるかどうかはTask 1の確認結果で決める。

## タスク

### Task 0: 判断事項の確定（ユーザー）

- [ ] 採用基準を決める。**推奨: AとTask 1で確認できたBだけを転記する。** Bを確認するには、ネットワークの Allowed domains に上記ホストを追加する必要がある。追加しない場合はAだけで更新する。
- [ ] 新しい列を決める。**推奨: Haiku 5.5、Atria-Dawn-Preview は即時追加。MiMo-V2.6-Pro/Flash、Gemini 4 Argon、Muse Spark 1.3、Reflection Beam は確認でき次第追加する。** それ以外は保留。列を追加するのは、主表の行（DeepSWE自己報告を除く）に入る値が1つ以上確定したモデルに限る。
- [ ] AAの値で既存列の空欄（τ³・SciCode・HLE）を埋めるかを決める。**推奨: 埋める。**「ベンダー値がなければAA」という既存の方針に沿う。configを併記し、Task 1でAAを目視確認できたものに限る。
- [ ] GLM-5.2 の HLE を修正するかを決める（Task 1で確定した場合）。

### Task 1: B区分の一次確認（ネットワーク許可後）

- [ ] DeepSWE公式JSONを取得し、`docs/sources/deepswe-v1.1-20260910.json` と機械照合する。比較する項目は model、effort、n_passed、n_attempted、n_tasks_attempted、pass@1、pass@4、CI、n_runs。changelogで9/03以降の追加の有無を確認する。
- [ ] Epoch AIの[DeepSWE review](https://epoch.ai/benchmarks/deepswe/review)の公開日、評価、件数を確認する。SWE-Bench Pro / Verified / HLE のreviewも同時に確認する。
- [ ] Haiku 5.5 のシステムカードのTable 8.1.A（SWE-Bench Pro 64.8かどうか）と、DeepSWE節の有無を確認する。
- [ ] AAの値を確認する。対象は新モデル（Haiku 5.5、Gemini 4 Argon、Muse Spark 1.3、MiMo-V2.6-Pro、Mistral Large 4）と、§3の空欄候補。HLE、SciCode、τ³-Banking を `?models=<slug>` でChromiumで描画し、目視とスクリーンショットで確認する。既存値（S9〜S11）が変わっていないことも確認する。
- [ ] HFカードを直接確認する。対象は MiMo-V2.6-Pro-RL / Flash-RL、GLM-5.2（text-only脚注）、GLM-5.3-Flash（脚注）、Mellum2.1。
- [ ] 次のページを確認する: x.ai の Grok 4.5、Meta の Muse Spark 1.3 / 1.2 / 1.1、reflection.ai の Beam、cognition.com の SWE-2、fireworks.ai の Ember-1。
- [ ] TB2.1公式ボードを確認する（`tbench.ai/leaderboard/terminal-bench/2.1`）。9/03以降の掲載の有無と、Gemini 3.1 Pro の70.3/73.8の出所を調べる。
- [ ] LiveCodeBench Pro公式（最新四半期・掲載モデル）と、OpenAIの10月版addendum（`cdn.openai.com/pdf/gpt-6-october.pdf`）を確認する。
- [ ] 確認できなかった項目は区分Bのまま残し、転記しない。

### Task 2: `docs/model-comparison.md` の更新

- [ ] 冒頭の更新履歴に「2026-10-11」の段落を追加する。内容は、追加した列、空欄に追記した値、既存値の修正（あれば）、主表の列数。既存の値は修正分を除いて変更しない。
- [ ] 主表に列を追加し、セルを追記する。
  - 列の位置は提供元ごとの並びに合わせる。Haiku 5.5 は Sonnet 5.5 の後、Gemini 4 Argon は Gemini 3.1 Pro の前、MiMo は Kimi K3 の後、Atria は GLM-5.3 の後に置く。
  - 値は `[値][Sxx]` の形式で書く。条件付きの値は `(tools)`、`(high)`、`(128K)` などを括弧で付ける。
- [ ] DeepSWE節を更新する。
  - 再確認日と、公式原本が不変かどうかを書く。
  - 「既知の採点問題」の段落を追加する。内容はEpoch（確認できた範囲）、#105、#107。本runnerも同じverifierを使う点にも触れる。
  - 自己報告の別表に、§5のうちAまたは確認済みBの行を追加する。
- [ ] 「2026-10-11の追加について」節を新しく設ける（9/30節と同じ形式）。提供元ごとに、出典、条件、採用しなかった値、資料間の不一致（Gemini 3.5 Flash の53.9/55.1、Gemini 3.1 Pro の TB2.1 など）を書く。
- [ ] 「比較時の注意」を更新する。対象は§4の各項目、SWE-Bench Pro V2と別セットのHLE（Diamond/Rolling/Verified/text-only）の不採用、GDM-MRCR v2の128k averageの意味。
- [ ] 出典リストに S33 以降を追加する。リンク参照定義も追加する。仮番号は次のとおり（確定時に連番を振り直す）。
  - S33 Claude Haiku 5.5 発表ページ
  - S34 Haiku 5.5 システムカード
  - S35 Atria-Dawn-Preview GitHub
  - S36 MiMo-V2.6-Pro-RL HFカードと技術報告
  - S37 MiMo-V2.6-Flash-RL HFカード
  - S38 Gemini 3.7 Flash 評価PDF
  - S39 Gemini 3.6 Flash 評価PDF
  - S40 Gemini 3.5 Flash 評価PDF
  - S41 Gemini 4 Argon 評価PDF
  - S42 GLM-5.3-Flash（Z.ai GitHubとHFカード）
  - S43 SWE-Bench Pro V2（`scaleapi/SWE-bench_Pro-os` README）
  - S44 deep-swe Issue #105 / #107
  - S45 Kimi K3 GitHub README（DeepSWE自己報告）
  - S46以降はTask 1で確定したB区分の出典（Grok 4.5、Muse Spark、Beam、Epoch、AA追加分など）
- [ ] 既存の S9 / S10 / S11 / S22 / S23 の再確認日と注記を更新する（2026-10-11）。

### Task 3: 照合記録 `docs/sources/public-results-audit-20261011.md` の作成

- [ ] `public-results-audit-20260910.md` と同じ形式で書く。出典ごとに、確認した箇所、結果、区分（A/B）、取得方法（直接 / PDF画像 / GitHubミラーとハッシュ）を記録する。
- [ ] 確定しなかった項目と保留理由を書く（ネットワーク到達不可、転記元間の不一致、同一性が不明なものなど）。
- [ ] DeepSWE公式JSONを取得できた場合は、`docs/sources/deepswe-v1.1-20261011.json` として保存するかを判断する。採点値が不変なら保存せず、SHA-256と照合結果だけを記録する（9/30と同じ扱い）。

### Task 4: `README.md` の更新

- [ ] 冒頭のリンクを「他モデルの公開ベンチマーク比較（出典付き、2026-10-11更新）」とする。
- [ ] ベンチマーク表の `swe-bench-pro` の補足に「v1 public 731問に固定（2026-09-22公開のV2・642問とは別）」と追記する。本runnerは dataset revision で固定しているが、HFの既定configがV2に変わったため明記する。
- [ ] 「結果」節の参照カタログの記述を直す。現状は「DeepSWEは28モデル・70設定…既存19モデル」。カタログ（JSON）がこの範囲である点は残し、比較ページではMarkdownでN列まで拡張していることを書き足す。
- [ ] 任意: DeepSWE節に「既知の採点問題（#105・#107）は本runnerが固定したverifierにも当てはまる」と1文追加する。

### Task 5: Wiki の更新（既存3ページだけ）

- [ ] `Home.md`: 最終更新日を2026-10-11にする。各リンクの説明文を更新する（比較ページの列数、SWE-bench Proの追加値、DeepSWEの再確認日）。
- [ ] `Benchmark-Results.md`（SWE-bench Pro）:
  - 公開値の表に次の行を追加する。A: Gemini 3.6 Flash 58.7、Gemini 3.5 Flash 53.9、Atria-Dawn-Preview 59.6、（任意）Hy4 preview 65.7。確認できれば: Haiku 5.5、Grok 4.5 64.7、Reflection Beam 65.5、Muse Spark 1.1 61.5。
  - SWE-bench Proの公表値がない新モデルを `—` の行で追加する: Gemini 4 Argon、Muse Spark 1.3、MiMo-V2.6-Pro/Flash、Mistral Large 4。
  - 「比較時の注意」にV2公開、v1固定、他社による他社モデルの測定値を採用しないことを追記する。出典も追加する。実測値の表は変更しない。
- [ ] `DeepSWE-Comparison.md`:
  - 更新日を変える。公式公開値は再確認の結果を書く。原本が不変なら表は据え置き、新モデルが載っていれば行を追加する。
  - 自己報告の別表に§5の行を追加する。
  - 「既知の採点問題」節を追加する（#105・#107・Epoch）。DeepSeek V4.1 Flash の実測（59.29%）も同じverifierによる結果であることを書く。
  - 任意: ホスト上の `results/deepswe-full-8w-20260912-r1/` で、#107の対象タスクの合否とvitestのクラッシュ有無を確認し、結果を1行で記す。この確認はホスト上でのみ実施できる。

### Task 6: 検証

- [ ] `python -m unittest discover -s tests -v`、`ruff check .`、`mypy --ignore-missing-imports src`、`python scripts/update_model_comparison.py --check` がすべて成功すること。
- [ ] 主表の各行のセル数がヘッダーの列数と一致すること。簡単なスクリプトで `|` の数を数えて確かめる。
- [ ] 本文で使った `[Sxx]` に、すべてリンク参照定義があること。定義だけあって使われていない出典がないこと。
- [ ] 追記した値を1件ずつ照合記録と突き合わせること（値、条件、出典番号）。
- [ ] Wikiで比較ページと同じ値・同じ出典になっていることを確認する（Benchmark-ResultsのSWE-bench Pro行と比較ページのSWE-Bench Pro行の一致など）。

### Task 7: コミット・PR・Wiki反映（ユーザーの指示後）

- [ ] ブランチ `docs/october-2026-model-scores` でコミットし、PRを作る。本文には採用基準、追加・修正の一覧、保留項目を書く。
- [ ] Wikiはクローン済みのリポジトリでコミットし、ユーザーの承認後にpushする。pushできない場合はパッチを渡す。
- [ ] PRのCIを監視し、緑になるまで対応する。

## 採用しない値（記録のみ）

- Terminal-Bench 4.0 / 3.0 / Science の値（Haiku 5.5 39.2、Argon 57.4、Mythos 5.1 60.9 など）。
- HLE の with tools 値（no tools 値がない列は `(tools)` 付きで既存方針どおり）、text-only、HLE-Verified、HLE-Diamond、HLE-Rolling。
- SWE-Bench Pro V2 と HARD-51。Reflection Beam の「v2-Hard 77.2」。
- AA-LCR v1.1 の値（Beam 79.3 など）。
- 自前ハーネスによる τ³ 値（Atria 41.2、Beam 38.0）と、Sierra の音声版 τ³（Gemini 3.8 Live 35.1）。
- 他社が測定した他社モデルの値。Google PDF の他社列、Cognition / Atria / Hy4 / DeepSeek / Z.ai 表の比較列などが該当する。
- 学習曲線だけの値と、非公開チェックポイントの値（MiMo、DeepSeek-V4.1-Flash の SWE-Bench Pro 曲線など）。
- 検索要約・アグリゲーターだけで得た値（Step 5 Preview、Fugu Max / Ultra v2 の数値、SWE-Bench Pro V2 ボードなど）。
