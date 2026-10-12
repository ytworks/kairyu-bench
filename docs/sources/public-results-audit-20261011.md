# 公開結果の出典照合（2026-10-11）

対象は、2026-09-30以降に公開された他モデルの値のうち、本runnerの13ベンチマークに対応するものと、既存の空欄・注記です。公開値は各測定者の参照値で、Kairyuの実測や同条件の順位ではありません。転記は [`docs/model-comparison.md`](../model-comparison.md) のMarkdownだけで行い、[カタログ](../../src/kairyu_bench/data/public_results.json)と生成表は変更していません。

## 調査環境

- 到達できたのは `www.anthropic.com`、`storage.googleapis.com`、`github.com`、`raw.githubusercontent.com` だけでした。
- 次のホストには到達できませんでした（WebFetchは名前解決エラー、シェルはプロキシ拒否）。
  - モデル提供元: `huggingface.co`、`openai.com`、`www-cdn.anthropic.com`（システムカードPDF）、`x.ai`、Metaの各ドメイン、`reflection.ai`、`mistral.ai`、`z.ai`、`qwen.ai`、`api-docs.deepseek.com`、`mimo.xiaomi.com`
  - 評価・リーダーボード: `artificialanalysis.ai`、`www.tbench.ai`、`deepswe.datacurve.ai`、`labs.scale.com`、`epoch.ai`、`livecodebenchpro.com`
- このため値を次の区分で扱いました。
  - **A**: 一次資料を直接読んだ値。転記しました。
  - **B**: 一次資料の第三者による写しでしか読めなかった値。保留しました。
  - **C**: 検索要約だけで得た値。手がかりとしてのみ記録しました。

## 転記した値（区分A）

| 出典 | 確認した箇所と結果 |
| --- | --- |
| [S33 Claude Haiku 5.5](https://www.anthropic.com/claude-haiku-5-5) | ベンチマーク表のHLEは「45.9% no tools」。ページに埋め込まれたグラフデータ（`haiku55,…,45.9,Max`）で、effortのMax点と一致することを確認。with tools 57.4は主表外。システムカード（`www-cdn.anthropic.com/e1080d6b…`）には到達できなかった。 |
| [S34 Atria-Dawn-Preview](https://github.com/atria-asi/Atria-Dawn-Preview) | READMEの表の行 `SWE-bench Pro \| 59.6 …` と `Terminal-Bench 2.1 \| 78.3 …` を `raw.githubusercontent.com` から逐語で取得。τ³-Bench Banking 41.2は自前のtau2系ハーネスの値のため転記しなかった。 |
| [S35 Hy4 preview](https://github.com/Tencent-Hunyuan/Hy4-preview) | `assets/benchmark-appendix.jpg`（4960×10840）を切り出して目視。Hy4 preview列は SWE-bench Pro 65.7、Terminal-Bench 2.1 85.4、GPQA Diamond 92.3。HLEは「no tools, text-only」43.4で別セット。GitHubの初回コミットは2026-08-28。 |
| [S36](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-7_flash_model_evaluation.pdf) / [S37](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-6_flash_model_evaluation.pdf) / [S38](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-5_flash_model_evaluation.pdf) Gemini Flash評価PDF | 結果表の画像を目視し、測定条件は本文テキストで確認（下記）。 |
| [S39 GLM-5.3-Flash](https://github.com/zai-org/GLM-5/blob/main/resources/bench_53.png) | Z.ai GitHubの評価図を目視。GLM-5.3-Flashは TB2.1 84.3、HLE w/ Tools 55.3、DeepSWE v1.1 63.4。条件（Claude Code 2.1.207、full set、judge）はHFカードの脚注によるもので、第三者の写しでのみ確認（区分B）。 |
| [S40 Gemini 4 Argon](https://storage.googleapis.com/deepmind-media/gemini/gemini_4_argon_model_evaluation.pdf) | 結果表の画像に、主表の行に対応する項目はない。DeepSWE v1.1 77.9は本文で「self computed, using a mini-swe agent harness」と明記。 |
| [S32 Gemini 3.8 Flash](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-8_flash_model_evaluation.pdf) | 表の画像でDeepSWE v1.1 73.7を確認。本文に「Results for Gemini 3.8 Flash are self computed, and use a mini-swe agent harness with high thinking」とある。 |
| [S41 Kimi K3 README](https://github.com/MoonshotAI/Kimi-K3) | DeepSWE 67.5（Kimi Code）。脚注に「all remaining scores are from the official DeepSWE leaderboard, under which Kimi K3 attains 67.3 with the mini-SWE-agent harness. We report the DeepSWE v1.1 tasks.」とある。 |
| [S42 SWE-Bench Pro](https://github.com/scaleapi/SWE-bench_Pro-os) | READMEのNews「(9/22) SWE-Bench Pro V2 released. 642 validated tasks … The HuggingFace `default` config is now V2; v1 is kept as config `v1` / tag `v1.0`.」を確認。本runnerは `harness_data.py` でdataset revision `7ab5114…`（v1）を固定して読み込むため、影響はない。 |
| [S43 #105](https://github.com/datacurve-ai/deep-swe/issues/105) / [S44 #107](https://github.com/datacurve-ai/deep-swe/issues/107) | いずれもopenで、Datacurveの回答なし（2026-10-11）。#105は26件の採点回避（例: `tests/conftest.py` だけを変えたパッチでreward 1、空パッチはreward 0）。#107は合格134件中95件がvitestのクラッシュによる合格で、少なくとも47件は実際には不合格と推定。他の20タスクも同じvitest採点だとしている。 |
| [S45 Gemini 3.1 Pro](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-1_pro_model_evaluation.pdf) | 本文に「gemini-3.1-pro-preview with default sampling settings」と「SciCode results are sourced from Artificial Analysis.」の記載を確認。 |
| [S46 terminal-bench-2-1](https://github.com/harbor-framework/terminal-bench-2-1) | READMEに「Community submissions are currently closed for Terminal-Bench 2.1. Only submissions run by the maintainers will be added to the leaderboard at this time.」とある。 |
| [S26 Fable 5.1 / Mythos 5.1](https://www.anthropic.com/claude-fable-and-mythos-5-1) | Terminal-Bench 4.0は「60.9% (Mythos 5.1)」。比較ページの「61%」を訂正した。 |

### Gemini Flash評価PDF（S36〜S38）の詳細

| モデル | SWE-Bench Pro (Public) | TB2.1 (Terminus-2) | HLE | CharXiv no tools / with tools | MRCR v2 8-needle 128K avg |
| --- | ---: | ---: | ---: | --- | ---: |
| Gemini 3.7 Flash（S36） | — | 85.8 | —（HLE-Verified 53.6のみ） | 84.5 / 88.7 | 97.0 |
| Gemini 3.6 Flash（S37） | 58.7 | 78.0 | —（S36の表ではHLE-Verified 51.2） | 85.2 / 89.4 | 91.8 |
| Gemini 3.5 Flash（S38） | 53.9（S37では55.1と再掲） | 76.2 | 40.2（full set, text + MM） | 84.2 / 84.9（with toolsはS37の値） | 77.3 |
| Gemini 3.1 Pro（S38・S37の比較列） | 54.2（既存S19と一致） | 70.3（S38）/ 73.8（S37）: 公式ボードまたはAAからの引用のため不採用 | 44.4（既存と一致） | 83.3 / 83.2 | 84.9（既存と一致） |

本文の条件:

- 「Terminal-Bench 2.1 results for Gemini 3.6 Flash and 3.5 Flash are self computed and for other models are reported from the public leaderboard and Artificial Analysis.」（S37）
- 「SWE-Bench Pro results for Gemini models are self-computed. Scores are averaged over 5x runs」（S38）
- 「SWE-Bench Pro results for Gemini models are self-computed, follow official provider methodology, using an internal version of the Antigravity harness」（S37）
- 「Humanity's Last Exam results for Gemini models are self-computed.」（S38）
- 「GDM-MRCR v2 includes 128k results as a cumulative score」（S37）

Googleの表にある他社列（Claude、GPT、Grok、Muse Spark）は、Googleが測定または引用した値です。各社の一次値としては転記していません。

## 確認したが変更しなかったもの

- **DeepSWE v1.1**: 公式JSONには到達できなかった。第三者のリポジトリ [lprhodes/deepswe-tracker](https://github.com/lprhodes/deepswe-tracker) の `data/official/leaderboard-live.json`（2026-10-09取得）を、[保存原本](deepswe-v1.1-20260910.json)と機械照合した。
  - 70設定の構成と採点値（model、harness、effort、pass_at_1、pass_at_4、n_passed、n_attempted、n_tasks_attempted、n_tasks_passed_any、CI、n_runsなど）は全件一致した。差分はGPT-6 Astraのコスト欄だけ。
  - `generated_at` は `2026-09-22T06:27:15.860279+00:00`。
  - 参考確認のため、S23の再確認日は更新していない。
- **SWE-bench Verified**: [`SWE-bench/swe-bench.github.io` の `data/leaderboards.json`](https://github.com/SWE-bench/swe-bench.github.io/blob/master/data/leaderboards.json) で、Verifiedの最新エントリは2026-02-26。表のモデルは掲載されていない。
- **LiveCodeBench**: 公式サイトのリポジトリの最終更新は2025-08-01で、最新リリースはv6。表のモデルは掲載されていない。
- **Anthropic（Opus 5.5 / Sonnet 5.5 / Fable 5.1）**: 発表ページの値は主表と一致した。システムカードのリンク先は既存のS25〜S27と同じファイル。

## 保留した値（区分B・C）

次回、一次ページに到達できる環境で確認してから転記します。値は確認前の手がかりで、比較ページには載せていません。

| 対象 | 候補値 | 読んだ経路 | 区分 |
| --- | --- | --- | --- |
| Claude Haiku 5.5 SWE-Bench Pro | 64.8 | 検索要約。システムカードTable 8.1.Aで要確認 | C |
| GLM-5.2 HLE 40.5 の扱い | text-onlyの可能性 | HFカードの脚注「By default, we report the text-only subset; results marked with * are from the full set.」（revision固定の第三者ミラー経由）。40.5には*がない。Z.aiのGitHubの図にはtext-onlyの記載がない | B |
| MiMo-V2.6-Pro / Flash（Xiaomi、09-21） | TB2.1 89.9 / 87.6、DeepSWE自己報告71.9 / 67.9 | HFカードのミラー。技術報告PDFのSHA-256はミラーが記録したHF LFS oidと一致 | B |
| Muse Spark 1.3（Meta、09-02） | TB2.1 88.8、MRCR 98.5（256K–512K）/ 98.1（512K–1M）、DeepSWE自己報告75.4 | スコアカード画像の第三者による転記。ハーネス表記が転記元ごとに異なる | B |
| Muse Spark 1.2 / 1.1 | 1.2: TB2.1 82.9、MRCR 66.3 / 55.5。1.1: HLE 52.2、TB2.1 80.0、SWE-Bench Pro 61.5 | 同上 | B |
| Grok 4.5 | SWE-Bench Pro 64.7、TB2.1 83.3 | x.ai記事の機械取得コピー（GoogleのS37も「各社の自己報告値」として同値を引用） | B |
| Reflection Beam（10-05） | SWE-Bench Pro 65.5、Verified 80.9、TB2.1 80.1、HLE 36.2（no tools）、GPQA 90.5、DeepSWE 44.4 | 記事の機械取得コピー | B |
| Cognition SWE-2 / Fireworks Ember-1 | SWE-2: TB2.1 92.8、DeepSWE 73.0。Ember-1: TB2.1 82.0、Verified 92.2、DeepSWE 75.2 | 記事の機械取得コピー | B |
| JetBrains Mellum2.1-12B-A2.5B-Thinking（10-08） | LCB v6 82.0、Verified 47.0、SWE-Bench Pro 28.0、TB2.1 17.4、GPQA 64.6 | HF model-indexの写し | B |
| Mistral Large 4 Preview（10-06） | DeepSWE 61.7（主表の行に対応する値はない） | 記事本文の写し | B |
| DeepSeek-V4-Flash-0731 GPQA | 89.9（V4.1技術報告の「DS-V4-Flash」列。0731だとするのは推定） | HFカードのミラー | B |
| DeepSeek V4-Pro / V4-Flash DeepSWE自己報告 | 62.7 / 54.4（「DeepSWE v1.1」と明記） | 同上 | B |
| Artificial Analysis（S9〜S11の新しい値・空欄） | 例: Haiku 5.5 HLE 44.4 / SciCode 55.0、Gemini 4 Argon HLE 57.1 / SciCode 61.8、Muse Spark 1.3 τ³ 50.5。空欄候補はτ³がFable 5 38.1・Opus 5 42.1・GPT-5.6 Sol 44.3、SciCodeがOpus 5 56.4・GPT-5.6 Sol 57.1 | AAデータの第三者ミラー2系統（10-10、値は一致）。主表の既存のAA値に変化はない | B |
| Artificial Analysisの構成変更 | SciCodeは「Under review」が継続。τ³-Bankingは9月にIndexから外れ、9/08以降に公開されたフロンティアモデルには値がない。Anthropicのconfig名が「(Max, Default Fallback)」に変更。slug `qwen3-8-max` は後継の0902版を指す | 検索要約とミラー | B/C |
| Epoch AI「DeepSWE v1.1 review」 | Flawed、113問中23問以上で偽陰性 | 検索要約。公開日も09-07説と09-17説で一致しない | C |
| OpenAI GPT-6 Sol / Luna（October） | ChatGPT向けの10月版snapshot。主表の行に対応する値はない | 検索要約 | C |
| Sakana Fugu v1.1（07-24）/ Fugu Max・Ultra v2（09-11） | 主表のFugu列はv1.0。新版の数値は画像にしかない | 検索要約 | C |
| SWE-Bench Pro V2のleaderboard | V2・HARD-51の値 | 検索要約。値に矛盾があり、v1の行とは別セット | C（不採用） |

## 採用しなかった値（区分を問わない）

- Terminal-Bench 4.0 / 3.0 / Scienceの値: Haiku 5.5 39.2、Gemini 4 Argon 57.4、Mythos 5.1 60.9、Gemini 3.7 Flash（3.0）14.9など。
- HLEのtext-only、HLE-Verified、HLE-Diamond、HLE-Rollingの値。
- SWE-Bench Pro V2、HARD-51、Reflection Beamの「v2-Hard」。
- AA-LCR v1.1の値。
- 自前ハーネスのτ³値（Atria 41.2、Beam 38.0）と、Sierraの音声版τ³（Gemini 3.8 Live 35.1）。
- 他社が測定した他社モデルの値。Google・Atria・Hy4・DeepSeek・Z.ai・Cognitionの各表の比較列が該当する。
- 学習曲線だけの値と、非公開チェックポイントの値。

この照合は掲載値の出典確認です。値がローカルで再現されること、未掲載のモデルに公表値が存在しないこと、全モデルが同じ問題・ハーネス・judge・effortで測定されたことを意味しません。
