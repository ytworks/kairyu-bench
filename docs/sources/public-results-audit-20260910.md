# 公開結果の出典照合（2026-09-10）

対象は既存12項目・19モデルの公開値102件と、DeepSWE v1.1公式原本の70設定です。欠測126セルは0で補完していません。公開値は各測定者の参照値で、Kairyuの実測や同条件の順位ではありません。

既存値100件を元の提供元の資料（Opus 4.8の2件は同社の後続システムカード）と照合しました。確認日と範囲は[カタログ](../../src/kairyu_bench/data/public_results.json)の各行にも記録しています。

| 出典 | 確認した箇所と結果 |
| --- | --- |
| [S1 Sakana](https://sakana.ai/fugu-release/) | [表画像](https://sakana.ai/assets/fugu-release/benchmark-table.png)のFugu/Fugu Ultra各11値を確認。発表日は6月22日であり、旧記載7月23日を修正。HLEは表の47.2/50.0を採用。同記事の[概要図](https://sakana.ai/assets/fugu-release/benchmark-fugu-grid.png)ではFuguが48.5で、資料内に不一致がある。 |
| [S3 Anthropic](https://www-cdn.anthropic.com/2f9323abbcc4abe219577539efe19a623c9ca2bd/Claude%20Fable%205%20%26%20Claude%20Mythos%205%20System%20Card.pdf) | 評価サマリー（PDF表示251ページ）でFable/Mythos/Opus 4.8のSWE・Terminal・HLE・CharXivを確認。8.8（印刷ページ258）でMythos GPQA 94.1、5試行平均を確認。 |
| [S4 OpenAI](https://openai.com/index/gpt-5-6/) | GPT-5.6 Sol/GPT-5.5のSWE Pro、Terminal 2.1、GPQA、MRCR値を本文表で確認。MRCRは256K–512K。 |
| [S5 DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) / [S13 Vision](https://api-docs.deepseek.com/news/news260821/) | Terminal 82.7/83.9をモデルカード・[公式画像](https://api-docs.deepseek.com/img/v4_260821_benchmark_en.png)で確認。DeepSeek Harness Minimal、temperature=1、top_p=0.95。DeepSWE表記にはv1.1の明示がないため、DatacurveのモデルIDと推測で結び付けない。 |
| [S6 Qwen MAX](https://qwen.ai/blog?id=qwen3.8) | Full Benchmark Tableの6値を確認。CharXiv 93.5はコード実行あり（なし88.4）と脚注に明記。SWE Proは修正済み問題、TerminalはClaude Code・10試行平均・5時間。 |
| [S7 GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) | GPQA、HLE no tools、SWE Pro、Terminal 2.1のTerminus-2列を確認。別agentの最高値を混ぜない。 |
| [S8 Kimi](https://huggingface.co/moonshotai/Kimi-K3) | 採用7値を確認。HLE/CharXivはツールなし、effort=max。AAによる別測定の値へ置き換えない。 |
| [S9 HLE](https://artificialanalysis.ai/evaluations/humanitys-last-exam) | ブラウザーで描画されたチャートのGPT-5.6 Sol / max 49.5を確認。 |
| [S10 SciCode](https://artificialanalysis.ai/evaluations/scicode) | Fable 5 / adaptive max / Opus 4.8 fallbackを60.2→61.0へ更新。モデル選択でGLM-5.2 / maxを表示し50.0→51.2へ更新。AAの対象は288 test subproblems。旧値はカタログのprevious_publicationに保存。 |
| [S11 Tau3](https://artificialanalysis.ai/evaluations/tau3-banking) | Qwen3.8 Max 51.3、GLM-5.3 / max 50.3を確認。[GLM-5.2 / max](https://artificialanalysis.ai/evaluations/tau3-banking?models=glm-5-2)は27.0→34.6へ更新。97問、反復平均のpass@1。 |
| [S14 Qwen Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) / [S16 Qwen 27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 各モデルカードの採用5値/6値を確認。HLEのjudgeはGPT-4o。CharXivはコード実行あり。SWE Proは修正済み問題のClaude Code評価。追加DeepSWE報告は比較文書の別表に条件付きで掲載。 |
| [S15 GLM-5.3](https://z.ai/blog/glm-5.3) | 本文表でTerminal 88.2、HLE tools 62.5を確認。TerminalはClaude Code 2.1.207、HLE judgeはGPT-5.6-luna medium。 |
| [S17 Opus 4.8](https://www.anthropic.com/news/claude-opus-4-8) / [S18 Opus 5](https://www-cdn.anthropic.com/b514064af1408018e64b1ad24e7d5e75850b4ffd/Claude%20Opus%205%20System%20Card.pdf) | S18 pp.148–149でOpus 5のSWE Pro79.2/Verified96.0/HLE56.3、Fable HLE56.5を確認。S17のOpus 4.8 Pro69.2/HLE49.8もS18で照合。Opus 5の既定はadaptive max、5試行平均。 |
| [S19 Gemini](https://deepmind.google/models/gemini/pro/) / [S20 Gemma](https://ai.google.dev/gemma/docs/core/model_card_4) | Geminiの6値、Gemma 31Bの4値を確認。Gemini SWE ProはPublic、LCB ProはEloなので%表には不採用。MRCRは8-needle 128K。 |
| [S21 Astra](https://openai.com/index/gpt-6-astra/) | GPQA96.0、HLE tools57.2、MRCR 256K–512Kの100.0を確認。項目ごとにeffortをまたいだ最大値で、固定effortの一括測定ではない。 |
| [S23 Datacurve](https://deepswe.datacurve.ai/artifacts/v1.1/leaderboard-live.json) | [保存原本](deepswe-v1.1-20260910.json)の28モデル・70設定を全件照合。原本SHA、モデル、effort、分子/分母、採点問数、反復数、pass@1、pass@4、CIを生成器で検査。 |

今回の値として確定しなかった2件は次のとおりです。表中に元の確認日を付け、更新済みの値と区別しています。

| 保持した過去値 | 理由 |
| --- | --- |
| GLM-5.2 Long Context Reasoning 71.0（2026-08-11、[S12](https://artificialanalysis.ai/models/glm-5-2)） | 現行ページはAA-LCR v1.1へ移行しており、旧版の71.0を再確認できない。新版の値を旧版として転記しない。 |
| GPT-6 Astra Terminal-Bench 2.1 87.4（2026-09-05、[S22](https://www.tbench.ai/?version=2.1)） | 元のPRの公式2.1リーダーボード/APIと[公開ジョブ](https://hub.harborframework.com/jobs/17d1a7f6-3339-4670-8b70-3b145979f57f)の照合結果を保持。今回のWeb取得は4.0の画面を返し、2.1の元行を再取得できなかった。 |

この照合は掲載値の帰属確認です。原本の数値がローカルで再現されたこと、未掲載モデルに公表値が存在しないこと、全モデルが同じ問題・ハーネス・judge・effortを使ったことは意味しません。
