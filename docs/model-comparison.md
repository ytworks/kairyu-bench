# 他モデルの公開ベンチマーク比較

他モデルが公開しているスコアの参照表です。スコアの単位はすべて `%`。`—` は採用した出典で対応する公開値を未確認の項目であり、0点ではありません。DeepSWE行はDatacurve公式原本のみを使い、提供元による別条件の値は下に分けて記載しています。

初版の値と出典は [`kairyu@9a00f39` の参照カタログ](https://github.com/ytworks/kairyu/blob/9a00f39cd60581d70baadb76d7df2737671e790c/kairyu/bench/reference.py) から転記しましたが、2026-08-27 に全列を一次ソース(各社の発表ページ・システムカード・モデルカード)と照合し、Fable 5列の誤値を修正のうえ、Mythos 5 / Opus 5 / Opus 4.8 / GPT-5.5 / Gemini 3.1 Pro / Gemma 4 / DeepSeek-V4-Flash-Vision-Exp / Qwen3.8-27B / Qwen3.8-Flash-Next / GLM-5.3 の列を追加しました。各提供元や第三者機関が異なる条件で測定した値を含むため、`kairyu-bench` の実測値と厳密に同条件とは限りません。

2026-09-05にGPT-6 Astra列を追加しました。OpenAI発表の評価表を本文で確認し、GPQA Diamond、HLE（ツールあり）、MRCR v2を転記しました。Terminal-Bench 2.1は公式リーダーボードのCodex / highの値です。調査した出典で確認できなかった項目は空欄のままです。

2026-09-10に既存の公開値102件中100件を再照合し、SciCodeのFable 5を60.2→61.0、GLM-5.2を50.0→51.2、Tau3 BankingのGLM-5.2を27.0→34.6へ更新しました。残る2件は表中に元の確認日を表示しています。[照合記録](sources/public-results-audit-20260910.md)に対象ページ・条件・保留理由をまとめています。

2026-09-12にDeepSWEで公開結果がある18モデルの列を追加し、主表を37モデルへ拡張しました。DeepSWEの全28モデルを主表に揃え、最高pass@1とその推論設定を併記しています。追加列の他ベンチマークは未照合のため `—` としています。

2026-09-30に、9月公開の8モデル（Opus 5.5 / Sonnet 5.5 / Fable 5.1 / GPT-6.1 Sol / GPT-6 Sol / GPT-6 Luna / DeepSeek-V4.1-Flash / Grok 4.7）の列を追加し、主表を45モデルへ拡張しました。同月公開で既存列のGPT-6 AstraとGemini 3.8 Flashは、空欄だった項目のうち確認できた値だけを追記しています。既存の値は変更していません。新モデルの発表資料の多くはTerminal-Bench 4.0・FrontierCode・CursorBench・OSWorld等へ移行しており、本表の行に対応する値は限られます。出典・評価条件・採用しなかった値は[2026-09-30の追加について](#2026-09-30の追加について)にまとめています。

2026-10-11に、Haiku 5.5 / Atria-Dawn-Preview / Hy4 previewの列を追加し、主表を48モデルへ拡張しました。既存列のGemini 3.7 Flash / 3.6 Flash / 3.5 Flash / 3.1 ProとGLM-5.3 Flashでは、空欄だった項目のうち各社の一次資料で確認できた値だけを追記しています。既存の値は変更していません。今回の調査環境ではHugging Face・OpenAI・Artificial Analysis・Terminal-Bench公式・Datacurve公式などに到達できませんでした。第三者による写しでしか読めなかった値（Muse Spark 1.3、MiMo-V2.6、Reflection Beam、Artificial Analysisの新しい値など）は転記を保留しています。詳細は[2026-10-11の追加について](#2026-10-11の追加について)と[照合記録](sources/public-results-audit-20261011.md)にまとめています。

| Benchmark | Fugu | Fugu Ultra | Opus 5.5 | Sonnet 5.5 | Haiku 5.5 | Fable 5.1 | Fable 5 | Mythos 5 | Opus 5 | Opus 4.8 | GPT-6 Astra | GPT-6.1 Sol | GPT-6 Sol | GPT-6 Luna | GPT-5.6 Sol | GPT-5.5 | Gemini 3.1 Pro | Gemma 4 31B | DeepSeek-V4.1-Flash | DeepSeek-V4-Flash-0731 | DeepSeek-V4-Flash-Vision-Exp | Qwen3.8 MAX | Qwen3.8-27B | Qwen3.8-Flash-Next | GLM-5.2 | GLM-5.3 | Atria-Dawn-Preview | Kimi K3 | Hy4 preview | Grok 4.7 | Gemini 3.8 Flash | GPT-5.6 Terra | Grok 4.6 | GPT-5.6 Luna | Gemini 3.7 Flash | GLM-5.3 Flash | DeepSeek-V4-Pro | Muse Spark 1.2 | Sonnet 5 | Grok 4.5 | DeepSeek-V4-Flash | Muse Spark 1.1 | GPT-5.4 | Gemini 3.6 Flash | Gemini 3.5 Flash | Kimi K2.7 Code | Sonnet 4.6 | Gemini 3.1 Pro Preview |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SWE-Bench Pro | [59.0][S1] | [73.7][S1] | [89.9][S25] | [81.3][S27] | — | [81.2][S26] | [80.0][S3] | [80.3][S3] | [79.2][S18] | [69.2][S17] | — | — | — | — | [64.6][S4] | [59.4][S4] | [54.2][S19] | — | — | — | — | [67.7][S6] | [61.7][S16] | [62.5][S14] | [62.1][S7] | — | [59.6][S34] | — | [65.7][S35] | — | — | — | — | — | — | — | — | — | — | — | — | — | — | [58.7][S37] | [53.9][S38] | — | — | — |
| SWE-bench Verified | — | — | — | — | — | — | [95.0][S3] | [95.5][S3] | [96.0][S18] | [88.6][S3] | — | — | — | — | — | — | [80.6][S19] | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Terminal-Bench 2.1 | [80.2][S1] | [82.1][S1] | — | — | — | — | [84.3][S3] | [88.0][S3] | — | [82.7][S3] | [87.4 (2026-09-05)][S22] | — | — | — | [88.8][S4] | [85.6][S4] | — | — | [90.6][S31] | [82.7][S5] | [83.9][S13] | [86.6][S6] | [73.0][S16] | — | [81.0][S7] | [88.2][S15] | [78.3][S34] | [88.3][S8] | [85.4][S35] | — | [89.4][S32] | — | — | — | [85.8][S36] | [84.3][S39] | — | — | — | — | — | — | — | [78.0][S37] | [76.2][S38] | — | — | — |
| LiveCodeBench | [92.9][S1] | [93.2][S1] | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | [80.0][S20] | — | — | — | — | [90.3][S16] | [91.9][S14] | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| LiveCodeBench Pro | [87.8][S1] | [90.8][S1] | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| HLE | [47.2][S1] | [50.0][S1] | [64.4][S25] | [56.9][S27] | [45.9][S33] | [60.9][S26] | [56.5][S18] | [59.0][S3] | [56.3][S18] | [49.8][S17] | [57.2 (tools)][S21] | [52.9][S9] | [47.9][S9] | [38.5][S9] | [49.5][S9] | [41.4][S3] | [44.4][S19] | [19.5][S20] | [36.8][S31] | — | — | [43.6][S6] | [30.8][S16] | [35.9][S14] | [40.5][S7] | [62.5 (tools)][S15] | — | [43.5][S8] | — | [43.1][S9] | [47.8][S9] | — | — | — | — | [55.3 (tools)][S39] | — | — | — | — | — | — | — | — | [40.2][S38] | — | — | — |
| CharXiv Reasoning | [85.1][S1] | [86.6][S1] | — | — | — | — | — | [88.9][S3] | — | [80.5][S3] | — | — | — | — | — | — | [83.3][S38] | — | — | — | — | [93.5][S6] | [90.2][S16] | [90.6][S14] | — | — | — | [84.8][S8] | — | — | [86.2][S32] | — | — | — | [84.5][S36] | — | — | — | — | — | — | — | — | [85.2][S37] | [84.2][S38] | — | — | — |
| GPQA Diamond | [95.5][S1] | [95.5][S1] | — | — | — | — | — | [94.1][S3] | — | — | [96.0][S21] | — | — | — | [94.6][S4] | [93.6][S4] | [94.3][S19] | [84.3][S20] | [90.9][S31] | — | — | [92.6][S6] | [89.2][S16] | [91.7][S14] | [91.2][S7] | — | — | [93.5][S8] | [92.3][S35] | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| SciCode | [60.1][S1] | [58.7][S1] | [66.9][S10] | [61.0][S10] | — | [63.1][S10] | [61.0][S10] | — | — | — | [56.5][S10] | [54.2][S10] | [57.6][S10] | [54.6][S10] | — | — | [59.0][S19] | — | [51.9][S10] | — | — | — | — | — | [51.2][S10] | — | — | [58.7][S8] | — | [57.4][S10] | [56.6][S10] | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| τ-bench Banking | [21.7][S1] | [20.6][S1] | — | — | — | [47.2][S11] | — | — | — | — | [41.4][S11] | — | — | — | — | — | — | — | — | — | — | [51.3][S11] | — | — | [34.6][S11] | [50.3][S11] | — | [33.4][S8] | — | — | [44.9][S11] | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Long Context Reasoning | [74.7][S1] | [73.3][S1] | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | [71.0 (2026-08-11)][S12] | — | — | [74.7][S8] | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| MRCR v2 | [86.6][S1] | [93.6][S1] | — | — | — | — | — | — | — | — | [100.0][S21] | — | — | — | [91.5][S4] | [81.5][S4] | [84.9][S19] | [66.4][S20] | — | — | — | [92.9][S6] | — | — | — | — | — | — | — | — | — | — | — | — | [97.0][S36] | — | — | — | — | — | — | — | — | [91.8][S37] | [77.3][S38] | — | — | — |
| DeepSWE v1.1 | — | — | — | — | — | — | [69.91 (xhigh)][S23] | — | [73.65 (max)][S23] | [58.97 (max)][S23] | [74.12 (xhigh)][S23] | — | — | — | [72.67 (max)][S23] | [67.04 (xhigh)][S23] | — | — | — | — | — | [57.46 (xhigh)][S23] | — | — | [43.78 (max)][S23] | [68.96 (max)][S23] | — | [68.51 (max)][S23] | — | — | [73.83 (high)][S23] | [69.62 (max)][S23] | [67.48 (medium)][S23] | [67.19 (max)][S23] | [65.49 (medium)][S23] | [63.39 (max)][S23] | [62.83 (max)][S23] | [54.87 (xhigh)][S23] | [53.85 (max)][S23] | [53.76 (high)][S23] | [53.32 (max)][S23] | [53.32 (xhigh)][S23] | [51.77 (xhigh)][S23] | [46.68 (high)][S23] | [36.06 (high)][S23] | [30.53 (unspecified)][S23] | [29.93 (high)][S23] | [11.73 (high)][S23] |


## DeepSWE v1.1（2026-09-30再確認）

Datacurveの公式JSON（`generated_at: 2026-09-03T22:24:37.984682+00:00`）にある28モデル・70設定を、分子・分母・effort・pass@4・信頼区間・取得日・出典URL付きで[参照カタログ](../src/kairyu_bench/data/public_results.json)へ保存しています。2026-09-12に再取得したJSONは[2026-09-10取得原本](sources/deepswe-v1.1-20260910.json)とSHA-256が一致したため、原本・取得日を保持し、本ページに再確認日を記録しました。2026-09-30の再取得では `generated_at` が `2026-09-22T06:27:15.860279+00:00` に更新されていましたが、70設定の採点値（モデル・effort・成功数・採点数・対象問題数・pass@1・pass@4・CI上下限・反復数）は保存原本と一致し、9月公開の新モデルはまだ掲載されていませんでした。このため主表の新モデル列のDeepSWE行は `—` とし、提供元の自己報告は下の別表に分けています。下表には全70設定を掲載し、主表には各モデルの最高pass@1の設定を表示します。同点なら原本の先の行を採用します。effortが異なる値を同条件の順位と解釈しないでください。[S23]

データセットは113問、各設定は4反復です。Opus 4.8 / maxは採点対象111問・429試行で、問題数も他の行と異なります。pass@1は成功試行/採点試行で、API・verifier・network障害の除外により分母が452未満の行があります。context超過とagent timeoutは失敗として数えます。pass@4は一度でも成功した問題数/対象問題数です。CIは公式の上下限を転記した95%区間で、半幅は4反復のスコア間の標準誤差 `1.96 × std(runs) / sqrt(4)` です。スコアとCIは小数第2位に丸めています。`unspecified` は原本のreasoning_effortが `null`（未指定）を意味します。[S23]

公開評価はPier / mini-swe-agent / Modal、本runnerは固定Pier / mini-swe-agent 2.4.6 / Dockerです。公開側の正確なcommit・agent版・image digestが一致すると確認できないため、参照値はローカル実測の平均値やdeltaへ混ぜません。v1.1はcommit済みの変更を独立した検証環境で採点し、v1の結果とは区別します。[S24]

`deepseek-v4-flash` は主表の `DeepSeek-V4-Flash`、`gemini-3-1-pro-preview` は `Gemini 3.1 Pro Preview` に対応します。日付付きの `DeepSeek-V4-Flash-0731` や `DeepSeek-V4-Flash-Vision-Exp`、preview表記のない `Gemini 3.1 Pro` との同一性は確認できないため、それぞれ別列にしています。[S23]

2026-10-11追記: Googleの評価PDFは、Gemini 3.1 Proの結果をmodel-id `gemini-3.1-pro-preview` で測定したと明記しています。つまり `Gemini 3.1 Pro` 列（S19）の値もpreview IDによる測定です。列の構成は変えていませんが、両列は同じmodel IDの測定です。同PDFは、S19のSciCode 59%がArtificial Analysisからの引用であることも明記しています([S45])。

2026-10-11には公式JSONに到達できませんでした。そこで、第三者が公式JSONを日次で取得しているリポジトリ（[lprhodes/deepswe-tracker](https://github.com/lprhodes/deepswe-tracker)、2026-10-09取得分）の写しを保存原本と機械照合しました。70設定の採点値はすべて一致し、新しいモデルの掲載もありませんでした（`generated_at` は2026-09-22のまま）。ただし第三者の写しによる参考確認なので、S23の再確認日は更新していません。

公式リポジトリには、採点に関する未解決の報告が2件あります。いずれも2026-10-11時点でDatacurveの回答はなく、公式原本の値への影響は確定していません([S43], [S44])。

- [#105](https://github.com/datacurve-ai/deep-swe/issues/105)（2026-10-02）: パッチがテスト基盤（例: `tests/conftest.py`）を書き換えると、機能を実装しなくてもreward 1になる事例を26件報告しています。
- [#107](https://github.com/datacurve-ai/deep-swe/issues/107)（2026-10-05）: `claude-code-by-agents-recursive-delegation` で、vitestのworkerがクラッシュすると全テスト合格と採点される不具合です。公開verifier出力のある合格134件のうち95件が該当し、うち47件以上は実際には不合格と推定されています。

本runnerは同じtask定義（commit `0b9fabb`）とverifierを使うため、ローカル実測にも同じ影響が出る可能性があります。

| Model ID | Effort | pass@1 (%) | Success / scored trials | Scored tasks | pass@4 (%) | Passed / scored tasks | 95% CI (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt-6-astra` | xhigh | [74.12][S23] | 335/452 | 113/113 | 80.53 | 91/113 | 71.25–76.98 |
| `gemini-3-8-flash` | high | [73.83][S23] | 330/447 | 113/113 | 85.84 | 97/113 | 72.41–75.24 |
| `claude-opus-5` | max | [73.65][S23] | 327/444 | 113/113 | 88.50 | 100/113 | 69.78–77.52 |
| `gpt-6-astra` | high | [73.23][S23] | 331/452 | 113/113 | 82.30 | 93/113 | 69.81–76.65 |
| `gpt-6-astra` | max | [73.23][S23] | 331/452 | 113/113 | 79.65 | 90/113 | 72.40–74.06 |
| `claude-opus-5` | xhigh | [73.15][S23] | 327/447 | 113/113 | 85.84 | 97/113 | 70.09–76.22 |
| `claude-opus-5` | high | [72.83][S23] | 327/449 | 113/113 | 87.61 | 99/113 | 70.88–74.77 |
| `gpt-6-astra` | medium | [72.79][S23] | 329/452 | 113/113 | 82.30 | 93/113 | 70.20–75.38 |
| `gpt-5-6-sol` | max | [72.67][S23] | 327/450 | 113/113 | 85.84 | 97/113 | 69.84–75.50 |
| `gemini-3-8-flash` | medium | [71.02][S23] | 321/452 | 113/113 | 83.19 | 94/113 | 68.74–73.30 |
| `gpt-5-6-sol` | xhigh | [70.73][S23] | 319/451 | 113/113 | 85.84 | 97/113 | 69.91–71.55 |
| `claude-fable-5` | xhigh | [69.91][S23] | 316/452 | 113/113 | 88.50 | 100/113 | 66.67–73.16 |
| `claude-fable-5` | max | [69.72][S23] | 304/436 | 113/113 | 84.07 | 95/113 | 65.69–73.76 |
| `gpt-5-6-terra` | max | [69.62][S23] | 314/451 | 113/113 | 88.50 | 100/113 | 67.07–72.18 |
| `gpt-5-6-sol` | high | [69.40][S23] | 313/451 | 113/113 | 86.73 | 98/113 | 67.97–70.83 |
| `glm-5-3` | max | [68.96][S23] | 311/451 | 113/113 | 87.61 | 99/113 | 65.94–71.98 |
| `claude-opus-5` | medium | [68.90][S23] | 308/447 | 113/113 | 89.38 | 101/113 | 67.73–70.08 |
| `claude-fable-5` | high | [68.60][S23] | 295/430 | 113/113 | 86.73 | 98/113 | 67.48–69.73 |
| `kimi-k3` | max | [68.51][S23] | 309/451 | 113/113 | 89.38 | 101/113 | 63.98–73.05 |
| `grok-4-6` | medium | [67.48][S23] | 305/452 | 113/113 | 84.07 | 95/113 | 65.20–69.76 |
| `gpt-5-6-luna` | max | [67.19][S23] | 301/448 | 113/113 | 90.27 | 102/113 | 63.20–71.18 |
| `gpt-5-5` | xhigh | [67.04][S23] | 303/452 | 113/113 | 88.50 | 100/113 | 60.57–73.50 |
| `gpt-6-astra` | low | [67.04][S23] | 303/452 | 113/113 | 79.65 | 90/113 | 65.73–68.34 |
| `grok-4-6` | xhigh | [66.74][S23] | 301/451 | 113/113 | 84.96 | 96/113 | 64.56–68.92 |
| `gemini-3-7-flash` | medium | [65.49][S23] | 296/452 | 113/113 | 83.19 | 94/113 | 62.40–68.57 |
| `claude-fable-5` | medium | [65.37][S23] | 285/436 | 113/113 | 83.19 | 94/113 | 60.95–69.79 |
| `gemini-3-7-flash` | high | [65.27][S23] | 295/452 | 113/113 | 82.30 | 93/113 | 63.48–67.05 |
| `grok-4-6` | high | [65.19][S23] | 294/451 | 113/113 | 84.96 | 96/113 | 63.65–66.72 |
| `gpt-5-5` | high | [64.38][S23] | 291/452 | 113/113 | 90.27 | 102/113 | 61.26–67.50 |
| `glm-5-3-flash` | max | [63.39][S23] | 284/448 | 113/113 | 84.96 | 96/113 | 59.01–67.77 |
| `deepseek-v4-pro` | max | [62.83][S23] | 284/452 | 113/113 | 88.50 | 100/113 | 56.50–69.17 |
| `gpt-5-6-sol` | medium | [61.06][S23] | 276/452 | 113/113 | 80.53 | 91/113 | 59.48–62.65 |
| `gpt-5-6-terra` | xhigh | [60.18][S23] | 272/452 | 113/113 | 80.53 | 91/113 | 58.05–62.30 |
| `claude-fable-5` | low | [59.58][S23] | 258/433 | 113/113 | 81.42 | 92/113 | 56.79–62.38 |
| `claude-opus-4-8` | max | [58.97][S23] | 253/429 | 111/113 | 79.28 | 88/111 | 57.21–60.74 |
| `claude-opus-5` | low | [58.13][S23] | 261/449 | 113/113 | 84.96 | 96/113 | 55.80–60.46 |
| `qwen3-8-max` | xhigh | [57.46][S23] | 258/449 | 113/113 | 83.19 | 94/113 | 54.80–60.12 |
| `gpt-5-6-luna` | xhigh | [56.86][S23] | 257/452 | 113/113 | 78.76 | 89/113 | 54.69–59.03 |
| `muse-spark-1-2` | xhigh | [54.87][S23] | 248/452 | 113/113 | 81.42 | 92/113 | 52.74–56.99 |
| `claude-opus-4-8` | xhigh | [54.36][S23] | 243/447 | 113/113 | 80.53 | 91/113 | 50.65–58.08 |
| `gpt-5-5` | medium | [53.98][S23] | 244/452 | 113/113 | 77.88 | 88/113 | 51.43–56.54 |
| `claude-sonnet-5` | max | [53.85][S23] | 238/442 | 113/113 | 78.76 | 89/113 | 49.61–58.08 |
| `gemini-3-7-flash` | low | [53.76][S23] | 243/452 | 113/113 | 77.88 | 88/113 | 51.17–56.35 |
| `gpt-5-6-terra` | high | [53.76][S23] | 243/452 | 113/113 | 80.53 | 91/113 | 49.43–58.09 |
| `grok-4-5` | high | [53.76][S23] | 243/452 | 113/113 | 77.88 | 88/113 | 51.48–56.04 |
| `deepseek-v4-flash` | max | [53.32][S23] | 241/452 | 113/113 | 80.53 | 91/113 | 49.75–56.89 |
| `muse-spark-1-1` | xhigh | [53.32][S23] | 241/452 | 113/113 | 79.65 | 90/113 | 50.28–56.35 |
| `claude-opus-4-8` | high | [51.77][S23] | 234/452 | 113/113 | 77.88 | 88/113 | 47.21–56.33 |
| `gpt-5-4` | xhigh | [51.77][S23] | 234/452 | 113/113 | 77.88 | 88/113 | 50.27–53.27 |
| `claude-sonnet-5` | xhigh | [49.67][S23] | 224/451 | 113/113 | 75.22 | 85/113 | 46.21–53.12 |
| `claude-opus-4-8` | medium | [48.67][S23] | 220/452 | 113/113 | 76.11 | 86/113 | 46.43–50.91 |
| `claude-sonnet-5` | high | [48.23][S23] | 218/452 | 113/113 | 79.65 | 90/113 | 43.72–52.74 |
| `gemini-3-6-flash` | high | [46.68][S23] | 211/452 | 113/113 | 75.22 | 85/113 | 42.98–50.39 |
| `gpt-5-6-sol` | low | [45.35][S23] | 205/452 | 113/113 | 71.68 | 81/113 | 42.97–47.74 |
| `gpt-5-6-luna` | high | [44.25][S23] | 200/452 | 113/113 | 75.22 | 85/113 | 41.33–47.17 |
| `glm-5-2` | max | [43.78][S23] | 197/450 | 113/113 | 76.99 | 87/113 | 42.05–45.50 |
| `grok-4-6` | low | [41.65][S23] | 187/449 | 113/113 | 69.03 | 78/113 | 39.33–43.97 |
| `claude-opus-4-8` | low | [40.80][S23] | 184/451 | 113/113 | 68.14 | 77/113 | 39.33–42.26 |
| `claude-sonnet-5` | medium | [39.78][S23] | 179/450 | 113/113 | 64.60 | 73/113 | 36.65–42.91 |
| `glm-5-2` | high | [36.28][S23] | 164/452 | 113/113 | 68.14 | 77/113 | 31.53–41.03 |
| `gemini-3-5-flash` | high | [36.06][S23] | 163/452 | 113/113 | 63.72 | 72/113 | 32.10–40.03 |
| `gpt-5-6-terra` | medium | [35.11][S23] | 158/450 | 113/113 | 60.18 | 68/113 | 31.73–38.49 |
| `kimi-k2-7-code` | unspecified | [30.53][S23] | 138/452 | 113/113 | 61.06 | 69/113 | 30.03–31.03 |
| `claude-sonnet-5` | low | [30.51][S23] | 137/449 | 113/113 | 57.52 | 65/113 | 29.39–31.64 |
| `claude-sonnet-4-6` | high | [29.93][S23] | 135/451 | 113/113 | 56.64 | 64/113 | 25.84–34.03 |
| `gpt-5-5` | low | [26.99][S23] | 122/452 | 113/113 | 47.79 | 54/113 | 24.70–29.29 |
| `gpt-5-6-terra` | low | [24.05][S23] | 108/449 | 113/113 | 44.25 | 50/113 | 23.28–24.83 |
| `gemini-3-1-pro-preview` | high | [11.73][S23] | 53/452 | 113/113 | 28.32 | 32/113 | 10.24–13.21 |
| `gpt-5-6-luna` | medium | [11.28][S23] | 51/452 | 113/113 | 27.43 | 31/113 | 10.45–12.11 |
| `gpt-5-6-luna` | low | [1.55][S23] | 7/452 | 113/113 | 4.42 | 5/113 | 0.72–2.38 |

提供元による追加のDeepSWE v1.1報告もあります（上の5行は2026-09-10、続く8行は2026-09-30、末尾の5行は2026-10-11確認）。これらの値を公式原本の70設定に追加したり、主表のDeepSWE行や最高値の選択対象に混ぜたりはしていません。採点数・除外数・effortが未公表の項目は推定しません。

| Model | 公表スコア (%) | 出典と評価条件 |
| --- | ---: | --- |
| Qwen3.8-27B | [42.2][S16] | Claude Code、temperature=1、top_p=0.95、256K。反復数・採点数・CIは記載なし。 |
| Qwen3.8-Flash-Next | [58.7][S14] | Claude Codeとmini-swe-agentの高い方（このモデルはmini）。temperature=1、top_p=0.95、256K。反復数・採点数・CIは記載なし。 |
| Qwen3.8 MAX | [56.6][S6] | 両agentの高い方（このモデルはClaude Code）。temperature=1、top_p=0.95、256K。反復数・採点数・CIは記載なし。 |
| Opus 5 | [68.8][S18] | システムカードpp.148–149。max effort、5試行平均。公式原本の4反復とは別条件。 |
| GLM-5.3 | [66.9][S15] | mini-swe-agent、temperature=0.95、top_p=1、6h、400K。反復数・採点数・CIは記載なし。 |
| Opus 5.5 | [74.2][S25] | 2026-09-30確認。システムカード§8.3。adaptive thinking max effort、5試行平均。ハーネス・採点数・CIは記載なし。 |
| Sonnet 5.5 | [71.0][S27] | 2026-09-30確認。システムカード§8.3。adaptive thinking max effort、5試行平均。ハーネス・採点数・CIは記載なし。 |
| Fable 5.1 | [67.4][S26] | 2026-09-30確認。システムカード§8.3。max effort、5試行平均。隠しテストが単一の参照解に依存し、より厳密な実装が失敗した例があると注記。 |
| GPT-6.1 Sol | [75.2][S29] | 2026-09-30確認。発表ページの図のデータ値（high）。low 64.4 / medium 73.0 / xhigh 71.9 / max 71.9。反復数・採点数は記載なし。 |
| GPT-6 Sol | [68.8][S28] | 2026-09-30確認。max。GPT-6.1 Solの発表図ではlow 37.2〜max 68.8。反復数・採点数は記載なし。 |
| GPT-6 Luna | [66.6][S28] | 2026-09-30確認。max。反復数・採点数は記載なし。 |
| Grok 4.7 | [71.0][S30] | 2026-09-30確認。high effort（同表の他ベンチマークはxhigh）。反復数・採点数は記載なし。 |
| DeepSeek-V4.1-Flash | [74.2][S31] | 2026-09-30確認。mini-SWE harness、reasoning_effort=100、各問題8サンプル、temperature=1.0、top_p=0.95、1M。scaffold別では65.5〜74.2。 |
| Gemini 4 Argon | [77.9][S40] | 2026-10-11確認。Googleによる自己計測。mini-swe agentハーネス、最高のthinking設定。反復数・採点数・CIは記載なし。 |
| Gemini 3.8 Flash | [73.7][S32] | 2026-10-11確認。評価PDFの自己計測値。mini-swe agent、high thinking。公式原本ではhighが73.83。 |
| Kimi K3 | [67.5][S41] | 2026-10-11確認。Kimi Codeハーネス。同じ脚注は、公式leaderboard上のmini-SWE-agentでの値を67.3としている。現行の公式原本ではmaxが68.51。 |
| Gemini 3.7 Flash | [65.3][S36] | 2026-10-11確認。Googleによる自己計測。mini swe agent、LiteLLM 1.96、high thinking。公式原本の最高値はmediumの65.49。 |
| GLM-5.3 Flash | [63.4][S39] | 2026-10-11確認。Z.aiの評価図の値。モデルカードの脚注ではmini-swe-agent、temperature=0.95、top_p=1.0、6h、400K（脚注は第三者の写しでのみ確認）。公式原本ではmaxが63.39。 |

既存12行・19モデルの値と出典は同じJSONに保存されています。2026-09-12に追加した18モデルの他ベンチマークは未照合のため `—` としています（Gemini 3.8 Flashのみ2026-09-30に一部追記）。2026-09-30と2026-10-11に追加した列と値はカタログJSONへ登録しておらず、本ページだけに出典付きで記載しています。カタログでは欠測をnullとし、条件差を各行に記録しています。`verified-primary`は測定提供元の本文・表との今回の照合、`cross-checked-same-publisher`は同じ提供元の別資料での確認、`historical-transcription`は元の確認日を維持した過去値、`verified-against-snapshot`は保存したDeepSWE原本との照合です。元の取得日と今回の再確認日は区別し、変更した旧値もJSONに残しています。τ-benchの公開Tau3 Bankingとrunnerのtau2-bench、Long Context ReasoningとLongBench v2は特に区別が必要です。

以下の折りたたみには保存済みカタログの参照表を残しています。`python scripts/update_model_comparison.py` の再生成対象はこの2表です。上の拡張表と全設定の詳細表は、出典を確認して本ページへ追記したものです。

<details>
<summary>保存済みカタログの参照表（2026-09-10取得時）</summary>

19モデルの横断表と、DeepSWE各モデルの代表設定です。この横断表ではGemini 3.1 Pro列にpreview値を注記付きで掲載していますが、上の拡張表ではPreviewを独立列に分けています。

<!-- BEGIN public-comparison -->
| Benchmark | Fugu | Fugu Ultra | Fable 5 | Mythos 5 | Opus 5 | Opus 4.8 | GPT-6 Astra | GPT-5.6 Sol | GPT-5.5 | Gemini 3.1 Pro | Gemma 4 31B | DeepSeek-V4-Flash-0731 | DeepSeek-V4-Flash-Vision-Exp | Qwen3.8 MAX | Qwen3.8-27B | Qwen3.8-Flash-Next | GLM-5.2 | GLM-5.3 | Kimi K3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SWE-Bench Pro | [59.0][S1] | [73.7][S1] | [80.0][S3] | [80.3][S3] | [79.2][S18] | [69.2][S17] | — | [64.6][S4] | [59.4][S4] | [54.2][S19] | — | — | — | [67.7][S6] | [61.7][S16] | [62.5][S14] | [62.1][S7] | — | — |
| SWE-bench Verified | — | — | [95.0][S3] | [95.5][S3] | [96.0][S18] | [88.6][S3] | — | — | — | [80.6][S19] | — | — | — | — | — | — | — | — | — |
| Terminal-Bench 2.1 | [80.2][S1] | [82.1][S1] | [84.3][S3] | [88.0][S3] | — | [82.7][S3] | [87.4 (2026-09-05)][S22] | [88.8][S4] | [85.6][S4] | — | — | [82.7][S5] | [83.9][S13] | [86.6][S6] | [73.0][S16] | — | [81.0][S7] | [88.2][S15] | [88.3][S8] |
| LiveCodeBench | [92.9][S1] | [93.2][S1] | — | — | — | — | — | — | — | — | [80.0][S20] | — | — | — | [90.3][S16] | [91.9][S14] | — | — | — |
| LiveCodeBench Pro | [87.8][S1] | [90.8][S1] | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| HLE | [47.2][S1] | [50.0][S1] | [56.5][S18] | [59.0][S3] | [56.3][S18] | [49.8][S17] | [57.2 (tools)][S21] | [49.5][S9] | [41.4][S3] | [44.4][S19] | [19.5][S20] | — | — | [43.6][S6] | [30.8][S16] | [35.9][S14] | [40.5][S7] | [62.5 (tools)][S15] | [43.5][S8] |
| CharXiv Reasoning | [85.1][S1] | [86.6][S1] | — | [88.9][S3] | — | [80.5][S3] | — | — | — | — | — | — | — | [93.5][S6] | [90.2][S16] | [90.6][S14] | — | — | [84.8][S8] |
| GPQA Diamond | [95.5][S1] | [95.5][S1] | — | [94.1][S3] | — | — | [96.0][S21] | [94.6][S4] | [93.6][S4] | [94.3][S19] | [84.3][S20] | — | — | [92.6][S6] | [89.2][S16] | [91.7][S14] | [91.2][S7] | — | [93.5][S8] |
| SciCode | [60.1][S1] | [58.7][S1] | [61.0][S10] | — | — | — | — | — | — | [59.0][S19] | — | — | — | — | — | — | [51.2][S10] | — | [58.7][S8] |
| τ-bench Banking | [21.7][S1] | [20.6][S1] | — | — | — | — | — | — | — | — | — | — | — | [51.3][S11] | — | — | [34.6][S11] | [50.3][S11] | [33.4][S8] |
| Long Context Reasoning | [74.7][S1] | [73.3][S1] | — | — | — | — | — | — | — | — | — | — | — | — | — | — | [71.0 (2026-08-11)][S12] | — | [74.7][S8] |
| MRCR v2 | [86.6][S1] | [93.6][S1] | — | — | — | — | [100.0][S21] | [91.5][S4] | [81.5][S4] | [84.9][S19] | [66.4][S20] | — | — | [92.9][S6] | — | — | — | — | — |
| DeepSWE v1.1 | — | — | [69.91][S23] | — | [73.65][S23] | [58.97][S23] | [74.12][S23] | [72.67][S23] | [67.04][S23] | [11.73 (preview)][S23] | — | — | — | [57.46][S23] | — | — | [43.78][S23] | [68.96][S23] | [68.51][S23] |
<!-- END public-comparison -->

<!-- BEGIN deepswe-comparison -->
| Model ID | Effort | pass@1 (%) | Success / scored trials | Scored tasks | pass@4 (%) | 95% CI half (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gpt-6-astra` | xhigh | [74.12][S23] | 335/452 | 113/113 | 80.53 | ±2.87 |
| `gemini-3-8-flash` | high | [73.83][S23] | 330/447 | 113/113 | 85.84 | ±1.42 |
| `claude-opus-5` | max | [73.65][S23] | 327/444 | 113/113 | 88.50 | ±3.87 |
| `gpt-5-6-sol` | max | [72.67][S23] | 327/450 | 113/113 | 85.84 | ±2.83 |
| `claude-fable-5` | xhigh | [69.91][S23] | 316/452 | 113/113 | 88.50 | ±3.24 |
| `gpt-5-6-terra` | max | [69.62][S23] | 314/451 | 113/113 | 88.50 | ±2.56 |
| `glm-5-3` | max | [68.96][S23] | 311/451 | 113/113 | 87.61 | ±3.02 |
| `kimi-k3` | max | [68.51][S23] | 309/451 | 113/113 | 89.38 | ±4.54 |
| `grok-4-6` | medium | [67.48][S23] | 305/452 | 113/113 | 84.07 | ±2.28 |
| `gpt-5-6-luna` | max | [67.19][S23] | 301/448 | 113/113 | 90.27 | ±3.99 |
| `gpt-5-5` | xhigh | [67.04][S23] | 303/452 | 113/113 | 88.50 | ±6.47 |
| `gemini-3-7-flash` | medium | [65.49][S23] | 296/452 | 113/113 | 83.19 | ±3.09 |
| `glm-5-3-flash` | max | [63.39][S23] | 284/448 | 113/113 | 84.96 | ±4.38 |
| `deepseek-v4-pro` | max | [62.83][S23] | 284/452 | 113/113 | 88.50 | ±6.33 |
| `claude-opus-4-8` | max | [58.97][S23] | 253/429 | 111/113 | 79.28 | ±1.76 |
| `qwen3-8-max` | xhigh | [57.46][S23] | 258/449 | 113/113 | 83.19 | ±2.66 |
| `muse-spark-1-2` | xhigh | [54.87][S23] | 248/452 | 113/113 | 81.42 | ±2.12 |
| `claude-sonnet-5` | max | [53.85][S23] | 238/442 | 113/113 | 78.76 | ±4.24 |
| `grok-4-5` | high | [53.76][S23] | 243/452 | 113/113 | 77.88 | ±2.28 |
| `deepseek-v4-flash` | max | [53.32][S23] | 241/452 | 113/113 | 80.53 | ±3.57 |
| `muse-spark-1-1` | xhigh | [53.32][S23] | 241/452 | 113/113 | 79.65 | ±3.04 |
| `gpt-5-4` | xhigh | [51.77][S23] | 234/452 | 113/113 | 77.88 | ±1.50 |
| `gemini-3-6-flash` | high | [46.68][S23] | 211/452 | 113/113 | 75.22 | ±3.70 |
| `glm-5-2` | max | [43.78][S23] | 197/450 | 113/113 | 76.99 | ±1.73 |
| `gemini-3-5-flash` | high | [36.06][S23] | 163/452 | 113/113 | 63.72 | ±3.97 |
| `kimi-k2-7-code` | unspecified | [30.53][S23] | 138/452 | 113/113 | 61.06 | ±0.50 |
| `claude-sonnet-4-6` | high | [29.93][S23] | 135/451 | 113/113 | 56.64 | ±4.09 |
| `gemini-3-1-pro-preview` | high | [11.73][S23] | 53/452 | 113/113 | 28.32 | ±1.48 |
<!-- END deepswe-comparison -->

</details>

## 2026-10-11の追加について

2026-09-30以降に公開された資料を調べ、一次資料を直接読めた値だけを主表の対応する行に転記しました。今回の調査環境では、Hugging Face、OpenAI、Artificial Analysis、Terminal-Bench公式、Datacurve公式、x.ai、Metaなどに到達できませんでした。到達できたのはAnthropic、Googleのstorage.googleapis.com、GitHubだけです。第三者の写しでしか読めなかった値は保留とし、[照合記録](sources/public-results-audit-20261011.md)に一覧で残しています。

- **Claude Haiku 5.5（2026-10-07公開）**: HLEは発表ページの表にあるno tools値45.9です。同ページのeffort別グラフのMaxの点（Low 30.9〜Max 45.9）と一致します。with toolsの57.4は主表に入れていません。システムカードには到達できず、SWE-Bench Proなどは確認できていません([S33])。
- **Atria-Dawn-Preview（上海AI Lab。GLM-5.2をベースにしたopen-weight）**:
  - SWE-Bench ProはOpenHands v1.23.0での値です。ネットワーク無効、4.5時間・250ステップの制限で測定され、試行数とsubsetの記載はありません。
  - TB2.1はClaude Code v2.1.224で、4回の平均です。
  - 同じ表のτ³-Bench Banking 41.2は自前のtau2系ハーネス（user simulator GPT-5.2）による値です。Artificial AnalysisのTau3 Bankingではないため転記していません([S34])。
- **Hy4 preview（Tencent、open-weight、2026-08公開）**: 9月の調査で見落としていたため追加しました。
  - SWE-Bench Proはswe-agent scaffold、TB2.1はClaude Code（最大500ターン、12時間）での値です。GPQAを含め、いずれも最高のreasoning設定です。
  - HLEはtext-onlyの値しかないため転記していません。比較列にある他社モデルの値（*付き）はTencentによる測定なので採用していません([S35])。
- **Gemini 3.7 / 3.6 / 3.5 Flash、Gemini 3.1 Pro**: 各モデルの評価PDFの結果表（画像）から転記しました([S36], [S37], [S38])。
  - TB2.1はTerminus-2によるGoogleの自己計測、CharXivはno tools、MRCR v2は8-needle 128K averageです。
  - SWE-Bench Pro（Public）は社内版Antigravityハーネスによる自己計測で、3.5 Flashは5回の平均です。
  - 3.5 FlashのSWE-Bench Proは、3.6 FlashのPDFでは55.1と再掲されています。資料間で一致しないため、主表には3.5 Flash自身のPDFにある53.9を採用しました。
  - 3.5 FlashのHLEは「full set, text + MM」の値で、no toolsの明記はありません。
  - Gemini 3.1 ProのCharXiv 83.3は、3.5 Flashと3.6 Flashの両PDFで一致するGoogleの自己計測値です。
- **GLM-5.3 Flash**: Z.aiのGitHubにある評価図から、TB2.1の84.3とHLE with toolsの55.3を転記しました。条件（TB2.1はClaude Code 2.1.207、HLEはfull setでjudgeがGPT-5.6-luna medium）はHugging Faceのモデルカードの脚注によるもので、今回は第三者の写しでのみ確認しています([S39])。
- **主表の行に対応する値がない新モデル**:
  - Gemini 4 Argon（2026-09-30）の評価PDFには、DeepSWE v1.1（自己計測、上の別表に掲載）以外に主表の行と対応する項目がありません([S40])。
  - Muse Spark 1.3、MiMo-V2.6-Pro / Flash、Reflection Beam、Mistral Large 4 Previewなどは、一次ページに到達できなかったため列を追加していません。
- **ベンチマーク側の変化**:
  - SWE-Bench ProのV2（642問、HARD-51 subset）が2026-09-22に公開され、Hugging Faceの既定configがV2になりました。主表のSWE-Bench Pro行と本runnerはv1（public 731問）を対象とし、V2の値は混ぜていません([S42])。
  - Terminal-Bench 2.1はコミュニティからの提出を停止しています（メンテナが実行した分だけを掲載）。公式ボードにも到達できなかったため、2026-09-03以降の新規掲載は確認できていません([S46])。
  - DeepSWE v1.1の採点不具合の報告（#105・#107）は、上のDeepSWE節に記載しています。
- **訂正**: 下の2026-09-30の節で、Mythos 5.1のTerminal-Bench 4.0を「61%」と記載していましたが、発表ページの値は60.9%です([S26])。

## 2026-09-30の追加について

9月公開モデルの値を一次資料と第三者の独立測定で確認し、主表の行に対応するものだけを転記しました。

- **Anthropic（Opus 5.5 / Sonnet 5.5 / Fable 5.1）**: SWE-Bench ProとHLEは各システムカードのTable 8.1.A（Opus 5.5はp.174、Sonnet 5.5はp.109、Fable 5.1はp.167）の値です。adaptive thinking・max effort・5試行平均が標準構成です。HLEはno tools値を採用し、with tools値（Opus 5.5 67.7、Sonnet 5.5 64.5、Fable 5.1 65.0）は主表に入れていません([S25], [S26], [S27])。
- Fable 5.1のHLE with toolsは、Fable 5.1のシステムカードでは65.0、Opus 5.5のシステムカードと発表ページでは65.6と記載されており、出典間で一致しません（主表はno toolsの60.9で、両資料とも同値）。
- 後続のシステムカードでは、既存列のOpus 5のHLE no toolsが56.6、Fable 5が57.8と再掲されています。主表は従来の出典の値（56.3 [S18] / 56.5 [S18]）のまま変更していません。
- Mythos 5.1は、主表の行に対応する単独の公表値がありません（Terminal-Bench 4.0の60.9%のみ。2026-10-11に61%から訂正）。このため列を追加していません([S26])。
- **OpenAI（GPT-6.1 Sol / GPT-6 Sol / GPT-6 Luna）**: 発表ページには主表の行に対応する評価表がありません。HLEとSciCodeはArtificial Analysisの独立測定（いずれもmax）です。DeepSWEの自己報告値は上の別表に記載しています([S28], [S29])。
- **Grok 4.7**: 発表ページで主表の行に対応するのはDeepSWE（high、別表）のみです。HLEとSciCodeはArtificial Analysisの独立測定（xhigh）です([S30])。
- **DeepSeek-V4.1-Flash**: 値はreasoning_effort=100（最大）、temperature=1.0、top_p=0.95です。Terminal-Bench 2.1はDeepSeek Harness（Minimal mode）、各問題3試行、ネットワークなしの値です（同条件のClaude Codeでは88.0、mini-SWEでは90.3）。HLEは全問の36.8で、text-onlyサブセットでは39.1、with toolsでは63.9です([S31])。
- **Gemini 3.8 Flash**: Terminal-Bench 2.1（Terminus 2）とCharXiv Reasoning（no tools）はGoogleによる自己計測です。モデルカードのHLE-Verified 54.9は、1,811問の別セットのためHLE行に入れていません。HLE行はArtificial Analysisの独立測定（high）です([S32])。
- **Artificial Analysisの値（S9 / S10 / S11）**: Anthropicモデルは「max with fallback」（セーフガード介入時は別モデルへフォールバック）です。GPT-6 Astra / GPT-6.1 Sol / GPT-6 Sol / GPT-6 Luna / DeepSeek-V4.1-Flashはmax、Grok 4.7はxhigh、Gemini 3.8 Flashはhighです。SciCodeは独立監査（SciCode-Verified）を受けて「Under review」となっており、データセット修正後に再測定される可能性があります。τ³-Bankingの表示21モデルにOpus 5.5・Sonnet 5.5・GPT-6系（Astra以外）・Grok 4.7・DeepSeek-V4.1-Flashは含まれていません。
- Artificial Analysisの長文脈評価はAA-LCR v1.1へ更新されています（例: Kimi K3は88.7）。既存のLong Context Reasoning行（v1時点の値）と比較できないため転記していません。
- Terminal-Bench 2.1公式リーダーボードを2026-09-30に再確認しましたが、GPT-6 Astra（2026-09-03掲載）より新しいモデルの掲載はありません([S22])。

## 2026-08-27の修正について

`kairyu` 参照カタログから転記していたFable 5列の4値を一次ソース照合により修正しました。

- SWE-Bench Pro 80.3 → **80.0**: システムカード([S3] Table 8.1.A)ではFable 5=80.0、80.3はMythos 5の値。Anthropicのlaunch記事[S2]の表は「Mythos 5 / Fable 5」統合列で高い方を表示していたため混同されていました。
- Terminal-Bench 2.1 88.0 → **84.3**: 88.0はMythos 5の値([S3])。旧出典(S8=Kimiモデルカード)の帰属も誤りでした。
- HLE 59.0 → **56.5**: 59.0はMythos 5(no tools)の値([S3])。Fable 5単体の56.5はOpus 5システムカード([S18])に記載。
- GPQA Diamond 91.3 → **削除**: どの公式資料にも確認できず。AnthropicはMythos 5の94.1のみ公表し([S3])、GPQA Diamondを飽和ベンチマークとして今後の報告を停止すると明言しています。

Mythos 5はFable 5と同一の基盤モデルで、セーフガード(一部領域でOpus 4.8へフォールバック)を外したものです。Fable 5の公表値は本番セーフガード込みの実測です([S3])。

## 比較時の注意

- HLEとCharXivは、このrunnerでは対象モデル自身による自己採点です。
- 表のHLE値は原則 no tools 測定です(Anthropic系・Gemini・Gemmaはno toolsと明記。ただしGemini 3.5 Flashの40.2は「full set, text + MM」とだけ記載され、no toolsの明記はありません)。GPT-6 Astraの57.2、GLM-5.3の62.5、GLM-5.3 Flashの55.3はツール使用あり(with tools)の公表値で、表中にも `tools` と明記しています。no tools値との直接比較には適しません。ツールあり同士もツール構成・ハーネスの一致は未確認です(ツールありのGLM-5.2はZ.aiのchartでは54.7)。
- GLM-5.2のHLE 40.5は、text-onlyサブセットの値である可能性があります。Z.aiのモデルカードの脚注に「既定はtext-only subset、*付きがfull set」とあり、40.5には*が付いていないためです。ただし2026-10-11時点では、この脚注を第三者の写しでしか確認できていません。値は変更せず、一次ページで再確認するまで保留しています。HLE-Verified、HLE-Diamond、ScaleのHLE-Rolling、text-onlyの値は、いずれも別の問題セットとして主表に入れていません。
- CharXiv ReasoningのMythos 5(88.9)とOpus 4.8(80.5)はno tools値(with toolsは各93.5 / 89.9)。CharXivのQwen3.8 MAXはコード実行あり93.5（なし88.4）で、Qwenによる一部正解アノテーションの修正もあります。Qwen3.8-Flash-Nextの90.6とQwen3.8-27Bの90.2("CharXiv RQ")は "With CI" として公開された値です。Gemini 3.1 Pro / 3.5 Flash / 3.6 Flash / 3.7 FlashはGoogle自己計測のno tools値です。with toolsの値は、3.1 Proが83.2、3.5 Flashが84.9、3.6 Flashが89.4、3.7 Flashが88.7です（Googleのwith toolsは検索とコード実行を使用）。
- τ-benchは、このrunnerでは対象モデル自身をuser simulatorにも使用します。
- Long Context Reasoning行の公開値に対し、このrunnerは代替としてLongBench v2を実行するため直接比較できません。
- MRCR v2の測定コンテキストは、GPTが8-needle 256K–512K、Qwenが256K、GeminiとGemmaが8-needle 128K、このrunnerは4K–128Kです。Gemini 3.5 / 3.6 / 3.7 Flashは8-needleの128K average（128Kまでの累積スコア）です。Googleは3.6 Flash以降、これを「GDM-MRCR v2」と表記しています。
- GPT-6 AstraのMRCR v2は8-needle 256K–512Kの100.0を採用しています。512K–1Mでは96.3であり、コンテキスト長を混同しないでください。OpenAIの発表値はreasoning effortをまたいで各評価項目で得られた最大値で、単一の固定effortを指定した一括測定とは扱えません。研究環境/APIと製品版ChatGPTではsystem promptやツールが異なり得ます([S21])。
- GPT-6 AstraのTerminal-Bench 2.1はCodex 0.151.0 / high、2026-09-03掲載、87.4 ± 1.8%（95%信頼区間）の公式リーダーボード値です([S22])。公開ジョブは各問題5試行・各effort 445試行で、このrunnerの既定agent（Terminus-2）・1試行と同条件ではありません。
- GPT-6 AstraのTerminal-Bench 4.0（57.9）とDeepSWE v1.1（74.1）は、Terminal-Bench 2.1やSWE-Bench Proへ転記していません([S21])。
- SWE-bench VerifiedのFable 5は5試行平均、このrunnerは1試行です。
- Terminal-Bench 2.1のOpus 4.8はlaunch時に74.6と公表され、その後のシステムカード([S3])で82.7に更新されています。Opus 5はTB2.1を公表せず後継のFrontierBench v0.1に移行。Gemini 3.1 ProについてGoogle自身が公表したのはTerminal-Bench 2.0の68.5だけです。後続のGemini 3.5 Flash / 3.6 Flashの評価PDFは3.1 ProのTB2.1を70.3 / 73.8と載せていますが、いずれも公式ボードやArtificial Analysisからの引用で、値も一致しません。このためTB2.1行は空欄のままです。
- TB2.1のハーネスは提供元ごとに異なります。Gemini 3.5 / 3.6 / 3.7 / 3.8 FlashはTerminus-2、Atria-Dawn-PreviewはClaude Code v2.1.224の4回平均、Hy4 previewとGLM-5.3 FlashはClaude Codeです。
- Gemini 3.1 ProのLiveCodeBench ProはElo(2887)での公表のため、%基準の本表には記載していません。
- Gemma 4の値は31B Dense(instruction-tuned)のものです。
- Qwen3.8-Flash-NextとQwen3.8-27BのLiveCodeBench値はv6として公開されたものです(Fugu行のバージョンとの一致は未確認)。Qwenの3モデルのSWE-Bench ProはClaude Code・temperature=1・top_p=0.95・256Kで、問題の修正後に再評価した値です。固定した元のpublic splitと同一ではありません。Qwen3.8-27BのTerminal-Bench 2.1はTerminusハーネスでの測定です。
- Gemini 3.1 ProのSWE-Bench Pro値はPublicサブセットの公表値です。Gemini 3.5 / 3.6 FlashもPublicで、社内版Antigravityハーネスによる自己計測です。Atria-Dawn-PreviewはOpenHands、Hy4 previewはswe-agentでの値で、どちらもsubsetの明記はありません。
- SWE-Bench Pro行はv1（public 731問）を前提とした各社の公表値です。2026-09-22に公開されたV2（642問）とHARD-51の値は別の問題セットのため、混ぜていません。本runnerもdataset revisionをv1に固定しています([S42])。
- 2026-09-10の照合ではAnthropicのPDF、OpenAIの本文表、QwenとZ.aiの本文表、DeepSeekの公式画像を確認しました。SakanaのHLEは表画像47.2を採用しています。同じ記事の別の概要図には48.5とあり、出典内の不一致を照合記録に残しています。

## 出典

- **S1** — [Fugu release](https://sakana.ai/fugu-release/), Sakana AI, primary, published 2026-06-22, retrieved 2026-08-11
- **S2** — [Claude Fable 5 and Claude Mythos 5](https://www.anthropic.com/news/claude-fable-5-mythos-5), Anthropic, primary, published 2026-06-09, retrieved 2026-08-11 (表は「Mythos 5 / Fable 5」統合列のため、個別値はS3を参照)
- **S3** — [Claude Fable 5 & Claude Mythos 5 System Card](https://www-cdn.anthropic.com/2f9323abbcc4abe219577539efe19a623c9ca2bd/Claude%20Fable%205%20%26%20Claude%20Mythos%205%20System%20Card.pdf), Anthropic, primary, published 2026-06-09, retrieved 2026-08-27 (Table 8.1.A。Opus 4.8/GPT-5.5等の competitor 値は各開発元の公表値からの転載と明記)
- **S4** — [Introducing GPT-5.6](https://openai.com/index/gpt-5-6/), OpenAI, primary, published 2026-07-09, retrieved 2026-08-27 (付録のカテゴリ別評価表。GPT-5.5値も同表由来)
- **S5** — [DeepSeek-V4-Flash-0731 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), DeepSeek, primary, published 2026-07-31, retrieved 2026-08-11
- **S6** — [Qwen3.8 launch benchmark table](https://qwen.ai/blog?id=qwen3.8), Qwen, primary image transcription, published 2026-08-03, retrieved 2026-08-11
- **S7** — [GLM-5.2 model card](https://huggingface.co/zai-org/GLM-5.2), Z.ai, primary, published 2026-06-16, retrieved 2026-08-11
- **S8** — [Kimi K3 model card](https://huggingface.co/moonshotai/Kimi-K3), Moonshot AI, primary, published 2026-07-29, retrieved 2026-08-11
- **S9** — [Humanity's Last Exam leaderboard](https://artificialanalysis.ai/evaluations/humanitys-last-exam), Artificial Analysis, third-party, rolling leaderboard, retrieved 2026-08-11; 2026-09-30追加分（GPT-6.1 Sol / GPT-6 Sol / GPT-6 Luna = max、Grok 4.7 = xhigh、Gemini 3.8 Flash = high）
- **S10** — [SciCode leaderboard](https://artificialanalysis.ai/evaluations/scicode), Artificial Analysis, independent measurement, retrieved 2026-09-10 (Fable 5 / adaptive max / Opus 4.8 fallback; GLM-5.2 / max); 2026-09-30追加分（Opus 5.5 / Sonnet 5.5 / Fable 5.1 = max with fallback、GPT-6 Astra / GPT-6.1 Sol / GPT-6 Sol / GPT-6 Luna / DeepSeek-V4.1-Flash = max、Grok 4.7 = xhigh、Gemini 3.8 Flash = high。同日再確認でFable 5の61.0は変更なし。SciCode-Verified監査によりUnder review）
- **S11** — [Tau3 Banking leaderboard](https://artificialanalysis.ai/evaluations/tau3-banking), Artificial Analysis, third-party, rolling leaderboard, retrieved 2026-08-11 (GLM-5.3値は 2026-08-27 取得、Qwen3.8 MAX値も同日再確認で変更なし); rechecked 2026-09-10 (Qwen3.8 Max, GLM-5.3 / max, GLM-5.2 / max); 2026-09-30追加分（Fable 5.1 = max with fallback、GPT-6 Astra = max、Gemini 3.8 Flash = high。同日再確認でQwen3.8 Max 51.3・GLM-5.3 50.3は変更なし）
- **S12** — [GLM-5.2 intelligence analysis](https://artificialanalysis.ai/models/glm-5-2), Artificial Analysis, third-party, published 2026-06-16, retrieved 2026-08-11
- **S13** — [DeepSeek-V4-Flash-Vision-Exp release](https://api-docs.deepseek.com/news/news260821/), DeepSeek, primary image transcription, published 2026-08-21, retrieved 2026-08-27
- **S14** — [Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), Qwen, primary, published 2026-08-26, retrieved 2026-08-27
- **S15** — [GLM-5.3 launch benchmark chart](https://z.ai/blog/glm-5.3), Z.ai, primary image transcription, published 2026-08, retrieved 2026-08-27
- **S16** — [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B), Qwen, primary, published 2026-08, retrieved 2026-08-27
- **S17** — [Introducing Claude Opus 4.8](https://www.anthropic.com/news/claude-opus-4-8), Anthropic, primary image transcription, published 2026-05-28, retrieved 2026-08-27
- **S18** — [Claude Opus 5 System Card](https://www-cdn.anthropic.com/b514064af1408018e64b1ad24e7d5e75850b4ffd/Claude%20Opus%205%20System%20Card.pdf), Anthropic, primary, published 2026-07-24, retrieved 2026-08-27 (Table 8.1.A。Fable 5のHLE 56.5もここに記載)
- **S19** — [Gemini 3.1 Pro model page](https://deepmind.google/models/gemini/pro/), Google DeepMind, primary, retrieved 2026-08-27
- **S20** — [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4), Google, primary, published 2026-07-30, retrieved 2026-08-27
- **S21** — [GPT-6 Astra: A new generation of intelligence](https://openai.com/index/gpt-6-astra/), OpenAI, primary, retrieved 2026-09-05 (本文末のAcademic / Long Context表と評価条件。HLEはwith tools、MRCR v2はコンテキスト長別)
- **S22** — [Terminal-Bench 2.1 leaderboard](https://www.tbench.ai/?version=2.1), Terminal-Bench, primary (benchmark organizer), listed 2026-09-03, retrieved 2026-09-05 (Codex / GPT-6 Astra / high。公式フロントエンドの公開APIで2.1データを確認。[公開ジョブ](https://hub.harborframework.com/jobs/17d1a7f6-3339-4670-8b70-3b145979f57f)でハーネスと試行数を確認)、rechecked 2026-09-30（新モデルの掲載なし）、2026-10-11は公式ボードに到達できず未再確認（[S46]でコミュニティ提出の停止を確認）

- **S23** — [DeepSWE v1.1 leaderboard JSON](https://deepswe.datacurve.ai/artifacts/v1.1/leaderboard-live.json), Datacurve, primary（ベンチマーク提供元による他社モデル評価）, rolling leaderboard, generated 2026-09-03T22:24:37.984682+00:00, retrieved 2026-09-10, rechecked 2026-09-12（全28モデル・70設定。再取得JSONと保存原本のSHA-256が一致）, rechecked 2026-09-30（generated 2026-09-22T06:27:15.860279+00:00。70設定の採点値は保存原本と一致し、新モデルの追加なし）。2026-10-11は公式JSONに到達できず、第三者による写し（lprhodes/deepswe-tracker、2026-10-09取得）との照合のみ（採点値は一致）
- **S24** — [DeepSWE v1.1 — A revision of DeepSWE v1](https://deepswe.datacurve.ai/blog/deepswe-v1-1), Datacurve, primary, published 2026-06-14, retrieved 2026-09-12（v1.1の実行・採点方式とv1との差分）
- **S25** — [Claude Opus 5.5 System Card](https://www-cdn.anthropic.com/fc1b44717c85dc068bc6ba5024219938094694bd/Claude%20Opus%205.5%20System%20Card.pdf), Anthropic, primary, published 2026-09-22, retrieved 2026-09-30（Table 8.1.A p.174、§8.3 DeepSWE。[発表ページ](https://www.anthropic.com/claude-opus-5-5)とも照合）
- **S26** — [Claude Fable 5.1 & Claude Mythos 5.1 System Card](https://www-cdn.anthropic.com/0339e6a7c5c7b87f5c07798616dc32c215d14235/Claude%20Fable%205.1%20%26%20Claude%20Mythos%205.1%20System%20Card.pdf), Anthropic, primary, published 2026-09-01, retrieved 2026-09-30（Table 8.1.A p.167、§8.2–8.3。[発表ページ](https://www.anthropic.com/claude-fable-and-mythos-5-1)とも照合）
- **S27** — [Claude Sonnet 5.5 System Card](https://www-cdn.anthropic.com/870c8f525702625d2c62fc6dd04c857e3250bec1/Claude%20Sonnet%205.5%20System%20Card.pdf), Anthropic, primary, published 2026-09-28, retrieved 2026-09-30（Table 8.1.A p.109、§8.3 DeepSWE。[発表ページ](https://www.anthropic.com/claude-sonnet-5-5)とも照合）
- **S28** — [Introducing GPT-6 Sol and Luna](https://openai.com/index/introducing-gpt-6-sol-and-luna/), OpenAI, primary, published 2026-09-22, retrieved 2026-09-30（DeepSWE v1.1の本文記載値）
- **S29** — [Introducing GPT-6.1 Sol](https://openai.com/index/introducing-gpt-6-1-sol/), OpenAI, primary, published 2026-09-29, retrieved 2026-09-30（DeepSWE v1.1図のデータ値）
- **S30** — [Introducing Grok 4.7](https://x.ai/news/grok-4-7), SpaceXAI, primary, published 2026-09-21, retrieved 2026-09-30（ベンチマーク表。DeepSWEはhigh effort）
- **S31** — [DeepSeek-V4.1-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), DeepSeek, primary, published 2026-09-10（[発表](https://api-docs.deepseek.com/news/news260910/)）, retrieved 2026-09-30（Comparison with frontier modelsとagent scaffold別の表）
- **S32** — [Gemini 3.8 Flash model card](https://deepmind.google/models/model-cards/gemini-3-8-flash/), Google DeepMind, primary, published 2026-09-02, retrieved 2026-09-30（評価表と[評価方法PDF](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-8_flash_model_evaluation.pdf)）、2026-10-11に評価方法PDFでDeepSWE v1.1の自己計測値73.7（mini-swe agent、high thinking）を確認
- **S33** — [Claude Haiku 5.5](https://www.anthropic.com/claude-haiku-5-5), Anthropic, primary, published 2026-10-07, retrieved 2026-10-11（ベンチマーク表のHLE no tools 45.9。effort別グラフのMax点と一致。[システムカード](https://www.anthropic.com/claude-haiku-5-5-system-card)は到達不可で未確認）
- **S34** — [Atria-Dawn-Preview](https://github.com/atria-asi/Atria-Dawn-Preview), Shanghai AI Laboratory（Atria Team）, primary（開発元GitHubのREADME「Evaluation Results」表と同梱PDF）, retrieved 2026-10-11（SWE-bench Pro: OpenHands v1.23.0。Terminal-Bench 2.1: Claude Code v2.1.224、4回平均）
- **S35** — [Hy4 preview](https://github.com/Tencent-Hunyuan/Hy4-preview), Tencent Hunyuan, primary image transcription（`assets/benchmark-appendix.jpg`）, published 2026-08-28（GitHub初回コミット）, retrieved 2026-10-11（最高reasoning設定。SWE-bench Proはswe-agent、TB2.1はClaude Code）
- **S36** — [Gemini 3.7 Flash Model evaluation](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-7_flash_model_evaluation.pdf), Google DeepMind, primary image transcription, results as of August 2026, retrieved 2026-10-11（TB2.1 Terminus 2、CharXiv no tools、GDM-MRCR v2 8-needle 128k average、DeepSWE自己計測）
- **S37** — [Gemini 3.6 Flash Model evaluation](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-6_flash_model_evaluation.pdf), Google DeepMind, primary image transcription, results as of July 2026, retrieved 2026-10-11（SWE-Bench Pro Public・社内版Antigravityハーネス、TB2.1 Terminus-2、CharXiv no tools、GDM-MRCR v2 128k average）
- **S38** — [Gemini 3.5 Flash Model evaluation](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-5_flash_model_evaluation.pdf), Google DeepMind, primary image transcription, results as of May 2026, retrieved 2026-10-11（SWE-Bench Pro Public・5回平均、TB2.1 Terminus-2、HLE full set text + MM、CharXiv no tools、MRCR v2 8-needle 128k average。Gemini 3.1 ProのCharXiv no tools 83.3も同表）
- **S39** — [GLM-5.3-Flash benchmark chart](https://github.com/zai-org/GLM-5/blob/main/resources/bench_53.png), Z.ai, primary image transcription（開発元GitHub）, published 2026-08, retrieved 2026-10-11（評価条件は[モデルカード](https://huggingface.co/zai-org/GLM-5.3-Flash)の脚注。今回は第三者の写しでのみ確認）
- **S40** — [Gemini 4 Argon Model evaluation](https://storage.googleapis.com/deepmind-media/gemini/gemini_4_argon_model_evaluation.pdf), Google DeepMind, primary, published 2026-09-30（results as of October 2026）, retrieved 2026-10-11（DeepSWE v1.1はmini-swe agentによる自己計測）
- **S41** — [Kimi K3 README](https://github.com/MoonshotAI/Kimi-K3), Moonshot AI, primary（開発元GitHub）, retrieved 2026-10-11（DeepSWEはKimi Codeハーネスで67.5。脚注ではmini-SWE-agentで67.3、DeepSWE v1.1 tasks）
- **S42** — [SWE-Bench Pro README](https://github.com/scaleapi/SWE-bench_Pro-os), Scale AI, primary, news dated 2026-09-22, retrieved 2026-10-11（V2: 642 tasks、HARD-51、HFの `default` configがV2、v1はconfig `v1` / tag `v1.0`）
- **S43** — [deep-swe issue #105: DeepSWE v1.1 grading defects and test infrastructure bypasses](https://github.com/datacurve-ai/deep-swe/issues/105), opened 2026-10-02（Tokenless）, retrieved 2026-10-11（open、Datacurveの回答なし）
- **S44** — [deep-swe issue #107: A crashed vitest worker is graded as all tests passed](https://github.com/datacurve-ai/deep-swe/issues/107), opened 2026-10-05, retrieved 2026-10-11（open、Datacurveの回答なし）
- **S45** — [Gemini 3.1 Pro Model evaluation](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-1_pro_model_evaluation.pdf), Google DeepMind, primary, published February 2026, retrieved 2026-10-11（全結果をmodel-id `gemini-3.1-pro-preview` で測定、SciCodeはArtificial Analysisからの引用と明記）
- **S46** — [terminal-bench-2-1 README](https://github.com/harbor-framework/terminal-bench-2-1), Terminal-Bench / Harbor, primary, retrieved 2026-10-11（"Community submissions are currently closed for Terminal-Bench 2.1."）

[S1]: https://sakana.ai/fugu-release/
[S2]: https://www.anthropic.com/news/claude-fable-5-mythos-5
[S3]: https://www-cdn.anthropic.com/2f9323abbcc4abe219577539efe19a623c9ca2bd/Claude%20Fable%205%20%26%20Claude%20Mythos%205%20System%20Card.pdf
[S4]: https://openai.com/index/gpt-5-6/
[S5]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
[S6]: https://qwen.ai/blog?id=qwen3.8
[S7]: https://huggingface.co/zai-org/GLM-5.2
[S8]: https://huggingface.co/moonshotai/Kimi-K3
[S9]: https://artificialanalysis.ai/evaluations/humanitys-last-exam
[S10]: https://artificialanalysis.ai/evaluations/scicode
[S11]: https://artificialanalysis.ai/evaluations/tau3-banking
[S12]: https://artificialanalysis.ai/models/glm-5-2
[S13]: https://api-docs.deepseek.com/news/news260821/
[S14]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
[S15]: https://z.ai/blog/glm-5.3
[S16]: https://huggingface.co/Qwen/Qwen3.8-27B
[S17]: https://www.anthropic.com/news/claude-opus-4-8
[S18]: https://www-cdn.anthropic.com/b514064af1408018e64b1ad24e7d5e75850b4ffd/Claude%20Opus%205%20System%20Card.pdf
[S19]: https://deepmind.google/models/gemini/pro/
[S20]: https://ai.google.dev/gemma/docs/core/model_card_4
[S21]: https://openai.com/index/gpt-6-astra/
[S22]: https://www.tbench.ai/?version=2.1

[S23]: https://deepswe.datacurve.ai/artifacts/v1.1/leaderboard-live.json
[S24]: https://deepswe.datacurve.ai/blog/deepswe-v1-1
[S25]: https://www-cdn.anthropic.com/fc1b44717c85dc068bc6ba5024219938094694bd/Claude%20Opus%205.5%20System%20Card.pdf
[S26]: https://www-cdn.anthropic.com/0339e6a7c5c7b87f5c07798616dc32c215d14235/Claude%20Fable%205.1%20%26%20Claude%20Mythos%205.1%20System%20Card.pdf
[S27]: https://www-cdn.anthropic.com/870c8f525702625d2c62fc6dd04c857e3250bec1/Claude%20Sonnet%205.5%20System%20Card.pdf
[S28]: https://openai.com/index/introducing-gpt-6-sol-and-luna/
[S29]: https://openai.com/index/introducing-gpt-6-1-sol/
[S30]: https://x.ai/news/grok-4-7
[S31]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
[S32]: https://deepmind.google/models/model-cards/gemini-3-8-flash/
[S33]: https://www.anthropic.com/claude-haiku-5-5
[S34]: https://github.com/atria-asi/Atria-Dawn-Preview
[S35]: https://github.com/Tencent-Hunyuan/Hy4-preview
[S36]: https://storage.googleapis.com/deepmind-media/gemini/gemini_3-7_flash_model_evaluation.pdf
[S37]: https://storage.googleapis.com/deepmind-media/gemini/gemini_3-6_flash_model_evaluation.pdf
[S38]: https://storage.googleapis.com/deepmind-media/gemini/gemini_3-5_flash_model_evaluation.pdf
[S39]: https://github.com/zai-org/GLM-5/blob/main/resources/bench_53.png
[S40]: https://storage.googleapis.com/deepmind-media/gemini/gemini_4_argon_model_evaluation.pdf
[S41]: https://github.com/MoonshotAI/Kimi-K3
[S42]: https://github.com/scaleapi/SWE-bench_Pro-os
[S43]: https://github.com/datacurve-ai/deep-swe/issues/105
[S44]: https://github.com/datacurve-ai/deep-swe/issues/107
[S45]: https://storage.googleapis.com/deepmind-media/gemini/gemini_3-1_pro_model_evaluation.pdf
[S46]: https://github.com/harbor-framework/terminal-bench-2-1
