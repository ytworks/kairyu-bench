# 他モデルの公開ベンチマーク比較

他モデルが公開しているスコアの参照表です。単位はすべて `%`。`—` は調査した出典で公開値を確認できなかった項目であり、0点ではありません。

初版の値と出典は [`kairyu@9a00f39` の参照カタログ](https://github.com/ytworks/kairyu/blob/9a00f39cd60581d70baadb76d7df2737671e790c/kairyu/bench/reference.py) から転記しましたが、2026-08-27 に全列を一次ソース(各社の発表ページ・システムカード・モデルカード)と照合し、Fable 5列の誤値を修正のうえ、Mythos 5 / Opus 5 / Opus 4.8 / GPT-5.5 / Gemini 3.1 Pro / Gemma 4 / DeepSeek-V4-Flash-Vision-Exp / Qwen3.8-27B / Qwen3.8-Flash-Next / GLM-5.3 の列を追加しました。各提供元や第三者機関が異なる条件で測定した値を含むため、`kairyu-bench` の実測値と厳密に同条件とは限りません。

2026-09-05にGPT-6 Astra列を追加しました。OpenAI発表の評価表を本文で確認し、GPQA Diamond、HLE（ツールあり）、MRCR v2を転記しました。Terminal-Bench 2.1は公式リーダーボードのCodex / highの値です。調査した出典で確認できなかった項目は空欄のままです。

<!-- BEGIN public-comparison -->
| Benchmark | Fugu | Fugu Ultra | Fable 5 | Mythos 5 | Opus 5 | Opus 4.8 | GPT-6 Astra | GPT-5.6 Sol | GPT-5.5 | Gemini 3.1 Pro | Gemma 4 31B | DeepSeek-V4-Flash-0731 | DeepSeek-V4-Flash-Vision-Exp | Qwen3.8 MAX | Qwen3.8-27B | Qwen3.8-Flash-Next | GLM-5.2 | GLM-5.3 | Kimi K3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SWE-Bench Pro | [59.0][S1] | [73.7][S1] | [80.0][S3] | [80.3][S3] | [79.2][S18] | [69.2][S17] | — | [64.6][S4] | [59.4][S4] | [54.2][S19] | — | — | — | [67.7][S6] | [61.7][S16] | [62.5][S14] | [62.1][S7] | — | — |
| SWE-bench Verified | — | — | [95.0][S3] | [95.5][S3] | [96.0][S18] | [88.6][S3] | — | — | — | [80.6][S19] | — | — | — | — | — | — | — | — | — |
| Terminal-Bench 2.1 | [80.2][S1] | [82.1][S1] | [84.3][S3] | [88.0][S3] | — | [82.7][S3] | [87.4][S22] | [88.8][S4] | [85.6][S4] | — | — | [82.7][S5] | [83.9][S13] | [86.6][S6] | [73.0][S16] | — | [81.0][S7] | [88.2][S15] | [88.3][S8] |
| LiveCodeBench | [92.9][S1] | [93.2][S1] | — | — | — | — | — | — | — | — | [80.0][S20] | — | — | — | [90.3][S16] | [91.9][S14] | — | — | — |
| LiveCodeBench Pro | [87.8][S1] | [90.8][S1] | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| HLE | [47.2][S1] | [50.0][S1] | [56.5][S18] | [59.0][S3] | [56.3][S18] | [49.8][S17] | [57.2 (tools)][S21] | [49.5][S9] | [41.4][S3] | [44.4][S19] | [19.5][S20] | — | — | [43.6][S6] | [30.8][S16] | [35.9][S14] | [40.5][S7] | [62.5 (tools)][S15] | [43.5][S8] |
| CharXiv Reasoning | [85.1][S1] | [86.6][S1] | — | [88.9][S3] | — | [80.5][S3] | — | — | — | — | — | — | — | [93.5][S6] | [90.2][S16] | [90.6][S14] | — | — | [84.8][S8] |
| GPQA Diamond | [95.5][S1] | [95.5][S1] | — | [94.1][S3] | — | — | [96.0][S21] | [94.6][S4] | [93.6][S4] | [94.3][S19] | [84.3][S20] | — | — | [92.6][S6] | [89.2][S16] | [91.7][S14] | [91.2][S7] | — | [93.5][S8] |
| SciCode | [60.1][S1] | [58.7][S1] | [60.2][S10] | — | — | — | — | — | — | [59.0][S19] | — | — | — | — | — | — | [50.0][S12] | — | [58.7][S8] |
| τ-bench Banking | [21.7][S1] | [20.6][S1] | — | — | — | — | — | — | — | — | — | — | — | [51.3][S11] | — | — | [27.0][S12] | [50.3][S11] | [33.4][S8] |
| Long Context Reasoning | [74.7][S1] | [73.3][S1] | — | — | — | — | — | — | — | — | — | — | — | — | — | — | [71.0][S12] | — | [74.7][S8] |
| MRCR v2 | [86.6][S1] | [93.6][S1] | — | — | — | — | [100.0][S21] | [91.5][S4] | [81.5][S4] | [84.9][S19] | [66.4][S20] | — | — | [92.9][S6] | — | — | — | — | — |
| DeepSWE v1.1 | — | — | [69.91][S23] | — | [73.65][S23] | [58.97][S23] | [74.12][S23] | [72.67][S23] | [67.04][S23] | [11.73 (preview)][S23] | — | — | — | [57.46][S23] | — | — | [43.78][S23] | [68.96][S23] | [68.51][S23] |
<!-- END public-comparison -->


## DeepSWE v1.1（2026-09-10取得）

Datacurveの公式JSON（生成日時2026-09-03）にある28モデル・70設定を、分子・分母・effort・pass@4・信頼区間・取得日・出典URL付きで[参照カタログ](../src/kairyu_bench/data/public_results.json)へ保存しました。[取得原本](sources/deepswe-v1.1-20260910.json)のSHA-256もカタログで固定しています。各モデルの最高pass@1の設定を下表と上の横断表に表示し、同点なら原本の先の行を採用します。全設定はJSONに残しています。effortが異なる値を同条件の順位と解釈しないでください。

データセットは113問、各設定は4反復です。Opus 4.8 / maxは採点対象111問・429試行で、問題数も他の行と異なります。pass@1の分母は採点対象の試行で、API・verifier・network障害の除外により452未満の行があります。context超過とagent timeoutは失敗として数えます。pass@4は一度でも成功した問題数/対象問題数。CIは4反復のスコア間のばらつきによる95%区間です。これらは公式公開値であり、このPRのローカル実測ではありません。[S23]

公開評価はPier / mini-swe-agent / Modal、本runnerは固定Pier / mini-swe-agent 2.4.6 / Dockerです。公開側の正確なcommit・agent版・image digestが一致すると確認できないため、参照値はローカル実測の平均値やdeltaへ混ぜません。DeepSeek-V4-Flashと日付付き0731版を同一視せず、Geminiの公開IDはpreviewと明記しています。表にないモデルへ値を補完しません。

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

既存12行・19モデルの値と出典も同じJSONへ移し、欠測をnull、条件差を各行に保存しました。既存値の取得日は元の調査日を維持しています。`historical-transcription`は過去の出典付き転記、`verified-against-snapshot`は今回保存した原本との照合を示し、既存全値を今回再検証したという意味ではありません。τ-benchの公開Tau3 Bankingとrunnerのtau2-bench、Long Context ReasoningとLongBench v2は特に区別が必要です。

表の再生成は `python scripts/update_model_comparison.py`、出典整合性と差分検査は `--check` で行います。

## 2026-08-27の修正について

`kairyu` 参照カタログから転記していたFable 5列の4値を一次ソース照合により修正しました。

- SWE-Bench Pro 80.3 → **80.0**: システムカード([S3] Table 8.1.A)ではFable 5=80.0、80.3はMythos 5の値。Anthropicのlaunch記事[S2]の表は「Mythos 5 / Fable 5」統合列で高い方を表示していたため混同されていました。
- Terminal-Bench 2.1 88.0 → **84.3**: 88.0はMythos 5の値([S3])。旧出典(S8=Kimiモデルカード)の帰属も誤りでした。
- HLE 59.0 → **56.5**: 59.0はMythos 5(no tools)の値([S3])。Fable 5単体の56.5はOpus 5システムカード([S18])に記載。
- GPQA Diamond 91.3 → **削除**: どの公式資料にも確認できず。AnthropicはMythos 5の94.1のみ公表し([S3])、GPQA Diamondを飽和ベンチマークとして今後の報告を停止すると明言しています。

Mythos 5はFable 5と同一の基盤モデルで、セーフガード(一部領域でOpus 4.8へフォールバック)を外したものです。Fable 5の公表値は本番セーフガード込みの実測です([S3])。

## 比較時の注意

- HLEとCharXivは、このrunnerでは対象モデル自身による自己採点です。
- 表のHLE値は原則 no tools 測定です(Anthropic系・Gemini・Gemmaはno toolsと明記)。GPT-6 Astraの57.2とGLM-5.3の62.5はツール使用あり(with tools)の公表値で、表中にも `tools` と明記しています。no tools値との直接比較には適しません。ツールあり同士もツール構成・ハーネスの一致は未確認です(ツールありのGLM-5.2はZ.aiのchartでは54.7)。
- CharXiv ReasoningのMythos 5(88.9)とOpus 4.8(80.5)はno tools値(with toolsは各93.5 / 89.9)。CharXivのQwen3.8 MAX値は元のlaunch tableに解釈上の曖昧さがあり93.5を採用、Qwen3.8-Flash-Nextの90.6とQwen3.8-27Bの90.2("CharXiv RQ")は "With CI" として公開された値です。
- τ-benchは、このrunnerでは対象モデル自身をuser simulatorにも使用します。
- Long Context Reasoning行の公開値に対し、このrunnerは代替としてLongBench v2を実行するため直接比較できません。
- MRCR v2の測定コンテキストは、GPTが8-needle 256K–512K、Qwenが256K、GeminiとGemmaが8-needle 128K、このrunnerは4K–128Kです。
- GPT-6 AstraのMRCR v2は8-needle 256K–512Kの100.0を採用しています。512K–1Mでは96.3であり、コンテキスト長を混同しないでください。OpenAIの発表値はreasoning effortをまたいで各評価項目で得られた最大値で、単一の固定effortを指定した一括測定とは扱えません。研究環境/APIと製品版ChatGPTではsystem promptやツールが異なり得ます([S21])。
- GPT-6 AstraのTerminal-Bench 2.1はCodex 0.151.0 / high、2026-09-03掲載、87.4 ± 1.8%（95%信頼区間）の公式リーダーボード値です([S22])。公開ジョブは各問題5試行・各effort 445試行で、このrunnerの既定agent（Terminus-2）・1試行と同条件ではありません。
- GPT-6 AstraのTerminal-Bench 4.0（57.9）とDeepSWE v1.1（74.1）は、Terminal-Bench 2.1やSWE-Bench Proへ転記していません([S21])。
- SWE-bench VerifiedのFable 5は5試行平均、このrunnerは1試行です。
- Terminal-Bench 2.1のOpus 4.8はlaunch時に74.6と公表され、その後のシステムカード([S3])で82.7に更新されています。Opus 5はTB2.1を公表せず後継のFrontierBench v0.1に移行。Gemini 3.1 ProはTerminal-Bench 2.0のみ公表(68.5)のためTB2.1行は空欄です。
- Gemini 3.1 ProのLiveCodeBench ProはElo(2887)での公表のため、%基準の本表には記載していません。
- Gemma 4の値は31B Dense(instruction-tuned)のものです。
- Qwen3.8-Flash-NextとQwen3.8-27BのLiveCodeBench値はv6として公開されたものです(Fugu行のバージョンとの一致は未確認)。両者のSWE-Bench Pro値はQwenがClaude Codeハーネスで測定したと注記しています。Qwen3.8-27BのTerminal-Bench 2.1はTerminusハーネスでの測定です。
- Gemini 3.1 ProのSWE-Bench Pro値はPublicサブセットの公表値です。
- AnthropicとOpenAIの表(画像)からの転記値は、それぞれの発表ページ・システムカードPDFを直接確認して照合済みです。DeepSeek-V4-Flash-Vision-ExpとGLM-5.3の公式値は発表ページのベンチマーク画像からの転記です(第三者転記と照合済み)。

## 出典

- **S1** — [Fugu release](https://sakana.ai/fugu-release/), Sakana AI, primary, published 2026-07-23, retrieved 2026-08-11
- **S2** — [Claude Fable 5 and Claude Mythos 5](https://www.anthropic.com/news/claude-fable-5-mythos-5), Anthropic, primary, published 2026-06-09, retrieved 2026-08-11 (表は「Mythos 5 / Fable 5」統合列のため、個別値はS3を参照)
- **S3** — [Claude Fable 5 & Claude Mythos 5 System Card](https://www-cdn.anthropic.com/2f9323abbcc4abe219577539efe19a623c9ca2bd/Claude%20Fable%205%20%26%20Claude%20Mythos%205%20System%20Card.pdf), Anthropic, primary, published 2026-06-09, retrieved 2026-08-27 (Table 8.1.A。Opus 4.8/GPT-5.5等の competitor 値は各開発元の公表値からの転載と明記)
- **S4** — [Introducing GPT-5.6](https://openai.com/index/gpt-5-6/), OpenAI, primary, published 2026-07-09, retrieved 2026-08-27 (付録のカテゴリ別評価表。GPT-5.5値も同表由来)
- **S5** — [DeepSeek-V4-Flash-0731 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), DeepSeek, primary, published 2026-07-31, retrieved 2026-08-11
- **S6** — [Qwen3.8 launch benchmark table](https://qwen.ai/blog?id=qwen3.8), Qwen, primary image transcription, published 2026-08-03, retrieved 2026-08-11
- **S7** — [GLM-5.2 model card](https://huggingface.co/zai-org/GLM-5.2), Z.ai, primary, published 2026-06-16, retrieved 2026-08-11
- **S8** — [Kimi K3 model card](https://huggingface.co/moonshotai/Kimi-K3), Moonshot AI, primary, published 2026-07-29, retrieved 2026-08-11
- **S9** — [Humanity's Last Exam leaderboard](https://artificialanalysis.ai/evaluations/humanitys-last-exam), Artificial Analysis, third-party, rolling leaderboard, retrieved 2026-08-11
- **S10** — [SciCode leaderboard](https://artificialanalysis.ai/evaluations/scicode), Artificial Analysis, third-party, rolling leaderboard, retrieved 2026-08-11 (再確認 2026-08-27、値変更なし)
- **S11** — [Tau3 Banking leaderboard](https://artificialanalysis.ai/evaluations/tau3-banking), Artificial Analysis, third-party, rolling leaderboard, retrieved 2026-08-11 (GLM-5.3値は 2026-08-27 取得、Qwen3.8 MAX値も同日再確認で変更なし)
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
- **S22** — [Terminal-Bench 2.1 leaderboard](https://www.tbench.ai/?version=2.1), Terminal-Bench, primary (benchmark organizer), listed 2026-09-03, retrieved 2026-09-05 (Codex / GPT-6 Astra / high。公式フロントエンドの公開APIで2.1データを確認。[公開ジョブ](https://hub.harborframework.com/jobs/17d1a7f6-3339-4670-8b70-3b145979f57f)でハーネスと試行数を確認)

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
