# 他モデルの公開ベンチマーク比較

他モデルが公開しているスコアの参照表です。単位はすべて `%`。`—` は公開値なしであり、0点ではありません。

値と出典は [`kairyu@9a00f39` の参照カタログ](https://github.com/ytworks/kairyu/blob/9a00f39cd60581d70baadb76d7df2737671e790c/kairyu/bench/reference.py) から転記しています。各提供元や第三者機関が異なる条件で測定した値を含むため、`kairyu-bench` の実測値と厳密に同条件とは限りません。

| Benchmark | Fugu | Fugu Ultra | Fable 5 | GPT-5.6 Sol | DeepSeek-V4-Flash-0731 | Qwen3.8 MAX | GLM-5.2 | Kimi K3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SWE-Bench Pro | [59.0][S1] | [73.7][S1] | [80.3][S2] | [64.6][S4] | — | [67.7][S6] | [62.1][S7] | — |
| SWE-bench Verified | — | — | [95.0][S3] | — | — | — | — | — |
| Terminal-Bench 2.1 | [80.2][S1] | [82.1][S1] | [88.0][S8] | [88.8][S4] | [82.7][S5] | [86.6][S6] | [81.0][S7] | [88.3][S8] |
| LiveCodeBench | [92.9][S1] | [93.2][S1] | — | — | — | — | — | — |
| LiveCodeBench Pro | [87.8][S1] | [90.8][S1] | — | — | — | — | — | — |
| HLE | [47.2][S1] | [50.0][S1] | [59.0][S2] | [49.5][S9] | — | [43.6][S6] | [40.5][S7] | [43.5][S8] |
| CharXiv Reasoning | [85.1][S1] | [86.6][S1] | — | — | — | [93.5][S6] | — | [84.8][S8] |
| GPQA Diamond | [95.5][S1] | [95.5][S1] | [91.3][S2] | [94.6][S4] | — | [92.6][S6] | [91.2][S7] | [93.5][S8] |
| SciCode | [60.1][S1] | [58.7][S1] | [60.2][S10] | — | — | — | [50.0][S12] | [58.7][S8] |
| τ-bench Banking | [21.7][S1] | [20.6][S1] | — | — | — | [51.3][S11] | [27.0][S12] | [33.4][S8] |
| Long Context Reasoning | [74.7][S1] | [73.3][S1] | — | — | — | — | [71.0][S12] | [74.7][S8] |
| MRCR v2 | [86.6][S1] | [93.6][S1] | — | [91.5][S4] | — | [92.9][S6] | — | — |

## 比較時の注意

- HLEとCharXivは、このrunnerでは対象モデル自身による自己採点です。
- τ-benchは、このrunnerでは対象モデル自身をuser simulatorにも使用します。
- Long Context Reasoning行の公開値に対し、このrunnerは代替としてLongBench v2を実行するため直接比較できません。
- MRCR v2の公開値はGPTが256K–512K、Qwenが256K、このrunnerは4K–128Kです。
- SWE-bench VerifiedのFable 5は5試行平均、このrunnerは1試行です。
- CharXivのQwen値は元のlaunch tableに解釈上の曖昧さがあり、`kairyu` の参照カタログでは93.5を選択しています。

## 出典

- **S1** — [Fugu release](https://sakana.ai/fugu-release/), Sakana AI, primary, published 2026-07-23, retrieved 2026-08-11
- **S2** — [Claude Fable 5 and Claude Mythos 5](https://www.anthropic.com/news/claude-fable-5-mythos-5), Anthropic, primary, published 2026-06-09, retrieved 2026-08-11
- **S3** — [Claude Fable 5 & Claude Mythos 5 System Card](https://www-cdn.anthropic.com/2f9323abbcc4abe219577539efe19a623c9ca2bd/Claude%20Fable%205%20%26%20Claude%20Mythos%205%20System%20Card.pdf), Anthropic, primary, published 2026-06-09, retrieved 2026-08-12
- **S4** — [Introducing GPT-5.6](https://openai.com/index/gpt-5-6/), OpenAI, primary, published 2026-07-09, retrieved 2026-08-11
- **S5** — [DeepSeek-V4-Flash-0731 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), DeepSeek, primary, published 2026-07-31, retrieved 2026-08-11
- **S6** — [Qwen3.8 launch benchmark table](https://qwen.ai/blog?id=qwen3.8), Qwen, primary image transcription, published 2026-08-03, retrieved 2026-08-11
- **S7** — [GLM-5.2 model card](https://huggingface.co/zai-org/GLM-5.2), Z.ai, primary, published 2026-06-16, retrieved 2026-08-11
- **S8** — [Kimi K3 model card](https://huggingface.co/moonshotai/Kimi-K3), Moonshot AI, primary, published 2026-07-29, retrieved 2026-08-11
- **S9** — [Humanity's Last Exam leaderboard](https://artificialanalysis.ai/evaluations/humanitys-last-exam), Artificial Analysis, third-party, rolling leaderboard, retrieved 2026-08-11
- **S10** — [SciCode leaderboard](https://artificialanalysis.ai/evaluations/scicode), Artificial Analysis, third-party, rolling leaderboard, retrieved 2026-08-11
- **S11** — [Tau3 Banking leaderboard](https://artificialanalysis.ai/evaluations/tau3-banking), Artificial Analysis, third-party, rolling leaderboard, retrieved 2026-08-11
- **S12** — [GLM-5.2 intelligence analysis](https://artificialanalysis.ai/models/glm-5-2), Artificial Analysis, third-party, published 2026-06-16, retrieved 2026-08-11

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
