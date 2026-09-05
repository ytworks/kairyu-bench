# GPT-6 Astraの公開ベンチマークスコア

確認日: **2026-09-05**。GPT-6 Astraは2026-09-03に発表されました([発売日の公式記事][release])。モデルIDは `gpt-6-astra` です([公式モデルページ][model])。

OpenAIの発表本文とTerminal-Bench公式リーダーボードを確認した参照資料です。以下は提供元・評価主催者の公表値で、`kairyu-bench` の実測値ではありません。モデル名・評価版・ツール・推論設定を併記し、調査した出典で確認できない値は `—` としています。モデル横断の一覧は [公開ベンチマーク比較](model-comparison.md)、ローカル実測は [Wiki](https://github.com/ytworks/kairyu-bench/wiki/Benchmark-Results) を参照してください。

## このrunnerの12項目との対応

| ベンチマーク | Astraの公開値 | 測定条件と参照先 |
| --- | ---: | --- |
| SWE-bench Pro | — | [OpenAI発表][S21]に同名の値を確認できず。DeepSWE v1.1の値は別評価 |
| SWE-bench Verified | — | 調査した出典で対応するスコアを確認できず |
| Terminal-Bench 2.1 | **87.4%** | Codex / high。公式表示は87.4 ± 1.8%、95%信頼区間。掲載日2026-09-03([S22]) |
| LiveCodeBench | — | 調査した出典でv6の対応値を確認できず |
| LiveCodeBench Pro | — | 調査した出典でpass@1の対応値を確認できず |
| HLE | **57.2%** | **with tools**。no toolsの値は確認できず([S21], Academic) |
| CharXiv Reasoning | — | 調査した出典で対応するスコアを確認できず |
| GPQA Diamond | **96.0%** | [S21], Academic。個別のreasoning effortは表に記載なし |
| SciCode | — | 調査した出典で対応するスコアを確認できず |
| τ-bench Banking | — | 調査した出典でこのrunnerのbanking条件に対応するスコアを確認できず |
| Long Context Reasoning | — | 調査した出典で対応値を確認できず。このrunnerはLongBench v2で代替 |
| MRCR v2 | **100.0%** | 8-needle、256K–512K。512K–1Mでは**96.3%**([S21], Long Context) |

HLEのツールあり値はツールなし値と直接比較できません。MRCR v2もこのrunnerの4K–128Kとは測定範囲が異なります。SWE-bench Pro / Verified / DeepSWE、Terminal-Benchの2.1 / 4.0 / Science 0.1はそれぞれ別の評価です。

## Terminal-Bench 2.1の推論設定別スコア

[公式リーダーボード][S22]のCodexによる2026-09-03掲載結果です。公開ジョブの設定はCodex **0.151.0**、dataset `terminal-bench/terminal-bench-2-1` の `ref=6`、各問題5試行です。各effortで `n_trials=445`（89問×5試行と整合）、5設定の合計は2,225試行です。スコアと95%信頼区間の半幅はサイト表示に合わせて小数1桁へ丸めています。比較表ではこの掲載結果中で最も高いhighを採用しています。

| reasoning effort | スコア | 95%信頼区間の半幅 |
| --- | ---: | ---: |
| high | 87.4% | ±1.8ポイント |
| medium | 87.0% | ±1.9ポイント |
| low | 86.7% | ±1.8ポイント |
| max | 86.7% | ±1.4ポイント |
| xhigh | 85.8% | ±1.4ポイント |

実行期間は2026-08-31〜09-01（UTC）で、掲載日とは異なります。公開ジョブ全体には65件のエラーも記録されているため、ここではジョブ全体の平均値から再計算せず、主催者が掲載した設定別の値を採用しています。

推論設定による差には信頼区間の重なりがあります。この表だけでhighがmaxより常に優れるとは判断できません。このrunnerのTerminal-Benchは既定がTerminus-2、各問題1試行です。Codexを使った5試行のこれらの測定と同条件ではありません。

## OpenAIが公表したその他の主な評価

出典はすべて [OpenAI発表本文のカテゴリ別評価表][S21]（2026-09-05確認）。このrunnerに未実装の評価も含みます。

| ベンチマーク | スコア | 版・条件 |
| --- | ---: | --- |
| Terminal-Bench 4.0 | 57.9% | Coding。2.1とは別評価 |
| Terminal-Bench Science 0.1 | 64.6% | Academic。科学研究の作業を評価 |
| DeepSWE v1.1 | 74.1% | Coding。SWE-bench Proとは別評価 |
| FrontierCode 1.1 Extended | 64.5% | Coding。開発者メッセージの条件は発表脚注8を参照 |
| FrontierCode 1.1 Main | 53.3% | Coding。同上 |
| FrontierMath Tier 4 (v2) | 97.6% | Academic。冒頭の98%は丸めた値 |
| ARC-AGI-3 | 99.9% | Abstract reasoning。Responses APIハーネスの設定変更は発表脚注1を参照 |
| ARC-AGI-2 | 95.0% | Abstract reasoning |
| Agents' Last Exam | 59.3% | Computer Use |
| OSWorld 2.0 | 72.6% | v2026.08.08、offline set、partial score |
| ScreenSpot-Pro | 92.7% | no tools |
| AutomationBench | 41.4% | Professional |
| BenchCAD | 95.9% | Professional |
| BrowseComp | 91.5% | Professional |
| HealthBench Professional | 63.4% | length-adjusted |

同じ発表にはArtificial Analysisの指標も転載されていますが、Intelligence Index **v4.1.1=61.2**、Coding Agent Index **v1.4=67.0**は指数であり、正答率の%ではありません。更新される外部リーダーボードの別バージョンと混同しないでください。

## 比較・転記のルール

- OpenAI発表の値は、推論設定をまたいで各評価項目で得られた最大値です。「全項目をmax設定で測定した値」や「GPT-6 Astra Proの値」とは記載されていません。Terminal-Bench 2.1のhigh等は主催者側の別測定です。
- OpenAIは研究環境またはAPIで評価しており、製品版ChatGPTとはsystem promptや利用できるツールが異なり得ると注記しています。個別の設定が不明な項目を同条件とみなしません。
- 検索結果の要約ではなく、取得したページ本文・公式リーダーボードの公開データを採用します。取得後に値が更新される可能性があるため、再利用時には確認日と評価版を残します。
- `—` は調査範囲内で未確認という意味です。0点や、他の場所にも一切公開されていないことを意味しません。別ベンチマークや別単位の値から補完しません。

## 出典と取得方法

- **S21** — [GPT-6 Astra: A new generation of intelligence][S21]、OpenAI、モデル提供元の一次資料、2026-09-05取得。末尾のカテゴリ別評価表・共通注記・脚注を確認。
- **S22** — [Terminal-Bench公式リーダーボード（2.1）][S22]、評価主催者の一次資料、2026-09-05取得。Codex / GPT-6 Astra、掲載日2026-09-03。サイトの初期表示版と2.1を混同しないよう版を指定し、公式フロントエンドが使用する公開APIの該当行で確認。
- [取得したTerminal-Benchの公開データ（抜粋JSON）](sources/gpt-6-astra-terminal-bench-2.1-2026-09-05.json) — 読取APIのURL・リクエスト条件、5設定のスコアと信頼区間、ジョブの版・試行数を保存。
- [Terminal-Benchの公開ジョブ][job] — 公式リーダーボードからリンクされる評価成果物。
- [GPT-6 Astraモデルページ][model] — OpenAI、2026-09-05取得。モデルIDの確認用で、スコアの出典には使用していません。
- [Safety overview: GPT-6 Astra][release] — OpenAI、2026-09-03公開、2026-09-05取得。発表日の確認用。

[S21]: https://openai.com/index/gpt-6-astra/
[S22]: https://www.tbench.ai/?version=2.1
[job]: https://hub.harborframework.com/jobs/17d1a7f6-3339-4670-8b70-3b145979f57f
[model]: https://developers.openai.com/api/docs/models/gpt-6-astra
[release]: https://openai.com/index/safety-overview-gpt-6-astra/
