# 2026-11-07: canonical WSP 30/60 の全問運用監査

新規 WSP、GPU 計測、候補作成、serving 変更は行っていない。現 source と既存 raw の ID・件数・全採点・個別 token/sample 条件を照合した。過去の候補順位や採用判断は根拠にしていない。

正規 tier30/60 は **2338 / 4841 問**。14 領域をすべて含み、診断 probe **96 / 190 問**も実行対象である。dev/test は同じ件数の別問題群、tier は入れ子。manifest SHA は `a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b`。コード 90 問だけの screen は canonical tier30 ではない。

| 領域 | tier30 | tier60 |
|---|---:|---:|
| agents | 10 | 20 |
| code | 90 | 198 |
| commonsense | 900 | 1780 |
| diagnostic | 96 | 190 |
| expert | 48 | 98 |
| facts | 120 | 240 |
| instructions | 170 | 416 |
| knowledge | 180 | 360 |
| long-context | 50 | 110 |
| math | 135 | 251 |
| multilingual | 187 | 390 |
| reasoning | 76 | 152 |
| safety | 108 | 216 |
| tools | 120 | 300 |
| vision | 48 | 120 |

| Code source | tier10 問数 | tier30 問数 / 元 dev score % | tier60 問数 / 元 dev score % |
|---|---:|---:|---:|
| humaneval-plus | 10 | 30 / 96.666667 | 60 / 98.333333 |
| mbpp-plus | 10 | 30 / 76.666667 | 60 / 76.666667 |
| livecodebench | 6 | 20 / 75.000000 | 50 / 74.000000 |
| scicode | 0 | 10 / 30.301587 | 28 / 32.687075 |
| 合計 | 26 | 90 | 198 |

コード領域は **source 平均の単純平均**。tier10 は 3 source、tier30/60 は SciCode を含む 4 source であり、異なる tier の点数を同条件の順位にしない。全 32 source の件数、code の全問題 ID・個別得点、元モデル dev/test の全 6 run は [数値証拠](evidence-2026-11-07-f02/wsp-large-tier-audit.json) に保存した。

| 元モデル dev run (2026-11-04) | 完了/対象 | Code % | Overall % | wall 分 | load 秒 | chat / logprob calls |
|---|---:|---:|---:|---:|---:|---:|
| wsp-base-10-dev-1-20261104 | 778/778 | 84.444444 | 81.223722 | 10.567 | 106.8 | 559 / 3062 |
| wsp-base-30-dev-1-20261104 | 2338/2338 | 69.658730 | 78.768381 | 33.972 | 109.9 | 1752 / 9193 |
| wsp-base-60-dev-1-20261104 | 4841/4841 | 70.421769 | 76.604069 | 59.775 | 107.9 | 3759 / 18141 |

各 run は completed、未採点 0、現 manifest と全選択 ID・上限・samples が一致。実時間には dataset load を含む。source_finished_seconds は同時進行した各 source の終了時刻で、加算できない。30 分版の最終終了は Terminal-Bench (2038.3 秒) だった。

当時の [環境記録](2026-11-04-gemma4-weight-editing.md) は元 checkpoint BF16、8GPU / TP2×4、MTP4、native thinking、generation config temp 1 / top_p .95 / top_k 64 を記載する。ただし raw run.json は concurrency・OCI image・全 runtime 設定を保存しておらず、厳密な現条件一致は証明できない。現 [CLI](../../src/kairyu_bench/wsp/cli.py) の既定 256 chat / 32 logprob / 4 agent を、過去 run の実指定の証拠として代用しない。

現在の AV checkpoint (`5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf`) は canonical dev10 の全完了が 2 回、code 87.777778% / 26 問は同得点。[反復監査](2026-11-07-av-repeat-audit.md) の通り capture・runner・重複負荷が異なり、現条件と厳密一致する先行 repeat は none。151 歴史 run と fresh AV を確認した範囲で、AV の canonical dev30/dev60 は **0 回**。追加 1560 / 4063 問、code の追加 64 / 172 問を含む全体は未測定で、SciCode も AV 未測定。過去の 13 同点 run を AV の反復とは数えない。

8GPU の同一 checkpoint TP2×4、256/32/4 を使うなら、まず AV の全 replica と chat/logprob の同一性を固定し、**AV 全問 dev30 → 候補 全問 dev30** を同じ runner/source/capture/generation/agent 条件で比較する。続く tier60 には **AV 全問 dev60 と候補 全問 dev60** の新しい組が必要。旧元モデル値や code90 screen はこれらの baseline を置き換えない。元モデルとの現在条件の改善も主張する場合は、元モデルも同じ全問条件で再測定する。現在の mixed checkpoint gateway をそのまま全 replica 評価に使わない。構成変更後の dev10 との連続比較が必要なら AV dev10 も再測定する。test は調整に使わず最終確認へ分離し、既に実行済みの test を未見と呼ばない。

TP2×1、32/8/1 を続行する時間の**粗い概算**は tier30 **83.5–96.5 分**、tier60 **147.0–169.8 分**。式は旧全問 tier wall × (実測 AV dev10 wall 1559.1–1800.7 秒 / 旧元モデル dev10 wall 634.0 秒)。モデル・capture・runner・他 GPU 負荷の差を含むため速度倍率の因果推定ではなく、SciCode の逐次生成・長文・agent mix も未校正。30/60 の AV 実測時間や上限保証ではない。8GPU へ戻す場合も旧 34/60 分を保証せず、最初の AV 全問を実測して計画を更新する。いずれも問題数・既存条件の削減で時間を合わせない。

SciCode の現 [source](../../src/kairyu_bench/wsp/sources/code.py) は、各問題を手順順に生成し、そのモデル自身の先行コードを後続 prompt に含める。sample 得点は **合格した tested steps / tested steps**、item は sample 平均、source は item 平均。全問題成功の boolean だけではなく部分点を保持する。公式指定の author steps (13,5), (62,0), (76,2) は zero-based で生成/採点から除かれる既存仕様であり、今回の除外ではない。どれかの生成 step が token cap で打ち切られた sample は 0 点。現 [scorer](../../src/kairyu_bench/wsp/scorers/scicode_score.py) は 300 秒/step、timeout は失敗、harness/H5 読込の環境不良は例外として区別する。300 秒は現 WSP 実装の条件で、公式全実験と同じ 1800 秒制限だとは称さない。

再現: [CPU 監査 script](scripts/fresh-wsp-large-tier-audit.py) を PYTHONPATH=src で実行し、project / repository / legacy-results / 新規 output を指定する。旧結果は変更せず、数値 JSON は入力・source・helper SHA を保持。準備時の CLI help の /work 権限拒否と None area の一時 sort 失敗も記録した（ベンチマーク実行 0）。
