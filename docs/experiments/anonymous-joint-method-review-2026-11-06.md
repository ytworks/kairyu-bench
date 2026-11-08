# 匿名共同編集レビューとreserve設計（2026-11-06）

ユーザーの再開承認後、次周期の共同編集案についてClaude Opus 5.5の匿名セカンドオピニオンを取得した。この文書は方法論の助言と、問題本文・解答・成績を読まないreserve metadata監査を記録する。新しい重みの作成、モデル切替、WSP・GPU実行は、この作業では行っていない。

## レビューの範囲と本人確認

- 実行日時: 2026-11-05T15:47:01.602747+00:00 → 2026-11-05T15:47:41.741773+00:00。
- 要求モデル、実assistant model、usageのmodel keyはすべて `claude-opus-5-5`。別モデルへのfallbackなし。
- 匿名A〜Hの公開一般機構、codingと関連能力を優先する目的、中程度の周辺損失を許容する方針、反証方法だけを送った。
- 実候補ID、保存重みの数値・座標、実験成績、問題文、軌跡、ローカルpath、個人情報は送っていない。
- CLIのsafe/restricted mode、明示的な一般reviewer system prompt、空の作業directory、tools/MCP無効、session保存無効を使用した。tool使用0。推論blockとraw streamは保存していない。
- usage: inputTokens 2、cacheCreationInputTokens 2003、cacheReadInputTokens 0、outputTokens 3275。記録された費用はUSD 0.081532。
- 以前拒否されたprivate測定データ送信を再試行したものではない。既存のprivate candidate card収集helperも外部送信に使っていない。

これは**設計案への意見**であり、実候補の品質・測定結果・重み座標をレビューした証拠ではない。後から具体化したspecへ無条件にレビュー済みの印を移さない。

## A〜Hへの意見

| 設計 | 査読者の優先度 | 主な失敗モード | 選択を変える最小観測 |
|---|---|---|---|
| A インターフェースを保つ共同編集 | 2 | 演習の表層形式に過適合する恐れがある。その結果、未知のスキーマや長い思考の後で、名前が微妙に置換されたり引数が欠落したりする。 | 独立作成の未使用インターフェース課題を自由生成させ、構造化呼び出しの完全一致率と実行可否の差を、元モデルと現行の両方に対して見る。 |
| B アルゴリズム不変条件の共同編集 | 3 | 成功系の実装ファミリーだけが強化され、失敗系の修正にはつながらない恐れがある。一般的な推論が劣化する恐れもある。 | 失敗系ファミリーで実行テストが成功に転じる割合を見る。あわせて、成功系が崩れる割合とファミリー別の分母を報告する。 |
| C ツール工程の共同編集 | 1 | 呼び出し過多になると非呼び出し判断が崩れる。逆の場合は過少呼び出しになる。多ターンでは軌跡が分岐し、単ターン指標と乖離する。 | 自由生成の多ターン課題で、四指標を分離した混同行列を見る。特に、不要な呼び出しの増減を確認する。 |
| D 検索と実装を両立させるadapter方向 | 6 | 検索能力の見かけの保持が、評価データとの重複による可能性がある。また、群の切り出しで方向間の相互作用が壊れる。 | 重複が否定できる新規文脈検索課題で、抽出前後の差を見る。同時に、インターフェース・実行の後退がどう変化するかを確認する。 |
| E 選択的な出力校正 | 8 | 全タスクへ広く微小な歪みが生じる。BF16丸めで正則化した解が保存されないこともある。尤度改善を能力改善と誤認する危険もある。 | BF16で保存した実ゲインを用いて、自由生成でコーディング実行率とツール指標を見る。温度別にも確認する。 |
| F 共通利得方向と損失修復 | 7 | 共通性が構築手順の副産物にすぎない可能性がある。補償項同士が干渉し、利得を相殺することもある。 | 共通成分のみ、補償のみ、両方の三条件で実行結果を比較する。 |
| G 目的別adapter方向の共同再結合 | 5 | 丸めで小係数方向が消失する。目的間の重み付けが探索で使ったデータに過適合する。 | 方向ごとの保存後の残存率を、明示的な分母で報告する。そのうえで、保存後モデルの自由生成成績を見る。 |
| H 相補的なセットの累積 | 4（C・A・Bの結果に依存） | 集合間の干渉で利得が非加法的になる。校正は合成後にしか価値が判定できない。 | 群単位で一集合ずつ除いた比較と、縮小係数を変えた比較を行い、各集合の限界寄与を見る。 |

査読者はC→A→Bを先行し、G/Dを有界に比較してからHで累積し、E/Fは補助比較に置く案を出した。これは実測値を渡さない状態の順位である。既存候補の測定結果を否定したり、今後の実測順を自動決定したりするものではない。

採用判断へ取り入れる点は、自由生成の実行結果、元モデルと現行候補の両方との比較、引数・非呼出・完遂率の分離、BF16丸め後の実差分、組合せ後の群単位除去である。activationの大きさ、尤度、marginだけで能力を判定しない。長いthinkingに対して先頭tokenだけの採取が不十分になり得る点も指摘された。

助言はそのまま実行規則にしない。Fへの同norm無作為対照は必要時の単一診断であり、無作為な1座標探索へ戻る理由にはしない。Eへの温度別比較は行うとしても別の診断であり、正式WSPのsampling条件は変えない。Hの部品が単独で明確に勝つことを必須にせず、独立の弱点を狙う根拠があれば中立な部品を含む有界な相互作用試験を認める。最終的には共同セットの追加便益で判断する。

## metadataだけで調べたreserve

固定manifest SHA256は `a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b`。tierはitem単位で入れ子である。dev/testは既知の同一問題groupで分ける一方、tier境界はgroup単位ではない。したがってIDの差集合だけでは翻訳・別表示を除けない。

| 段階 | 件数 |
|---|---:|
| test10 | 778 |
| test30 | 2338 |
| test30からtest10の同一IDを除く | 1560 |
| 使用済みtest10と全dev metadataとの既知group交差を除く | −10 |
| metadataだけではcross-source同一性が未解決の族を隔離 | −59 |
| 提案reserve（まだ評価・gate有効化なし） | 1491 |

既知group交差10件はMGSMの別言語8件、MMLU-Pro／日本語ProXの同一question IDが2件だった。MMMUの標準表示と画像表示も設定名を除いて照合した。隔離59件はGSM8K-Platinum 34件と、上記除外後のMGSM 25件である。この2source間の正規groupは英語問題文のhashに依存し、保存manifestのIDだけからは復元できないため、本文を開かず保守的に除いた。

提案reserveは14能力領域と分布probeを保つ。code64、tools80、instructions115、agents8を含む。1,491表示は1,490 underlying groupで、reserve内部には同一groupの別表示1組がある。残す場合は独立問題として二重に数えず、group単位の不確実性評価を使う。未確認の意味的な重複や公開donorの訓練重複がないと証明したわけではない。

## 最終gateへ固定する前の条件

このtest30/test60は元モデルでは既に実行され、manifest作成時の出力長calibrationにもtestが含まれる。したがって「誰も評価していない新データ」「訓練汚染のないデータ」とは呼べない。今回提案できるのは、**新候補の評価・選択へ使っていないreserve**である。現在の監査ではrunの識別metadataとmanifestだけを読み、問題本文、解答、responses、report、個別scoreを読んでいない。過去の閲覧履歴を完全には復元できない点も残る。

1. この選択ID一覧・group規則・除外理由・source重み・過度な劣化の基準を先に固定し、以後の設計・縮小・選抜へreserve成績を使わない。
2. 元モデル・選抜候補・sampling・runtime・manifest／selection hashを凍結してから、同条件で一度比較する。元モデル側も現runtimeとの同条件性を確認する。
3. 1,491件の部分集合は公式full tier30と呼ばない。実行側がこの選択契約を検証・保存できることを確認してから有効化する。
4. 結果を見て候補を編集した場合はこのreserveを使用済みにし、次の最終確認には別集合を予約する。
5. 完全に新しい問題が必要なら、今回の差集合をその代用にせず、未使用のpinned問題集合とgroup metadataを用意する。

## 再利用する道具

今回の一回実行から再利用部分を `scripts/claude-method-review.py` へ整理した。元の取得済み回答を新scriptで再送信してはいない。既定はローカルpreviewだけで、出力directoryや外部sessionを作らない。prompt全体を匿名内容として確認した後のSHAを、送信時に明示する。

```sh
python scripts/claude-method-review.py \
  --prompt /mnt/nvme/<work>/anonymous-methods.txt \
  --output /mnt/nvme/<work>/review-new

# previewで表示したSHAと、ローカルで確認したpromptが一致する場合のみ:
python scripts/claude-method-review.py \
  --prompt /mnt/nvme/<work>/anonymous-methods.txt \
  --output /mnt/nvme/<work>/review-new \
  --send --approved-prompt-sha256 <reviewed-sha256>
```

`--claude-binary`で既存CLIを指定できる。既定要求モデルは `claude-opus-5-5`、fallbackなし。timeout・費用上限も引数で固定する。出力先の既存directoryは拒否し、失敗・timeoutを含め同じ場所への自動再送信はしない。実assistant modelとusageのモデルが一致せず、想定labelが揃わない場合は成功扱いにしない。

パス・URL・credentialらしい文字列のlintは補助にすぎず、任意の私有IDや測定値を検出できる保証はない。SHA確認も匿名性の証明ではない。各promptの一般機構・目的・反証方法以外の情報がないことを送信前に確認する。raw stream、thinking block、stderr本文、環境変数はreceiptへ保存しない。CLIの認証情報は既存の仕組みを使い、コピー・表示しない。

reserve監査は `scripts/wsp-reserve-audit.py` で再現できる。入力はmanifest metadataだけで、dataset loader、問題本文、解答、responses、成績、ベンチ実行を使わない。

```sh
PYTHONPATH=src python scripts/wsp-reserve-audit.py \
  --manifest src/kairyu_bench/data/wsp/manifest.json \
  --reserve-tier 30 --seen-test-tier 10 --seen-dev-tier 60 \
  --output /mnt/nvme/<work>/reserve-audit-new.json
```

既存出力は上書きしない。今回の再利用CLI出力は、最初の独立監査と同じ1,491 ID・selection hashになった。道具が確定するのは既知groupのmetadata交差であり、閲覧履歴や意味的な重複・訓練汚染まで認証しない。

合成テスト8件で、preview無送信、prompt/hash guard、モデル・usage不一致、tool使用の拒否、thinking等の非保存、timeoutの一回終了、翻訳・別表示除外、未解決族の隔離、本文field拒否、重複ID拒否を確認した。Claudeへの追加問い合わせは0件だった。

## 保存証拠

以下は実験artifact rootからの相対pathである。

- `analysis/resumed-joint-cycle-20261106/claude-opus55/anonymous-design-prompt.txt`
- `analysis/resumed-joint-cycle-20261106/claude-opus55/review-receipt.json`
- receipt SHA256: `1c84cd9cbe2e87e085639b290aa83fdfce9c2fe39a591ecd41b1e6777e555ded`
- prompt SHA256: `951b35407f45cc81aaa363a0233d3337b7426411372ba4c12a8ce1683e5f380e`
- `analysis/resumed-joint-cycle-20261106/reserve-metadata-audit.json`
- selection SHA256: `5c05fc8769c2b5ead41502648cd3ed7088ce24fa830a1355548338b7b06f692b`
- `analysis/resumed-joint-cycle-20261106/audit_reserve_metadata.py`

[前回の実験記録](2026-11-04-gemma4-weight-editing.md)と[観測ツールの説明](../wsp-weight-observation.md)は別に維持する。
