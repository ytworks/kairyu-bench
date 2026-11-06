# 構造別donor方向の匿名手法レビュー（2026-11-06）

全text投影の低ランクdonor方向に、目的ごとのsigned係数を共同で付ける6案について、Claude Opus 5.5からセカンドオピニオンを取得しました。送信は一般化した機構・目的・反証方法のみで、モデル名、実checkpoint座標、評価問題・答案、候補の実測成績は含めていません。これは**手法への助言**で、候補の効果やcode >90を検証したものではありません。

## 実モデル確認と形式エラー

既存helperのreceiptはverified_requested_model=false、labels_match=falseを保持しています。回答のreviewsが期待したlistではなく、A–Fをキーとするdictだったためです。再送はしていません。

別の読取監査で、A–Fが過不足なく存在し、各案にcritiqueとcheapest_falsificationがあることを確認しました。assistant_modelsとmodel_usageはともに **claude-opus-5-5のみ**、returncode=0、is_error=false、tool_uses=0です。観測されたモデルは要求と一致していますが、元helperが成功したようにreceiptを書き換えていません。

実行71.046秒、記録費用$0.146436。モデル使用量はinputTokens=2、cacheCreationInputTokens=2,141、outputTokens=6,465、cacheReadInputTokens=0。ツール・MCP・web searchは使用せず、raw stream・thinking本文・環境変数は保存していません。

## 6案と反証に役立つ助言

| 案 | 仮説 | 主な懸念と確認候補 |
|---|---|---|
| A balanced | 全方向でコード・科学の穏やかな共同目標 | 最小Frobenius normは機能保全ではない。active constraints、trust limit、実BF16で残る変位、fold間の再現を確認する |
| B contracts | 科学API・shape契約を強める | 小さいfoldの表層表現や負係数への外挿に過適合し得る。目的別の実行成功と、学習に使わないfoldへの転移を見る |
| C numerics / algorithms | 数値・アルゴリズムを強める | 正しい答案の尤度が数値安定性や実装成功を保証しない。Aとのcost-weighted cosineと実行成績で別方向の価値を調べる |
| D MLP | MLP投影だけで実装目標を満たす | 同じ制約・trust範囲で比較し、必要な実変位の違いをscopeの効果と混同しない |
| E attention | 注意投影で実装・引数処理を改善 | Q/KとV/Oを区別して予測と実BF16有限差を点検。reference trackingや不要呼出への副作用を見る |
| F late layers | decoder後半に限定した補正 | 出力語彙・形式への傾きだけで尤度が上がる可能性。前半を含む案との実行比較、既存native編集との相互作用を測る |

査読は全同係数が既存scalar mergeのbytesを再現すること、zero係数がsigned zeroを含む元bytesを保つこと、FP32演算順と一度のBF16丸めを固定することを重視しました。連続係数の予測と、実モデルの変更数・norm・donor方向への投影・直交残差を分けて記録する助言です。

全方向とMLP限定などを比較するとき、名目目標が同じでも実変位量は同じとは限りません。fold別cross-fit、少数のnormを合わせた対照、係数を縮小した有限差は、必要性と費用を見て追加する診断候補です。全部のcontrolを無条件に増やす実行計画にはしていません。

## 採用する範囲と訂正

- 純donorと既存native編集を重ねたparentは、別々に実BF16 finiteと生成コード実行で確認します。tensorが非重複でも機能上の加算性は保証されません。
- 回答の「Bの0.02はAの科学0.01の4倍」は算術誤りです。実際は2倍であり、その記述は採用しません。
- MLP限定の制約が不成立でも、言えるのはその基底・測定点・目標・trust範囲で成立しなかったことです。「注意機構が能力に必須」とは結論しません。
- 全purpose floor維持を絶対条件とする助言は、新しいhard gateへ変更しません。ユーザー方針はコードを主、関連能力を副とし、他分野の多少の損失は得失を示したうえで許容するものです。
- 新しいreserveを複数controlで必ず測るという提案も、既存の確認契約を自動で置き換えません。既知参照と独立した確認を区別し、選択後の都合のよい再調整を避けるという方法上の注意として扱います。
- donorの学習元や評価との重複は未解決です。レビューからclean training dataや未見転移を主張できません。

[Code90計画](2026-11-06-code90-plan.md)と
[今回のmatched screen結果](2026-11-06-code90-matched-screens.md)に従って実測で判断します。proxyの改善だけを公式実行成功の代わりにはしません。

## 証跡

実験artifact rootからの相対名です。

- analysis/code90-resume-20261106/opus-structured-donor-review/prompt.txt
- analysis/code90-resume-20261106/opus-structured-donor-review/receipt.json（原本不変）
- analysis/code90-resume-20261106/opus-structured-donor-review/audited-receipt.json（dict形式・全label・観測モデルの別監査）

prompt SHA256: b06eaf61b6be23297d6b24699184d7550bd606262d6776b546c372881a9acd75。
匿名性はprompt全体の人手確認を伴う運用で、pattern lintだけの機械的保証ではありません。
