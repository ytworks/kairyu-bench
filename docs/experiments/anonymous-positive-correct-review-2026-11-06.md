# 正答尤度を直接扱うdonor編集の匿名手法レビュー（2026-11-06）

correct-minus-foil marginだけを改善した2案でcorrect NLLが悪化したため、
正答の平均logPとmarginを別々に制約する6案をClaude Opus 5.5へ相談した。
送信は匿名化した一般機構・目的・反証方法だけで、候補の実測成績や評価答案の
査読ではない。**尤度の改善からコード能力の改善を結論しない。**

> 後続のframe監査で、旧proxyが公開final境界ではなくthinking ONのassistant-start条件であると判明した。このレビューはその発見前の一般手法助言であり、旧42foldからの6連続案は未適用のままretireした。公式WSPの得点はこの自作proxyとは別である。旧数値・receiptは保持し、境界を修正した新しい収集へ読み替えない。証跡はanalysis/code90-resume-20261106/public-objective-source-audit.jsonとpublic-final-frame-v3/。

## 実モデルと記録

receiptのverified_requested_modelとlabels_matchはいずれもtrue。
assistant_modelsとmodel_usageはclaude-opus-5-5のみで、A–Fの6回答を確認した。
returncode=0、tool_uses=0、fallbackなし。88.766秒、記録費用$0.174604。
usageはinputTokens2、cacheCreationInputTokens2,387、outputTokens7,775、
cacheReadInputTokens0。raw stream・thinking本文・環境変数は保存していない。
元receiptは変更せず、匿名性はpattern lintだけの保証ではなく送信内容の確認に依存する。

## 6案への助言

| 案 | 機構 | 有用な反証・診断案 |
|---|---|---|
| A | 全410方向で正答尤度を改善 | 正答・foil・marginを別計測。logitの一様な校正や答案形式のpriorでも改善し得る |
| B | 科学目的を強める | Aとの差が実BF16で残るかを見る。数値tokenのprior改善と問題依存の計算を区別する |
| C | algorithm/implementationを強める | code勾配とのcost正規化cosineと候補間cosineから、実質同じ方向の重複を調べる |
| D | MLP投影に限定 | 連続normとBF16実変位を分ける。制約不成立を「注意機構が能力に必須」と一般化しない |
| E | Q/K/V/Oに限定 | Q/Kの非線形性とV/Oを区別。先頭・末尾だけのKLは途中位置や自由生成の変化を見落とす |
| F | 後半層に限定 | 出力分布の校正だけで目的を満たしていないかを確認。小さいnormだけで優位としない |

全案で、両foldをfitしているため制約達成は未見の証拠ではない。
正答logPが増えても、foilも同程度に増えることや、文脈に依存しない形式priorは残り得る。
重みの実BF16丸め、元からの変更量、復帰の一致、実生成・実行をそれぞれ確認する。
学習donorの元モデル差と評価データ重複は未解決のままである。

## 採用する範囲と助言の限界

今回の実装では、同じケース・公式frame・source・方向順序で正答logPとmarginを
分けて記録し、84制約と許容損失を明示する。continuous predictionと実BF16効果を分離し、
元重み復帰と実行成績を採否の根拠にする。レビューreceiptもplanへSHA固定する。
安価なcost重み候補cosine、algorithm/code勾配cosine、binding constraintsと
係数boundへの到達を診断値として追加した。cosine自体は能力得点ではない。

回答にある0.7/0.9等のcosine・尤度説明率・線形性の数値、複数scale/null/control、
全位置KLや温度fitは**提案**であり、新しい必須gateとして採用していない。
特に同一forwardのA/A誤差と、異なる問題のcase bootstrap不確実性は別の量である。
小さい局所予測だから不可能、またはmarginが最大案を超えないから無価値とは扱わない。
ユーザー方針はコードを主、関連能力を副とし、他領域の多少の損失は具体的な得失と
一緒に判断するもの。全fold非退行を新たに要求しない。

reserveに複数候補やnullを必ず投入する提案も、現在の確認契約を上書きしない。
既知集合、fitした校正、別familyの転移、独立した最終確認を区別する。
校正文の修正はrevision2として別に保存し、旧結果や既見集合を未見に戻さない。
compositionは部品の効果を加算せず、実際に組み合わせた重みで測る。

[目的補正の実測根拠](2026-11-06-positive-correct-donor-amendment.md)と
[dev90・既知64・full14の結果](2026-11-06-code90-matched-screens.md)を参照。
Bのfull14純損失と既知64未改善は保持し、今回の手法レビューを成功の代用にしない。

## 証跡

実験artifact rootからの相対名：

- analysis/code90-resume-20261106/opus-positive-correct-review/prompt.txt
- analysis/code90-resume-20261106/opus-positive-correct-review/receipt.json

prompt SHA256: 04f6c3c7c1c5267f267d213a87ec211205f895f88c270388dc9bedc1a94aa09d。
receipt SHA256: 304dd018e65286978ffee99667b491ae67ad62c1b4c6d907acc4d8f089db7127。

## 公開final境界を修正した後の匿名amendment

追加の一般手法レビューでは、A（全方向）、B（科学重点）、C（algorithm/implementation重点）、
D（MLP）、E（attention）、F（後半層）の6案を、明示的な空thought閉鎖の後に
公開答えを評価する条件で再検討した。候補の実測値・重み座標・問題本文は送っていない。
receiptはverified_requested_model=true、labels_match=true、assistant_modelsと
model_usageはclaude-opus-5-5のみ。returncode=0、tool_uses=0、fallbackなし、
82.064秒、$0.168356。usageはinputTokens2、cacheCreationInputTokens2,606、
outputTokens7,375、cacheReadInputTokens0。原本receiptは変更していない。

主な助言は、公開答え境界・source・token・padding・serial batchingを揃えること、
飽和した正答logPへ不可能な固定利得を要求しないこと、B/CがAと同じ制約になる場合は
重複を明示して除外することだった。A/Fの校正・形式prior、B/Cのfitした文章への過適合、
DのBF16丸め消失、EのQ/KとV/Oの非線形性を、それぞれ実際のコード能力と区別する。
これらは一般的な反証仮説であり、この候補で原因が確定したという査読ではない。

今回採用するheadroom方針は、元のfold平均正答logPをL≤0、H=−Lとして、
主目的の要求利得をH≤0.004なら0、それ以外はmin(τ, 0.25H)とするもの。
τは通常0.01、重点目的0.02。式・scope・許容損失は候補の予測を見て変更せず、
元のheadroomと展開後の制約をplanへ保存する。これは実験上の分解能・予算の選択であり、
ノイズ推定やユーザーから要求された数値ではない。各caseのheadroomも記録するが、
それを見てfold目標を後から調整しない。同じscopeかつ全制約が一致する案はsolve前に除外する。
採否には、元重みから一度だけ丸めた実BF16変位、実測した正答logP・foil・margin、
復帰検証、公開生成の実行結果と公式WSPを使う。logPの改善は正解率の改善を保証しない。

回答中の「necessary」はそのまま運用要件にはしない。ユーザーはコード90を目指す
継続を既に承認しており、レビューが述べた別の実行承認待ちは追加しない。
既存の全410行列の実BF16 replayと重複する巨大な事前丸め計算、未収集のtoken別勾配分解、
温度fit、Q/K専用・V/O専用の追加solve、複数scaleの大量診断は今回は必須化しない。
連続解の予測と実丸め効果は引き続き別々に記録する。数値最適化の未収束を、
大きな係数だけを根拠に数学的な不可能性へ読み替えない。

[境界発見と3例oracleの実測](2026-11-06-public-answer-frame-audit.md)は、
この一般手法レビューと別の証拠である。強制した空thoughtという条件は通常の
thinking生成後の答えと異なり、両foldをfitした制約達成も未見への汎化ではない。
コードを主とする実際の得失で判断する方針と、他領域の多少の損失を許容する方針は維持する。
旧frameの6案を復活させたり、既見集合を未見と呼び直したりしない。

追加receipt: analysis/code90-resume-20261106/opus-final-frame-amendment-review/receipt.json。  
prompt SHA256: 8c1a6cf4b12b0a28ae1737d6f5a46ce582b8912a00e22d50d6d86a80945fcef2。  
receipt SHA256: 8b8e7191f9c9822dec2ee528cce534fc2b51260d5d9bac45c5f38fc707eab026。

