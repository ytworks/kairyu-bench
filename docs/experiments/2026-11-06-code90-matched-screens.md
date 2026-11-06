# Code90：dev90の利得と既知64問への転移確認（2026-11-06）

**native案Bのdev90での利得は、既知64問では確認できませんでした。code >90は未達です。**
4条件のdev90と、B・0397対照の既知64問は全て完了し、各runの未採点・API errorは0でした。Bは広い能力への影響を調べる暫定候補として保持し、F・Gは採用していません。

[再開計画](2026-11-06-code90-plan.md)、
[科学コード校正と行列pilot](2026-11-06-science-matrix-pilot.md)、
[前回0397の固定確認](2026-11-06-frozen-confirmation.md)を併読してください。

## 比較条件と分母

Bはscience-native-contracts-v1、Fはscience-matrix-balanced-v2、
Gはscience-joint-matrix-numerics-native-contracts-v2。対照は0397です。
自作校正での微分・実BF16 finite測定から選んだ共同編集を、公式WSPの実行成功で測りました。自作16学習case・8転移caseの点数をWSPに加算していません。

| 集合 | split / tier | HE+ | MBPP+ | LiveCodeBench | SciCode | 各条件の採点数 |
|---|---|---:|---:|---:|---:|---:|
| dev90 | dev / 30 | 30 | 30 | 20 | 10 | 90/90 |
| 既知64 | test / 30、固定確認subsetのコード部分 | 20 | 20 | 14 | 10 | 64/64 |

dev90のmanifest SHA256はa3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b、
既知64は52a76039574df920138e7b88247955859af89865b2aa565fb11b00a0682177a6。
各比較内で全ID・分母・manifest SHAが一致します。既知64は以前に成績を見た参照集合で、独立した未見holdoutではありません。保存上のsplit=testは変更せず、dev専用の反復証拠ツールも使用していません。

各条件は **1つのTP2 replicaへ直接接続、chat concurrency32**。dev90は4条件を同時に、既知64はB・0397を同時に測定しました。native thinking ON、checkpointの公式generation設定を使い、今回だけのsampling変更はありません。起動元は共通commit 5e3f2df0c7f455013af3fa2d7abb3719858f808f。dev90の実outer containerは4つとも同じimage sha256:4fd52f9993e5d5a51918dd091958b6b408129f9b4d7c2f257ec4b116c53231f7で、runtime configと16 filesystem layersの一致をinspect証跡で確認しました。異なるbuild attestationログや事前buildのimage IDから推定したものではありません。

前回0397の既知64 **82.5476** は4 replica配信経由のfull14評価内の値で、今回の1 replica・コードだけの条件と異なります。今回の対照は **77.0119**。旧82.5476を新しいpaired baselineに置き換えたり、その差をbatch構成または重みだけの因果効果と決めつけたりしません。

## dev90：nativeの3問利得、科学コードは不変

codeは4 sourceの等重み平均です。SciCodeは各問題の成功段階率を10問で平均し、全段階の単純成功率や完全正解問題数とは区別します。数値は0–100点、差とCIはpercentage pointsです。

| 条件 | code | 対照差 [paired 95% CI] | HE+ | MBPP+ | LCB | SciCode |
|---|---:|---|---:|---:|---:|---:|
| 0397対照 | 70.4921 | — | 29/30 | 22/30 | 15/20 | 36.9683 |
| B native | 73.4087 | +2.9167 [0.0000, +6.6667] | 30/30 | 23/30 | 16/20 | 36.9683 |
| F matrix | 67.5754 | −2.9167 [−7.5000, +1.6667] | 28/30 | 23/30 | 12/20 | 40.3016 |
| G joint | 70.7698 | +0.2778 [−1.6667, +2.5000] | 29/30 | 22/30 | 15/20 | 38.0794 |

以下は公式scoreの変化です。0.5閾値を跨ぐ「flip」だけに縮約しません。

| 条件 | 利得 | 損失 |
|---|---|---|
| B | HE151、MBPP641、LCB abc390_g：各0→1 | なし |
| F | MBPP801：0→1、Sci35：1/3→2/3 | HE95、LCB3777・abc399_c・arc195_b：各1→0 |
| G | Sci35：1/3→2/3 | Sci72：7/9→5/9 |

Bの科学10問・39段階の合否は対照と全て同じでした。F・Gは35.2を回復（対照の例外hintはValueError）。Gは72.7・72.8を失い、両方のhintはAssertionErrorです。Gの公式threshold flipは獲得1・喪失0ですが、Sci72が0.5以上に残るためであり、退行なしを意味しません。

科学の合計段階成功数は対照18/39、B18/39、F19/39、G17/39。これは診断値で、上表の公式source平均とは別です。完全に解けた科学問題は各条件1/10、段階timeoutは全条件0でした。

## 既知64：利得と損失が入れ替わり、科学平均も増えず

| 条件 | code | 対照差 [paired 95% CI] | HE+ | MBPP+ | LCB | SciCode |
|---|---:|---|---:|---:|---:|---:|
| 0397対照 | 77.0119 | — | 18/20 | 16/20 | 12/14 | 52.3333 |
| B native | 76.4762 | −0.5357 [−8.5119, +7.3810] | 19/20 | 16/20 | 11/14 | 52.3333 |

| 種類 | 変化した項目 |
|---|---|
| 利得 | HE163、LCB abc399_d、MBPP279：各0→1。Sci21：0→1/3 |
| 損失 | LCB abc397_f・abc388_e、MBPP635：各1→0。Sci24：1/3→0 |

実scoreは4利得・4損失。公式threshold flipの3/3には、0.5未満のSci21・24の変化が入りません。科学の平均は同じでも、21.2回復と24.2喪失があります。

Sci15.2は **両条件とも公式300秒timeout**、0397対照の26.3もtimeoutでした。Bの26.3はValueError hintで不合格です。これらは公式採点器が失敗とした段階で、未採点API errorではありません。生成完了後の約300秒の待ちと整合しますが、sanitized metadataだけで環境・数値・アルゴリズムのどれが原因かは確定しません。正解・stderr本文・隠れたreasoningを使った再採点はしていません。

## 所要時間と判断の範囲

| 集合・条件 | generation秒 | wall秒 | leaf API calls | input / output tokens |
|---|---:|---:|---:|---:|
| dev90・0397 | 300.7 | 355.3 | 119 | 97,086 / 348,578 |
| dev90・B | 291.9 | 363.5 | 119 | 95,420 / 335,524 |
| dev90・F | 283.4 | 352.6 | 119 | 96,759 / 330,645 |
| dev90・G | 277.6 | 351.4 | 119 | 96,935 / 342,286 |
| 既知64・0397 | 255.5 | 560.8 | 87 | 64,916 / 280,427 |
| 既知64・B | 255.2 | 557.9 | 87 | 65,249 / 280,086 |

call数・token数は保存済みleaf response metadataに基づき、coverageは全件。SciCodeのaggregate sampleを二重加算していません。全callのfinish_reasonはstopです。呼出時間の単純和は並列wall時間とは別の値です。

公式paired bootstrapは2,000再標本化、seed=20261104。各条件1回の生成なので、CIはこの問題集合の不確実性を表し、生成の反復変動までは測りません。候補探索で複数の条件を見たこと、既知64の過去露出も残ります。全比較でCIは0を含み、一般的な能力向上・重みだけの因果効果・目標達成を断定しません。

Bは元モデルとの新しいfull14対応比較で副作用を確認する**暫定候補**です。F・Gはこの結果では採用していません。B対元モデルの各1 TP2 replicaへの配置を準備中で、本記録にその評価結果はまだ含みません。旧full10の実績は614–636秒ですが、今回の1 replicaへの負荷集中を考えた15–25分は計画上の目安に過ぎません。tier名はhard wall上限ではなく、既存agent timeout倍率48の長時間化リスクも別途管理が必要です。

今回測ったのはcodeのみで、他13分野・全体overall・NLL/KL probeは未測定です。dev10 full14にはSciCodeがなく、今後の広域確認だけで4 sourceのcode >90を達成扱いしません。

## 証跡

実験artifact rootからの相対名です。各JSONにrun ID・全選択ID・manifest SHA・入力hash/stat・数値metadataを保持しています。

- analysis/code90-resume-20261106/matched-code90-final-analysis.{json,md}
- analysis/code90-resume-20261106/matched-code90-comparisons/*-official.{json,md}
- analysis/code90-resume-20261106/matched-code90-comparisons/*-evidence.json（devのみ）
- analysis/code90-resume-20261106/matched-code90-runner-image-proof.json
- analysis/code90-resume-20261106/known64-first-pair-final-analysis.{json,md}
- analysis/code90-resume-20261106/known64-first-pair-official.{json,md}
- analysis/code90-resume-20261106/full14-runtime-risk.json

比較は既存公式wsp compare --partialを使用しました。partialはコードsourceのみの選択を表し、90問または64問の中に欠測があるという意味ではありません。
