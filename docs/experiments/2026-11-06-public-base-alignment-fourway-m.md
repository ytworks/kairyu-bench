# M: 公開改変baseと固定adapterの4条件比較・事前契約（2026-11-06）

**これは実施前の方法・入力固定記録であり、Mの能力結果ではない。**
[Lの公式dev90](2026-11-06-k1-exported-text-c1-review.md)はfresh original比−3.517857 pp、
出力token約1.47倍で不採用だった。次は係数を探索せず、同じexported-unit text LoRAの
効果が公開改変baseに依存するかを4条件で区別する。
Lの失敗をbase不一致の証明とせず、公開baseへの移植が成功するとも仮定しない。

## 公開baseの固定と分かっていないこと

対象Pは [coder3101/gemma-4-31B-it-hereticの固定revision](https://huggingface.co/coder3101/gemma-4-31B-it-heretic/tree/ea846e2b6d58fc9f943c462c19a1b60891407bf6)、
ea846e2b6d58fc9f943c462c19a1b60891407bf6（公開履歴2026-04-09）。
これはK1 adapter upload（2026-04-12）より前の公開revisionだが、
adapter configのtraining revisionはnullであり、真の学習weightsやローカル変更は不明である。
最新公開headのtemplate変更を学習時templateの証拠にしない。

既存の公開metadata＋全header監査では、1188 stored tensorの名前・BF16 dtype・shape・
サイズがoriginalと一致し、tensor payload合計は62,546,177,752 Bだった。
shard分割とindex weight mapは異なるため、公開Pのindexを保持する。
header照合は値の同一性や能力の互換性を証明しない。
pre-adapter revisionと監査時headの公開LFS OIDは一致するが、
それも実download後のlocal fullhash検証の代わりではない。

公開configの構造差はtext_config.moe_intermediate_size=nullの有無だけ。
tokenizer.jsonとgeneration_config.jsonはoriginalとbyte同一だが、
chat_template.jinjaとtokenizer_config.jsonは異なる。
「tokenizer全設定が同一」「著者protocolを再現」とは述べない。

| 公開P weight file | Bytes | 公開LFS SHA256（local検証前のpin） |
|---|---:|---|
| model-00001-of-00002.safetensors | 49,923,154,850 | 57338b5e4a55441e20853ce7aedb3d93b6fee5f5095ac67e2610d6cbd3206f38 |
| model-00002-of-00002.safetensors | 12,623,183,414 | 607d371a9e209fecbd12d53256fe8052f340c6ea8ce3218553c62ac95a678b2e |

## 固定する4条件と推論protocol

| Arm | 保存weights | 意味 |
|---|---|---|
| O | 検証済みoriginal BF16 | 基準 |
| O+D | 既存Lと同じ固定text D、exported coefficient1 | originalへのcross-base transfer |
| P | 公開固定BF16 baseのみ | 公開baseの効果 |
| P+D | 同じtext DをPから一度merge | baseによるadapter効果の変化 |

DはLでlossless抽出したF32 text820 factors／410 pairsで、vision378 factorsを明示除外したもの。
rank32、alpha32、普通LoRA scale1、係数1を維持し、再fitや振幅gridは行わない。
P+DはPを直接基点に作り、丸め済みO+Dへ後からbase差分を加えない。
同じ因子でもbaseのBF16格子が異なるため、保存後の実deltaは各baseに条件付けられる。

全条件でoriginalのconfig、generation_config、tokenizer.json、tokenizer_config、
chat_template、processor_configをbyte同一で使用する。
公開Pの元metadataはprovenanceとして別に保存し、書き換えて著者の配布物と偽らない。
これは**元protocolに固定した公開base view**であり、Heretic/K1著者のdeployment再現ではない。
native thinking ON、公式sampler、caps、scorer、direct TP2と各c32の比較条件を揃える。
4armは同じ比較窓で測り、過去Lのscoreを新しい4条件のO+D実測として代用しない。
正式run ID・endpoint・tree・counter bindingは実作成／配信後の実証跡で固定する。

## 事前に固定する比較と解釈

主比較は **P+D − O**。公式codeの4-source等重み平均を維持する。
同じ固定90項目（HE30／MBPP30／LCB20／SciCode10）を全armで採点し、
新しい未加重overallや成功宣言用の重みを作らない。
known64の90超という目標をこのdev90で置き換えない。

副比較はP−O、(O+D)−O、(P+D)−P。
相互作用は、各source内の同じpaired項目について
[(P+D)−P]−[(O+D)−O]を求め、4-source等重みで記述する。
不確実性を示す場合もpaired/source構造を保ち、
独立性の根拠なしに4armのvarianceを単純加算しない。
公式paired CIを優先し、相互作用は機序の証明や自動採用gateにしない。

SciCode部分点、binary flip、成功集合のunionを区別する。
best soloを上回る累積改善に、union外の新成功を必須としない。
全arm単回stochastic出力での個別得失を因果的なmodule機能へ帰属しない。
正の相互作用も真のtraining base一致や「DにはPが必要」の証明ではない。
Pの修復・悪化、template条件との相互作用はこの設計だけでは分離できない。

既存authored24は構造smokeであり目的指標ではない。
長さ、cap、unknown、形式、実usage、wallをofficial scoreと併記する。
保存callと起動discovery probe、独立HTTP回数、思考tokenを混同しない。
追加logit probe、独立GPU oracle、seed/grid、著者templateの追加armをこの計画だけで必須化しない。
新しい承認待ちや全領域非退行基準を設けず、coding優先・関連能力と多少の他分野損失を個別に報告する。

## 最小のbytes/provenance契約と現在のplan

base-view planはheretic-public-base-official-protocol-v1/plan.json、SHA
00b3d5c5f2016ac9d9ade5c2ddf88b367c2ea9581418d3191013c9bb79e41182。
plan作成時のstatusはprepared_not_executedで、base viewの取得・検証用入力を固定したもの。
rootはこのSHAの取得planを承認済み。これはdownload完了や4armの実配信／公式run完了を意味しない。

公開Pの2weight filesを指定LFS SHAでlocal検証し、独立ファイル1コピーと公開indexを保持する。
上記6つの元protocol metadataをコピーする計画で、original weightsのhardlink/copyは行わない。
公開weight filesは計62,546,338,264 B。viewは2shards＋公開index＋6元metadataの9files、
予定checkpoint容量62,578,656,419 B、300 GiB reserve込み必要空き384,701,203,619 B。
部分downloadは完成品とせず失敗receiptと共に保持する。

後続mergeは同じDの元／派生hash・ordinary scaling・numerical code/kernelを固定し、
baseごとに保存BF16の実変更数/norm、非対象span、全file hashを記録する。
target数と実変更tensor数は丸めno-opがあり得るため同一と仮定しない。
既存の全CPU recipe verificationとfreeze/stat継続を再利用し、
余分な第三GEMMや新しいfingerprint測定を追加しない。
この文書作成ではweight download、候補作成、配信、モデルcallは実行していない。

## Opus 5.5方法レビュー

匿名の公開一般機構だけを1回送信し、actual assistantとusageはclaude-opus-5-5のみ。
json_reviewsのM一致、verified_requested_model/review_usableともtrue、tools0。
67.5981秒、$0.135252。privateスコア・norm・候補ID・座標・課題データは未送信。
公開finalをreceiptに保存し、raw stream／thinking blocks／環境は未保存。

4条件によるbase主効果とadapter/base相互作用の切分けを支持する意見だった。
input/output provenance、同じoriginal protocol、paired得失・不確実性は採用する。
変更tensor数=target数、追加logit指紋必須、追加revision/precision/seedや
未加重item平均への置換は、そのまま新要件にはしない。
方法上の妥当性は能力を実証したという意味ではない。

artifact名はanalysis/code90-resume-20261106/からの相対名。

| 証拠 | SHA256 |
|---|---|
| k1-training-base-feasibility-v1/summary.json | 8a070417f39d340cd2ab6fe26514bc0f9a1467a0e6cb10af4d09979627c24702 |
| 同pre-k1-tree.json | 5335b48c3495dd6d8c20c8984f83fc44e045a864ec81d29e5c39f96ca9f8a0e3 |
| 公開P index | 390b7f9a0a502e4526cf20862d22954c911cc318d8aba2cdb745c6fbd98264f6 |
| 公開P pre-K1 template | 2dfbfc7d538912f4ea11d29d85b4e25d7bc26386e53f57529f1d707c28b5828c |
| opus-base-alignment-fourway-m-review/receipt.json | 046a57848316a3bf2ad33d803538ee0de047056eaf71497071798b3f7db1dd26 |
| 同independent-summary.json | 745ba2e5731580aa5dcb13941c06cc4db700141a00b52c31ae8b20d56e2cb45d |
| 同REVIEW.md | 7892de96991b79454f39977e85d2b9167afa84867b4435933e1086aafa30b550 |
| 匿名prompt | 9e1a89412ba35c2f4d7377d975ba0c6296f2bd81b87239897ce0bf6952380de1 |
