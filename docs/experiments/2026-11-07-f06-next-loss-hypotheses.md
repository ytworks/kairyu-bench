# F06の応答形式から考える次の損失設計（未実装・未試行）

F06の優劣は、進行中の全2338問の正式採点まで保留する。以下は追加model call・GPU・学習・source変更なしの判断資料であり、候補採用や原因の断定ではない。次の親は、その時点で同条件の完全評価により選ばれた最高モデルとする。既存F04を必ず親にする予約はしない。

**確認した事実。** 現trainerは正規thinking ONのuser入力にassistant reference codeを直接連結し、prefix直後の最初のcode tokenから全codeと公式turn終端までをCE対象にする。promptだけをmaskし、コード導入部・本文・終端を分離していない（train_lora.py 73–136、266–276行）。Gemmaのtemplateは非empty reasoning/reasoning_contentがある時だけthought channelを描き、ON生成prefix自体はmodel turnで止まる（239–243、381–391行）。したがって「人工empty thoughtを入れなかった」ことは「ON時の初手分布を保存した」ことを意味しない。直接codeを最初の正解tokenとして繰り返し教える経路がある。trainの同じ52例のCEは52/52低下したが、形式への適合とアルゴリズム改善を分離できない。

保存F04メタデータはcode90/90問に119/119非empty reasoning nodesを記録する。F06途中snapshotは491/2338応答、code26/90。同ID/sample/pathの477 leafでreasoning文字数中央値5475→0、code36 leafは全てF06側が空文字だった。非codeで両側contentを観測できた同じ408 leafのコード先頭外観はF04 2/408→F06 370/408だった。別分母としてF06全441 leafでは393件であり、F04 content欠測33はNoneのまま保持する。これは出力形式変化を支持する観測だが、完了順の偏り、異なる生成prefix・agent軌跡、外観判定の限界がある。返却reasoningは内部計算そのものではなく、reasoning token数は未観測。疎通の公開answer記録も確認し、one-word要求へ関数定義とprintのcodeが返り、finish=stop、completion15 tokensだったことを固定した（reasoning本文は記録対象外）。全成績や内部推論能力の消失とは断定しない。

| 未実施の案 | 必要入力・loss対象 | 最小性と限界 |
|---|---|---|
| A: code本体だけのloss | 同じ非重複TRAIN204を全件保持。正規ON full frameも保持し、referenceの導入import・最初の関数signatureと先頭docstringまでを教師contextとして与え、その後のcode suffixだけをCE対象にする。公式turn終端もlossから外す。ASTとtoken境界を全件記録し、境界不明・空targetを黙って除外しない。 | 変更はframeのloss-mask/target位置だけ。追加親forward・新教師dataなし。初手codeと停止tokenへの直接教師圧力は外せるが、全410共有行列へのbody勾配は初手分布も間接的に変え得る。思考contextのないcode continuationを学ぶ点も残り、保存の保証ではない。 |
| B: A＋初手分布の親KL | 同じTRAIN204のON promptのみを、次の親のadapter無効・no-grad状態で各1回prefillし、最初のassistant tokenのsoftcap後の全語彙分布p0を固定保存。学生側の同位置pθにKL(p0 || pθ)を加え、body CEと併用する。λは実行前固定（最小案は1.0で候補内探索なし）。reference code・turn終端を初手の教師にしない。 | 学生の初手logitsは既存full forwardから使える。追加は教師prefill204回で、現在のprompt合計6602 tokens、最長64 tokens。これは訓練用教師計算であり、親WSP再測定や能力得点ではない。現ログにはp0が無いため計算コストはゼロではない。初手KLは後続reasoning、tool選択、非code入力全体を拘束せず、λの妥当性も未実測。 |

**次試行を増やさない進め方。** F06の全得失を見た後、AかBを1つだけ事前固定し、rank/scope・204/51 split・順序・optimizer・64stepを同じにして変更点をlossへ絞る。Aを試してから必ずBも試す二段階の予約はしない。形式変化が主要な損失として残るなら、Bは短い教師prefillで直接その初手分布を制約できる一方、Aが最小の実装変更である。どちらも学習CE低下だけでは採用せず、同条件の正規WSP全問を既存最高モデルの完了結果と比較する。親のWSP再実行は不要。公式code/source平均に加え、他領域の得失、finish/tool-call、reasoning文字数とcode外観を同ID/pathで比較し、partial・欠測・異なるprefixを別に記す。

WSPの問題・正解・保存reasoning本文を教師やKLアンカーへ転用しない。既存51 holdoutは勾配0のまま。独立した非codeアンカーを後で追加する場合は別のdata変更になるため、ここでは第三案へ広げない。今回の確認はsourceと既存メタデータのみで、思考本文を読まず、現在稼働中の学習・評価ソースを変更していない。

根拠は[入力SHAと行番号](records/2026-11-07-f06-next-loss-hypotheses/inputs-with-smoke-v2.json)、[F04メタデータ](records/2026-11-07-f06-next-loss-hypotheses/F04_reasoning_metadata.json)、[F06途中snapshot](records/2026-11-07-f06-next-loss-hypotheses/F06_partial_metadata.json)、[訓練ログ監査](records/2026-11-07-f06-next-loss-hypotheses/training_audit.json)に固定した。
