# 公開TRAINと複数回更新の実行条件（読み取り調査）

今回確認したNVMe cacheには、そのまま独立した公開コーディングTRAINとして使える実データを確認できなかった。モデル読込み・GPU処理・download・container作成・学習コード追加は0。古い候補やLoRA採否は根拠にしていない。調査範囲とpermission不足は[記録](records/2026-11-07-public-train-and-optimizer-readonly/report.json)に限定しているため、ホスト全体に存在しないという断定ではない。

| 既存データ | 現物・元source・license | 学習へ使う際の境界 |
|---|---|---|
| MBPP+ | evalplus/mbppplus revision b2d74c…、TEST378。cached cardはApache-2.0。生成sourceはGoogle sanitized-mbpp.jsonを元にする | 元MBPPのID2–809を含み、601以上も108件。元MBPP trainというsplit名だけでは非重複にならない。元MBPP train本体・その固定revision/ライセンスは今回は未確認 |
| SciCode | SciCode1/SciCode revision4510f6…、validation15/test65、cached card Apache-2.0 | WSP adapterはtest65を母集団にする。validationはTRAINではなく15件だけで、独立性・近重複を今回検証していない |
| LCB | ローカルJSON builderはtrain342だが、問題/contest/公開・非公開テストの評価用cache | 自動生成split名のtrainを学習許可と解釈しない。参照実装列もない。公式評価問題の転用不可 |

**MBPPの汚染防止は具体的に可能だが未実施。** 現manifestの全10/30/60・dev/testからMBPP+120 ID、probe-code160 ID、重複52、和集合228 IDを得た。うち73 IDは601以上。probeは同じMBPPの正答codeを測るため必ず含める。まず原task_idを対応させて全228を除外し、さらに全評価sourceの正規化prompt・code/AST近重複と同問題派生を排除する。より保守的にはMBPP+母集団378全部を除外する。現TEST cacheの残余をTRAINと呼び替えない。将来の公開TRAIN取得ではsource revision、license、元split、行ID、除外理由を凍結し、参照codeと付属testsの整合・曖昧仕様・環境依存・不正なassertをCPUで確認する必要がある。今回は行本文の大量閲覧や実行をしていないので、未取得TRAINの答え品質は未評価。

**実image af9の可用性。** metadataだけでTorch2.14.0+cu130、Transformers5.20.2、Accelerate1.15.0、safetensors0.8.0を確認。PEFT・TRL・bitsandbytes・DeepSpeed・datasetsは未導入。Adam/AdamW、Trainer、torch FSDP sourceは存在し、Gemma4 sourceはgradient checkpointingを宣言する。これは学習動作・PEFT Gemma4互換・SM120量子化optimizerを実測した証拠ではない。4推論replicaが8GPU全部を使用するため、学習とWSPを同時に押し込まず、承認した1replicaの停止/同CID復帰境界を再利用する。

最小の別方針は、実行時点の最高complete親から、多様な非重複TRAINを使って**複数optimizer更新**を行うこと。まず256–512件程度（独立64件程度を別途保持）、最大1024 token、microbatch1とaccumulation4–8、固定64–128更新という小さな上限を置ける。この数字は開始用の予算案で、必要十分な学習量や速度の実測ではない。学習損失の改善だけで採用せず、自由生成と実行テストで転移を確認し、既存と同条件の全WSP30を1回比較する。新しいベース再測定や公式問題削減は不要。

追加依存を最小化する案は361 native vectorをAdamW等で繰り返し更新する方法で、FP32の2momentは約10.65MB、既存faststore保存も使える。ただし表現自由度は既存native編集と同じで、一般能力への上限は未確認。容量を増やす代案はLoRAで、現物headerの全410 attention/MLP projectionにrank8なら61,214,720 parameter、BF16adapter122.4MB、FP32の2moment489.7MB（gradient/master copy/activationを別途加算）。PEFT導入・対象module対応・merge後の保存検証が必要で、matrixを変えるためnative-only faststoreでは保存できない。

BF16本体は62,546,177,752 bytes（約58.25GiB）。95GiB単GPUで本体と小adapter stateが収まる計算でも、長いsequenceの学習activationは未検証で、既存290 token以下のnative sensitivity peak62.26GiBを1024 token訓練へ外挿しない。全31B parameterのBF16weight+gradientだけで約116.5GiBになるため、full fine-tuneは単GPUでは成立せず、追加の分散/optimizer設計が必要。checkpoint原本と共有safetensorsはimmutableにし、実更新はRAMと新規候補へ限定する。native thinkingのchat境界・回答tokenだけのloss maskを正しく扱い、空thoughtを強制した既存代理目的をそのまま広域学習の目的にしない。

記録梱包時、旧gemma2へのmetadata再読取りが配信切替中に終了1となったため再試行せず、それ以前の成功したmetadata/source読取りを保存した。サービスへ変更は加えていない。
