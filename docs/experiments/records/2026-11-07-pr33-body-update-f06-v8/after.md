Gemma 4の保存重みを累積的に編集し、公式source平均のコード得点90超を目指すツールと実験記録です。**目標は未達です。最高の完了比較モデルはF04（dev30 code72.93650794、overall76.99108643、全2338採点・未採点0）で、F06の全問dev30は現在実行中・未採用です。** 元モデルやF04のベンチマークを追加で回していません。

F06はF04を親に、独立の公開TRAIN204件を使って410 text projectionへrank8の更新を64step学習した新候補です。norm/scalar/vision/embedding/headを凍結し、通常BF16の別checkpointに保存しました。全410行列・8,193,921,391要素で実差分を確認し、保存ΔL2=3.7080033175、選択410行列の完全分母2651.4095893に対する相対値0.001398502643です。保存／連続値のL2比0.9635242872は方向cosineではありません。全量create/verifyが完了し、元F04不変・非対象byte不変・独立inodeを確認しています。

- [学習設計と実装](docs/experiments/2026-11-07-f04-lora-training-implementation.md)：FP32 factors、固定AdamW/accumulation4/seed0、4step技術pilot後に同じoptimizerで64step。GPU実測64/64、256 forward/backward、410全勾配、保留51件への勾配0、peak63.7545GiB、学習wall461.425秒。
- [独立参照の検証](docs/experiments/2026-11-07-mbpp-reference-quality-and-frames.md)：元374件の全除外理由を保持し、benchmark全poolと近似重複を除いてtrain204/holdout51へ固定。全255参照・765テストPASS。公式templateのtoken列を独立照合、切詰め0。小さい原テストは全仕様の保証ではありません。
- [全学習ログ](docs/experiments/2026-11-07-f04-mbpp-lora-training-log-audit.md)：全64step、全204 ID、全410の勾配/活性/変更量、同一52件のCE再提示結果。CE低下は訓練への適合であり、能力得点ではありません。活性は訓練時のもので、benchmark live hook観測とは区別します。
- [保存の全試行と実差分](docs/experiments/2026-11-07-f06-materialization.md)：初回は読み取り専用copyへの書込みで失敗。親には触れず、新しい独立copyだけowner-writeを付ける最小修正と回帰テストを追加。同じ学習済み更新で再保存し、create444.002秒・verify297.805秒で成功。失敗原本と全argv/出力/SHAを保持。
- [実Claude Opus 5.5レビュー](docs/experiments/2026-11-07-f06-opus55-review.md)：実応答モデルをreceiptで確認、tools0。固定全dev30を1回行う価値があるという条件付き判断。直接code教師による思考抑制、丸め、他領域の損失、標本の小ささを指摘。採用・不採用と理由をすべて明記しています。
- [F06配信](docs/experiments/2026-11-07-f06-deployment-completed.md)：4replicaを同じ保存F06へ切替、TP2×4/MTP4/native BF16、chatとlogprobを確認、他23container不変。疎通のready指示にはコードで回答したため、疎通成功を指示準拠成功と呼んでいません。
- [全問dev30起動](docs/experiments/2026-11-07-f06-dev30-launch.md)：run `gemma4-fresh-f06-dev30-dp4-20261107-v1`、全2338／code90／14領域＋96診断、F04と同じmanifest/全IDs・固定f91 runner・256/32/4 concurrency。除外/only/問題削減0。初回plan writerの改行不備は起動前失敗として残し、同条件で正常起動しました。部分結果は順位付けしません。
- [F04失敗の事前分類](docs/experiments/2026-11-07-f04-code-failure-taxonomy.md)、[既存reasoning記録](docs/experiments/2026-11-07-f04-reasoning-evidence.md)：通常コード13失敗＋Sci18失敗stepを全件保存。原因未確定を推測で埋めません。同ID/sample/pathのreasoning文字・bytes・finishをF06と比較し、総completion tokensをreasoning tokensとは呼びません。

このPRには重みinspect/diff/spectrum/align、実行時capture/merge/compare/view、native/matrix/finite-response、保存editor/overlay/LoRA、frozen data/quality/frame検証、GPU所有と復旧、全問監査・明示したrunだけを対象にするprogress CLIを含めています。共通機能とモデル固有recipeを分け、今回作成した道具・依存条件・実行source・全失敗も同じPRに残します。[一覧と操作入口](docs/wsp-tool-inventory.md)、[学習recipe](scripts/experiments/gemma4-purpose/README.md)、[保存recipe](scripts/experiments/gemma4_lora/README.md)。

既存結果も撤回・失敗を含めて保持しています。F04はAVとコード同点でoverallのみ+0.2710。[F04全完了比較](docs/experiments/2026-11-07-f04-complete-official-result.md)。F05はnative19の16/19完全成功・264/304 fixtureが親と全件同じで、不採用・公式未採点です。[F05選別](docs/experiments/2026-11-07-f05-native19-screen.md)。F03は2337/2338と未完了を残し、完了モデルに順位付けしていません。異なるtier10の87.7778とtier30の72.9365を混同しません。

CPU最終検証[R13](docs/experiments/2026-11-07-cpu-validation-r13.md)：586 tests＝527 PASS＋59 optional skip、失敗0、Ruff・mypy74・data・tracked shell31 PASS、590source前後一致、48.04秒・各1回。CPU検証、実モデル64step学習、保存byte検証、まだ未確定の能力評価を明確に分けています。[監視CLIの修正と誤追記記録](docs/experiments/2026-11-07-wsp-progress-cli-fix.md)も含め、既存の失敗記録は上書きしていません。

容量確保は未稼働の旧scalar候補1個だけを対象とし、全12fileのfresh hashと元モデルへの2byte逆復元hash、全metadata/source/specを保存して完全復元を検証しました。その専有2payload62.546GBだけをarchive表現へ移し、元モデル・AV・F04・共有faststore・全サービス・全実験結果を保持しています。[復元方法・全ガード・可視性の限界・失敗を含む記録](docs/experiments/2026-11-07-scalar-l41-minus4-archive-plan.md)。

**未完了**：F06全dev30の採点と採否、90超への継続改善、最終60/test確認、編集モデルのDeepSWE113問×1周・32並列、wiki追記。元モデルよりDeepSWEが改善した場合だけHugging Faceへ重みを登録します。現時点で90達成・最終DeepSWE改善・公開済みとは主張しません。
