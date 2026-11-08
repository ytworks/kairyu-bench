# F04新親の契約19件：実回答を変更目標へ使う

同条件の全2,338問でF04はコード72.9365079365（AVと同点）、総合76.9910864328（+0.2710271413）。コード得点を改善したとは称さず、同点の中で総合が高いF04を次の編集元とした。基準WSPの再測定は0。

旧A–F train12と新G–I train6を保持し、F2実回答のoff-by-oneに対応する別対照J1を追加した。train19は入力SHA74e1cd2000fda0abdc03f4454b18f4231b9891360af266391744e13976f199f1。J1はF2と同じ公開prompt/correctでfoilが異なるため、19生成には同じpromptの2回答を含む。独立19能力とは扱わない。旧transfer6と新transfer3はモデル・勾配に未使用。

既存生成ownerのschema2で親treeとordered IDs、train-only frameを明示できるよう限定拡張した。旧planは旧12/AV固定検査を維持。転用・重複・非singleton拒否と実SIGTERM中のlock継続を既存3テストで検証し、0.344秒・全成功、Ruff成功。旧実行ソースも保存した。

新しいCPU tokenizer recipeは既存prepare_framesを呼び、19件すべて最大290token・切り詰め0で1024制限内と確認した。モデルforward/GPUアクセス0、内部8.3867秒。F04のprivate8ファイルをfresh hashし、shared62.5GBは既存full hashと全stat/inode/nlink連続性を確認した。新規全量hashとは称さない。

F04 native thinkingで19回答を一度生成し、owner26.3995秒、全19回答、エラー/空回答/長さ打切り/再試行0。model POST19、tool0。4レプリカと23保護containerは同一、終了後idle確認・lock解放済み。入力token 2104、出力token 23374（thinking含むAPI計数）。回答SHA4105a5c87f9c54a4fa552a31927ce1bf8e330a5831a3a78d23cb0b1802cbc07f。

この記録は自由生成の完了だけで、契約実行採点・感度・次候補・公式得点とは別。採点の全19結果は別記録へ保存する。teacher-forced空thought診断と自由thinking回答の違いを保持する。

[入力・owner前後・全回答・全receipt](records/2026-11-07-f04-behavior19-native/SHA256SUMS)。最初に存在しないtest名を読もうとしてexit1になったread-onlyコマンドがあったが、モデル/生成処理の失敗ではない。
