# Gemma 4 31Bの重み編集実験記録（2026-11-04〜05）

この実験では、静的な重み観測と小規模な目的別プローブから編集候補を作り、WSPのdevで比較した。WSPで実測した編集候補は**22種類、反復・部分評価・広域評価を合わせて45実行**だった。最終選抜候補のWSP testでは、コードと指示追従は元モデルと同点、ツールは改善した一方、14領域のoverallは低下した。最終DeepSWEはユーザーの指示で停止し、全113課題の比較は未完了である。

現在は重み観測ツールの整備へ移行している。**能力探索のWSP・DeepSWEの再開は、ツール完成後のユーザーの明示承認を受けて行う。** 使用済みtestを、次の探索の未見最終評価として扱わない。次周期の承認前に、新しい未見の最終評価計画を決める。ツール動作を確認する短い合成入力の実forwardは別枠で承認された検証であり、以下の候補数・WSP実行数へ加えない。

汎用ツールの操作と観測値の読み方は[重み観測ガイド](../wsp-weight-observation.md)、能力評価の定義は[WSP](../wsp.md)を参照する。

## 対象と比較条件

| 項目 | 記録 |
|---|---|
| 元モデル | `google/gemma-4-31B-it@842da379`、BF16 |
| 推論構成 | 8GPU、TP2×4 replicas、MTP4 |
| WSP | 固定manifest、devとtest、公式または規則ベースの採点 |
| サンプリング | 元checkpointのgeneration config、temperature 1.0 / top_p 0.95 / top_k 64 |
| 思考設定 | 稼働replicaでnative thinking ON、追加thinking-token capなしを確認 |
| DeepSWE | v1.1、113課題、mini-swe-agent 2.4.6、各1試行、32並列 |
| 成果物 | 元重み、編集spec、manifest、モデル識別hash、試行結果、未採点理由を保存 |

「追加thinking-token capなし」は確認したサーバー設定の範囲を表す。数値のEffort 100や、総出力・文脈長の上限解除を意味する測定は行っていない。最終比較では113課題のchecksum・task画像ID・agent版・実行設定を照合した。元モデルR3当時のserving Docker image IDは記録不足で、完全一致を独立には確定できなかった。

## 実測数の数え方

候補は編集後checkpointの識別子で数えた。同じ候補のscreen、反復、full評価を別の候補に数えていない。元モデルと最終testは、次の45回のdev実行から除外した。

| 編集群 | 候補数 | dev実行数 | 候補 |
|---|---:|---:|---|
| 層scalar | 6 | 6 | L23/L28/L41の各±4 BF16 ULP |
| 疎な目的別編集 | 3 | 3 | `sparse-code-margin3`、`sparse-action-margin3`、両者の6要素組合せ |
| 全層から選ぶ目的別joint | 6 | 17 | balanced-small（A）、balanced-medium（B）、workflow-language（C）、programming-set（D）、algorithm-set（E）、chat-set（G） |
| 最終norm校正 | 3 | 9 | −1%、+1%、+0.5% |
| 公開donor方向 | 2 | 6 | text行列への係数1、0.25 |
| 異なる編集の組合せ | 2 | 4 | D＋norm +1%、donor 0.25＋norm +1% |
| **合計** | **22** | **45** | 全候補でmaterialization manifestとWSPのcheckpoint labelを照合 |

実験root直下のspecは37個あり、そのうち15個はWSP未実測だった。内訳は注意系10、FFN系4、D＋norm +0.5%の1個である。spec作成、CPU上の実forward、GPU上の有限差分、checkpoint作成、WSP測定はそれぞれ異なる段階として記録した。

Fの行列joint、減衰、再線形化、H/H-smallの有限応答実験も実施したが、WSP候補数には含めていない。H系は専用GPUでの短い合成forwardを含むため、「合成プローブはすべてCPU」とは記録していない。

## 最初に観測した重み

全1188個の保存浮動小数tensorをチャンク読みし、31,273,088,876個のBF16値、非有限値0、Frobenius norm `97779.1491644232`を確認した。この全体値はvision側も含み、共有embeddingはファイルに保存された分を1回数えている。モデル実装上の参照回数や論理parameter数とは集計範囲が異なる。

小さいnorm・scalar tensorは全値を読み、行列は層・head・行・列のRMS、方向のcosine、出力座標への重みenergyの集中を調べた。近隣層との比や大きいnormは編集案を作る手掛かりに使った。大きい値を異常と決めつけたり、特定の座標を「coding neuron」と同定したりする根拠にはしていない。

稼働実装も確認した。Gemma 4のRMSNormは正規化後にweightを掛ける。ここで確認した演算は `1 + weight` 形式ではない。layer scalarはblock後のresidual全体へ掛かるため、branchだけを変えるpost-attention/post-FF normとは作用点が異なる。最終layer scalarは後段の最終normで一様な倍率が相殺されやすい。

## 試した手法と観測結果

### 小さいscalar・疎な編集

最初はL23/L28/L41のscalarを±4 ULP変えた6候補を測った。20問の短いコード集合は17/20のままで、ツール・指示追従の変化には元モデルの過去実行でも現れた回復が多かった。1回だけの正誤反転を能力改善として採用しなかった。

続いて、コードと操作・指示に関する合成のtarget/foil marginから、3要素ずつの方向とその6要素組合せを作った。独立の合成例では方向によってコード・日本語・推論が下がり、WSPでも十分な利得は得られなかった。部品の予測を足した結果と、組合せを実際にBF16で評価した結果が一致しない例があった。

### 全層の目的別joint

全60層のnorm、Q/K norm、layer scalarを対象に、複数目的・複数foldの感度を測った。注意headやMLP出力列の方向も観測し、目的を保ちながら全体L2を小さくする共同編集を求めた。

Aは2309要素・61tensorを編集し、総ΔL2は約 `0.000124616` だった。L28の1要素・4 ULP編集より総ΔL2が約63倍小さくなった。「少数要素」と「小さい重み差」は別々に測る必要がある。

Aの最初の95項目screenではコード18/20になったが、反復では17/20に戻った。獲得したMBPP/801は元モデルの過去dev30でも成功しており、新しい到達点と断定できなかった。BFCLのoptional引数や不要呼出しの失点、広域評価の損失も残った。

Dは実装工程、必須呼出し、optional引数、不要な呼出しの抑制を目的へ加えた。Eはアルゴリズム実装・指示対象・事実保持を追加した。Gはraw文字列と公式chat frameの条件差を修正して全目的を測り直した。Gの有限差分では実装などの目的が上がる一方、推論・日本語の一部foldが下がった。目的関数の線形予測だけで採用すると、こうした損失を見落とす。

Eの難しいコード部分では、LiveCodeBenchの新規成功とSciCodeの部分得点向上があった。同時に元モデルが成功していた問題の失点もあり、全体の利得としては限定的だった。

### 行列jointと有限応答

Fでは全60層の410行列から層別に座標を取り、共同方向を作った。50,004座標の候補、0.25/0.5の減衰、1回の再線形化を実際のBF16 forwardで調べた。目的別の線形予測と有限変化の食い違い、fold間の方向の不安定さが残り、WSPには進めなかった。

H/H-smallでは少数の全層方向を固定し、同じdevice・dtype・chat frame・batchのbaseと正負endpointから応答を測って共同解を求めた。Hは事前のsolver時間上限に達し、H-smallも測定した範囲で適格な解が得られず閉じた。WSP用checkpointは作らなかった。

これらは探索した範囲についての結果である。一般の行列編集や有限差分法が改善できないことを示すものではない。

### 最終normの校正

最終RMSNormの全5376値を−1%、+1%、+0.5%で変えた。devで最も目的に沿った結果を示した+1%を、最終評価用に選抜した。

この実装では最終norm、tied head、`30 * tanh(logit / 30)` のsoftcap、temperatureの順に処理される。理想算術で正の一様倍率はlogit順位を保つが、sampling分布は変わる。softcapがあるため、全領域でtemperature変更と等価にはならない。BF16への丸めによって座標ごとの実倍率にも差が生じる。結果は出力校正の仮説として解釈した。

| +1%候補の実差分 | 値 |
|---|---:|
| 変更tensor | `model.language_model.norm.weight` 1個 |
| 変更BF16値 | 5376 |
| ΔL2 | 25.008668011967437 |
| 元の全保存浮動小数重みのFrobenius norm | 97779.1491644232 |
| 相対L2 | 0.00025576688103425227（0.0255766881%） |

元と凍結後のsafetensorsからこの5376値を直接読み、全要素の変更数とΔL2を再計算してmanifestと一致することを確認した。相対L2の分母は表の全重みnormであり、変更要素数の割合には置き換えられない。

+0.5%ではΔL2を約47%減らせたが、+1%で得た難しいコードや広域の利得を維持できなかった。+0.25%は多数の値がBF16で変わらず、方向のcosineも下がったため、追加のWSP候補にはしなかった。

### 公開donor方向と組合せ

公開adapter `Tsunamayo7/gemma4-31b-ja-agent-coder` を固定revisionで取得し、safetensorsのheader・全値の有限性・text 410行列への対応を検証した。係数1と0.25でtext行列だけへ反映し、tokenizer・template・vision側は元checkpointを使用した。公開学習データの全内容は確認できず、評価問題との重複は不明である。

| 係数 | BF16で変わった値 | 実ΔL2 | 未丸め方向とのcosine |
|---|---:|---:|---:|
| 1 | 22,574,585,664 | 34.358892 | 0.99215 |
| 0.25 | 12,569,638,846 | 8.696031 | 0.92840 |

さらに小さい係数は、丸め後の方向が大きく崩れた。係数1ではNLLが下がった一方、実行するコード・ツール・指示追従の得点が低下した。関数名の大小文字を変えてentry pointを失う例や、不要なtool呼出しがあった。NLL改善だけで採用しなかった。

D＋norm +1%と、donor 0.25＋norm +1%を共同checkpointとして測った。後者の107項目部分評価は、norm単独に対して長文が約10ポイント上がった一方、コード18/20→17/20、BFCL36/40→35/40、IFBench14/15→13/15となった。部品の利得を単純加算できないことが確認された。

## devから最終評価への移行

短いscreenはコード20、BFCL40、IFBench15、分布probe20の計95項目だった。最初からコード17/20で、差が出る余地が小さかった。そのため、有望な共同編集にはfull dev10の778項目・14領域や、dev30のLiveCodeBench20＋SciCode10を追加した。107項目のcore＋長文評価は別の部分集合として扱った。編集候補のfull dev30/60は実施していない。

最終norm +1%の主な結果は次の通り。差は百分率のポイント差で示す。

| 評価 | 観測 |
|---|---|
| dev95の2回とfull dev10 | MBPP/801、BFCL multi_turn_base_29の成功が再現。MBPP/801は元の過去成功例もある |
| full dev10 | overall +0.076773ポイント、code +3.333333、tools +2.5。overallの95%区間は0を含む |
| full dev10の損失 | expert −6.25、長文 −6.25、推論 −2.173913、多言語 −1.792929、数学 −1.875 |
| hard code部分30 | LCBは15/20で同数。arc195_bを獲得しabc399_cを失点。SciCodeは平均+0.111111、72が7/9→8/9 |
| 凍結後test10、778件 | overall 85.130499%→82.360074%（−2.770425ポイント）、code 91.111111%で同点、指示82.083333%で同点、tools77.5%→87.5% |

hard code部分はこの候補で1回だけ測った。SciCode/14の回復は元dev60でも観測されていた。過去の最大値を超えない回復も、頻度が再現すれば改善候補になり得るが、単発回復から信頼性向上を確定することはできない。

当初は非退行を重く見ていたが、後にユーザーが目的に沿うトレードオフを許容する方針へ変更した。選抜時にはコード重視・agent重視・広域補助の3通りの重み付けを感度分析として使った。これらは意思決定の補助で、WSPの公式overallや統計的検定の代用にはしなかった。

最終testは選抜後に1回使用し、その結果で候補を選び直していない。調査初期には既存ドキュメントのtest集計表を閲覧したことも記録した。次周期では今回のtestを未見扱いせず、開発用証拠と新しい最終評価用証拠の分離を改めて設計する。

### 移行できなかった理由についての仮説

次の要因は調査候補であり、個々の影響を切り分ける追加実験は行っていない。

- 多数の候補からdevで選ぶ過程と、少数反復による偶然の正誤反転。
- 95項目の短いコード集合と、repositoryを編集する長いagent作業との課題差。
- 最終normの分布校正が長い生成・複数tool呼出し・引数の表記へ及ぼす影響。
- 合成margin、NLL、実行テストで測る性質の違い。
- 一部のMGSMで、Bengali数字とASCII数字の数値同値を公式採点が拾わない表記上の問題。

最後の項目は診断として別記し、元と候補の公式スコアは保存した。数字表記だけで、広域の損失すべてを説明した扱いにはしていない。

## 最終DeepSWEの停止状態

元モデルR3は113件を採点し1件成功した。編集モデルの最終実行は、ユーザーの停止指示により2026-11-05 12:00:36 UTCに終了した。

| 編集モデルの停止記録 | 値 |
|---|---:|
| 計画した課題 | 113 |
| 採点済み | 109 |
| 採点済み成功 | 0 |
| 未採点 | 4 |
| 記録されたagent手数 | 7844 |
| 空patch | 71 |

未採点4件は、`aiomonitor-task-snapshots-diff`の既存の`VerifierTimeoutError`と、停止に伴う`sql-formatter-bigquery-pipe-formatting`、`vulture-persistent-analysis-cache`、`wasmi-trap-coredumps`の`CancelledError`である。runの正式状態は`incomplete`として保持した。未採点を0へ補完せず、113件完走のpass@1は報告しない。

元の唯一の成功だったClaude再帰委譲課題は、編集後ではF2P7/7→0/7、P2P31/31維持となった。編集patchの93.1MBには`backend/core`のバイナリ差分が含まれ、テキスト部分は412行追加・65行削除だった。coreの内容や推論本文を調査・公開する必要はなかった。

HTTPX cookie保存ではF2P0/115→106/115、P2P1281/1281維持という部分進展があったが、Client統合・不正cookie・置換・複合header処理など9件が残った。Exprではvm/parserのbuild failureにより多数のテストが実行されなかった。aiomonitorは新suite29件終了後のCLI snapshot保存テストで止まり、正式rewardを得ていない。部分ログの成功数を課題全体の成功へ換算していない。

停止前には「112採点＋1未採点」での報告コードを準備していたが、実行していない。正式停止状態は上表の109＋4である。編集モデルの最終Wiki追記・Hugging Face登録・公開用posthash証跡の発行は行っていない。

## 運用上の損失と修正

### 約10時間の引継ぎ待機

matched95 driverの終了は2026-11-04 16:00:12.293795 UTC、初候補driverの開始は翌日02:04:21.049555 UTCだった。この間に**10時間4分9秒の測定間隔**が生じた。次段の実行を人の応答・引継ぎに依存させた運用上の待機として記録する。全期間のGPU利用率は記録していない。

以後は候補を登録するときに部分評価・同checkpointでの条件付き後続評価を準備し、単一owner、明示pause gate、実子processとlockの確認で継続した。SIGSTOPでwrapperを止めたつもりでも子queueが動いた事例があり、pause markerだけで停止を判断する方法を修正した。

### 評価以外の時間

初scalar候補ではWSP本体69.2秒に対してdriver合計375.3秒だった。切替194.6秒、事前hash59.16秒、事後確認47.65秒などが大きかった。最後のdonor＋norm候補ではmaterialize/verify422.0秒、切替179.71秒、WSP352.3秒、driver合計1004.96秒だった。短いscreenを増やすだけでは、wall timeが短くなるとは限らなかった。

同一候補の後続devでは、保存したrecipe証拠と新しい全hash・config・donor・code・稼働identityを照合し、dense行列の再生成検証を省略する経路を使った。手動検証・最終freeze・公開では全recipe検証を維持した。

### 保存容量とnested mount

重みはNVMeへ置き、変更のないshardだけをSHA照合後にhardlinkした。変更shardには独立inodeを用意した。元重み・稼働中・選抜候補・共有poolを保護し、退役時もspec・manifest・結果を残した。パラメータ編集量とcheckpointファイルの書込み量は別に測る必要があった。

WSP holdoutでは、新しいNVMe結果pathがnested Docker daemonへ共有されておらず、Terminal-Bench2件のreward読取が失敗した。元trial ID・時刻に対応する公式reward・CTRF・logの6ファイルをSHA照合して回収し、公式読取・採点で復元した。追加推論0・追加verifier実行0、他776件のitem/response行はbyte単位で保持した。元partialと例外も保存した。後続DeepSWE前にhost→daemon→nested containerの同一path共有を検証した。

### 外部レビューの範囲

Opus 5.5の実使用モデルを確認したセカンドオピニオンは、匿名化した設計・方法論の範囲だった。候補の私的な座標や実測結果を継続送信する個別レビューは、承認範囲の制約から実施していない。匿名レビューを、具体的な候補の効果を実測で検証したレビューとして扱わなかった。汎用的なRMSNormの説明が稼働実装と異なる例もあり、演算の確認には実コードを使った。

## 観測ツールへ反映すること

この試行では、問題ごとのlayer/token activationをライブ採取していない。静的重み・合成forwardのmargin・公式成績は保存したが、当時のactivationを復元した証拠とはしない。追加した観測ツールは次周期のためのもので、今回の22候補・45dev実行へ遡って実測データを足していない。

1. 全体・選択tensor・明示groupの分母を区別し、変更要素数とΔL2を並べる。
2. rawの方向だけでなく、保存dtypeへ丸めた後の差、消えた変更、方向cosineを測る。
3. normやtensor名から機能を断定せず、対応軸を明示した構造的な整列を観測する。
4. 全要素の集計と、quantile・SVDなどのsample統計を分けて出す。
5. checkpoint全体をGPUへ載せず、必要なtensor・軸・チャンクをCPUで読む。
6. 元・部品・組合せの差分を同じschemaで保存し、共有shardと実差分を区別する。
7. 静的な観測だけで能力改善を宣言せず、再開承認後の対応あり評価で検証する。

探索終了は、測定した縮小・共同編集が目的の総合増分を保てなかったことと、経過時間を踏まえた担当rootの判断だった。その後のユーザー停止指示で実験を止め、ツール整備へ移った。全重み空間を探索し尽くしたという結論は出していない。

## 停止後のツール検証との区別

能力探索を止めた後、別途承認された短い合成入力の実モデルforwardで観測ツールを検証した。TP2×4replicaの8GPU測定では、warm c32の3回中央値は採取OFF 1.424930秒、ON 1.534793秒（7.710%増）だった。一方、初回c1は18.068%増、c32初回warmupは19.012%増で、時間増10%以内を全面達成したとはしない。応答hashにも不一致があり、出力不変を断定しない。

32合成入力と64 request-rankの結合を確認したが、成績はすべてnullで、22候補・45dev実行へ追加しない。全8workerのqueue drainとerror/drop 0を確認した一方、停止時のshutdown acknowledgementは5/8に留まり、captureはpartialのまま保存した。採取範囲、MTP受理、保存容量、負荷と未達条件の詳細は[観測ガイド](../wsp-weight-observation.md)を参照する。WSP・DeepSWEの再開は引き続き明示承認を要する。

## 保存した証拠

以下は実験成果物rootからの相対pathである。大きな重み・生の推論ログ・core binaryをこのリポジトリへ複製していない。

| 証拠 | path |
|---|---|
| 22候補・45runの独立照合 | `reporting/search-closure-independent-audit.json` |
| 全保存重みのnorm・非有限値 | `analysis/global-weight-norms.json` |
| 初期静的観測 | `analysis/dev-weight-review.md` |
| 注意/FFNの構造観測 | `analysis/attention-head-observations.json`、`analysis/ffn-channel-observations.json` |
| 全層目的joint | `analysis/whole-text-purpose-dependencies/` |
| 行列・再線形化 | `analysis/stratified-matrix-purpose-dependencies/`、`analysis/stratified-matrix-purpose-relinearized/` |
| H/H-small | `analysis/finite-joint-basis-h-gpu/`、`analysis/finite-joint-basis-h-small-gpu/` |
| donorとBF16差分 | `analysis/public-adapter-donor-audit/` |
| 選抜・凍結 | `dev-finalist.json`、`frozen-candidate.json` |
| 最終testの対応あり比較 | `reporting/final-holdout/comparison.json` |
| holdoutの元reward復旧 | `provenance/holdout-verifier-recovery.json` |
| 匿名Opusレビュー | `provenance/claude-opus55/` |
| 正式なユーザー停止 | `final-user-cancelled.json` |
| 各候補の条件と時間 | `<run-id>.status.json`、`provenance/<run-id>-thinking.json` |
