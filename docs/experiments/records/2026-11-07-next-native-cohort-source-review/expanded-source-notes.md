# 次のnative観測を既存機能で組む場合の境界

新G/H/Iの6 trainはreference検証済みだが、選ばれた親での自由生成の合否は別の観測である。失敗へ向ける改善目標と既存成功例の保持制約をcase単位で同時に宣言し、LL/margin平均の上昇だけで候補を選ばない。F03のprobe-code NLL改善と公式code90低下は、この区別が必要な実例である。以下はsource/`--help`のみの確認で、GPU・モデル・追加ベース測定は0。

**(a) 完全評価済みbestがAVのままなら、旧12の感度は再利用可能な証拠だが、現在の既存CLIだけではnative-onlyの12＋6結合を完結できない。** `compose_purpose_collections.py:18,133–176` はfamiliesをnative＋matrixに固定し、各segmentに両familyのcorrect-logprob/margin、計4出力を要求する。現在のAV12にはnative2出力しかない。`propose_native_objectives.py:212–239` も各objectiveに1 directoryを受け取り、全cohort・cases SHA・frameを厳密照合する。`joint.read_collection`も1 directory入力である。旧配列を新測定と称したり、matrixゼロを補ったりする回避は不可。

必要なGPU観測自体は**新GHI6だけをAVで2objective収集**すればよい（serialならLL6＋margin12＝18 forward/VJP、旧12の再収集不要）。旧測定と同じ凍結collector/runtime/SDPA/FP32退避・BF16・seed・frame規約・421列・source値を維持する。A–FとG–Iはpurposeが非重複で、36制約は既存128上限内。既存composerが既に持つ親/全source配列bytes/列配置/同runtime/ordered union/各segment receipt検査をnative-onlyへ適用できれば十分で、新optimizerは不要。ただしその限定対応は未実装であり、この調査では作らない。旧12の成功10例と失敗A1/F2を全て保持し、新6の合否に応じて改善・許容損失を明示する。transfer3は勾配入力にしない。

**(b) F04が同条件の完全結果で新bestになれば、AVの勾配を新親の勾配として使えない。** 最短の数値処理は、A–I全18 singletonを1つの新case入力にして、既存`native.py`をcorrect-logprobとmarginで各1回実行し、その2出力をv1 `propose_native_objectives.py`へ直接渡す手順である。segment結合は不要。共通条件は `--mode chat-public-answer --answer-batching serial --max-cases-per-microbatch 1 --saved-activation-storage cpu-fp32 --max-frame-tokens 1024 --text-attention-backend sdpa --cuda-memory-cap-gib 88 --seed 0`。tokenizer-onlyで全frame長を確認し、F04の9ファイル内容証拠＋stat、実frame/sourceをpinする。shared payloadの既存full hash＋stat継続を新規全量hashと称さない。

collector自体は18件に対応するが、既存`fresh-av-native-gpu4-20261107.py:24,270,291–312,486–490`はAV path/treeとA–F/12件を固定しているため、そのままは使えない。今の安全なlock・exact owned cleanup・same-CID復帰を残し、宣言した親とcohortを受ける限定parameter化が必要である。native collectorを増殖する必要はない。正式WSP終了・global lock解放前にGPUを奪わず、実施時は専用GPU4、同af9 image・network none・host memory128GiB・readonly checkpoint、同じtext SDPAの2objective順次収集を用いる。これは新親の新感度観測であり、旧ベースの反復評価ではない。

**(c) F2の実失敗off-by-oneは、既存marginのfoilでは直接測っていない。** 現F2 correctは`values[k-1::k]`、foilはzero削除後の同sliceであり、実生成の`values[0::k]`と異なる。既存correct-logprobは正解全tokenの平均LL、marginはその平均LLから指定foilの平均LLを引く。失敗に沿う対照を測るなら、元ケースを保持した上で、正解と実off-by-oneを対にする新しいversion/IDを作りreferenceで反例を検証する。元F2を黙って置換せず、追加ならtrain総数も明示する（18のまま追加したとは称さない）。

foilを変えるとcase SHA・frame回答hashが変わり、`native.py:179–186`のpaired-pack最大長から決まるserial padding幅も変わり得る。correct文字列が同じでも、既存LL勾配と新marginをframe同一と見なせない。`propose_native_objectives.py:299–307`は両objectiveのframe/coverage/runtimeを照合するので、新対照は2objectiveを揃えて観測する。`finite_replay.prepare_frames:312–318`が明記する通り、このframeはthinking ONテンプレートに空thought終端を強制したteacher-forced診断であり、自由なthinking生成の成功を測るものではない。最終判断は新ケースの実行合否と、全既存対象を保った公式WSPの得失で行う。

[source SHAとhelp保存](records/2026-11-07-next-native-cohort-source-review/)（実装変更なし）。
