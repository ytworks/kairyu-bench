# 次のnative観測：既存機能の利用境界

G/H/Iの6 trainはreference検証済みだが、親モデルの自由生成で解けるかは未測定。まずその合否を確認し、失敗への改善と旧成功例の保持をcase単位で同時に宣言する。LL/margin増加を能力改善とは扱わない。F03のprobe-code NLL改善と公式code90低下もこの区別を支持する。本調査はsource/`--help`のみ、モデル・GPU・追加ベース測定0。

**AVがbestのままの場合。** 旧12のLL/margin421列を保持し、AVで新GHI6だけを2objective収集する数値上の条件は揃う。serialならLL6＋margin12＝18 forward/VJPで、旧12の再収集は不要。同じ凍結collector、BF16/SDPA/seed/frame/退避方式、source値、列配置をpinし、旧12＋新6の36制約を宣言する。旧成功10例も保持し、transfer3は勾配へ送らない。

ただし**現在の既存CLIだけではnative-only segment結合を完結できない**。`compose_purpose_collections.py:18,133–176`はnative＋matrixを固定し、各segmentに4出力receiptを要求する。`propose_native_objectives.py:212–239`と`joint.read_collection`は各objectiveに1 directoryを受ける。既存composerの親/bytes/列/runtime/ordered-union検査をnative-onlyへ適用する限定対応が必要。zero matrixの捏造や旧感度を新測定と称する回避は不可。新optimizerは不要だが、今回は実装しない。

**F04が完全な同条件評価で新bestになる場合。** AV感度は新親へ流用できない。A–I全18 singletonを1入力として既存`native.py`をLL/margin各1回、新親で観測し、v1 `propose_native_objectives.py`へ直接渡すのが最短で、結合処理は不要。共通条件はserial/1case、cpu-fp32退避、text SDPA、全frame≤1024、88GiB allocator cap、seed0。tokenizer-only frame確認、F04全9内容証拠＋fresh stat、sourceをpinする。sharedのprior-full-hash＋statを新規全量hashと称さない。

既存GPU ownerはAV path/treeとA–F/12を固定している（`fresh-av-native-gpu4-20261107.py:24,270,291–312,486–490`）。親/cohortの限定parameter化が必要だが、lock・owned cleanup・same-CID復帰はそのまま再利用できる。正式WSP終了とlock解放後だけ、GPU4、af9 image、network none、128GiB host、readonly checkpointで2目的を順次実行する。

**F2の対照を実失敗に合わせる場合。** 正解`values[k-1::k]`に対し旧foilはzero削除で、実生成の`values[0::k]`とは違う。元ケースを保持し、新version/IDでoff-by-one対照を追加・reference検証すれば既存LL/marginで測れる。追加するなら18からの総数変更も明示する。foil変更はcases/frame hashを変え、serialでもpaired-packの最大長に由来するpadding幅が変わり得る（`native.py:179–186`）。正解が同じでも旧LLと新marginを同一frame扱いせず、両目的を揃えて新対照を観測する（`propose_native_objectives.py:299–307`）。

このframeは空thought終端を強制したteacher-forced診断（`finite_replay.py:312–318`）であり、自由thinkingの成功ではない。新ケース実行合否と全既存対象を保つ公式WSPで利得・損失を測る。[source SHA/help/詳細根拠](records/2026-11-07-next-native-cohort-source-review/)を保存、実装変更なし。
