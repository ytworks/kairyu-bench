# Gemma 4の目的別感度を収集する実験レシピ

過去の全層感度計測を、再実行可能な入力・出力へ分離したものです。モデル固有のレシピであり、
観測から「コーディングのニューロン」を同定するものではありません。
既定は入力検証だけです。モデルのロードと微分には明示的な --run が必要です。
checkpointの保存・書換え、候補の採用、WSPの実行は行いません。

## 入力と準備

リポジトリをインストールしたPython環境に、対象Gemma 4をサポートする
torch、transformers、safetensors、accelerateが必要です。
これらは通常の観測CLIの必須依存には追加しません。
checkpointはローカルHFディレクトリ、全モデルファイルは読み取り専用にしてください。
ネットワークからモデルやコードを取得せず、local_files_only=Trueを使用します。

ケース本文をコードに埋め込まず、別のJSONを --cases に指定します。

~~~json
{
  "implementation": [
    [{"id": "case-a", "prompt": "公開可能な入力", "correct": "正の継続", "foil": "対照の継続"}],
    [{"id": "case-b", "prompt": "別の入力", "correct": "正の継続", "foil": "対照の継続"}]
  ]
}
~~~

目的名→fold→ケースの構造です。fold数・目的名は固定しません。
任意のexpected_marginは、同じ実行条件で保存した既知のmarginと照合する値です。
過去のraw観測のcases、positive / negative / marginキーも読めます。
異なるframe・dtype・batch条件の値を同じ基準値として使わないでください。

--runには事前に全ファイルのSHA256を計測したcheckpoint receiptと、そのreceipt自身のSHA256が必要です。
receiptはCheckpoint.identity(hash_files=True)のJSON（またはそれをcheckpointキーへ置いたJSON）です。
収集側はreceiptのhashと全ファイルinventory / device / inode / size / mtime / ctimeを照合し、
終了時も継続性を検査します。収集側自身が重み全payloadを再hashしたとは扱いません。
出力は新しいディレクトリに限定し、既存結果や入力配下への上書きを拒否します。

## nativeと仮想gain

~~~bash
python scripts/experiments/gemma4-purpose/native.py \
  --checkpoint /nvme/model --cases /nvme/cases.json --output /nvme/new-native \
  --mode chat-public-answer
~~~

これはpreviewです。同じ引数へ以下を追加すると実行します。

~~~text
--checkpoint-attestation /nvme/checkpoint-identity.json
--attestation-sha256 RECEIPT_SHA256 --run
~~~

native.pyはtext decoderの全scalar / vector parameterとlayer_scalarを対象にします。
既存実測と同じBF16 autogradの値をFP32で累積し、原本BF16値との一致を確認します。
--virtualを指定すると、全層のO投影入力headごとのunit gainとMLP-down入力列ごとのunit gainも微分します。
これらは実行時の仮想乗数であり、行列全要素それぞれの微分ではありません。
投影の列群L2を、unit gain変更に対応する重みL2の費用として別途保存します。

- chat-public-answer: 公式thinking ON chat-templateの公開assistant回答部分だけをteacher-forceし、
  正と対照それぞれの全継続tokenの平均log probabilityの差を計測します。
  思考を生成・採点しません。閉じるturn tokenも採点外です。
- raw-next-token: chat-template無しのpromptで、正と対照の最初の継続tokenのsoftcap後logit差を計測します。
  token境界が変わる場合や最初のtokenが同じ場合は、黙って除外せず拒否します。
  長い継続の正しさを測る値ではありません。

この2モードは単位も条件も異なります。同じ観測として結合しないでください。

## 層化した実行列座標

~~~bash
python scripts/experiments/gemma4-purpose/matrix.py \
  --checkpoint /nvme/model --cases /nvme/raw-cases.json --output /nvme/new-matrix
~~~

matrix.pyは実q/k/v/o/gate/up/down行列ごとに512座標を決定的に層化抽出します。
headと行・列の帯を保った標本です。Gemma 4の形状に合わなければ拒否します。
対象モデルの元実験では410行列でしたが、収集結果の実数を記録し、410を前提に補完しません。
forward入力とbackward出力の積から真のdL/dW[row,column]を計算し、
FP32積・FP64のtoken/batch和で集めます。全行列のparameter gradientは保持しません。
先頭・末尾の対象行列の計8座標で、zero-additive probeのautogradと照合します。
目的関数はraw-next-tokenだけであり、chat計測とは混ぜません。

## 実行資源と成果物

既定はCPU、--threads 4です。GPUは --device cuda:0 --gpu-exclusive-approved が必要で、
見えるGPUがちょうど1枚かつ70GiB以上空いていることを検査します。
このflagはロックの代替ではありません。配信・ベンチとの排他と復帰は所有者が外側で管理してください。
当レシピは既存配信コンテナを操作しません。

主な成果物:

- provenance.json: 入力・コードhash、checkpoint statとreceipt、mode / device / threads / seed。
- frame-evidence.json: ケースIDとtemplate / prefix hash・token数。ケース本文や生成思考は保存しません。
- coverage.json: 実際に観測したtensor・層・座標・virtualとnativeの区別。
- source-native-values.npzまたはsample-plan.npz: 計測座標と元の実値。
- gradient-目的-fold.npzまたは.npy: 数値配列だけ。読み取り時はallow_pickle=False。
- progress.jsonとsummary.json: margin、参照差、時間、丸め条件、未使用tensor。
- 途中で失敗した場合のfailure.json: incompleteと例外種別。未完の結果を完成扱いしません。

局所一次微分は、BF16で実際に変わる値、有限forward、生成能力の変化を保証しません。
次段では元checkpointから一度だけBF16へ丸めた共同編集を作り、
有限差と目的別の得失を検査してから公式devを測定します。
このcollectorは旧実験の全fold非退行という採用ルールや固定目的を持ち込みません。
目的・許容損失・評価条件は次段の実験計画で明示します。

## 局所Gから共同編集を提案する

joint.pyは上の完了済みcollector出力から、一つの候補specを作ります。
既存finite_responseが「実測した有限端点の応答」を扱うのに対し、このレシピは
「一つのcheckpointにおける局所微分G」を使います。両者を同じ実測値として扱いません。
実行はCPU数値処理だけで、モデルをロードしません。

~~~json
{
  "id": "explicit-purpose-proposal",
  "cost_metric": {"type": "euclidean"},
  "constraints": [
    {"purpose": "implementation", "fold": 0, "minimum_delta": 0.02},
    {"purpose": "implementation", "fold": 1, "minimum_delta": 0.01},
    {"purpose": "general", "fold": 0, "minimum_delta": -0.01}
  ],
  "rounding": {"relative_radius": 0.025, "absolute_radius": 0.000001, "rounds": 16}
}
~~~

これは形式例で、推奨閾値ではありません。minimum_deltaはそのcollectorのmargin単位です。
負値は明示的な予測損失許容です。目的・fold・非退行制約・半径を暗黙に追加しません。
同じcollection内の計測済みfoldだけを参照できます。
cost_metricはEuclidean L2、または明示した正のfloorを持つrelative対角費用
（各座標の費用1/max(abs(base), floor)）です。後者でも実際の絶対delta L2を別に報告します。

~~~bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/experiments/gemma4-purpose/joint.py \
  --collection /nvme/completed-native --plan /nvme/purpose-plan.json \
  --output /nvme/new-proposal
~~~

既定は計算と要約表示だけです。--write-specを付けると新しい出力ディレクトリへreport.jsonと
候補specを保存します。重みcheckpointは作りません。
nativeはindices、層化行列はcoordinatesで、実際のBF16 set_valuesを保存します。
virtual gainは単独の保存重み座標ではないため、このsolverでは実体化しません。

非負双対の連続最小L2解を使い、明示した回数だけBF16丸め後の残差を補正します。
毎回、同じ元BF16値＋蓄積したF64提案をF32→BF16 ties-to-evenへ一度だけ丸めます。
既に丸めた候補へ増分を重ねません。半径は丸め前の連続提案に適用され、
丸め後の実変位が半径を超える場合があることをreportへ明示します。
離散BF16の大域最小性は保証しません。制約未達・変更消失もそのまま報告し、
予測が満たされたことをモデル改善や採用資格とは扱いません。
非収束は非収束として拒否し、実行可能な解が存在しないとは断定しません。

分母は自動で全モデルとみなしません。represented_source_l2はnative対象値、または行列の標本座標だけです。
任意のplan.reference_norm = {"l2": ..., "scope": "..."}を指定した場合は、
caller_declared_reference_normとして明示し、その分母に対する比を別名で記録します。
元実験の全保存tensorノルムを固定値で埋め込んでいません。
specは正規editorで元原本と再検証し、実有限forward→公式dev→未見評価へ進めてください。
行列は通常64座標までです。これを超えるspecには--allow-batched-matrixの明示指定が必要です。
その場合も512座標/tensor・合計65,536座標以内に限り、sample-plan SHA付きmatrix_batch specを出します。
未opt-inまたは上限超過ではreportのみを書き、specを出さず終了コード2を返します。
per-tensor / total件数、必要なopt-in、上限適合性をreport.editor_compatibilityで確認できます。
標本数209,920全体の編集を既存editorで適用できるとは扱いません。

## 移植の範囲

| 旧実験の機能 | このレシピ |
|---|---|
| 全層native gain、O-head / down-column仮想gainのraw目的微分 | native.pyのraw-next-tokenと--virtual |
| thinking ON公開回答全tokenのnative目的微分 | native.pyのchat-public-answer |
| 全投影行列を層化した真の座標微分とzero-additive照合 | matrix.py |
| native / sampled matrixの局所G最小L2・BF16残差補正 | joint.py |
| 候補RAM適用、有限forward、元checkpointへ復元 | 既存weights/finite_replay.py側の明示固定spec経路 |
| WSP実行・採用・全自動探索 | 収録対象レシピでは実行しない |

旧スクリプトに埋め込まれたケース本文、固定サーバーパス、固定の2fold・目的・旧採用条件は移植しません。
旧matrixの座標を再現する場合は--seed 20261105を明示します。
既定seedは0です。式・座標抽出と小行列autogradはCPU検査しますが、
移植後にGemma全体のforwardを再実行したことを意味しません。

## 保存済みchat微分の費用比較

compare_costs.pyは過去のchat-native collector形式（cases / scan-coverage / observationsと数値tensor辞書）を、
対応するbasis.npz・basis provenance・frame-evidenceと照合して読むCPU専用レシピです。
cases SHA、元checkpoint identity、template frame、tensor順序・shapeを確認します。
Torchファイルは自己生成した保存感度に限定し、weights_only=True / map_location=cpuで読みます。
実モデル、GPU、forward、ベンチは使いません。新しいデータをこの旧形式に変換する必要はありません。

~~~bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/experiments/gemma4-purpose/compare_costs.py \
  --legacy-chat /nvme/saved-chat --basis /nvme/saved-basis/basis.npz \
  --basis-provenance /nvme/saved-basis/provenance.json \
  --plan /nvme/cost-plan.json --output /nvme/new-cost-comparison
~~~

既定preview、--runで最大3profileを計算します。
plan.profilesにはid、cost_metric、group_mean_lower、fold_lowerを指定します。
後二つは計測した全目的の名前をキーとする有限値の辞書です。
各目的のfold平均と各foldの許容損失を明示し、固定の全fold非退行を加えません。
plan.roundingはjoint.pyと同じです。diagnostic_relative_floorは正の診断用floorです。
相対費用のcost_metricはtype=role_relative、floor_ratio、absolute_floorを指定します。
役割はtensor名のlayer番号だけを*へ置換したものです。
役割ごとのfloor=max(absolute_floor, floor_ratio×役割全座標の絶対値中央値)を使い、
cost=1/max(abs(weight), floor)とします。これはFisherやactivation摂動の実測ではありません。

任意のfixed_delta_floor_ratiosは既に得た実変位の費用を別floorで計るだけで、
追加の最適化や第4候補を意味しません。
reference_normを指定した場合は、そのscopeとartifact SHAをplanに残してください。
結果は3案の同じcoding目標に対する局所予測、実BF16丸め、役割別変位、
絶対L2と相対費用を保存します。既存の生成改善や失点を、この局所予測へ変換しません。
