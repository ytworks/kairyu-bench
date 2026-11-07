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
既定は従来のraw-next-tokenです。--mode chat-public-answerを指定するとnativeと同じ
公式thinking ON frame、公開correct/foil回答それぞれの全token平均log probability差を使います。
native/virtual gainの微分へ読み替えず、同じ行列の実座標に対する微分として保存します。
先頭tokenが同じ回答でも後続tokenまで採点します。異なるmodeのcollectionは混ぜません。

native/matrixの両方で --max-cases-per-microbatch 1 を指定すると、既存packingの順序を
保ってcorrect/foilの1組ずつに分割します。省略時は従来packingを維持します。
分割後も元foldの件数で勾配を割るため目的関数の重みは同じです。ただしBF16の
計算順序による丸め差はあり、実行時のbatch設定をprovenanceに保存します。
これはactivation checkpointingではなく、単一組でもGPUメモリを保証しません。
matrixのhead配置は現HFの per_layer_config（例05）または旧checkpointの
global head fieldsから解決し、実際のprojection形状と一致することを必須とします。

chatは共有の1600 paired-token packingでmicrobatchへ分け、各batchのmargin和を
元foldの例数で割って累積します。例数や回答token数の違うbatchを等重み平均しません。
初期pilotではprefix+correct/foilがそれぞれ512token以下のケースだけを受け付け、
超過はtruncateせず拒否します。1case/foldからGPUのpeak memoryを確認してください。
現在はeager/evalでgradient checkpointingを使いません。checkpoint再計算はhookと
zero-additive probeの二重作成を防ぐため拒否します。2k tokenへ黙って拡張しないでください。

~~~bash
python scripts/experiments/gemma4-purpose/matrix.py \
  --checkpoint /nvme/model --cases /nvme/authored-science-cases.json \
  --output /nvme/new-chat-matrix --mode chat-public-answer
~~~

上はpreviewです。実計測には既存のattestationと--run、GPUの場合は排他承認flagが必要です。
casesは明示された1ファイルのcaptured bytesをparse/hashし、前後statも照合します。
隣接dataset、benchmark結果、評価splitを自動探索しません。入力の出典・calibration用途は
呼出側で固定してください。ケースを改良や選択に使った後は独立holdoutではありません。
この道具だけで入力内容が評価データから独立していると証明するものではありません。

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
その場合も512座標/tensor・合計262,144座標以内に限り、sample-plan SHA付きmatrix_batch specを出します。
未opt-inまたは上限超過ではreportのみを書き、specを出さず終了コード2を返します。
per-tensor / total件数、必要なopt-in、上限適合性をreport.editor_compatibilityで確認できます。
標本数209,920全体の編集を既存editorで適用できるとは扱いません。

## 移植の範囲

| 旧実験の機能 | このレシピ |
|---|---|
| 全層native gain、O-head / down-column仮想gainのraw目的微分 | native.pyのraw-next-tokenと--virtual |
| thinking ON公開回答全tokenのnative目的微分 | native.pyのchat-public-answer |
| 全投影行列を層化した真の座標微分とzero-additive照合（raw / 公開chat回答） | matrix.py |
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

## code90: 自作の実行検証済み科学校正

science_cases.pyは、今回独立に作成したNumPy programの学習用16caseと、
別のproblem familyの転移確認用8caseを生成・CPU検証します。
既存benchmarkの問題・正解・reference実装や隠れたreasoningを取り込んでいません。
これは重み感度・実行校正のための入力であり、WSP scoreや未見評価の代用ではありません。
現在のSciCode失敗が数値・shape・signatureのどれに由来するか未分類のため、
特定の原因が実測で支配的だとは主張しません。

学習用は、重み付きmoment、積分、周期差分、確率正規化、ridge、Markov状態、
射影、密度、共分散、保存flux、陰的更新、batch統計、Verlet、group集約、
補間、batched線形系を扱います。数値algorithmと状態/保全を2目的に分け、
各目的は別familyからなる2foldです。各promptはinterface・shape・float64出力・
入力を変更しない契約を指定します。正しい実装は入力から計算する一般的な
algorithmであり、検証で得た定数配列を返しません。

各correct/foilを可変shape、float32/float64入力、決定的な乱数seedで実行します。
参照は独立のloopまたは解析的恒等式で作り、数値・shape・dtype・有限性・
入力保全を確認します。foilは数式、axis、dtype等の1箇所を変えた実行可能な
programで、少なくとも1つの検査に落ちない限りexportしません。
元の検証は各8trial、別seedのfixtureでも同じ契約を確認しています。
有限個の入力を通過したことは全入力に対する数学的証明ではありません。

~~~bash
# 既定はCPU検証だけ。出力やモデルを作らない。
python scripts/experiments/gemma4-purpose/science_cases.py

# 検証後、新しいdirectoryだけへ書く。
OPENBLAS_NUM_THREADS=1 python scripts/experiments/gemma4-purpose/science_cases.py \
  --write --output /nvme/science-calibration
~~~

出力は次の通りです。

- train-cases.json: purpose→fold→case形式。16caseだけをnative/matrix collectorへ渡す。
- transfer-cases.json: split=transfer、gradient_allowed=falseの別envelope。
  既存collectorのread_casesはこの形式を拒否する。gradient_allowed=falseや
  split=transfer/test/holdoutの明示ラベルはpurpose→foldへ再ラップしても拒否する。
  未申告データの独立性までは保証しない。学習gradient・係数fit・
  重み学習へ渡さず、別familyへの生成/実行の確認に使う。
- metadata.json: source hash、family一覧、seed、NumPy version、correct/foilの
  実行成否とtext長。正解全文やfixture入力は検証logへ重複出力しない。

転移familyはKalman、3D回転、log-domain HMM、周期Poisson、線形保存制約、
確率状態の共分散伝播、barycentric補間、熱的速度再調整です。学習用とのfamily名は
重複しませんが、基本的な数理primitiveは共有し、統計的独立性を保証しません。
この8caseを繰り返し見て候補を選ぶなら既見校正となり、新しい最終評価ではありません。

テスト実装はteacher-forcing用JSONへ含めません。prompt+correct/foilは最大2048
UTF-8 bytesに制限しますが、これはtoken数ではありません。今回のGemma公式tokenizer・
正規chat-public-answer frameをCPUだけで照合した学習16caseは、prefixを含め
141〜255tokensで、512token上限内でした。tokenizer/templateを変えるときは
再確認し、超過を黙って切り捨てないでください。

このutilityは自分のsource内の信頼済みprogramだけをexecします。
外部JSONやモデル生成codeを実行するsandboxではありません。
新しいmodel生成の実行確認には、別途隔離された実行環境が必要です。
NumPy以外の追加依存、モデルロード、ネットワーク、GPU、WSP呼出しはありません。


## code90: 生成されたpublic answerの隔離実行

science_sandbox.pyは自作科学校正16件・転移8件に対する保存済みpublic answerを
評価します。モデル/APIを呼ばず、生成codeをホストでexecしません。
既存のNumPy入りDocker image IDと明示Unix socket/daemon IDを指定し、
networkなし、UID1003、read-only、capabilitiesなし、GPUなし、host mountなしの
短命コンテナでsolveだけを実行します。image pullやdaemon設定変更はありません。
入力配列だけをstdinへ渡し、correct/foil/referenceや期待値は渡しません。
信頼済みscience_cases.sampleの期待値との数値比較はホスト側で行います。

~~~bash
python scripts/experiments/gemma4-purpose/science_sandbox.py \
  --cases /nvme/science-calibration/transfer-cases.json \
  --responses /nvme/generation/transfer-responses.json \
  --docker-socket /nvme/docker/run/docker.sock --daemon-id EXPECTED_ID \
  --image sha256:EXISTING_IMAGE_ID --output /nvme/new-execution-preview
# 内容を確認後、別のfresh outputと --run で実行する。
~~~

既定previewはDocker命令0で、fresh outputへpreview receiptを書きます。
--runは既存imageのIDとdaemonを照合してから各caseを直列実行します。
既定はCPU1、RAM512MiB、PID64、tmpfs16MiB、startupを含むcase wall15秒、
stdout+stderr合計1MiBです。停止時の削除対象は自分が生成したUUID名かつ
一致する所有ラベルのコンテナだけです。Docker自体の制御命令には別の20秒上限があります。
接続先や所有権が変わった場合は他のコンテナを操作せず停止します。
Dockerはprocess隔離であり、悪意あるcodeに対する形式的なsecurity証明ではありません。

responses JSONはschema_version=1、split=train|transfer、cases_sha256、
generation、responsesを持ちます。trainは既存purpose→fold mapping、
transferは既存cases配列envelopeをそのまま指定できます。
generationはcheckpoint_sha256/model_id/frame/enable_thinking/reasoning_saved=falseと
sampling条件を保存します。生成器はthinking_control=request|server-defaultも保存し、
server-default時のenable_thinkingはnullです（サーバー設定は別証拠で確認）。
HTTP失敗はerror_type=HTTPErrorと整数http_status（400–599）を記録でき、
public_answerとの同居や文字列/真偽値のstatusを拒否します。
sandboxは生成失敗をgeneration_error、passed=nullとして保持します。
例えばsampling=server_official_generation_config、
temperature=null、max_tokens=8192、seed=0です。
各responseはid、public_answer、request_id、finish_reason、seconds、
数値だけのusageを持ちます。隠れたreasoning本文は受け付けず保存しません。
cases内容はこのレシピの信頼済みcatalogと完全一致させ、SHAで生成入力と照合します。

抽出はplain Python、または回答全体が1つのPython fenced blockの場合だけです。
短い説明付きの正しい関数をformat failureにする可能性があり、WSPのcode sanitizerと
同一の指標ではありません。reportは元public answerと抽出codeのSHAを別に残し、
形式/構文、実行/interface、数値、shape、dtype、入力変更を区別します。
未知応答、timeout、sandbox/protocol errorはunknownで、勝手に0へ補いません。
数値比較はrtol=2e-6、atol=2e-7、float64 ndarray出力・有限性・shape一致を要求します。
入力保全検査はcontainer内のため、敵対的なprotocol偽装まで排除する採点器ではありません。

学習/転移は別reportで、転移caseをgradientへ渡しません。この8caseで候補を繰り返し
選ぶと既見校正になります。公式WSP score、未見確認、code90達成の代用にはしません。
現在の16件は短い数値kernelの基礎確認です。収束判定、複雑な境界条件、
多段の関数依存、数値特異点、確率的許容誤差、計算量・保全を含む大きな実装への
転移は別に確認が必要です。


science_generate.pyは、今回のone-off生成launcherから固定パスとDocker探索を除いた
再利用CLIです。API endpoint/model/checkpoint SHAを明示し、plain train mappingと
transfer cases配列envelopeの双方を受けます。元のcorrect/foilはAPIへ送信せず、
promptだけを送ります。既定の--thinking-control requestは従来どおり
chat_template_kwargs={"enable_thinking":true}を明示します。
--thinking-control server-defaultを明示した場合だけ、そのfieldを省略します。
このmodeはgateway等がそのoverrideを受け付けない場合のためで、
enable_thinking=null、frame=live-server-default-public-answerとして記録し、
生成器自身はthinkingの実状態を推定しません。実配信replicaの起動引数・
template/default設定と重みの同一性を別途固定してから使ってください。
temperature/top_pを省略してserverの公式generation_configを使い、
seed0/max_tokens8192が既定です。
checkpoint SHAは呼出側の申告であり、このCLIだけで配信中の重みを認証しません。
別途同じserving identity/readonly mount証拠を確保してください。

~~~bash
python scripts/experiments/gemma4-purpose/science_generate.py \
  --cases /nvme/science-calibration/transfer-cases.json --split transfer \
  --endpoint http://127.0.0.1:8000/v1 --model MODEL_ID \
  --checkpoint-sha256 CHECKPOINT_SHA256 --output /nvme/new-generation
# 既定previewはHTTP 0、receipt保存のみ。--runで明示実行。
~~~

最大32並列、既定8、socket操作timeout240秒、retry0、1case1choiceです。
urllibのtimeoutは各socket操作への制限で、request全体の厳密なwall deadlineではありません。
public content、request ID、finish_reason、数値token usage、経過秒だけを保存し、
reasoning本文・raw response・HTTP error本文は保存しません。usage欠測はnullです。
失敗はexception classをerror_typeへ保存し、HTTPErrorの場合は整数statusも
http_statusへ保存します。本文・header・URL・例外messageは保存しません。
失敗行は公開回答でも誤答でもなく未測定です。SIGINT等で中断した場合は
未着手futureを取り消しますが、実行中HTTPは個別timeoutまで待つ場合があります。
このutilityはAPI呼出しだけで、code実行・重み変更・WSP採点をしません。
生成結果はscience_sandbox.pyへ別途渡します。

## code90: advanced 自作科学校正

advanced_science_cases.py は、公開dev課題の失敗概念を参考に新しく書いた
16学習case＋8転移caseです。既存science_cases.pyの24caseと学習/転移分類は
変更していません。元benchmarkの参照実装・test・goldを取り込まず、
新しい式、入力契約、実装、数値検証を定義しています。dev-informedな校正であり、
独立した未見評価や公式WSP scoreではありません。

学習は4目的×4case、各目的2fold×2caseです。

| 目的 | 自作の実装要素 |
|---|---|
| state | リセット付き統計、逐次affine状態、容量/供給/流出の保存、ノルム制限付き反復更新 |
| counting | 重複イベントの保持、同値を含む辞書順top-k、等号/ゼロ費用を含む予算、境界を含む格子点計数 |
| simulation | 対エネルギーと力、周期的kick→drift、複数壁反射、非弾性衝突の運動量と散逸 |
| probability | 外部乱数を注入した採用/拒否連鎖、非復元抽出、区分密度のCDF、log生存確率 |

転移8caseはSchur補完、吸収連鎖の時間、障害物経路数、Hermite値/微分、
rank-one逆行列更新、Poisson-binomial分布、Heun step-doubling、whiten/recolorです。
単なる学習caseの変数名変更ではありませんが、数学的primitiveは共有し、
family名の分離だけで統計的独立性を保証しません。旧transfer8も新transfer8も
gradientへ渡さず、転移の失敗をそのまま学習caseへ転記しません。
繰り返し選抜に使った転移集合は既見校正として扱います。

~~~bash
python scripts/experiments/gemma4-purpose/advanced_science_cases.py
python scripts/experiments/gemma4-purpose/advanced_science_cases.py \
  --write --output /nvme/fresh-advanced-science
~~~

既存のCase、正/foil検証、split exporterを再利用します。correctは一般的な
計算手順で、乱数seedや期待値を埋め込んでいません。新しい数値入力・異なるshape・
float32/float64入力・等値/端点・ゼロ/非ゼロ等の有限条件で、
float64出力、shape、有限性、数値一致、入力非変更を検証します。
期待値は独立loop、解析的恒等式、または別のsolverから算出します。
それぞれのfoilは一般手順の1箇所を誤らせ、最低1条件で落ちることを必須にします。
この検証は全入力の正しさの証明ではなく、NumPy以外の複雑な科学依存や
長いmulti-file工程を網羅しません。

今回の8入力/caseの検証でcorrectは24×8すべて成功し、全24foilに失敗がありました。
別seedのCPU契約検査も通過しています。公式Gemma tokenizerと
chat-public-answer frameによる学習16caseのprefix＋correct/foilは最大288tokensで、
512token上限内です。モデルやtokenizer/templateを変える場合は再測定し、
超過を切り捨てません。テストコードはteacher-forcing入力へ含めません。

science_generate.pyは同じtrain mapping/transfer envelopeをそのまま受け、
correct/foilをAPIへ送りません。実行は既存science_sandbox.pyに
明示的なcatalog選択を追加するだけです。

~~~bash
python scripts/experiments/gemma4-purpose/science_sandbox.py \
  --catalog advanced \
  --cases /nvme/fresh-advanced-science/transfer-cases.json \
  --responses /nvme/advanced-generation/transfer-responses.json \
  --docker-socket /nvme/docker/run/docker.sock --daemon-id EXPECTED_ID \
  --image sha256:EXISTING_IMAGE_ID --output /nvme/fresh-advanced-execution
~~~

既定catalogはbasicです。catalogと全splitの公開caseが完全一致しなければ拒否し、
任意のPython moduleや外部のreferenceコードはロードしません。隔離条件・数値判定・
unknown扱い・preview/Docker 0・source/hash照合は既存実行器と共通です。
revision1（既定の --revision 1 / sandbox --catalog advanced）は保持します。
元モデルの公開回答を24件だけ生成し、隔離CPU実行したrevision1の生結果は
学習14/16、転移7/8、計21/24、unknown0でした。各8入力で177/192条件が成功しています。
しかし予算caseの公開文は「昇順」と書きながら入力が未整列で、Schur caseは
行列ブロックの寸法を明示していませんでした。この2件の失敗を能力低下の
根拠にはしません。生結果は変更せず、事後の条件明確な部分集合は別診断として
学習14/15、転移7/7と記録します。この除外は生成前の独立な評価設計ではありません。
残る反射caseの1入力の数値不一致は、float32の中間計算精度が原因の可能性が
ありますが、厳密な原因を再実行で確定してはいません。

--revision 2 / sandbox --catalog advanced-v2 は2問の公開文だけを明確化します。
予算caseは未整列入力を昇順ソートしてからprefixを選ぶこと、Schur caseは
非スカラーの各ブロック寸法、右辺ベクトル、連結した一次元出力を明示します。
全24件のcorrect/foil、数値入力生成、期待値、許容誤差は変更しません。
revision1のtrain/transfer出力は元artifactとbyte一致を確認しています。
v2は別出力へ保存し、旧結果をv2採点に読み替えません。

~~~bash
python scripts/experiments/gemma4-purpose/advanced_science_cases.py \
  --revision 2 --write --output /nvme/fresh-advanced-science-v2
# 生成は同じscience_generate.pyへv2 casesを明示する。
# 実行時は science_sandbox.py --catalog advanced-v2 を使用する。
~~~

旧120件＋新学習16件の136件を使用する場合も、revisionを固定した別入力にします。
転移8件は引き続きgradientから除きます。新版の生成、実行、gradientの実施状況は
それぞれのreceiptで区別し、旧版の結果で新版の改善を主張しません。

### Native correct-answer log probability

Native scalar/vector collection additionally accepts
`native.py --mode chat-public-answer --objective correct-logprob`.
The default remains `--objective margin`, including the existing raw-next-token
mode. The opt-in objective maximizes the mean token log probability of each
public correct answer. It retains the same correct/foil forward rows, padding,
official chat frame, softcap, microbatch partition, and original fold denominator;
foil likelihood is not part of the selected loss.

The shared `objective_variants.py` implementation is pinned in input provenance.
Only `expected_correct_logprob` is a reference for the new objective;
`expected_margin` retains its old meaning and is not reinterpreted. New records
use `correct_logprobs` / `mean_correct_logprob` and
`objective_kind: correct-logprob`, while legacy records retain their margin keys.
These native BF16-autograd derivatives are neither finite BF16 edit effects nor
free-generation accuracy. Existing margin-only joint solvers are not connected
to the new arrays: a downstream consumer must explicitly require the matching
objective, source, frame, case split, and fold layout before using them.


For long native calibration sequences, opt in to
`--answer-batching serial --max-cases-per-microbatch 1`. This supports
chat-public-answer native parameters only (`--virtual` is rejected).
Each correct/foil answer keeps the original pack padding width and every
original token. Margin collection evaluates correct then foil in separate
forward/VJP passes; correct-logprob needs only the correct pass.
Each answer gradient is converted to FP32 before accumulation with the original
fold denominator, and its graph is freed before the next answer. No case is
excluded or truncated. Provenance and fold records label this layout and the
actual forward count. The mathematical objective is unchanged, but BF16
batch shape and gradient reduction can differ from paired collection; the
tool does not claim bitwise paired/serial gradients. Paired remains the default,
including the donor collector. New data must be distinguished from earlier
paired measurements.

## Public-answer boundary diagnostic

旧chat-public-answer frameは、公式templateのenable_thinking=trueを使っていたが、
そのprefixはsystem内のthink flagとassistant-startで終わっていた。
assistantのthought終了を含まず、答案を応答開始へ直置きする条件付き尤度だった。
この条件と実生成の公開finalを同一視しない。旧成果物を変更せず保持する。

共有prepare_framesの新版はframe_protocol=
gemma4-public-answer-v2-forced-empty-thoughtを明記する。
公式templateがthinking OFF生成prefixで使う空thought区間
<|channel>thought\n<channel|>を、ONのassistant-startへ明示的に追加する。
public本文は閉じ記号の直後に続く。これは思考本文を生成・保存しない
teacher-forced条件であり、通常の思考ON生成で得た思考文脈の再現ではない。
旧frame証拠を新版の微分/予測へ混ぜない。今回の136caseでは新旧の答案token列は
同一で、prefixは4tokens増加、最大509tokensとなった。

public_answer_diagnostic.pyは、明示した自作caseを最大3件だけ使って境界と
logit計算を切り分ける。既定previewはモデル読込み・forward・出力作成0。
--run時も生成をせず、正答tokenの尤度・順位と差の数値だけを保存する。

~~~bash
python scripts/experiments/gemma4-purpose/public_answer_diagnostic.py \
  --checkpoint MODEL --cases AUTHORED_CASES_JSON \
  --checkpoint-attestation ATTESTATION --attestation-sha256 SHA256 \
  --case-id AUTHORED_CASE_ID --output FRESH_DIRECTORY
# 明示GPU承認後だけ --device cuda:0 --gpu-exclusive-approved --run
~~~

各caseを旧assistant-start/新forced-public-boundary、padding0/+8で測り、
公式Gemma4ForConditionalGeneration全体のlogitsとtext submodel+lm_headを照合する。
公式HFのhead dtypeのままsoftcapする経路と、旧customのFP32へupcastしてから
softcapする経路を別に記録する。異なるdtype間のbit一致を誤って要求しない。
同じanswer token位置と内容を使い、frameやpaddingの差を分離する。
最大512tokensで超過は拒否し、切り捨てない。既定120秒のbudgetは各forward前の
dispatch検査であり、モデル読込みや実行中forwardを中断するhard deadlineではない。

元checkpointは書き換えず、既存attestationとstat継続性を確認する。
fresh outputへ数値reportを保存し、失敗はincompleteとexception型だけを記録する。
診断の通過はコード能力向上でもWSP scoreでもなく、旧微分や候補を正当化しない。

実際の3問で公式whole-modelとの一致と境界の量的影響を確認した記録は、[公開回答frame監査](../../../docs/experiments/2026-11-06-public-answer-frame-audit.md)を参照してください。旧frameを新版の結果として再解釈しません。

## Coupled native objectives

[propose_native_objectives.py](propose_native_objectives.py) combines matched native correct-logprob and margin collections using the existing local-gradient solver. The [input and finite-replay guide](../../../docs/wsp-native-coupled-objectives.md) specifies full-checkpoint identity, serial cohort matching, explicit permitted losses, BF16 realization and selected-word restoration proof. Default preview reads metadata only. The same native values from different full checkpoints do not establish matching gradients; collect again after promoting a new incumbent.

### Coherent sampled matrices with both public-answer objectives

matrix.py also accepts --objective correct-logprob and --answer-batching serial,
using the same objective helpers and pair padding as native collection.
--sampling-mode mlp-triplets fixes shared hidden indices across gate/up rows
and down columns, with 512 sampled cells per matrix. Attention sampling and all
legacy defaults remain unchanged. The sampled derivatives still pass the
existing eight zero-additive coordinate probes.

The explicit v2 plan in propose_native_objectives.py matches four collections
on one complete current-parent identity, concatenates native and sampled O/MLP
columns, and reuses the existing two-objective optimizer/BF16 rounding.
Q/K/V remain diagnostic-only. Costs, permitted losses and block scales are
explicit owner inputs; actual matrix participation after rounding is reported
without forcing nonzero triplets. See the [v2 input and proof contract](../../../docs/wsp-native-coupled-objectives.md#optional-native-plus-sampled-matrix-plan).

V2 also supports explicit eligible_blocks: ["matrix"] to keep the accepted
parent's entire native block fixed. Native columns are absent from optimization
and output, while both objective row sets and the same four-collection identity
checks remain. Defaults are unchanged; no large-cost masking or forced edits
are used. Saved-output regression checks cover native/QKV bytes and BF16 sets.

## Behavior18: 固定行動契約の追加校正

behavior_cases.py と behavior_offline.py は12 train / 6 transferの小さな独立校正です。
初回生成はtrain12のみ。全18の著者correct/foilを先に固定16fixtureで隔離CPU検証し、
transferは候補spec/幅凍結後にAV/候補で各1回だけ測り、gradientへ渡しません。
既存Docker.run/WORKERを使用し、Cの要求関数名はAST適合後だけsolveへaliasします。
BのSymPy実使用などの方法要件は数値合格から推定しません。
[操作・全検証・既知の限界](../../../docs/experiments/2026-11-07-behavior18-implementation.md)を参照。
