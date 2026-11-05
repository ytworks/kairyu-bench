# 重み・activation観測ガイド

静的な重みの差、実行中に採取したactivationの概要、問題ごとの成績を同じ識別子でたどるためのツールです。重みを編集したり、観測値だけで次の編集を自動採用したりする機能はありません。

[WSP](wsp.md)が能力の得失を測り、このガイドのツールがその得失を調べる材料を保存します。[Gemma 4実験記録](experiments/2026-11-04-gemma4-weight-editing.md)には、試した手法と採用しなかった理由も残しています。

## 対応範囲

| 機能 | 入力・実行場所 | 範囲 |
|---|---|---|
| `wsp weights inspect/diff` | safetensors、ホストCPU | tensor名の選択・group化、全量数値集計、保存値の差 |
| `wsp weights spectrum/align` | safetensors、ホストCPU | 明示した行列・軸の標本診断 |
| `wsp observe init/merge/compare/view` | metadata・JSONL、ホストCPU | 採取設定、結果との結合、自己完結HTML |
| ライブactivation採取 | 起動時に明示追加するvLLM worker | 初版はGemma 4、vLLM 0.33のv1 / V2 runner、PP=1 |
| replay / teacher-forced | 観測modeとして識別 | このCLIはreplay・追加forwardを実行しない |

静的解析と表示schemaはモデル構造に依存しません。ライブadapterにはモデル・runtime固有の知識が必要です。初版はcontext parallelism、dual-batch overlap、多次元position、他モデルのlayerを受け付けません。未対応構造を推測して動かすことはありません。

静的解析と表示の検証には、小さな合成safetensors、合成観測、CPUの契約検査を使います。比較HTMLはChromiumで1440×1000と390×844の表示・filter・rank切替・セル詳細・差分拒否・外部通信0を確認しました。これらはモデル能力の測定ではありません。小さい合成tensorのCUDA graph経路と、実モデルでの採取ON/OFF・速度・VRAM測定は区別します。実モデルでの受け入れ結果が揃うまでは、低負荷や出力不変を達成済みとしません。2026-11-04開始の能力探索は停止済みで、WSP・DeepSWEの再開はツール完成後の明示承認を待ちます。今回のツール検証用に承認された短い合成入力の実モデルforwardは別枠です。実モデルの限定requestで採取・attribution・表示を確認した範囲は下記に記します。負荷測定の範囲と未達条件は下記に記し、能力改善の測定には数えません。

### 実記録で確認した範囲

2026-11-05の `live-c1-r4` では、実Gemmaに合成入力を送った初回1requestを独立に監査しました。追記中のcaptureから対象requestだけを読取snapshotとして取り出し、元のstatusを保持しています。能力の採点はありません。

| 確認項目 | 結果・範囲 |
|---|---|
| 採取 | TP2、各rank 11,712行、計23,424行 |
| site | 全60層のresidual/attention/mlpに、2channel・1headを加えた183site/rank |
| 位置 | 各rankで採取したprefill 32位置・decode 32位置。全prompt被覆率ではない |
| 重みとの構造対応 | HF tensor名・shape・軸範囲を照合。head0/16はO投影の列0:256 / 4096:4352に対応 |
| channel | 256記録の幅1統計とrank間同値を確認 |
| 表示 | 正規merge→HTML→Chromium。60層×32位置の表示、channel5375、head16、desktop/mobile、外部通信0 |
| 未確認・未達 | 全attention投影の再構成、因果的な能力帰属、性能条件の合格 |

このr4の `value` はreduction後のmeanを使っており、44/256記録で元dtypeに表現可能な `max_abs` とfloat32末尾が異なりました。最大絶対差は `5.960464477539063e-8`、相対差は `1.3934967180365296e-7` です。bit単位で元scalarと同じ値を記録したとは主張しません。原本は修正せず、この精度上の限界を監査結果へ残しています。

証拠は実験成果物rootの `tool-validation/live-c1-r4-observation-audit.json`、`live-c1-r4-merged.json`、`live-c1-r4-browser-checks.json` と同prefixのPNGです。source captureの終了や性能測定の合格を、選択requestの表示成功で代用しません。その後の修正検証と8GPU負荷測定は、下記で別の証拠として扱います。

改修後の `live-c1-r5` は両workerの終了・queue drainを確認してから圧縮原本をコピーし、先頭1requestの同じ23,424記録を監査しました。channel 256記録で `value = mean`、`abs(value) = rms = max_abs` が完全一致し、BF16表現可能性とTP間同値を確認しました。これは記録された幅1統計の整合性検査で、別に保存した全activation tensorとの照合ではありません。圧縮batchの軸を直接読んだ値と正規decoderも一致しました。同じ問題IDに複数requestがあるため、正規task mergeを別保存し、HTMLは先頭requestだけの明示的なprojectionにしています。

r5の正規gzip読込みからHTMLまでをChromiumでも再確認し、desktop/mobileのoverflow・JavaScript error・外部通信はいずれも0でした。証拠は `tool-validation/live-c1-r5-observation-audit.json`、`live-c1-r5-merged-task.json`、`live-c1-r5-merged.json`、`live-c1-r5-browser-checks.json` と同prefixのPNGです。r5の単一replica監査と、次の8GPU負荷測定は別の証拠です。

### 8GPUでの限定負荷測定

`live-c32-final` は同じGemma 4 checkpoint、TP2×4replica、MTP4、通常のcompiled/CUDA graph経路で検証しました。合成入力、`max_tokens=64`、temperature 0、seed `1700+case`、native thinking ONの短いAPI測定です。WSP・DeepSWE、長いagent実行、能力改善の評価ではありません。

| 測定 | 採取OFF | 採取ON | 時間増 | 範囲 |
|---|---:|---:|---:|---|
| c1 | 0.475740秒 | 0.561697秒 | 18.068% | 各側の初回1測定 |
| c32初回 / warmup | 1.484642秒 | 1.766897秒 | 19.012% | warm中央値から除外。モデル起動時間を含まない |
| c32 warm中央値 | 1.424930秒 | 1.534793秒 | 7.710% | a/b/cの3回 |

c32 warmの各回はOFFが1.463497 / 1.424930 / 1.408215秒、ONが1.601805 / 1.534793 / 1.464137秒でした。10%以内という時間増の目標に入ったのはこのwarm c32中央値です。c1と初回c32には当てはまらず、長文・別batch・別モデルで同程度になる保証もありません。200ms間隔のGPU使用メモリ標本では全GPUで最大値が22MiB増えました。一方、確保した固定観測bufferはrank当たり43,289,600 bytes（41.284MiB）です。標本最大とallocator内の確保量は別であり、22MiBを真のピーク増分とは呼びません。

同じcaseのON/OFF応答hash一致はc32のa/b/cで28/32・25/32・28/32でした。OFFのa対bは30/32、a対cは31/32の一致で、OFFの反復間にも不一致があり、今回の比較だけでは採取の影響と元の変動を分離できません。全出力のbit一致や出力不変を主張しません。本文・推論はこのレポートへ保存していません。

5測定すべてで、API完了後に確認した8workerのqueue depth・pending bytesは0、保存側の最終完了時刻から計算した `post_response_write_seconds` は0でした。statusがAPI終了後に更新されたことの確認には別途約0.74〜0.83秒かかっています。Docker停止後の `closed=true` は5/8workerだけで、3workerはdrain済みでもshutdown acknowledgementがありません。正規mergeは `partial_or_running` を保持します。error/drop 0を、完全closeの証拠へ読み替えません。

c32-aの32入力はcapture外の合成結果metadataへ結び、採点値は全件nullのままです。request×TP rankの64結合、全60層×3role、全183site/rank、HF weight名・axis範囲、gatewayからbackendへ同じtrace IDが届くことを確認しました。prefillはprefix cacheの影響を受けるため、上限各32位置を全requestで採ったとは扱いません。

c32-aで結合した観測は441,396件です（位置×site×rankの記録数で、token数や問題数ではありません）。rank1のraw MTP受理不明148,047件をrank0のsampler記録へ結合すると、不明0、受理434,808件・不受理6,588件となりました。不受理は既定表示から除外しますが原本には保持します。capture全体の設定・trace・worker metadata/eventは79,172,196 bytes、そのうち圧縮観測は76,440,700 bytesでした。この容量はcode snapshot、compile cache、派生HTML/JSONを含みません。

証拠は `tool-validation/final-performance-summary.json`、`observed-final-drain.json`、`observed-final-closed.json`、`observed-final-{c1,c32-a,c32-b,c32-c}.json`、対応する `baseline-final-*.json` と `*-gpu.json`、`gateway-trace-proof.json`、`live-c32-final-audit.json` です。全32問題の `live-c32-final-merged.json` と、1問題に絞った `live-c32-final-view.html` を別に保存しています。

## CPU側の準備

リポジトリ直下で、Python 3.11以上の環境にインストールします。NumPyは静的数値解析の依存です。

```sh
python3 -m venv /mnt/nvme/<work>/weight-tools-venv
/mnt/nvme/<work>/weight-tools-venv/bin/python -m pip install -e .
export KAIRYU_BENCH_PYTHON=/mnt/nvme/<work>/weight-tools-venv/bin/python
```

`wsp weights`と`wsp observe`はホストのこのPythonを使用します。これらのコマンド自身はDocker・API・GPUを起動しません。`wsp run`は従来のベンチ実行なので、同じ扱いにはなりません。

## 静的な重みを調べる

### 全体と層group

```sh
./kairyu-bench wsp weights inspect /mnt/nvme/<base> \
  --chunk-mib 8 --sample-size 4096 --top-k 8 \
  --output /mnt/nvme/<analysis>/base.json

./kairyu-bench wsp weights inspect /mnt/nvme/<base> \
  --pattern 'layers\.[0-9]+\.' \
  --group-pattern 'layers\.([0-9]+)\.' \
  --axis --output /mnt/nvme/<analysis>/layers.json
```

`--pattern`はtensor名への正規表現searchです。`--group-pattern`もsearchで、第1captureがあればその値、captureがなければ一致文字列をgroup名にします。非一致は`__ungrouped__`です。groupは利用者が定める構造上の区分で、能力名を自動推定しません。

BF16/F16/F32/F64の保存値をfloat64へ変換して集計します。整数・boolは数値normの集計から除外し、除外数を残します。未対応dtypeを復号したふりはしません。モデルのPythonコードやpickleは実行しません。

数値集計の全量とは、**選択した保存浮動小数tensorの全要素**を意味します。共有weightが実装内で参照される回数ではなく、保存されたtensor entry単位で数えます。`--pattern`で絞った結果をモデル全体のnormと呼ばないでください。

`--axis`はrank 2の各行・各列の全要素L2を計算します。軸長の上限などで計算しなかった場合はskip理由を出します。チャンク8MiBは入力payloadの単位であり、変換配列・軸集計・出力JSONを含むプロセス全体のRAM上限ではありません。

### 元と編集後の保存値を比べる

```sh
./kairyu-bench wsp weights diff /mnt/nvme/<base> /mnt/nvme/<candidate> \
  --group-pattern 'layers\.([0-9]+)\.' \
  --output /mnt/nvme/<analysis>/weight-diff.json
```

主に次を確認します。

- 保存bitが変わった要素数と、有限な数値が変わった要素数。
- 元・編集後・差のL2、各tensor/groupの差、最大差の座標。
- 相対ΔL2の分母。比較した全tensorを分母にする値と、変更tensorだけを分母にする値は別です。
- tensor名・shape・dtypeの不一致、非有限値、設定・tokenizer・template等のファイル差。

不一致を含む比較は既定で拒否します。意図して互換部分だけを見る場合は`--allow-partial`を明示し、結果の除外範囲も確認します。ゼロnormや非有限値で定義できない比・cosineは`null`として扱い、0へ置き換えません。

編集specの実数上の予測より、**保存dtypeへ丸めた後の実差分**を優先します。要素数が少なくてもL2が大きいこと、係数を縮めるとno-opや方向の崩れが起きることがあります。このdiffは既に存在するcheckpointを比較するだけで、編集specからcheckpointを作りません。

### 行列の標本と明示した軸対応

```sh
./kairyu-bench wsp weights spectrum /mnt/nvme/<base> \
  --tensor model.layers.0.mlp.up_proj.weight \
  --sample-rows 128 --sample-columns 128 --seed 0 \
  --output /mnt/nvme/<analysis>/spectrum.json

./kairyu-bench wsp weights align /mnt/nvme/<base> \
  --left model.layers.0.mlp.up_proj.weight --left-axis rows \
  --right model.layers.0.mlp.down_proj.weight --right-axis columns \
  --channels 128 --features 128 --seed 0 \
  --output /mnt/nvme/<analysis>/alignment.json
```

tensor名は対象checkpointで確認して指定します。例の名前をすべてのモデルへ適用することはできません。

`spectrum`の特異値・stable rank・effective rankは、記録した行列部分に対する値です。全行列を選んだ場合を除き、元の全行列のrank推定値とは呼びません。抽出した行・列のindex、seed、アルゴリズム、NumPy版を保存します。

`align`は「channel軸とfeature軸が同じ意味で対応する」と利用者が指定する診断です。両軸の長さが一致する必要があります。同じchannel index・同じfeature indexを両行列から採り、norm積とcosineを計算します。任意の置換対応、異なる軸の意味、ニューロンの機能は推定しません。GQA、fused QKV、TP shardなどの対応は先に確認してください。

### 全量・標本・I/Oの違い

| 出力 | 全量か標本か |
|---|---|
| inspect/diffのnorm・要素数・非有限数・最大座標 | 選択した互換tensorの全量 |
| quantile | tensorごとに既定4096の固定seed標本。小さいtensorは全量 |
| axis profile | 対応する行/列の全量。skip時は理由を表示 |
| spectrum | 既定128×128部分行列 |
| alignment | 既定128channel×128feature |

標本座標は重複せず、seedとtensor名に対して決定的です。モデルの構造・局在を均等に代表する保証はありません。

既定では全参照shardと設定ファイルの内容hashを取るため、`--pattern`で小さいtensorだけを選んでも全ファイル読みに時間がかかる場合があります。初期探索の`--no-hash`はこの読込みを省きますが、内容同一性を確定した成果物にはなりません。出力の数値読込み・hash読込み・推定I/Oと実測時間を分けて確認します。

## 実行中のactivationを採取する

以下は導入手順です。`observe init`は設定を作るだけで、servingやベンチを開始しません。

```sh
./kairyu-bench wsp observe init /mnt/nvme/<checkpoint> \
  --output-dir /mnt/nvme/<capture> --model <served-model-id> \
  --tp-size 2 --replicas 4 --mtp-tokens 4 --mode live
```

新しいcapture directoryを使用します。checkpoint識別を取り、`capture.json`へmode・topology・上限を保存します。既定はworker当たりGPU観測bufferの上限256MiB、capture全体のworker・client/APIログ・metadataへの割当容量2GiB、requestごとのprefill/decode先頭各32位置です。全token・全requestの均等標本ではありません。prefix cacheで再計算されなかったprompt位置は採取されず、全promptの被覆率を推定できる分母がない場合もあります。上限・drop・error・受理不明を結果と一緒に読みます。

serving imageにも同じパッケージを導入し、capture directoryをhost・API・worker・ベンチrunnerから同じ絶対pathで見えるようにします。replicaごとに識別子を分け、通常のvLLM起動引数に次を追加します。

```sh
export KAIRYU_BENCH_CAPTURE_CONFIG=/mnt/nvme/<capture>/capture.json
export KAIRYU_BENCH_CAPTURE_REPLICA=replica-0

# 既存の vllm serve 引数へ追加する部分:
# --worker-cls kairyu_bench.wsp.observe.vllm_worker.CaptureWorker
# --middleware kairyu_bench.wsp.observe.middleware.CaptureMiddleware
# --enable-request-id-headers
```

観測codeとsite選択に固有のNVMe内cache directoryを、servingの VLLM_CACHE_ROOT に指定してください。vLLMのAOT cacheがhookの存在をhashへ反映しない場合、通常実行用のcache再利用で観測hookが期待どおり実行されない可能性があります。compiled / CUDA graphを無効化して別条件へ置き換えず、独立cacheで通常構成のwarmupを行い、実際にsite記録が出るか確認します。この注意点だけで、今回観測したattribution失敗の原因を確定したとは扱いません。

ベンチ側は通常の`wsp run`へ`--observe-config /mnt/nvme/<capture>/capture.json`を追加します。trace IDが問題・API callとworker requestを結びます。API gatewayを通す場合、W3C `traceparent`を転送する必要があります。転送できないrequestを順序やpromptの類似性で勝手に問題へ割り当てず、unattributedとして残します。gateway自身のtracingとSDK設定も必要です。Terminal-Bench / SWE-bench Proにはnative設定でtrial単位のtraceparentを渡します。同trial内のbackend callは別request_idで区別し、request_scope、item_id、trial_id、run_idを保存します。Harbor retry ordinalが得られない場合はnullを保持し、順序から推測しません。実際のtrace結合数と未結合数を確認してください。

初版はtarget decoder各層の次を採ります。

| site | 実際の位置 |
|---|---|
| residual | layer scalar適用後のblock出力 |
| attention | post-attention normより前のattention出力 |
| mlp | post-feedforward normより前のMLP出力 |

記録はmean・RMS・最大絶対値・非有限値数・std・zero_fraction・outlier_fractionの概要です。stdとfractionは各位置の有限channelを対象とし、outlierは既定で絶対値がそのrow RMSの6倍を超えるものです。しきい値はcapture設定とsite metadataへ保存します。これは値の大きさの定義で、能力上の異常や重要度の判定ではありません。生のactivation配列、prompt/回答/推論本文、全attention行列は保存しません。draftモデルのactivationは採取しません。対応weightはsite metadataへ明示します。runtime_parameter_namesはvLLM内部名、weight_names/checkpoint_weight_namesはGemmaの明示的なfusion対応を実checkpoint indexで確認した名前です。未知の対応は推測せず、該当tensorがない場合は表示します。

MTPの`accepted`は`true / false / null`を区別します。rank0のsampler記録を同一replica・step・requestへ結合してdraftの採否を補います。これはsamplerでの受理であり、公開応答のEOS以後の打切りと同義ではありません。未確認を受理済みとして数えません。TP rankの数値を足し合わせて全モデルactivationと呼ぶこともしません。

### 必要なchannel・headだけ詳しく採る

概要で対象を絞った後、capture設定の `detail_channels` と `detail_heads` に対象を明示できます。例:

```json
{
  "detail_channels": {
    "layer.0.residual": [19],
    "layer.0.mlp": [7]
  },
  "detail_heads": {
    "layer.0.attention": [0]
  }
}
```

channelのindexはそのactivationの軸、headの指定はTP rank内のquery-head indexです。headの表示には `global_head = rank × num_local_heads + local_head` を使い、rankも残します。channelは幅1の値を `metrics.value`、headは通常の概要metricとして記録します。指定数もGPU buffer上限に含め、超過は起動前に拒否します。

site metadataの `parent_site_id`、`activation_axis`、`checkpoint_axes` が、元のsiteと保存重みの軸・範囲を示します。画面でその対応を確認してから解釈してください。tokenをbin化した `value` はbin内の平均なので、個々の値を見るときは `merge --token-bin-size 1` を使います。詳細採取も、全headの因果的重要度を同定する操作ではありません。

### 保存物

```text
<capture>/
  capture.json
  requests-<pid>.jsonl
  server-requests-<pid>.jsonl
  workers/<replica>-tp<rank>/
    sites.json
    observations.jsonl.gz
    acceptance.jsonl
    status.json
```

`observations.jsonl.gz`は、1 GPU実行分のsite・位置・metricをまとめた`activation_batch`を、gzip level 1の連結memberとして保存します。共通のsite軸・metric軸と、`request_id / token_position / phase / speculative_index / accepted`の行軸を持ちます。正規`merge`が論理recordへ展開します。ファイルの行数や圧縮batch数は、観測した位置×site数ではありません。旧`observations.jsonl`も読めますが、同じworker directoryに両形式がある場合は二重集計を避けるため拒否します。

JSON化と圧縮はworker内の専用CPU threadで行います。既定queueはworker当たり最大8 batch・推定payload 64MiBで、処理中のbatchも上限に含めます。この値はPython object・圧縮用一時領域を含むプロセスRSSの上限ではありません。queue満杯、disk割当上限、writer停止による欠測はdropとして残します。全captureの2GiB割当にはworker event、client/APIログ、metadataを含めます。

`status.json`で論理`records_written`と`batches_written`、保存bytes、drop、errors、closedを確認します。`writer_queue`はdepth/pending bytes、high-water mark、enqueued/completed、drop、serialize/compress/CPU時間、drain時間、close timeoutを記録します。API応答の完了だけでは観測保存の完了を意味しません。正式なmergeはwriter drain・確定した終了状態の後に行い、途中snapshotを使う場合は範囲と未完状態を明示します。書込み途中でgzip memberが切れたファイルは、正規readerが不完全として拒否します。

負荷測定ではAPI応答時間、応答後に残ったwriter処理、close時のdrain、保存完了までのwall timeを分けます。応答後に待ってからcloseした場合、closeのdrain時間だけでは応答後の処理量を表せません。capに達した観測を欠測として残し、生成の成績へ0点を足しません。

## 成績との結合とHTML

```sh
./kairyu-bench wsp observe merge /mnt/nvme/<base-capture> \
  --results /mnt/nvme/<base-wsp-run> --token-bin-size 64 \
  --output /mnt/nvme/<analysis>/base-observation.json

./kairyu-bench wsp observe merge /mnt/nvme/<candidate-capture> \
  --results /mnt/nvme/<candidate-wsp-run> --token-bin-size 64 \
  --output /mnt/nvme/<analysis>/candidate-observation.json

./kairyu-bench wsp observe compare /mnt/nvme/<analysis>/base-observation.json \
  /mnt/nvme/<analysis>/candidate-observation.json \
  --output /mnt/nvme/<analysis>/comparison.json

./kairyu-bench wsp observe view /mnt/nvme/<analysis>/comparison.json \
  --weights /mnt/nvme/<analysis>/weight-diff.json \
  --output /mnt/nvme/<analysis>/comparison.html
```

HTMLは外部library・CDN・サーバーを使いません。取得したJSONを内包するので、共有前にはpath・request ID等のmetadataも共有範囲に合うか確認してください。1回のcaptureだけでも表示できます。

画面では次の順に調べます。

1. 基準/比較の実験、能力領域、出典、問題IDを選ぶ。未採取の問題も成績一覧に残す。
2. 1問題と、基準/比較それぞれのrequestを選び、metric・phase・構造の種類・site・token位置を絞る。call/trial/replicaの識別も表示する。既定のTP rankは0。
3. heatmapは選択したbackend requestだけを表示し、別call/trialの同位置を平均しない。セルを選び、sample数・位置範囲・request・受理状態・siteに対応する重みを見る。
4. 下部でmode、checkpoint、runtime、採取上限・欠測を照合する。

問題表は先頭300件、heatmapは96行×128列まで表示します。filterで範囲を切り替えます。全層を見る場合は構造の種類をresidual/attention/mlpへ絞ると、異なるsiteを混ぜず層を並べられます。大きいtraceを詳しく調べる場合は`merge --task <item-id> --token-bin-size 1`で対象問題の詳細reportを別に作れます。元の圧縮JSONLまたは旧JSONLは保持します。

`metric_sample_counts[metric]`（未記録なら`sample_count`）を重みにしてRMSを二乗平均平方根、meanを加重平均、max_absを最大、nonfinite_countを合計として表示します。差分は対応binの指標差の加重平均です。sample数はtoken数や問題数と同じとは限りません。

### 比較できないものを差分にしない

通常のlive生成では元と候補のtoken列が分岐します。同じ問題・同じtoken位置だけでは、同じ入力に対するactivation差とは言えません。画面はそれぞれを表示できますが、静的重みの詳細も、添付inspect/diffの内容hashが選択実験と一致した場合だけ結びます。未hash・別checkpointの値を活性の根拠として表示しません。

対応あり差分には明示的な`pair_key`と一致条件が必要です。

一致条件には問題/出典、mode、site/layer/role、phase、token種別・位置・範囲、sample数、受理確認、TP rank/軸対応、runtimeを含みます。曖昧な重複keyや条件不足は、差分を作らず理由を表示します。live・replay・teacher-forcedを混ぜません。teacher-forcedの好ましい応答に対する値を、実際の自由生成時の挙動として報告しないでください。

`observe compare`は表示用の結合です。WSPの公式paired score、統計的検定、候補の採否は従来の`wsp compare`と実験計画で扱います。

## 観測を編集案へつなげる記録

新しい編集案ごとに、次の項目をMarkdownへ残します。

| 項目 | 記録内容 |
|---|---|
| 目的 | どの能力・問題群の何を変えたいか。別領域の許容損失も事前に記す |
| 観測 | weight group/軸、問題ID、site、sample範囲、元/候補/反復のartifact参照 |
| 仮説 | 複数箇所を共同変更する理由、予測される得失、反証される条件 |
| 実差分 | 保存dtype、変更数、総/局所ΔL2、分母、no-op・方向保持、hash |
| 比較 | 元、部品、組合せ、既存候補。同条件と欠測を明示 |
| 結果 | 問題別・領域別の得失、信頼性、時間、予測と違った点 |
| 判断 | 採用/保留/棄却と理由。次に減らす変更、除去比較、停止条件 |
| 証拠の利用区分 | dev/使用済みtest/未見最終評価。閲覧済みtestを未見へ戻さない |

normの大きさ、activationの頻度、attention量、weightとの積は編集仮説の材料です。因果的な「能力を担うneuron」の同定ではありません。sampleで見えなかった領域を無変化と推定せず、目的別の共同編集と対応あり実行評価で反証します。

時間は採取・重み読込み・hash・候補作成・serving切替・生成・採点・待機に分けます。小さい評価を多数回行う前に、切替コストを含めた次段の条件を登録します。単発の既知問題回復、NLL低下、合成margin増だけで能力改善を宣言しません。

## 実モデルでの採取を有効にする前の受け入れ確認

能力評価の再開前に、ツール検証として承認された範囲で、同じruntime・sampling・batch条件の採取OFF/ONの短いA/Aを行います。出力差、wall time、VRAM、採取数/drop、trace結合率、TP/MTPの受理、CUDA graph下のhook動作を確認します。数値上の負荷目標は、実測前に達成済みの性能として書きません。

停止・quota到達・未対応runtime・flush失敗を識別できることも確認します。既に使用したholdoutの結果を見て次の候補を選ぶ場合、その集合は開発用証拠です。最終的な効果の確認には、承認前に別の未見評価計画を決めます。
