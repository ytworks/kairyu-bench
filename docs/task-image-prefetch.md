# Task image prefetch

`scripts/prefetch-task-images.py` は標準ライブラリだけで実行できる、明示操作型の
キャッシュ準備ツールです。通常のベンチマークから自動実行されません。

明示したUnix socketとdaemon IDを確認して、taskに必要なimageを取得します。
tag名だけの一致は採用せず、既存imageとpull後のimageの両方について、期待する完全な
`repository@sha256:...` が `RepoDigests` に存在することを要求します。
Docker configのImage IDも別に記録し、registry digestと混同しません。
tagが別digestへ移動していれば不合格です。既存の不一致imageは上書き・削除しません。
pull後の不一致imageも自動削除しません。

## 入力

task/ref JSON:

```json
{"records":[{"task":"task-a","ref":"registry.example/team/image:fixed"}]}
```

期待digest JSON:

```json
{"records":[{"task":"task-a","ref":"registry.example/team/image:fixed",
 "registry_digest":"sha256:<64 lowercase hex>",
 "digest_reference":"registry.example/team/image@sha256:<64 lowercase hex>"}]}
```

2入力のtask/ref集合は完全一致が必要です。期待digest入力は
`records[].task/ref/registry_digest/digest_reference` schemaを受けます。
証跡の他fieldは実行入力にせず、記載されたsource path等は辿りません。
両入力file全体と実行scriptをhash/stat固定し、実行中はstat、最後はhashまで再確認します。
入力に資格情報を含めないでください。

## previewと実行

```bash
python3 scripts/prefetch-task-images.py \
  --docker-socket /nvme/docker/run/docker.sock \
  --expected-daemon-id EXPECTED_ID \
  --tasks tasks.json --expected-digests expected.json \
  --output /nvme/prefetch/UNIQUE_RUN \
  --min-free-path /nvme --min-free-bytes 536870912000 \
  --max-parallel 1 --issue-interval-seconds 2 \
  --max-retries 3 --backoff-seconds 5 --max-backoff-seconds 120
```

既定はpreviewです。入力・socketの種類・空き容量を確認し、新規outputへreceiptだけを
保存します。Docker命令は0ですが、preview用output directoryとreceiptのfile書込みは行います。
daemonや既存cacheを確認した扱いにはしません。
実取得は同じ引数に `--apply` を加え、別のfresh outputを指定します。
image操作はinspectとpullだけです。applyで実際に発行したpullはreceiptの各imageのattemptsへ記録します。
UID、GPU、containerの実行、daemon設定は変更しません。image cacheが不変であるという意味ではありません。

- 最大同時取得は既定1、許容1〜32。取得開始の発行間隔は全worker・再試行で共有します。
- retryはpullのrate limit（ECR Rate exceeded/429等）とtimeoutだけです。
  認証失敗、digest不一致、一般的なネットワークエラーを無条件に再試行しません。
  retriesは初回を除く追加回数で、上限まで指数backoffします。
- pull timeoutは既定900秒、info/inspectは30秒。途中timeout後もimageが完成していれば
  digestを照合し、同じimageを不要に再取得しません。
- 空き容量は各pull直前と直後に確認します。未知のdownloadサイズを予約する仕組みでは
  ないため、並列downloadがfloorを超えて消費しないという保証ではありません。
  `min-free-path` が実daemonの保存filesystemに対応することは呼出し側の責任です。
- daemon ID/socket変更、入力変更、digest不一致、容量不足、恒久取得失敗で追加dispatchを
  止めます。SIGINT/SIGTERMでもこの呼出しが生成したDocker CLI childだけを終了させます。
  daemon側の共有download処理まで停止したとは主張せず、daemon/containerをkillしません。
- 全imageを照合した場合だけcompleted。未着手/中断はincomplete/cancelledのままです。
  同じrefを使う複数taskは1回だけ取得し、receiptのtask/ref対応を保持します。
- 生のDocker stdout/stderrは保存せず、一時fileを閉じて削除します。receiptには
  command結果分類・試行数・時間・ID/digest・入力証拠を残します。
- fresh outputの再開/上書きはありません。再実行には新outputを使います。

## CPU契約テスト

```bash
python3 -B -m unittest discover -s tests -p 'test_prefetch_task_images.py' -v
```

Docker全呼出しはfakeです。preview呼出し0、別daemon拒否、rate limitからの再試行と
正/誤digest、既存imageの照合、timeout上限・認証非retry、入力変更を確認します。
本物のregistryや稼働中の取得jobには接続しません。
