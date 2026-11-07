# Gemma 4 direct-weight restart — 2026-11-08

指定元モデルからの再出発。過去の仮説、候補順位、採用判断、学習実装・設定・データは引き継がない。作業・計算・整理はすべてSSH先で実施した。保存重みの直接編集だけを扱い、追加学習、勾配、optimizer、正解に適合する係数学習は使わない。**候補001の対象診断は[0,0,1]/10で既定guard未達、不採用。正規全問評価は未実行で退行は未確認。元モデル復帰と候補削除は完了し、採用・改善・tier全問クリアを主張しない。**

以下の記録先はSSH先の作業ルート `/mnt/nvme/kairyu/wsp/gemma4-coding-20261104` を基準とする。問題の答えや巨大なraw出力はこの文書へ複製しない。

## 確定した元モデル測定

元モデルは `/mnt/nvme/kairyu/model-volumes/gemma-4-31b-8gpu/models/gemma-4-31b-it`。保存ファイルのtree hashは `cab440f871a32b5ea1c12364db7dd35ac8768a83b413ca24bbc18d6478eccbd9`。

正規10分版dev全問run `wsp-direct-original-dev10-full-20261108-r1` は778項目完了。能力746問の内訳は正解577、不正解168、部分点1、別途診断32。未採点0、通信失敗0、能力問題の未解決169。診断NLLに独自の合否は作らない。公式compositeは0.8099938281843849、generation 585.8秒。4 replicas × TP2、MTP4、chat/logprob/agent concurrencyは256/32/4、observer OFF。既存測定との条件同一性を確定できなかったため、このfresh baselineを採用した。

`restart-direct-20261108/active-baseline.json`、`active-baseline-all-items-status.jsonl` に条件、問題別分類、hashを保存。manifest配列の固定順で選んだ最初の未解決10問は次のとおり（全て `arc-challenge:`）。順序は変更しない。

| 問題ID | 元canonical | 候補r1 | 候補r2 | 候補r3 | 状態 |
|---|---:|---:|---:|---:|---|
| Mercury_7163818 | 0 | 0 | 0 | 0 | 未解決 |
| Mercury_SC_401278 | 0 | 0 | 0 | 0 | 未解決 |
| LEAP_2007_8_10417 | 0 | 0 | 0 | 0 | 未解決 |
| Mercury_SC_400401 | 0 | 0 | 0 | 0 | 未解決 |
| MDSA_2010_4_20 | 0 | 0 | 0 | 0 | 未解決 |
| Mercury_7218400 | 0 | 0 | 0 | 0 | 未解決 |
| Mercury_7206938 | 0 | 0 | 0 | 0 | 未解決 |
| Mercury_7074848 | 0 | 0 | 0 | 1 | 未解決 |
| Mercury_402348 | 0 | 0 | 0 | 0 | 未解決 |
| Mercury_SC_413624 | 0 | 0 | 0 | 0 | 未解決 |

## 順伝播観測と限界

元重み固定、単一replicaのobserver ON診断は1/10。observer OFF対照3回は全て0/10で、3回目の前に元モデルprocessを再起動した。ONの正解増は保存重みの改善に数えない。ON/OFF初回の最大choice loglik差は6.129667、元の全問baselineとOFF間にも差がある。compile、BF16丸め、async batching、再起動をこの測定だけで分離できない。

ONでは40 choice requestsの指定prompt-tail位置を両rank・60層のresidual/attention/MLPで取得し、対象nonfiniteは0。capture上限は両rankとmetadata合計256MiB、observer全体の予約は2GiBであり、per-rank上限と混同しない。hookはattention/MLPのpost-norm前を見ているため、RMSからresidualへの実寄与や担当能力、編集方向を断定できない。token tのlogprobはhidden位置t−1に対応するため、tail32のcoverageだけで全継続tokenの予測位置を観測したとも断定しない。得点は公式の継続loglik合計／選択肢文字数で採点し、採点器・promptは変更していない。

`arc-observation-tail-plan/activation-tail-summary.json`、`independent-observer-audit.json` と各runの問題別出力を保持。旧head準備snapshotは未実行の履歴として残し、tailと混同しない。

## 登録済みtrial 001

`restart-direct-20261108/trial-001-final-branches/registration.json` と `edit-spec.json` に、最後のtext block59の `post_attention_layernorm.weight` と `post_feedforward_layernorm.weight` を同時に公称127/128倍する一つの有限操作を事前登録した。各vectorは5376要素。係数、方向、座標のscore fittingやsweepはしない。遅い段階のbranch更新を弱める仮説は探索的であり、現観測がその方向を支持したとの主張はない。

実Claude Opus 5.5へのレビューで、requested model・assistant応答modelがともに `claude-opus-5-5` と一致し、fallbackなしをreceiptで確認した。判定は条件付き承認。元OFF／候補OFF各3回、再起動を含む元対照、固定screen、実runtimeのnorm/loader semantics、保存dtypeでの実差分検証を条件に反映した。screenは追加の公式合格条件ではなく、全問評価へ進める根拠の確認である。候補の有望な対象結果だけで採用しない。

書込予定BF16 bytesを独立の有理数演算・nearest-even丸めと照合し、両vector合計10752要素の変更が一致した。実保存比率は一律127/128ではなく0.9895833333〜0.9947643979。正のBF16保存コードの整数差（`stored_code_decrement`）はattentionで1減4201／2減1175、feedforwardで1減887／2減4489。各before/after hashと丸めtieは `rounded-proposal-verification.json` に記録。独立proofは書込予定bytesの検証であり、保存後は別の `chain-verification.json`／`chain-execution.json` で全12ファイルの元→候補stream再構成一致とexit0も確認した。

## 必要な非学習ツールと再現手順

操作は現行sourceと `--help` を確認して使う。`wsp observe init ... --prefill-policy tail --decode-offsets 0` は各requestのprompt末尾を採取し、head既定を維持する。policyはconfig／worker／record／reportへ保存し、head/tailのprefill差分比較を拒否する。immutable worker依存closure9ファイル、capture config、source hashを保持した。詳細は [観測配置手順](../wsp-observation-deployment.md)。

保存編集とstream検証の基本形（登録済みの新規pathへ置き換え、SSH先で実行する）：

```sh
python -m kairyu_bench.wsp.weights.editor create ORIGINAL CANDIDATE \
  --spec EDIT_SPEC --manifest EDIT_MANIFEST --reserve-bytes RESERVED_BYTES
# previewと容量台帳の照合後だけ、同引数に --apply を付ける。
python -m kairyu_bench.wsp.weights.chain ORIGINAL CANDIDATE \
  --manifest EDIT_MANIFEST
```

editorは保存dtypeへ丸めたbefore/after bytes、変更範囲、全保持ファイルhashを記録し、独立した候補ファイルだけを書き換える。chainは各段階のbefore一致と保存済みafter bytesを元モデルからstream replayし、途中の親checkpointがなくても最終bytesを照合する。第3の完全コピーは作らない。親削除前にこの検証を完了し、累積manifestとeditor／checkpoint／chain sourceの版・hashを残す。詳細は [保存重み編集手順](../wsp-weight-editing.md)。

今回の一回限りの補助手順を [records/2026-11-08-direct-restart-t001](records/2026-11-08-direct-restart-t001) へ元scriptと同bytesで保存した。`script-snapshot.json` が対応元・hashを示す。

- `verify-rounded-proposal.py`：固定127/128、正のnormal BF16、shape5376という今回の範囲で丸めを独立照合する。重みは書き換えずproof JSONを出す。
- `create-and-verify.py`：今回の登録・review・容量予約・元OFF対照の存在を確認し、既存editorを呼び、容量を監視して既存chainで検証する。普遍的な実験管理toolではない。

これらは元trialの再現証拠であり、archiveからそのまま再実行しない。再実行には新trial・fresh候補/manifest/出力pathと容量台帳を先に登録し、hardcoded作業rootと前提を明示的に確認する。入力を置換した版は別hashで保存する。`python <登録trial>/verify-rounded-proposal.py`、そのproofと全guardの確認後に `python <登録trial>/create-and-verify.py` の順。既存結果を上書きせず、失敗・中断もownerが記録して同trial内で整理する。

単一replicaの診断切替は既存 `docs/experiments/scripts/redeploy-replica.py --help` を使う。`single-replica-switch-plan.json` にpreview/apply/元モデル復帰argvを記録した。現runtimeのcommand/env/image/GPU/CPU/IPC/network/healthcheckを同helperで読み取り照合済み。preview/applyは別fresh出力先、復帰は新baselineを使う。mixed gatewayで正規全問評価を行わない。旧 `evaluate_candidate.py` は旧state、自動materialization、dense関連経路を含むため今回使わない。

## 各試行の容量管理

元モデルを保持し、追加完全checkpointは採用中1＋候補1が上限。初採用前の今回候補は1個だけ。追加重み・差分・補助行列を合計150GiB以下、再生成cache100GiB以下、NVMe空き200GiB以上を維持する。`storage-state.json` と実fileを開始／作成前／終了／再開時に照合し、不要物削除と台帳更新が終わるまで次候補を作らない。

observerの要約と必要raw証拠を確保した後、不要な派生merge/cacheを同工程で削除した。`arc-observation-tail-plan/storage-closeout-outcome.json` の削除実割当は783953920 bytes（747.63671875MiB）。元データ・唯一の評価証拠は保持した。

元filesystemのreflink非対応とroot所有元fileへのhardlink拒否を確認したため、今回は理由を記録したprivate copy fallback。copy上限62578686256 bytes（58.280943GiB）にmanifest・filesystem余裕各1MiBを足し、58.282896GiB（約58.283GiB）を予約した。編集はprivate候補shardへのseek書込みで、全shard一時コピーは作らない。writer終了、評価終了、mount／共有参照と元モデル非該当を確認してから棄却候補を削除する。

候補OFF対象測定3回は[0,0,1]/10、各回40 choice requests。3回目だけ `Mercury_7074848` が正解となり、他9問は全回不正解だった。元OFFは[0,0,0]/10。事前登録した全3回正解のguardを満たさず、不採用とする。対象10問は全て未解決扱いで、全体の未解決169も変えない。候補の正規全問評価は未実行、退行は未確認。保存編集は2 tensor／10752要素。元モデル4 replicasはhealthyへ復帰済み。生成・削除した候補のregular実割当は62578728960 bytes、directory8192 bytesを含め62578737152 bytes（58.280991GiB）。追加完全checkpointは0。`storage-closeout.json` にwriter/FD/mount参照ゼロと削除を記録した。30分版・60分版、最終DeepSWE全113問・32並列・1周、wiki追記、Hugging Face登録へはまだ進んでいない。

評価量は40 continuation requests／回×3＝120 completion requests、tokenizeは81／回×3＝243、合計363 logprob client API calls。候補3回のrun timestamp経過合計は7.095010秒、runner process経過合計は11.362986秒であり、生成／起動時間と混同しない。終了照合時はNVMe空き2665.30983GiB、追加重み保守上限32.14117GiB、cache予約91.97288GiB。確定要約の唯一の保持先はSSH先 `restart-direct-20261108/trial-001-final-branches/final-scientific-summary.json`（SHA256 `b1fc903bb4fd5ca3d282374208bc07ce8255f4156c88f036337ec5ed9f336bac`）。

事前登録した元モデルOFFの2回token診断は完了し、各40 completion＋81 tokenize、capture complete、公式0/10だった。40対のtoken列・hash・継続境界は一致し、欠損／非finiteは0、全80 sumと全20 itemのargmax／scoreは公式結果と完全一致した。220位置中64位置（11 choice／8問）で返却token logprobが変動し、最大絶対差5.0473985672、正解option marginの実行間差は最大0.2389380485。入力や集計の不一致は本2観測で検出されなかったが、runtime原因や決定性は特定できない。新しい改善候補・解決数の増加を示す結果ではない。詳細・唯一の証拠保持先／hash・失敗経路の制限は [response記録手順](2026-11-08-token-logprob-diagnostic.md) を参照。診断を重み改善や全問評価の代わりにしない。
