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

## T002：最終block残差枝の固定ゼロ化（全問評価 IN PROGRESS）

この節は進行中のドラフト。`wsp-direct-t002-candidate-dev10-full-20261108-r1` は全778問・dev・tier10の正規評価を実行中で、`full-evaluation-execution.json` は `running`、PID 2599158。全問結果、問題別退行、採否、復元・cleanupは未確定。採用モデルはまだなく、確定済み元baselineの577正解／168不正解／1部分点（能力746問）＋NLL診断32問、未解決169を更新しない。対象診断を全問クリアや一般的改善とは扱わない。

指定元モデルから、layer59の `post_attention_layernorm.weight` と `post_feedforward_layernorm.weight` を保存BF16の正のゼロへ直接変更した。今回の推論構造観測に基づく、最後の2残差枝を共同除去する固定1介入である。係数は探索・学習していない。同じtensor対で棄却したT001の127/128も試行履歴へ含め、0をこの対・最終blockについての最終値とし、結果に合わせた中間係数・片枝・層scanへ進まない。

保存差分は2 tensor、10752要素、実際に変化した21476 bytes（対象span21504 bytes）、ΔL2=268.1548464079132。両5376要素は全て `0x0000`、元は正の有限BF16だった。独立literal-zero proof、生成時の全file検証、元モデルからのstreaming chainで全12 filesを照合し、変更範囲外の一致を確認した。第3の完全コピーは作らず、freeze時の再読込は対象21504 bytesだけ。候補treeは `80c94f003d180959e6d8f0ff3a622520de0a7ff5cfd466e64f26c056467b59d0`。

現imageの対象は通常のRMSNormで、`normalized(x) * weight`、+1 offsetなし。2残差加算後のPLEはconfigで無効であり、最後にscalarを乗算する。保存scalarは149/4096=0.036376953125、eps=1e-6、eps/c²=0.0007556964100716184。既存tail観測のlayer58 residual RMSに対する実数式上の振幅差は最大約0.0957%だが、BF16丸め・未観測位置・score不変性を保証しない。MTP4、共有KV、prefix/chunked-prefill条件を維持したため、測るのはこの保存編集の**固定target+MTP4 runtimeにおける総効果**である。純target-only、logit-lens、能力担当、compiled kernel内部の同一性を主張しない。

実際のClaude Opus 5.5へ事前・補足・実生成候補のレビューを依頼した。実生成候補の初回応答は形式上unusableとして保存し、再依頼receiptで応答モデル `claude-opus-5-5` と本人一致・利用可能性を確認した。最終レビューは登録済み正規全問確認への条件付き進行のみを認め、採用を認めていない。禁止手法は実行していない。

対象10問のObserver OFF診断は[7,7,7]/10、3回目の前に同候補processを再起動した。下表の元canonical scoreは全て0。7問は全3回の公式score1かつ各回marginが元OFF5回の最大marginを超え、事前登録した全問確認の起動条件を満たした。この条件は公式合格定義や統計的noise上限ではない。token別対照は記録のある元2回だけを使い、5回分のtoken証拠があるとは扱わない。

| 対象ID | r1 | r2 | r3 | 診断条件 |
| --- | ---: | ---: | ---: | --- |
| arc-challenge:Mercury_7163818 | 1 | 1 | 1 | 満足 |
| arc-challenge:Mercury_SC_401278 | 1 | 1 | 1 | 満足 |
| arc-challenge:LEAP_2007_8_10417 | 0 | 0 | 0 | 未達 |
| arc-challenge:Mercury_SC_400401 | 0 | 0 | 0 | 未達 |
| arc-challenge:MDSA_2010_4_20 | 1 | 1 | 1 | 満足 |
| arc-challenge:Mercury_7218400 | 1 | 1 | 1 | 満足 |
| arc-challenge:Mercury_7206938 | 1 | 1 | 1 | 満足 |
| arc-challenge:Mercury_7074848 | 1 | 1 | 1 | 満足 |
| arc-challenge:Mercury_402348 | 1 | 1 | 1 | 満足 |
| arc-challenge:Mercury_SC_413624 | 0 | 0 | 0 | 未達 |

未採点・通信失敗・capture validation errorは診断3回で0。記録は各40 continuation、2MiB上限（計6MiB）、tokenizeは各81。計120 completion＋243 tokenize＝363 client callsで、transport retry数は推定しない。診断のrun timestamp経過合計は6.972428秒、outer process経過合計は11.531419秒。正規全問評価は同じf91 runner、256/32/4、4 replicas、全領域・全778問を維持し、token wrapperやactivation hookを追加していない。

T002候補regular本体の実割当・累積生成は62578728960 bytes（58.280983GiB）、directory8192 bytesを含む候補全体は62578737152 bytes（58.280991GiB）、累積削除は0。保持は指定元＋評価中候補1、採用中0である。全追加weightの厳密な総実量は未確定で、whole-root保守上界90.424038GiBに進行中の非weight予約5GiBも重ねて計上した上界は95.424038GiB。cacheの予約込み上界96.974764GiBも実使用量ではない。全問開始release時のNVMe空き2607.027218GiBと合わせ、150GiB／100GiB／空き200GiBの各制約を守る予約を維持し、新checkpoint生成は禁止している。区分訂正は `storage-accounting-clarification.json` に記録し、総量・評価releaseは変更していない。最終使用量・cleanupはPENDING。`storage-full-monitor.jsonl` は2026-11-07 22:08:15 UTC以降 `storage-probe-delta-v1` と従来の完全行が混在する。147行目の基準行番号/hash・set/delete/null/継承規則に従って復元し、差分行を独立した全状態と扱わない。

preflight初回は、checkerがhelper receiptのディレクトリ末尾 `-` を落としていたため失敗した。失敗JSON・stdout/stderr・sourceを保持し、v2で登録argvの `--output` から実パスを読むよう修正した。v2は全4候補のRO mount・runtime・readiness・対象保存bytesと、元gateway image／4接続先の一致を確認した。未実行のfull replica launcherにも同じ前service-path誤りがあり、1箇所を修正して旧新hashとexact diffを保存した。どちらもモデルの失敗として数えたり、失敗記録を上書きしたりしていない。

途中観測の訂正も保持した。2026-11-07 19:40:58 UTCに固定した `responses.jsonl` の先頭6,162,113 bytes・77行だけを対象とする `partial-response-observation-001.json` は、実際の `sample.content` を未存在の `sample.text` と取り違えたため、文字数・反復指標が無効だった。件数・finish reason・token数は影響を受けていない。`partial-response-observation-002.json` は同じprefixを `content` で読み直した訂正版で、後続追加行を含めない。001のsource・結果を変更せず保持し、002に誤りと旧hashを記録した。

別の固定prefix（2026-11-07 20:52:49 UTC取込、先頭16,231,734 bytes・436行）では、130件の失敗を `canonical-partial-failures-001.json` が例外allowlistの `ApiError` 欠落により `other_unscored_error` と記録した。同じprefixの `canonical-partial-failures-001-classification.json` で、外側 `ApiError`、HTTP 502、内側 `ReadTimeout` を各130件確認し、通信失敗へ分類を訂正した。元記録は不正解へ分類しておらず、上書きせず保持した。これらは固定部分の観測・分類であり、正式採点、全体結論、失敗原因の推定には使わない。生出力の複製、新規推論、再試行、評価条件の変更は行っていない。

実行中の限定監査では、現gatewayの `httpx.post(timeout=self._timeout_s)` から例外をHTTP502へ変換する静的な可能経路を確認した。当該失敗との一意な対応、実timeout override、実retry回数・経過は未確定であり、constructor既定60秒とbenchmark clientの7200秒を混同しない。source読取のUID1003 `PermissionError` も保持し、続くcontainer UID0での読取は指定sourceだけに限定した（host root・sudoは未使用）。別のagent-tail監査時点では、残るTerminal-Bench 2 taskのcontainerとrunner子processが生存し、result/trajectoryは未生成だった。元baselineと `max_turns=250`、`agent_timeout_multiplier=48`、LLM timeout1200秒、task定義timeout3600秒が一致するが、実token上限・手数・待機phase・実retry・timer適用は未確認で、48×3600秒を終了deadlineとは扱わない。限定 `docker top` のexit1も記録した。両監査とも新規推論・runtime変更・正式比較は0である。

T002の唯一の証拠rootはSSH先 `restart-direct-20261108/trial-002-final-branch-ablation/`。下記は参照であり、結果・レビュー・raw captureの追加コピーではない。

| 唯一の保持先（T002内） | SHA256 |
| --- | --- |
| `edit-spec.json` | `86a0facdfff4c556ec1c01ec832cbda0a4418c45fb2a15107d225dc3022e1abd` |
| `literal-zero-proof.json` | `f7d81dec3ccb38e83c920d82e66ee461ef4f0395119f4809b367f29045b4f305` |
| `chain-verification.json` | `e1b7ddbc86337063b5c5a0e6029113ac26a3b9ae26d9f21d1079aa643fb56cda` |
| `candidate-freeze.json` | `154808704472c7321fcabeedf496b5da4d22b874c8f49b0a9bd52b52e8b65bda` |
| `candidate-token-comparison.json` | `c89fc92210fa9b37baf3f196fdd59a60b5ca2a6ed94b74e63bb15b7f87d00186` |
| `runtime-zero-ablation-audit.json` | `2555eeccec4157bf5bbda4954fa2a3ffd3b1c01d2e971b215806189f57c886bb` |
| `mtp-kv-prefix-chunked-addon.json` | `23b8f279a803e9b48228c768b8e3b36f11127a849d9fbe6c5bef8a7c83f304c1` |
| `opus55-realized-candidate-review-retry/receipt.json` | `a52b40f2f7b4b5ccc38bc07c9c783e03bcd3bb66e3235deae1f28b0551c79e6d` |
| `full-serving-preflight-v2.json` | `1b25b543378c44194990e1fe8922314bd482cf9a67a7b686f5d00b6b51fe40a4` |
| `canonical-full-evaluation-plan.json` | `93cbe42e03b0bd6a03b067b95ccc63cdb4d76c1a764a87b526e08363482a3679` |
| `storage-full-evaluation-release.json` | `244c970a2c0c33af904dde48c1d6f21b02b3b949bf6e912a2f7f0ecda0af32e7` |
| `partial-response-observation-001.json`（誤参照を含む原記録） | `b3561744995ab288381dab15181b91e339e13bf9c650a1f306463b23b3c36565` |
| `partial-response-observation-002.json`（同prefixの訂正） | `011d9b8dc2fd3cc7d59943badf0615a516f36bc6bdd97f54a2a92c45ca28ae8d` |
| `canonical-partial-failures-001.json`（初期分類） | `a092b3f2e34389fa4ce89f4b1e061cfeb141a78170f744717681dc773be2d8d0` |
| `canonical-partial-failures-001-classification.json`（同prefixの分類訂正） | `115281b4002e654a0b0c13ea386f6fb76f990e6d327018672d8eea428b88e599` |
| `storage-accounting-clarification.json` | `8fba18906394635c3e37dfdf267d85d686e8a7dd9053b594b8bfb05bba03daa7` |
| `gateway-timeout-readonly-audit.json` | `eb5070975061ffac69ca30ddabf752f22ac5af1ac617fe107dd9fbcacd015945` |
| `agent-tail-readonly-audit.json` | `ec0fff8d7d4104969f1a91ca0f1e1dd248b20d608e713f5eeba7430e0d4817e1` |

非学習の再現・運用script12本と訂正版の部分観測script1本を、[PR用records](records/2026-11-08-direct-restart-t002/README.md) へ元SSH trialと同一inodeのhardlinkで収録した。[script-snapshot.json](records/2026-11-08-direct-restart-t002/script-snapshot.json) に元/収録path、hash、device/inodeと検査結果を記録した。下表はこの固定13本であり、実行中のsource bytesは変更していない。`PYTHONDONTWRITEBYTECODE=1` とNVMe TMPDIRを使い、必要なものは既存 `infra/weight-tools-ci/bin/python` と `PYTHONPATH=<worktree>/src` で実行する。過去の完了trialを再実行しない。再現には新trial・fresh候補/manifest/outputと容量予約を先に登録し、hardcoded pathを変更した版のhashを新しく記録する。deployment/evaluationの具体的argvは登録JSONに従う。

| SSH上の固定script | 用途・使用順 | SHA256 |
| --- | --- | --- |
| `verify-literal-zero.py` | 生成前：固定+0 bytesとeditor予定bytesを独立照合 | `1e767e82256d5c4b0cdd97915c2a6d227dcb666480572cc7ffdfffdbe2d5aefe` |
| `create-and-verify.py` | 容量・科学guard後：候補生成→全file chain検証 | `f295cddd946412f0b74d272137548313ffcb944c419cbf725ddf32fca7d29887` |
| `verify-and-freeze.py` | chain exit0後：対象span確認→元/候補freeze | `8edc01a41f29cbb137dd8a3d895f54405633878cc1c6cf8d03a6a975dc408526` |
| `run-replica-operation.py` | 初回単一replica preview/applyと復元用argv実行 | `52197b84e9f6f8cd6cb7a27bb7236740cfd8597ef5fbf7363454058f3fe9075f` |
| `audit-live-load.py` | 初回、または candidate-live-load-after-restart.json 指定でr3前確認 | `cc26cb6a8ccf54fd34c0fec40bcd189e74b4ad546c05503837b6c4d3c78cc33b` |
| `run-diagnostic.py` | 引数1/2/3：登録済み対象診断を各1回 | `5798a2abb6cb04213101dfa07489ce017913d4f02ed30b5a04bf956f6823d942` |
| `restart-before-r3.py` | r1/r2完了後：同候補・同containerを再起動 | `13e94163d90e808634e52d93f4b80d5559380ab22fd7b0618914baec5e3dba13` |
| `compare-fixed-three-token-runs.py` | 3回完了後：公式sum/scoreとtoken対照を検証 | `af4378412dcd3315db916ff559c7f54d5a1fd95dede15b9c709bb4655d60ea89` |
| `check-full-serving-preflight-v2.py` | 全4候補切替後：identity/readiness/gatewayを確認 | `e3d641715376daaf046b28d952f656927ec12ab690765c56e843ebfcd258df17` |
| `run-canonical-full.py` | guard後：正規778問を1周（現在実行中） | `a5f44210105dabf7aa82047721d80c268fe6eb520359f5c58254c458fddf52c6` |
| `compare-canonical-full.py` | 全問終了後：全ID・問題別得失・退行を比較（待機） | `152ff0f7d67abc16d757a0a782c015fc3c326c195e39e65b904cd7f9dd7e3a52` |
| `run-full-replica-operation.py` | serviceとpreview/apply/restore_preview/restore_applyを指定（未実行の復元用launcher） | `bc51be2e3dae34cebe465bc28404df7ab8b50442e87b8120f251d5c2e5d821fb` |
| `partial-response-observation-002.py` | 固定prefixのcontent文字数等を訂正観測（001の原記録は保持） | `555111cf806a48b97961e5714303ca4f5c05f74d034fdef5404627cd46b24ef8` |

`run-replica-operation.py` のmodeは `preview/apply/restore_preview/restore_apply`、`run-diagnostic.py` は `1/2/3`。`audit-live-load.py` は初回を引数なし、再起動後を `candidate-live-load-after-restart.json` 指定で別保存する。full replica launcherは `gemma-N` とmodeを指定する。全4候補への今回の切替はrootが登録argvをinlineで順番に実行し、このlauncherは未実行である。今回は全問終了前に固定sourceのPR収録を行った。元pathとrecords側は同一inodeなので、以後いずれもin-placeで編集しない。修正が必要なら別inodeの新version fileを新規作成し、旧新hash・理由・適用対象をrevision記録へ残す。実行中sourceと固定snapshotは維持する。結果・レビュー・manifest・raw評価は複製せず、上記SSH保持先とhashを参照する。全問終了、正式比較、採否、必要な復元・cleanupはPENDING、候補未採用、採用判断済みbaselineの未解決数は169のままである。今回の収録段階ではstage・commit・pushは行っていない。


## T003：最終attention枝だけを元の保存値へ戻す（棄却・復元／整理中）

この節は前節の固定snapshot以後の記録である。T002は元の正規10分版を全778項目で実行し、2026-11-08 03:31:06 UTCに `partial` で終了した。能力746問は正解192、不正解322、通信失敗230、その他未採点2。比較は改善30、退行220、比較不能232であり、全問クリアではない。ARC100問は元モデル63正解からT002の66正解へ変わり、改善12・退行9だった。T002は累積作業baseとして保持し、この全問確認の未完と退行を残した。証拠はT002の `canonical-full-comparison.json` と `arc-official-saved-reuse.json` にあり、採点結果を再生成していない。

T003はT002から開始し、`model.language_model.layers.59.post_attention_layernorm.weight` だけを指定元モデルの保存BF16 bytesへ戻した。最終post-feedforward normはT002の正のゼロのままである。既存editorの `set_values` に渡す値は指定元モデルの当該保存tensorだけから得ており、正解データに合わせた値・係数の探索や学習、勾配、optimizer更新は使っていない。保存dtypeへ戻したbytesと元tensorの完全一致を独立照合した。大きさや相関から能力の担当を断定する操作でもない。

対象は5376要素、span10752 bytes、実際に変わった保存bytesは10746、ΔL2は87.59597497324575。事前見積り・容量release後、既存editorの独立copy経路で別checkpointを生成した。生成・全file検証は210.113秒、指定元→T002→T003の2段階streaming chain検証は192.663秒で、双方exit0。保持12 filesのhash、段階ごとの保存dtype丸め、再構成器 `native-edit-chain-v1` の実装識別を検証し、追加の完全再構成コピーは作っていない。freezeは完了したchain hashとstat連続性を使用し、対象norm bytesも照合した。GPU内bytesの読戻しを実施したとは主張しない。候補treeは `60c2e41c39382d867dd03977171e6a76345c5a28f54df1ff6b90dcb3af702ced`。

生成後に実際のClaude Opus 5.5へレビューを依頼し、requested model、assistant model、model usageの `claude-opus-5-5` 一致と利用可能な応答を確認した。レビューとroot判断は、登録済み固定編集の同条件診断への進行を記録している。禁止手法を取り入れておらず、レビューを全問完了や採用の証明とは扱わない。

### 同条件ARC100診断

T002の既存canonical ARC出力は4個のLP endpointを使用していた。一方、今回の候補は `gemma-0` だけへ切り替えるため、既存結果を単一endpoint候補との同条件比較に流用しなかった。切替前に親T002を `gemma-0` 単独で100問1周だけ測り、その後同じendpointのT003を100問1周だけ測った。新しい経路条件に対応する測定であり、確定済み同条件の無理由な再測定ではない。3回反復を新たな採否条件にはしていない。

両runは固定runner `f91bdcf…`、正規manifestのdev/tier10 ARC100問、公式scorer、TP2・MTP4、LP並列32、chat並列256、agent並列4、observer OFFを維持した。model discoveryもLP endpointも `http://gemma-0:8000/v1` を指定し、候補とT002が混在するgatewayへ送っていない。公式CLI引数の親子差分はrun IDとlabelだけで、候補labelは実treeへ事前bindingした。全100問により今回の対象10問、親の正解66問、既存退行を一度に確認した。これは診断subsetであり、全778問評価の代わりではない。

| 同条件run | 正解 | 不正解 | 未採点／通信失敗 | 公式generation | outer |
| --- | ---: | ---: | ---: | ---: | ---: |
| `restart-direct-t003-parent-t002-arc100-dev10-r1-20261108` | 66 | 34 | 0／0 | 4.9秒 | 6.331秒 |
| `restart-direct-t003-candidate-arc100-dev10-r1-20261108` | 62 | 38 | 0／0 | 5.6秒 | 6.948秒 |

親の100個の公式scoreは、参考値である既存4endpointのT002 canonical ARCと全件一致した。各runのlogprob API callsは1198、chat API callsは0。候補の全loglikは有限、lengthsは正で、非有限値・欠損・公式errorは0だった。元応答、公式採点、全100問の得失はそれぞれの唯一のresultsと比較JSONへ保持し、再採点していない。

候補は親に対して改善8・退行12、純減4。親の正解66問のうち54問を維持した。対象10問は次の4問が新正解となり、6問が未解決だった。

| 対象ID | T002 | T003 | 結果 |
| --- | ---: | ---: | --- |
| `arc-challenge:Mercury_7263183` | 0 | 0 | 未解決 |
| `arc-challenge:MCAS_2004_5_7` | 0 | 1 | 新正解 |
| `arc-challenge:LEAP_2007_8_10417` | 0 | 0 | 未解決 |
| `arc-challenge:Mercury_SC_400401` | 0 | 0 | 未解決 |
| `arc-challenge:Mercury_7234360` | 0 | 1 | 新正解 |
| `arc-challenge:MDSA_2007_8_17` | 0 | 1 | 新正解 |
| `arc-challenge:Mercury_SC_413624` | 0 | 1 | 新正解 |
| `arc-challenge:NCEOGA_2013_8_28` | 0 | 0 | 未解決 |
| `arc-challenge:Mercury_SC_407400` | 0 | 0 | 未解決 |
| `arc-challenge:Mercury_7217298` | 0 | 0 | 未解決 |

退行12問（全て `arc-challenge:`）は `Mercury_7163818`、`Mercury_SC_400708`、`Mercury_7206938`、`MEA_2013_8_18`、`LEAP_2001_8_10379`、`TIMSS_2003_8_pg47`、`Mercury_401014`、`TIMSS_1995_8_J1`、`Mercury_SC_413549`、`Mercury_7094290`、`Mercury_7174738`、`Mercury_SC_414156`。全12問を含む候補の不正解38問を `rejected-candidate-unresolved-queue.json` にmanifest順で保持した。これは棄却候補の派生記録であり、T002のactive queueを置換しない。棄却候補の改善8問をT002で解けた問題として扱わない。

採否の「1問でも正解が増えたら」は、追加の指定がない時点では**退行を含めた同条件の正解総数が1以上増えること**として扱った。対象の新正解だけでは総数増加とは扱わず、T003は66→62のため棄却した。対象10問すべての正解を採用条件にしたわけではない。T002の66正解を累積baseとして保持する。`gemma-0`のT002への復元とT003の証拠確認後削除は進行中であり、この記録は削除完了を主張しない。別の解釈をユーザーが明示した場合は、その後の判断へ反映する。

生成診断として選んだ `triviaqa:bb_6707` と `xstest:232` は未実行。両者の元公式max_tokens16384／samples1を変更せず、親のgateway結果と候補のdirect結果を同条件として比較しないため保留した。T003の正規全778問、test、30／60分版、DeepSWE、Hugging Face登録は実施しておらず、全tier完了や一般的改善を主張しない。

### 既存toolの使用と証拠

新しいeditor・scorer・推論wrapperは開発していない。既存の [保存重み編集CLI](../wsp-weight-editing.md)、`kairyu_bench.wsp.weights.chain`、`docs/experiments/scripts/redeploy-replica.py`、変更なしの `kairyu_bench.wsp.cli` を使用した。以下の登録JSONのargvが実行仕様であり、完了済みT003の保存先・run IDへ再実行しない。再現には新試行、新しい出力先、容量予約、候補freezeと提供先の確認を先に登録する。specの保存値、raw評価、重みはPRへ複製せず、SSH上の唯一の証拠を参照する。

| 工程 | 使用方法と固定argvの保持先 |
| --- | --- |
| 保存編集 | `python -B -m kairyu_bench.wsp.weights.editor create`。親T002、登録済み `edit-spec.json` の `set_values`、別候補path、`--manifest`、`--reserve-bytes 214748364800 --apply`。実argvは `creation-execution.json`。 |
| 元からの復元検証 | `python -B -m kairyu_bench.wsp.weights.chain`。指定元とT003にT002・T003の `edit-manifest.json` を順に渡す。実argvは `chain-execution.json`。 |
| replica切替 | 既存helperのpreview→apply。exact argvは `single-replica-switch-plan.json`、実行／検証は `deployment-apply-execution.json` と `deployment-apply/receipt.json`。 |
| ARC診断 | `python -B -m kairyu_bench.wsp.cli run` に `--only arc-challenge --tier 10 --split dev` と上記並列数・単一endpointを指定。exact container/CLI argvは `diagnostic-evaluation-plan.json` の `runs`、候補tree binding後のargvは `candidate-arc100-input-binding.json`。 |

現行 `./kairyu-bench wsp run --help` と旧token診断wrapperの `--help` を確認し、固定runner内のCLI／run／client／registry／logprob／text／manifestの8source hash一致を記録した。通常wrapperは `--only` をサポートするが全scorer準備でNLTK約73.7MBを展開するため、この2runでは既存のbounded Docker形から同じ公式CLIを起動した。旧token wrapperは旧対象10問が固定されているため今回使用も変更もしていない。初回preflightのhost Python import失敗（推論0）を保存し、既存非学習venvと `PYTHONPATH=<worktree>/src` でv2確認後に親を1回実行した。各run前にfreeze stat、RO mount、tree、runtime、health、idle、容量を確認し、候補は切替helper終了とservice lock解放後に開始した。固定13 scriptsとhardlink元は変更していない。

作成後容量照合（03:50:09 UTC）では、指定元＋採用中T002 1個＋候補T003 1個。T003の生成regular実割当は62,578,733,056 bytes、directoryは8192 bytes、重み削除は0だった。追加weight全量の厳密値は未確定で、保守上界159,705,311,536 bytesは150GiB以内、cache予約上界104,280,008,831 bytesは100GiB以内だった。両診断と計画・help・終了記録の証拠は218,115 logical bytes（inode一度計上の実割当286,720 bytes）、activation dump0。終了した診断containerの消滅を確認し、空の各tmp/xdgをその試行内で削除した。診断終了時NVMe空きは2548.672GiB。これらはT003 checkpoint削除完了の証拠とは別であり、採否後の重み整理は引き続き容量台帳で確認する。

唯一の証拠rootはSSH先 `restart-direct-20261108/trial-003-restore-final-attention/`。各評価runの唯一の出力は作業rootの `results/<run-id>/` にある。

| T003内の保持先 | SHA256 |
| --- | --- |
| `registration.json` | `3e77484400c6de5135e84610b4f6bda4baf54fb0b2a83b2a6650ff20fdaf01b2` |
| `saved-byte-preview.json` | `4486ed2dda4b3e70b1888594b0e2a6c6f369a6bf47c179a595054f86b9f7cc5b` |
| `creation-execution.json` | `7e503e39a3377f38803871664f37fce8836a33a89df2e37cb43f29f92849d214` |
| `edit-manifest.json` | `61da330daed65071044d5926525035901dd47ff8999924763f425ec1f60bc394` |
| `chain-execution.json` | `249f33c480fb640e911e0db3d29a14fc5c5023a5fd4d34a1a50121aa4f7337db` |
| `chain-verification.json` | `b71cd62bc320a46bf7dcf657fc450508ebcefa62344a1f146473072565ebf4fb` |
| `candidate-freeze.json` | `144567f73f0733f7807980fc575ee038c96552a7ab5aa5be6ebd4edb82bb74ce` |
| `candidate-freeze-proof.json` | `01199634354e7451f90edb6009f204524ec82bddc96b8066ece0a609f7edbbb5` |
| `opus55-review/receipt.json` | `ae0b4cc6e4bc9fd94f4c36934b1e241a7429bd68ddc7e6d7d970302fcd211759` |
| `diagnostic-evaluation-plan.json` | `00d54d43fb6afafd5beadf17ef456b575946e15e26fc14661d29a7b42f2551fe` |
| `candidate-arc100-input-binding.json` | `29acaf8f23548545403f75f4491a7b8b9fcd60536163cf15bbc3266822993fb4` |
| `parent-arc100-summary.json` | `9d3290c72351e545414e9b3b9e9393bef5b29c1e6f45e1f46d8db37748188678` |
| `candidate-arc100-summary.json` | `735b2edf6538efe02face64c7a451b158830804cf65743c3f98ca013055e6599` |
| `rejected-candidate-unresolved-queue.json` | `6d6dbbae0766d0adead0c10a8a2e95178ba311a5250a99d731c44cf7a2221385` |
| `storage-postcreation-reconciliation.json` | `b0ca60ec71b2fb9df2c0cf740a334ea7fc01a74ec3ce9f846711af8a670462ad` |
