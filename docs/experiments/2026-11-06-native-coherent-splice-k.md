# K: 保存済み native B と coherent H の byte 合成と4条件比較（2026-11-06）

K は、同じ元モデルから作った B の native 変更と H の係数1/16 LoRA 変更を、保存済み BF16 のまま一つに集める固定候補である。追加学習・係数探索・GEMM・再丸めは行わない。**4条件とも公式dev90の90/90採点を完了した。Kのcodeは70.908730、fresh original比+0.138889 pp（95% CI −4.027778〜+3.888889）で、改善確定・code90達成とはしない。Kは採用せず、この小さい不確実な差を根拠とするfull14への昇格も行わない。**

[J の公式dev90](2026-11-06-weighted-reasoning-c1-review.md) はfresh original比 +3.055556 ppだったが、95% CIは0を跨ぎ、既存B/H/Iを全て上回る固有成功は0だった。次はJを足すのではなく、既存BとHの成功を単一モデルに残せるかを検証する。異なる時刻の単体利得を足し算せず、B/H/K/originalを同時に測る。

## 固定した保存重みと変更量

元treeは cab440f871a32b5ea1c12364db7dd35ac8768a83b413ca24bbc18d6478eccbd9。
入力監査は元／B／Hの各12ファイルの記録と現在stat、元file hashの記録一致、
全571選択tensorのheader・shape・dtype・byte spanと非重複を確認した。
この事前監査自体は新しい全payload hashではない。実合成では後述のbytes検証を行う。

| 成分 | 変更tensor数 | 保存後の変更要素数 | 保存Δ L2 |
|---|---:|---:|---:|
| B: science-native-contracts-v1 | 161 | 49,053 | 0.003735683695257374 |
| H: coherent donor 1/16 | 410 | 4,328,720,639 | 1.888537495617679 |
| Kのbyte検証により継承した合成値 | 571 | 4,328,769,692 | 1.888541190360053 |

Bはlayer scalar 2、pre-feedforward norm 29、post-attention norm 58、
post-feedforward norm 56、K norm 5、input norm 11 tensor。
Hは既に元BF16へ1回mergeしたtext linear 410 tensorであり、Bとの重なりはない。
全tensorをコピーするが、上表の変更要素は元bytesと異なる値だけを数えた既存component実測である。

非重複なら変更数は加算、Δ L2は二乗和の平方根になる。
これは独立した数値再計算ではなく、hash固定したcomponent実測とbyte同一性に基づく継承である。
少ないnative要素の機能効果が小さいとは限らず、normとlinearが同じ計算経路で相互作用し得る。
Bの旧代理指標を新frameの学習証拠へ読み替えず、Bの既存公式実測と
[広い領域の損失](2026-11-06-code90-matched-screens.md)も保持する。

| 固定対象 | SHA256 |
|---|---|
| B tree | 564ae5fc1efdedd77180a91764308ccea222311e19c5e0a8c7a2262069a8138b |
| B manifest | aaf7ca16b9bf6df1f5052fac9b3e72db303d7cd069c42d8776ed9b20bb410f16 |
| B spec | 05fd80d19cfc56467e172e0f7bf14a74dae5f562d80b09228eca6b72204d7128 |
| H tree | 0f9775ba6a2406989407d86d365eed69e1684a4400faddf56bb3822fad6b98d3 |
| H manifest | b1eb4249f07baebb4796ec12e8d1097ea50b330f46a9e3128c0279dc801fc6fd |
| H spec | a72aecba942de401d814c06162065ea822c44268f9cacf5b76b5501a7f42f409 |
| H CPU recipe freeze | 8941b3271db0d2593871f114a53759d3ad4741741b10f0fd8d089bca6af37a8b |
| b-plus-h-composition-input-audit.json | 9b7c816bd012704c21d012807b0d6e14fe6ed15c1893f8daf120b0bcc6cb2ddc |

artifact名は analysis/code90-resume-20261106/ からの相対名である。
Bには独立freezeが無いため、合成時はfresh full-file hashを用いる。
HはCPU再生成済みfreezeとexact stat継続を照合し、最初のprivate copyでも全file hashを計算する。

## 合成の検証契約

再利用する [composer](../../scripts/experiments/compose-verified-checkpoints.py) は
明示planとSHA、同じ元tree、非重複full-tensor spans、同じlayout/configを要求する。
Hから独立ファイルを作り、Bの保存済みtensor bytesを指定spanへコピーする。
入力は変更せず、outputのsymlink・alias・上書きを拒否する。

output全fileと各選択tensorのhashを計算し、次の三つの仮想差戻しを検証する。

1. B/Hの全spanを元bytesに戻すとoriginalの全file hashになる。
2. Bのspanだけを戻すとHの全file hashになる。
3. Hのspanだけを戻すとBの全file hashになる。

既存のKと元bytesの同じstreamを使ってhashersを更新するため、この三条件のための追加payload読取やGEMMは不要である。
全差戻しはKの未編集領域を固定し、部分差戻しは各componentの宣言外変更も検出する。
たとえばBの宣言外変更をコピーしなくても、三番目と実Bのfull-file hashが不一致になる。
SHAの衝突耐性と入力の前後stat/hash継続を前提としたbytesの保証であり、関数出力の保証ではない。

manifest/freezeは排他的に公開し、失敗時はこの呼出しが所有する未公開出力だけを掃除する。
既存redeployが読めるfreezeへ実施したhash方式を記録する。
[実装と公開境界](../wsp-checkpoint-composition.md)を参照。
実Kの証跡は末尾に記録し、CPU fixtureの成功と区別する。

## Opus 5.5 の匿名方法レビュー

actual assistantとusageはともに claude-opus-5-5 のみ、
verified_requested_model=true、review_usable=true、tools 0、60.889秒、$0.116412。
Kをキーとするpublic final JSONは通常のlist schemaと異なり labels_match=false だが、
単一Kのpublic-final-textとして受領した。raw stream・thinking blocks・環境は保存していない。
匿名の公開一般機構だけの意見であり、非公開の候補測定や能力を査読したものではない。

固定済み保存成分のbyte合成は、余分な丸めを避けて累積仮説を直接調べる方法として妥当との意見だった。
採用するのは入力／配信同一性、実span非重複、三つの差戻し、同時4条件比較、
課題別得失と単回測定の不確実性の区別である。

入力normがprojectionの入力を変え、K normがattentionを変え、
post-normが残差の寄与を変えるため、parameterが非重複でも機能は線形加算ではない。
出力長・cap・形式の変化、同じ成功への冗長な寄与も確認する。
レビューが挙げたfinal normの一般論を、今回Bがfinal normを変更したという事実にはしない。

追加logit probe、独立した全重みnorm走査、係数探索、反復や新しい損失閾値は
このレビューによって必須化しない。ユーザーのcoding優先・他領域の多少の損失許容を維持する。
成功集合のunionを超える新成功は必要条件ではない。両単体の成功を一つに残して
best soloを上回れば正当な累積改善であり、同じ成功は二重計上しない。

receipt opus-native-coherent-splice-k-review/receipt.json のSHAは
efad51b13b2fe8d302358899f76652480a5867c0785e539e30892b898436b4d6、
匿名prompt SHAは b3bf4d50ded444b2cc3d00ccb37632f5ff177f6f5b37fb413d0b6511d5067978。

## 事前固定した4条件の公式比較と数え方

保存検証と既存authored smokeの構造確認後、fresh B、H、K、originalを
4つのdirect endpointで同時に実行する。各endpointはTP2、chat concurrency32。
native thinking ON、公式sampling/template/tokenizer/caps/scorer、固定bench source
5e3f2df0c7f455013af3fa2d7abb3719858f808f を揃える。
source選択はHE30／MBPP30／LCB20／SciCode10の同じ90項目、
既存manifest SHA a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。
run IDs、実際のloaded tree、container/image/config/layer列、開始時刻は実行証跡で固定する。

主指標は公式codeの4-source等重み平均である。K対B、K対H、K対originalを個別に
公式paired比較し、成績の良い単体との関係も明示する。過去単体や元モデルは揺らぎの文脈に限る。
B/Hの双方成功・Bのみ・Hのみ・双方未成功のセル別にKの維持／獲得／喪失を示し、
SciCodeのpartial scoreはそのまま別記する。binary thresholdのflipとfractional変化を混同しない。
cap/timeout/unknown、sampleのfinish reason、使用量、wall timeも併記する。
固定seedでもc32の出力同一性は保証されず、単回のdev再利用は未見確認ではない。
このdev90の90超も、既に成績を見たknown64の目標達成と同一視しない。

開始前／終了後に [serving counters](../wsp-serving-counters.md) を採り、
process identity・reset・missing・running/waitingを保ったまま保存sampleと照合する。
現在の公式sourceは、run作成前にmax_tokens=1のchat discovery probeを1回送る。
全39 SciCode stepまで通常完了する場合、保存leaf callsは119、probeを含むprocess終了は120となり得る。
probeのlength終了をbenchmarkのcap失敗へ足さず、実際の各run recordsとcounter差分で確認する。
単なるcounter差分からHTTP retry不在・他client不在・課題帰属・思考token数を断定しない。
startup probeの根拠は[Jのsource監査](2026-11-06-weighted-reasoning-c1-review.md#採点分離と追加process-eventのsource監査)に残した。

## CPU統合検証と固定plan

新しい合成工具を含む全CPU検証は495件中443成功、任意依存52skip、失敗0。
Ruff全treeとmypy 73 source filesはPASS、実行中のsource変更0だった。
6件の追加テストはbyte合成→配信freezeの契約、入力破損・未申告component差分、
出力競合と失敗時cleanupを保護する。追加の模型／GPU callsは0。
CI receipt cpu-integration-checkpoint-composition-v1/ci-results.json SHAは
536bfa458f158e285ceb33e1341c347a4549fca78b30dda1673e30d2f3d16438。

実作成用planは native-b-plus-coherent-h-v1/plan.json、SHA
085cd6df19b8d410e8278cbb86ece932099bf3f182f82dd0de59ded932b5e6c0。
同フォルダのpreview.jsonは571 tensors／4,328,769,692 changes、
62,578,686,256 bytesの独立コピーと300 GiB予約条件を確認した。
実作成・全hash検証の完了と、配信・smoke・公式比較の開始を次節に記録する。


## 実作成・配信・構造smokeの完了

合成applyは **522.8478秒** で完了した。元fresh full-file hash、
全選択tensor SHA、非対象span、独立private files、component仮想差戻しが全PASS。
Bはfresh hash、HはCPU freeze＋exact statとコピー時hashを用い、
K全12ファイルを新しくhashした。再GEMM、再丸め、RAM/GPU比較、モデル呼出しは0。

all→original、B span差戻し→H、H span差戻し→Bの全file identityを実Kで確認した。
保持componentは合計571 tensorで、4,328,769,692 changed elements、
Δ L2 1.888541190360053を継承する。これはcomponent実測＋bytes証明であり、
独立に全値を数値走査した新norm測定ではない。

| 完了証拠 | SHA256 |
|---|---|
| create-completion.json | cd7bd206dc5e40da6cc41a17dbe4b40b6bcbfc8ed585160e2ac1f220e751cf60 |
| K全file tree | fd8ba3ccb9237a133e9cdead352b5fda71bc7058520e4dfdd0957da3d77cf6ad |
| K composer manifest | 7480209804b372908e6efb75ad698608bb496baafed51d164ac20df59e141481 |
| freeze.json | 40be455c87375172837639f1c321d7430e4c6f5c9556b89ed6bcdc822f6df1cb |
| deployment-b/receipt.json | c17655dff24319c9445388681824cc456c4446b79ae447b75ed78bbbd6e6e4e9 |
| deployment-k/receipt.json | d80bd502804a58f29d200f4679db164e41eb2cec33eb857b6cde31cbb75ee411 |
| deployment-smoke-launch.json | be73124e0a9f0de8eee3c92746778f6289dff8d8b2a3f44ffd4a5df9e2a86295 |

ここからのartifact名は native-b-plus-coherent-h-v1/ 以下。
既存replicaのうちgemma-0をB、gemma-3をKへ更新し、Hとoriginalを保持した。
両redeployは正常完了し、Kを既存のauthored advanced-v2 24件へ進めた。

| 構造smoke | Historical original-v2 | K |
|---|---:|---:|
| Train / transfer | 15/16 / 8/8 | 15/16 / 8/8 |
| 合計pass / evaluated | 23/24 | 23/24 |
| Unknown /追加構造失敗 | 0 / — | 0 / 0 |
| Finish stop | 24 | 24 |
| Input tokens | 2,864 | 2,864 |
| Output tokens | 43,599 | 42,250 |

双方の唯一の数値失敗はscience-advanced-reflecting-wall-flightで、8数値trial中7成功。
保存された数値metadataだけを照合し、生成本文を読んで原因を推定していない。
出力tokenは思考と公開回答を分離した計測ではない。
同じcase/seed/validator条件だが元はhistorical referenceであり、fresh causalな能力差とはしない。
構造停止条件に該当せず、rootは固定dev90への移行を記録した。
smoke-numeric-summary.json SHAは9af6fe611895ad44d5621e6bd48d76946d9ad30a264503054d02c53ce071d7e9、
structural-smoke-decision.jsonは8678cc7f39b5f2352a762a74d1d21ecf4b398623b07b18d58c76083ab5665a4d。

## 公式4条件の開始記録

2026-11-06 **10:33:23 UTC** に4 childを起動し、
run metadataの開始時刻は10:33:27 UTCだった。
10:35:06 UTCの読取時点では全4runがrunningで、
同じ90項目・tier30/dev・manifest・model IDと各意図したtree labelを確認した。

| Arm | Run ID |
|---|---|
| B | gemma4-code90-native-b-c32-dev30-code-pairk-20261106 |
| H | gemma4-code90-coherent-h-c32-dev30-code-pairk-20261106 |
| Original | gemma4-code90-original-base-c32-dev30-code-pairk-20261106 |
| K | gemma4-code90-native-b-plus-coherent-h-c32-dev30-code-pairk-20261106 |

4つのdirect TP2 endpoint、各c32、MTP4、native ON、公式source5e3を維持。
launcherは実CID/readonly checkpoint/構成と、保存freeze・source/元B/Hのstatを照合した。
開始前counterは10:33:22 UTCに採り、全4endpointのrunning/waitingが0であることを確認した。
開始時点では終了counterと得点は未確定だった。完了後の照合結果を次節に記録する。

dev90-four-plan.json SHAはd96ac80d7f919ff2dffbcf867e3b235886dddd0f2da70c55d8b352466333b07e、
実bindingは33808ecc48a8bbf1f1eaacba88948eb30e6f425dc0d94c141176bc355de0dd29、
launch receiptはa2e1d0468df9b19a95105325f886a5ae7e7a18c1d2f3f80f61672a12ee0d58d7。
この開始記録は比較可能性の準備を示す。成績と採用判断は以下の完了結果による。

## 公式4条件の完了結果

全4runがcompleted、各90採点・未採点0、合計360採点で終了した。
同じ90 ID、manifest、公式source、配信treeとsource/stat継続を確認した。
以下のcodeは4-source等重み平均であり、90件の単純正解率ではない。
SciCodeは10問それぞれのstep成功率の平均で、完全成功数へ置き換えない。

| Arm | HE+ (30) | MBPP+ (30) | LCB (20) | SciCode mean (%) | 公式code (%) |
|---|---:|---:|---:|---:|---:|
| Original | 28/30 | 23/30 | 15/20 | 38.079365 | 70.769841 |
| B | 29/30 | 22/30 | 14/20 | 40.301587 | 70.075397 |
| H | 29/30 | 22/30 | 15/20 | 38.079365 | 70.769841 |
| K | 29/30 | 23/30 | 14/20 | 40.301587 | 70.908730 |

既存の公式paired比較をそのまま使用した（2000 resamples、seed 20261104）。
全90項目がpairedで、比較のpartialはcode以外13領域が未測定であることを表す。
未採点を0で補完した比較ではなく、overallはnullのままである。

| 比較 | Code差 (pp) | 公式95% CI (pp) | 公式0.5境界の獲得 / 喪失 |
|---|---:|---:|---:|
| K − Original | +0.138889 | [−4.027778, +3.888889] | 2 / 2 |
| K − B | +0.833333 | [−5.000000, +6.666667] | 4 / 3 |
| K − H | +0.138889 | [−5.416667, +5.833333] | 4 / 3 |

K対originalのsource差はHE+ +3.333333、MBPP+ 0、LCB −5、
SciCode +2.222222 ppだった。課題単位ではHumanEval/95、Mbpp/641を獲得し、
LCB/3777、Mbpp/801を失った。これにSciCode/72の5/9→7/9という部分改善が加わる。
公式flipの2獲得/2喪失だけではこの部分改善を表せない。

### 保存した4条件セルと累積の限界

二値のHE/MBPP/LCB計80問では、B/H双方成功64問中Kは61成功、
Bのみ成功1問・Hのみ成功2問はKが全て維持した。
双方失敗13問中Kは2成功だった。SciCodeは別に実数のまま扱う。

| B/Hの二値セル | 件数 | K成功 | K失敗 |
|---|---:|---:|---:|
| 双方成功 | 64 | 61 | 3 |
| Bのみ成功 | 1 | 1 | 0 |
| Hのみ成功 | 2 | 2 | 0 |
| 双方失敗 | 13 | 2 | 11 |

4条件間で値が異なった全11問を以下に示す。0/1は保存された公式値であり、
SciCodeの分数は公式step成功率である。問題本文・生成本文・hidden reasoningはこの集計に使用しない。

| ID | Original | B | H | K |
|---|---:|---:|---:|---:|
| HumanEval/116 | 1 | 0 | 1 | 1 |
| HumanEval/151 | 0 | 1 | 1 | 0 |
| HumanEval/95 | 0 | 1 | 0 | 1 |
| LCB/3777 | 1 | 1 | 1 | 0 |
| LCB/abc390_g | 1 | 0 | 0 | 1 |
| LCB/abc399_c | 0 | 1 | 1 | 0 |
| LCB/arc195_b | 1 | 0 | 1 | 1 |
| Mbpp/641 | 0 | 0 | 0 | 1 |
| Mbpp/801 | 1 | 0 | 0 | 0 |
| SciCode/35 | 2/3 | 2/3 | 1/3 | 2/3 |
| SciCode/72 | 5/9 | 7/9 | 8/9 | 7/9 |

B/Hの少なくとも一方がoriginalを上回った4問のうち、Kがその単体最大値まで
回復したのはHumanEval/95の1問だった。HumanEval/151、LCB/abc399_cは利得を失い、
SciCode/72はoriginalより良い7/9を保持するがHの8/9には届かない。
したがって「最大component scoreへ未回復3問」は「全利得を失った3問」ではない。

Kが両単体より良いのはLCB/abc390_gとMbpp/641、
両単体より悪いのはHumanEval/151、LCB/3777、LCB/abc399_cだった。
これはこの単回比較で観測した相互作用であり、重み機構の因果局在は示さない。
元重みへのbyte合成の正確さと、成功集合を単一モデルへ保持できることは別である。

### 使用量・終了理由・時間

保存された生成callは各119、logprob callは0。全sourceのtruncated/empty rateは0。
process counterは各120終了（stop119、length1、error/abort/repetition各0）で、
保存使用量との差は各prompt21 token／completion1 tokenだった。
これは事前に確認した起動時discovery probe 1回と一致し、
benchmark119生成のうち1件がcapに達したという意味ではない。

| Arm | 保存input tokens | 保存output tokens | report wall (s) | metadata開始→終了 (s) |
|---|---:|---:|---:|---:|
| B | 95,181 | 349,678 | 419.5 | 419.488267 |
| H | 96,343 | 336,519 | 374.3 | 374.267049 |
| Original | 98,428 | 347,285 | 451.5 | 451.545265 |
| K | 96,090 | 329,763 | 364.8 | 364.854504 |

全counterでprocess start identityは不変、missing seriesなし、前後running/waitingは0。
この整合だけでは観測されないreset、別client、HTTP retryの不存在や
思考／公開回答token別計測を保証しない。独立HTTP回数・agent stepsはここから推定しない。

最後に終了したoriginalは2026-11-06 10:40:59.193746 UTC。
次候補LのCPU作成はcreate-launchのISO時刻10:41:10.235677 UTCに開始し、
約11.042秒後である。4件の公式runとL作成の時間重複は0で、
Lと重なったのは終了後counter採取窓の一部だけだった。
この事実とは別に、同時採点・runtimeの揺らぎを含む単回wall差を
Kの因果的な高速化とは主張しない。

### 判断と追跡可能性

Kは固定dev90でも90に届かず、fresh original/Hとの差は+0.138889 ppに留まり、
全3paired CIが0を跨いだ。**Kを採用せず、今回の差を根拠にfull14へ進めない。**
候補のbyte合成方法は検証済みとして残すが、能力改善の結論とは分離する。
既に見たdev集合の単回比較であり、未見での一般化や既知64問での90達成ではない。
未測定のtools/instructions/agents等の得失も補わない。

以下は同じnative-b-plus-coherent-h-v1/以下の証跡。
独立レビューは90 ID・数値セル・CSV・source平均・公式flip・使用量・時間の整合を確認し、
新しい採点、モデル呼出し、bootstrap再実行は行っていない。

| 証跡 | SHA256 |
|---|---|
| dev90-four-final-summary.json | 645d3e943ff345bfbfe24d0fcec7807808f8b20e3e1d135025d5a8b2be5ac461 |
| dev90-four-numeric-evidence.json | 51389ddb99399461a6a279ce996ac1a0721cda7bf4f8c890e086ff0835f83c21 |
| dev90-four-cells.csv | 5a25e817f153696f7785cc8349983276be71b319b005767b0ff8853e26f6a89f |
| dev90-k-versus-base-evidence.json | f9c282ef3ce33fa88945c80d065286fe5fa4766ff63508981635e883fd7e84ec |
| dev90-k-versus-b-evidence.json | b726301593ca9a12524187caf010268c75fe7b02cbfb098c434b7ef67ba11642 |
| dev90-k-versus-h-evidence.json | c2fc395ec7a36d56eaf22c64a4d4092b9461b691645bb5a0e20e5540d6ad8a90 |
| dev90-four-usage-accounting.json | a31b7142c454039b3400adf153086a66b0810872508c3767f00a85a5e140ab1c |
| dev90-four-independent-numeric-review.json | 71006651111c3cb2849849700f10c2f106ae25a5b642ab67f8ad0e48a41d12ea |
