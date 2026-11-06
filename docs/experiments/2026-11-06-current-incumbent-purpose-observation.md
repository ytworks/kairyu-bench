# 現作業モデルPの目的別微分と次の変更セット仮説（2026-11-06）

> 状態追補：一時R昇格時にP微分を歴史観測へ移し、R両目的も新規収集した。その後Rの全14領域net lossを受け、ownerは同一の保存済みPへ復帰した。Pのwhole-checkpoint identity／9file stat継続性を確認した上で、この保存P微分を再利用する。RのGをPへ流用しない。元のP観測値・収集source fingerprintは変更しない。

現在Pを起点に取り直したcorrect-LLとmarginを、同一frameの歴史Oと比較した。**Pのnative保存値は同一でも微分は同一でない。旧O GをPの勾配として流用しない。** 候補作成・solver・モデル実行はこの分析では0。Rが昇格したらP Gは歴史観測へ移り、新Rの再観測か明示的な未検証transfer仮説が必要になる。

[Rの新観測と固定S/U提案](2026-11-06-current-r-purpose-observation.md)へ続く。以下のP数値・方法助言は当時の記録として保持する。

## 条件と計算法

自作train136例／21目的／42fold、native421 tensor／1,331,516座標。O/P両目的の保存native値は完全一致。cases SHA c666cb49…、forced-empty-thought frame SHA a6a94e19…、serial・1case microbatch・同じcollector実装／Torch／Transformersが一致。これは空thoughtを明示したteacher-forced公開回答の診断で、native ONの実生成reasoningそのものではない。

既存joint.read_collectionで読んだ配列をFP64集約。各目的は2foldの等重み平均（同目的内のfold例数は等しい）。cosineはEuclideanな保存weight座標上の内積を両normで除した値、zero normはnull。role-equal cosineは各role内cosineの単純平均で、座標尺度への感度を示す補助であり新しい最適化costではない。JSONには全42fold／136例の数値、21目的・全目的ペア、役割・層別norm²比率とP/O cosineを保存した。

独立agentが元summary4本だけからscalar集約を再計算し、42foldのgain/loss、科学6群の符号、目的等重みΔLL−0.4242256160／Δmargin−0.5251358637を照合した。この平均は公式scoreでも新しい成功指標でもない。

CPU8–15／BLAS2で21.42秒。collector出力hash一覧との94入力一致と前後statを確認。新しいcheckpoint hash scanは行わず、Pのfullhashは独立download証跡＋9file stat継続性を継承する。

## 観測

- correct-LLは42fold中15改善／27低下、marginは13改善／29低下。これを能力得失の件数へ変換しない。Pの公式code改善とこの短い自作proxyの一様な改善は一致していない。
- 科学6目的の平均correct-LLは全て改善したが、marginはnumerics／contracts／state／simulationの4目的で低下した。goodとfoilがともに動くため、片方だけの最大化は実行改善の保証にならない。
- lookupのP/O G cosineはcorrect-LL .0019／margin .0005、contractsのcorrect-LL .1900、advanced probabilityのmargin −.0308。保存native値が不変でも他の重み変更を通して局所応答が変化する。
- raw G norm²のpost-attention norm＋post-FFN norm比はcorrect-LLで82.1〜99.2%、marginで92.4〜99.3%。多くの目的のcorrect-LL上位層は53〜57付近だが、lookupは1〜4、countingは1/2/13/14寄り。これは尺度依存の感度であり、能力の因果局在やその層だけを編集すべき証拠ではない。
- 工具引数のcorrect-LL対margin G cosineは−.7651、call対no-callのmargin間は−.2492。片側目的の利得を「tool能力」とまとめない。
- cross-foldのmargin cosineはalgorithm .0710、contracts .0510、simulation .0071、probability .0253、referent −.0537と弱い。少数例のmean方向を頑健な機構と断定せず、実有限応答と実行を分けて確認する。

| 目的 | O correct-LL | P correct-LL | ΔLL | Δmargin | P/O G cosine (LL) |
|---|---:|---:|---:|---:|
| code | -0.53538 | -1.74042 | -1.20504 | -2.56785 | 0.5162 |
| tools | -23.27994 | -23.49494 | -0.21500 | +0.29232 | 0.9365 |
| instruction | -2.64574 | -2.22834 | +0.41740 | -0.77821 | 0.9277 |
| general | -8.60174 | -10.04577 | -1.44403 | -1.66881 | 0.9301 |
| reasoning | -6.33941 | -6.45877 | -0.11936 | -0.48873 | 0.9081 |
| lookup | -1.50395 | -4.80224 | -3.29828 | -3.66644 | 0.0019 |
| japanese | -15.24678 | -17.12648 | -1.87970 | -1.99812 | 0.7988 |
| implementation | -1.20246 | -1.26984 | -0.06738 | -0.03950 | 0.8930 |
| workflow | -2.75878 | -2.88787 | -0.12909 | -0.05090 | 0.6688 |
| tool_call | -2.64845 | -2.72326 | -0.07480 | +0.18240 | 0.6394 |
| tool_no_call | -3.03321 | -3.62338 | -0.59017 | -0.27554 | 0.8532 |
| tool_arguments | -2.06680 | -2.30937 | -0.24257 | +0.05266 | 0.8130 |
| algorithm | -1.12849 | -1.13162 | -0.00313 | -0.00517 | 0.9494 |
| referent | -4.41314 | -4.49729 | -0.08416 | -0.00404 | 0.9349 |
| fact_retention | -2.05568 | -2.18391 | -0.12822 | -0.01963 | 0.9233 |
| science_numerics | -0.94613 | -0.94352 | +0.00261 | -0.01243 | 0.8721 |
| science_contracts | -1.04630 | -1.01678 | +0.02952 | -0.01673 | 0.1900 |
| science_advanced_state | -1.25085 | -1.23840 | +0.01245 | -0.00148 | 0.8285 |
| science_advanced_counting | -1.25062 | -1.22600 | +0.02462 | +0.03345 | 0.3622 |
| science_advanced_simulation | -1.59998 | -1.56744 | +0.03255 | -0.00649 | 0.8818 |
| science_advanced_probability | -1.77279 | -1.71975 | +0.05304 | +0.01137 | 0.6405 |

単位は回答内mean log probabilityのnats/tokenとその差で、accuracy ppではない。toolsのcorrect-LL約−23等もこの固定回答診断の値で、公式tool成功率ではない。

## 次の変更セット仮説（未選択・未実体化）

**H1〜H7はこの文書内の仮説グループ名であり、歴史checkpoint Hや既存候補のIDではない。** 匿名方法レビューの単文字label S/T/U/V/W/X/Yは順にH1/H2/H3/H4/H5/H6/H7へ対応する。

以下は別々に判別可能な目的セットである。native Gが行列deltaやactivation covarianceを測ったわけではなく、matrixの因果方向を復元できない。Rの広いMLP変更と、これらのnative感度から将来作る案も混同しない。

| 仮説 | 根拠・依存要素 | 何を測れば判断が変わるか |
|---|---|---|
| H1 数値＋array契約＋state | numerics/contracts LL cosine .8348、role-equal .7811。連続変換・shape・状態保存を一組にする。 | good-NLLとmarginの有限応答を別記し、数値/shape/state実行と公式SciCode/LCBで実利を見る。LLだけ上がり実行が変わらなければ根拠が弱まる。 |
| H2 simulation＋probability | LL cosine .7427、role-equal .6716。保存則・安定計算を組にするがmargin方向は弱い。 | 不変量、nonfinite、境界実行を形式失敗と分ける。両foldのgood/foil変化が実生成へ届くか。 |
| H3 implementation＋algorithm＋workflow | LL cosine implementation/algorithm .5714、implementation/workflow .4144。LL対marginは一部逆向き。 | 実装保全・algorithm正答・最終提出を別々に見る。HE/MBPP/LCBの実得失で判定する。 |
| H4 call＋no-call＋arguments | margin call/no-call −.2492、call/arguments +.3403。意思決定と引数を単独で押さない。 | BFCLのcall/no-call/argumentsとinstruction成績を個別報告。呼出し率だけの増加を改善としない。 |
| H5 lookup＋referent＋fact retention | lookupはP/O方向がほぼ直交、他2目的と層分布も異なる。共有方向でなく複数componentの協調仮説。 | 2foldの整合と自作参照実行でまず根拠を確認し、選ばれた場合に公式long/facts/knowledgeを確認。短いlookupだけでは不足。 |
| H6 instruction＋reasoning＋general | margin instruction/reasoning .3804、LLでは .0498。回答尤度と識別の別軸。 | instruction/reasoningの公式得失と完了/長さを別記。general proxyを安全性の代用にしない。 |
| H7 counting＋state（低確度、保留） | LL cosine .3010だがcountingの2fold .1303、margin .1014。early/late構造も異なる。 | 少数例方向を即編集せず、ties/order/boundsの異なる自作family実行で支持が増えるか。係数gridへ進めない。 |

優先検討はH1（科学数値＋契約＋状態）、次にH3（実装＋algorithm＋workflow）。両方ともcorrect-LLとmarginを組にして得失を報告する。H2は別の科学機構、H4/H6は関連能力向け候補とする。これはSの採用・生成指定ではない。H5/H7は方向の弱さを明記して保留寄りとする。目的ごとの許容損失は選定時に明示し、全目的0損失やCI有意差を新gateにしない。

## 最短の次工程と限界

まずRの公式parent-relative判断を待つ。Pが残れば現在Gと保存P値を使用できる。R採用なら再観測の実績目安は今回2目的compute144.11秒、load/restore込み304.53秒で、将来時間の保証ではない。既存CPU solveを1目的セットだけ行う場合も未実行・未計時。既存finiteはbase→candidate→restoredでgood NLL/marginを測り、BF16丸め後の実deltaと局所予測を分離する。新kernel・係数grid・第三の全410GEMMはこの分析から要求しない。

公式dev90の直近Pは351.2秒、Nは345.4秒。これは将来の上限ではなく実測参考で、長い出力や別負荷で変わる。自作/finiteが良くても公式current-parent差分が得られるまで改善を主張しない。90点目標・未知確認・DeepSWE/HFは別段階。

21目的にはXSTest等の安全性目的がない。Pの歴史full14安全性低下を、このGで改善済みと主張できない。全層postnormの大きなraw感度、normやcosine、局所線形additivityだけでは実機構・意味・能力を保証しない。

## 関連記録

[Pの採用経緯とMの4条件](2026-11-06-public-base-alignment-fourway-m.md)、
[Nの公式比較とP維持](2026-11-06-incumbent-coherent-step-n.md)、
[RのMLP構造subset](2026-11-06-cumulative-mlp-math-step-r.md)、
[全公式スコア台帳](2026-11-06-all-official-scores.md)を参照。

## 証拠

同analysis起点 current-p-purpose-observation-v1/numeric-analysis-v1/ にsummary.json、parity.json、hypotheses.jsonを保存。原Gは変更せず、collector-output-hashes.json／completion.jsonを参照する。

| 証拠 | SHA256 |
|---|---|
| current-p-purpose-observation-v1/numeric-analysis-v1/summary.json | be4fc2897446332f8edfd383999cd1d8dca137838dd2a6650fcdd62a3a512bbc |
| current-p-purpose-observation-v1/numeric-analysis-v1/parity.json | 460ed8857e4b4f36367a9bb38d4f89dad7e1c63bf1e7a69ef6f24f47eade5c48 |
| current-p-purpose-observation-v1/numeric-analysis-v1/hypotheses.json | 6291a9bd8a0ef4c27a7523ff438e7fda76490fa29eb10ed368d7fa9a5f04d094 |
| current-p-purpose-observation-v1/collector-output-hashes.json | 1cfa2d0cf9e9aab67e4c4358667b49bd37ca742e7e3b3b109f4d4aab15ff3428 |
| current-p-purpose-observation-v1/completion.json | 85dd3a3701ee96a45eee3d3fdd9ef151a7ed04f8bc1726eee21226c9e54db889 |

## 匿名7方法レビュー

匿名の公開一般方法だけを1回送信し、actual assistant/model usageとも
claude-opus-5-5のみ、STUVWXYの全7label一致、verified_requested_model／review_usable=true、
tools0／fallbackなしを確認した。58.434秒、記録費用$0.118392。
usageはinput2／output5,122／cache creation1,993 tokens、cache read0／web search0。
私的測定値・norm・座標・model/checkpoint ID・問題・回答・reasoningは送っていない。
raw stream／thinking本文も保存していない。匿名性はpattern lintに加え送信全文を確認した範囲であり、レビューは実候補の品質査読ではない。

Opusの上位もS＝H1、U＝H3だった。受理する観点は次のとおり。

- 数値変換・shape/dtype・入力状態保存を一組にし、value／shape／dtype／mutationの実行結果を区別する。
- 実装・algorithm・提出を一組にし、完了率と完了した回答の実行成功率を混ぜない。短縮・format改善だけで能力向上としない。
- tool call/no-call/argumentsは相互依存として得失を別記する。lookup／instruction／generalは低コストの関連能力確認にも使えるが、安全性測定にはならない。
- 同じ固定自作実行集合、既存finite、fresh current-parent公式比較を再利用し、係数gridや大量の追加benchを既定にしない。改善の積み上げに、新規成功が既存componentのunion外であることを要求しない。

reviewは「既存finite evaluatorでnative ONの生成・実行も測る」と書いたが、
実装上のfiniteはteacher-forced診断である。
実生成は既存science_generate／science_sandboxの別経路へ読み替え、両者を統合済みとしない。
token span別尤度、新しいfoil、scaffold差分等の提案は未実装の条件付き助言であり、
今回の必須工程や観測済み結果にしない。
同程度のLL/margin変化をscale変化の確定証拠とせず、注意すべき可能性にとどめる。
親変更後の再観測と元baseのexact復元を維持し、追加のsaved-artifact reloadや全410再生成は要求しない。
有意差・全領域無損失・追加承認を新gateにせず、候補の選定・実行はrootの別判断に残す。

| review証拠（同analysis起点） | SHA256 |
|---|---|
| coupled-purpose-sets-stuvwxy-prompt.txt | d70c2dd8394d9502bd4be8004f48833516be4f7c3ce33f0c30c45b84a8c2704d |
| opus-coupled-purpose-sets-stuvwxy-review/receipt.json | fe0a334a179d3763e2f9f7a85d750de662d0fe43e30c823369e559e5c482d6ed |
| 同method-summary.json | 421848e3ba4d62212195fbb4aa7d31bd923801a3fc5d59756361203e482ff2f4 |

## 同一Pへの復帰と固定S/Uの再計算

Rのfull14 net lossによる撤回後、同一Pへ復帰した。current state SHA 2666c54246852bf328be8b6293ff3402f011403fc27ad28113c5bcddaf923c99、R撤回decision SHA 1619f5487e92716434641797eb93971a4ad88e181ccc9f105d96f5d788b1e528。Pの全9file statとfull-content attestation87975545…が収集時と一致するため、Pの136例／42fold両目的を再利用し、追加GPU観測は行わない。

S=numerics/contracts/advanced_state、U=implementation/algorithm/workflow、全421 native／1,331,516座標、Euclidean cost1。選択LLはH≤.004なら0、それ以外min(.01,.25H)、選択margin0、他LL−.01／margin−.02、radius.025＋1e−6／16補正／変更100万上限を維持する。Pで全選択LL下限は+.01。新しい親・出力IDを用いた別計算で、R結果の再ラベルではない。

current-p-native-su-proposals-v1/plan.json SHA 44b9b81dfc9f3366ea1595e1f1b757319fb31f005cbf08c1659ed9b7220e1807を先に固定し、metadata preview後14:40:58.406 UTCに一度だけCPU計算を開始した。CPU8–15／BLAS2／budget300秒、grid・自動再試行なし。保存collectorのfinite source214b3c7e…と現在9a5fb90f…は異なるfingerprintであり、後者のserial/threads/seed設定とselected-word検査の追加を明示する。frame/objective計算は変更しておらず、同一実装だったと遡及しない。結果を次に記す。

### P起点S/UのCPU結果（実応答未測定）

83.9932秒で終了、process returncode0。各案1回、grid・再solve・追加GPU観測0。全入力source/hash/stat、P9file stat、current state2666c542…の継続一致を確認した。

| 提案 | BF16提案変更数 / tensor | P基準ΔL2 | 選択LL予測範囲 | 選択margin最小 | 全84予測制約 |
|---|---:|---:|---:|---:|---|
| S | 6,911 / 86 | 0.000389206341952 | 0.0100008198–0.0101706632 | 1.6246224e-06 | true / 未達0 |
| U | 9,448 / 95 | 0.000506870669541 | 0.0100060614–0.0164227278 | 1.3388874e-06 | true / 未達0 |

これは保存P基準のBF16提案値をGへ戻した線形予測で、checkpoint実体化や実モデル応答ではない。R起点の数値・specとは別である。S/Uともinstruction marginの低下を許容しており、平均はS−.0196964／U−.0185920、最小はS−.0199897／U−.0196840。Uのtool_call margin平均−.0120521、instruction correct-LL片fold−.0064745も保持する。選択目的の予測利得とこれらの損失を実際のgood NLL／margin、次いでnative ON生成の公式成績で確かめる。tiny L2、LL単独改善、制約trueを能力改善や採用と呼ばず、全領域zero-lossやCI有意差を新たな必須gateにもしない。

| P提案証拠（analysis/code90-resume-20261106/起点） | SHA256 |
|---|---|
| current-p-native-su-proposals-v1/plan.json | 44b9b81dfc9f3366ea1595e1f1b757319fb31f005cbf08c1659ed9b7220e1807 |
| current-p-native-su-proposals-v1/proposals/report.json | 2e1fb0001b7d939f01da185fcb57cade41250452c2aefdeeb1ea18fe0763ec0a |
| current-p-native-su-proposals-v1/solve-completion.json | b47611bc67ef82fc5dfa3ccc1aa112aeb2511341c4af733a056a20352925cc29 |
| current-p-native-su-proposals-v1/proposals/current-p-science-coupled-s.json | 168e4851418cbc209f47e50881bcf3fb235c51ce29ed50af525584a3aaf40c0d |
| current-p-native-su-proposals-v1/proposals/current-p-implementation-coupled-u.json | efd4bc3b23dc0608e90a96167393a4b73175656c17ba704b3856ed9da6454cad |
